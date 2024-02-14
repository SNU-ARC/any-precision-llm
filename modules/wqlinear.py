import torch
import torch.nn as nn

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    exit('Please install any precision CUDA kernel extension from modules/kernels.')


class WQLinear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits=None, device=None):
        super().__init__()

        if supported_bits is None:
            supported_bits = [3, 4, 5, 6, 7, 8]
        if (not isinstance(supported_bits, list)) or (not supported_bits):
            raise RuntimeError('supported_bits must be a nonempty list.')
        if sorted(supported_bits) != supported_bits:
            raise RuntimeError('supported_bits must have sorted.')
        for bit in supported_bits:
            if bit < 3 or bit > 8:
                raise RuntimeError('Each element of supported_bits must range from 3 to 8.')

        self.in_features = in_features
        self.out_features = out_features
        self.supported_bits = supported_bits

        self.register_buffer(
            'qweight',
            torch.empty(
                (out_features, in_features // 32 * max(supported_bits)),
                dtype=torch.int32,
                device=device
            )
        )
        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty(
                    (out_features, 2 ** bit),
                    dtype=torch.float16,
                    device=device
                )
            )


    def forward(self, x, w_bits):
        if w_bits not in self.supported_bits:
            raise RuntimeError('Ensure that w_bits are contained within the supported_bits.')

        if x.shape[0] > 8:
            weight = dequant_kbit(self.qweight, self._buffers[f'lut{w_bits}'], w_bits)
            return torch.matmul(x, weight.T)
        else:
            return matmul_kbit(x, self.qweight, self._buffers[f'lut{w_bits}'], w_bits)
