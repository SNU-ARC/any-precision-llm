import torch
import torch.nn as nn
try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    exit('Please install any precision CUDA kernel extension from modules/kernels.')

class AnyPrecisionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, supported_bits=None, device=None, dtype=None):
        super().__init__()
        if supported_bits is None:
            supported_bits = [3, 4, 5, 6, 7, 8]
        if not isinstance(supported_bits, list):
            raise RuntimeError('supported_bits must be a list of integers.')
        if dtype is not None and dtype != torch.float16:
            raise RuntimeError('Only float16 is supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.supported_bits = supported_bits
        self.default_bit = max(self.supported_bits)

        self.register_buffer(
            'qweight',
            torch.empty(
                (8, out_features, in_features // 32),
                dtype=torch.int32,
                device=device
            )
        ) # size of buffer refined later

        for bit in [3, 4, 5, 6, 7, 8]:
            self.register_buffer(
                f'lut{bit}',
                torch.empty(
                    (out_features, 2 ** bit),
                    dtype=torch.float16,
                    device=device
                )
            ) # unsupported lut table will removed later

        if bias is not None:
            self.register_buffer(
                "bias",
                torch.empty(
                    (out_features, ),
                    dtype=torch.float16,
                    device=device
                ),
            )
        else:
            self.bias = None

    def refine_bits(self):
        self.qweight = self.qweight[:max(self.supported_bits)]

    def forward(self, x, w_bits=None):
        if w_bits is None:
            w_bits = self.default_bit

        if w_bits not in self.supported_bits:
            raise RuntimeError('Ensure that w_bits are contained within the supported_bits.')

        if x.numel() // x.shape[-1] > 8:
            weight = dequant_kbit(self.qweight, self._buffers[f'lut{w_bits}'], w_bits)
            x = torch.matmul(x, weight.T)
        else:
            x = matmul_kbit(x, self.qweight, self._buffers[f'lut{w_bits}'], w_bits)

        if self.bias is not None:
            x += self.bias

        return x

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
