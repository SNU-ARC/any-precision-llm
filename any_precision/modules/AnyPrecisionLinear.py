import torch
import torch.nn as nn

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    matmul_kbit, dequant_kbit = None, None


class AnyPrecisionLinear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, bias=True, precisions=None, device=None,
                 dtype=None):
        super().__init__()
        if dequant_kbit is None or matmul_kbit is None:
            raise ModuleNotFoundError('Please install any precision CUDA kernel extension from modules/kernels.')
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')
        if dtype is not None and dtype != torch.float16:
            raise RuntimeError('Only float16 is supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)
        self.supported_bits = supported_bits

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

    def prune_precisions(self):
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')

    def forward(self, x, **kwargs):
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision

        if x.numel() // x.shape[-1] > 8:
            weight = dequant_kbit(self.qweight, self._buffers[f'lut{w_bits}'], w_bits)
            x = torch.matmul(x, weight.T)
        else:
            x = matmul_kbit(x, self.qweight, self._buffers[f'lut{w_bits}'], w_bits)

        if self.bias is not None:
            x += self.bias

        return x

    def set_precision(self, precision):
        if precision not in self.precisions:
            raise RuntimeError('Ensure that w_bits are contained within the supported_bits.')

        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
