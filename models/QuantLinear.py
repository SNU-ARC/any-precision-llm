import numpy as np
import torch
import torch.nn as nn
try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    exit('Please install any precision CUDA kernel extension from modules/kernels.')

class AnyprecisionLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias,
                 supported_bits=None,
                 w_bits=None,
                 dev="cpu",
                 dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if supported_bits is None:
            supported_bits = [3, 4, 5, 6, 7, 8]

        self.supported_bits = supported_bits

        if w_bits is None:
            w_bits=supported_bits[0]
        elif w_bits not in self.supported_bits:
            raise RuntimeError('Ensure that w_bits are contained within the supported_bits.')

        self.w_bits = w_bits

        total_bits = [3, 4, 5, 6, 7, 8]
        self.register_buffer("qweight", torch.zeros((out_features,in_features),dtype=torch.int32,device="cpu"))
        for bit in total_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty(
                    (out_features, 2 ** bit),
                    dtype=torch.float16,
                    device="cpu"
                )
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    def refine_bits(self):
        supported_bits = self.supported_bits

    def forward(self, x, w_bits=None):
        if w_bits is None:
            w_bits = self.w_bits

        if w_bits not in self.supported_bits:
            raise RuntimeError('Ensure that w_bits are contained within the supported_bits.')

        if x.shape[0] > 8:
            weight = dequant_kbit(self.qweight, self._buffers[f'lut{w_bits}'], w_bits)
            return torch.matmul(x, weight.T)
        else:
            return matmul_kbit(x, self.qweight, self._buffers[f'lut{w_bits}'], w_bits)
