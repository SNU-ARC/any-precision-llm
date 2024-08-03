import torch

from setuptools import setup
from torch.utils import cpp_extension

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

setup(
    name="any_precision_ext",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "any_precision_ext", ["main.cu"],
            extra_compile_args={'nvcc': [f'-DCUDA_ARCH={cuda_arch}']},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
