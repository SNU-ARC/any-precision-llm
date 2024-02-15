from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="any_precision_utils",
    version="0.0.1",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "any_precision_utils", ["utils.cpp"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
