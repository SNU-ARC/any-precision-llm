from setuptools import setup
from torch.utils import cpp_extension


setup(
    name="any_precision_ext",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "any_precision_ext", ["main.cu"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
