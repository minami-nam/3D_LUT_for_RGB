from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# 공통 include 경로
incs = ["src"]

ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            name="trilinear_c._ext",           # models.py import 구조에 맞춤
            sources=[
                "src/trilinear_cuda.cpp",
                "src/trilinear_kernel.cu",
            ],
            include_dirs=incs,
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-O2",
                    "--expt-relaxed-constexpr",
                    "-allow-unsupported-compiler",
                ],
            },
        )
    )
else:
    ext_modules.append(
        CppExtension(
            name="trilinear_c._ext",
            sources=["src/trilinear.cpp"],
            include_dirs=incs,
            extra_compile_args=["-std=c++17"],
        )
    )

setup(
    name="trilinear_c",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["trilinear_c"],
)
