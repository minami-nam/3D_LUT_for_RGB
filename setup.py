from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ImageAdaptive3DLUT",
    version="0.1",
    packages=find_packages(),   # trilinear_c 패키지를 자동으로 찾습니다.
    ext_modules=[
        CUDAExtension(
            name="trilinear_c._ext",                  # trilinear_c/_ext 모듈로 설치
            sources=[
                "trilinear_cpp/src/trilinear_cuda.cpp",
                "trilinear_cpp/src/trilinear_kernel.cu",
            ],
            include_dirs=["trilinear_cpp/src"],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-O2",
                    "--expt-relaxed-constexpr",
                    "-allow-unsupported-compiler",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
