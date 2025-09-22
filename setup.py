from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ImageAdaptive3DLUT",
    version="0.1.0",
    packages=find_packages(),   
    ext_modules=[
        CUDAExtension(
            name="trilinear_c__ext",                  
            sources=[
                "trilinear_cpp/src/trilinear.cpp",
                "trilinear_cpp/src/trilinear_cuda.cpp",
                "trilinear_cpp/src/trilinear_kernel.cu",
            ],
            include_dirs=["trilinear_cpp"],
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
