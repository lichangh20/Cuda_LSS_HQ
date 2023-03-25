from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantize_forward_easy',
    ext_modules=[
        CUDAExtension(name='quantize_forward_easy', sources=[
            'quantize_forward_easy.cpp',
            'quantize_forward_easy_kernel.cu',
        ], include_dirs=["../cutlass-feature-2.10-updates_before_tagging/include"],
        extra_compile_args=["-std=c++17"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
