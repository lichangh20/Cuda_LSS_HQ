from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantize_forward_easy',
    ext_modules=[
        CUDAExtension(name='quantize_forward_easy', sources=[
            'quantize_forward_easy.cpp',
            'quantize_forward_easy_kernel.cu',
        ], include_dirs=["/home/ubuntu/lichangh20/ANN_project_advance/qmatmul/cutlass/include"],
        extra_compile_args=["-std=c++17"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
