from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantize_grad_input_speed',
    ext_modules=[
        CUDAExtension(name='quantize_grad_input_speed', sources=[
            'quantize_grad_input_speed.cpp',
            'quantize_grad_input_speed_kernel.cu',
        ], include_dirs=["/root/autodl-tmp/lichangh20/cutlass-feature-2.10-updates_before_tagging/include"],
        extra_compile_args=["-std=c++17"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
