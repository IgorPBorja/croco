# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import sys

def parse_extra_nvcc_args() -> list[str]:
    """
    Parses extra args (to be forwarded to nvcc on compilation)
    Those args are prefix by the '--custom=' prefix, like for e.g
    "--custom=-DMY_MACRO"

    It then modifies the sys.argv variable to remove
    these extra arguments, since they would probably
    break setuptools.
    """
    extra_nvcc_args = []
    processed_argv = []

    for raw_arg in sys.argv[1:]:
        if raw_arg.startswith("--custom="):
            nvcc_arg = raw_arg.removeprefix("--custom=")
            print(f"Found nvcc arg '{nvcc_arg}'")
            extra_nvcc_args.append(nvcc_arg)
        else:
            processed_argv.append(raw_arg)
    sys.argv[1:] = processed_argv
    return extra_nvcc_args

extra_nvcc_args = parse_extra_nvcc_args()

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# all_cuda_archs = [
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    # '-gencode', 'arch=compute_86,code=sm_86'
# ]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=(
                        ['-O3','--ptxas-options=-v',"--use_fast_math"] +
                        all_cuda_archs + extra_nvcc_args + ["-std=c++17"]
                    ), 
                    cxx=['-O3', "-std=c++17"])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
