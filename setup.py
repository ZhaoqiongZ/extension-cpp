import os
import torch
import glob
import platform  
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

library_name = "sycl_extension"
py_limited_api = True

IS_WINDOWS = (platform.system() == 'Windows')

if IS_WINDOWS:
    cxx_args = [
        "/O2",                        
        "/std:c++17",                 
        "/DPy_LIMITED_API=0x03090000" ,
    ]
    sycl_args = ["/O2", "/std:c++17"] 
else:
    cxx_args = [
        "-O3",
        "-fdiagnostics-color=always", 
        "-DPy_LIMITED_API=0x03090000"
    ]
    sycl_args = ["-O3"]

extra_compile_args = {
    "cxx": cxx_args,
    "sycl": sycl_args
}

assert(torch.xpu.is_available()), "XPU is not available, please check your environment"

# Source files collection
this_dir = os.path.dirname(os.path.curdir)
extensions_dir = os.path.join(this_dir, library_name)
sources = list(glob.glob(os.path.join(extensions_dir, "*.sycl")))

# Construct extension
ext_modules = [
    SyclExtension(
        f"{library_name}._C",
        sources,
        extra_compile_args=extra_compile_args,
        py_limited_api=py_limited_api,
    )
]

setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["torch"],
    description="Simple Example of PyTorch Sycl extensions",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)