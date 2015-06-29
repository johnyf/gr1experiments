import os
from distutils.core import setup
from Cython.Build import cythonize

os.environ['CC'] = "gcc"

setup(
    ext_modules = cythonize("simple.pyx")
)
