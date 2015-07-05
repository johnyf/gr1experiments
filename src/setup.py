import os
from distutils.core import setup
from Cython.Build import cythonize

os.environ['CC'] = 'gcc'

setup(
    ext_modules=cythonize('buddy.pyx', gdb_debug=True))
