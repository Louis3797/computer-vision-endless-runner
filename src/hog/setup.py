# setup.py

from distutils.core import setup
from Cython.Build import cythonize
import sys
import os
import pybind11



lib_folder = os.path.join(sys.prefix, 'libhog')



setup(ext_modules=cythonize("hog.pyx"))
