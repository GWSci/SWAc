#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup for Cython module."""

# Standard Library
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

# Third Party Libraries
import numpy
from Cython.Build import cythonize

EXTENSIONS = [Extension("model", ["model.pyx"], extra_compile_args=["-w"])]

setup(
    include_dirs=[numpy.get_include()],
    package_dir={"swacmod": ""},
    ext_modules=cythonize(EXTENSIONS),
)
