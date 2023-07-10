# -*- coding: utf-8 -*-

# Standard Library
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

# Third Party Libraries
import numpy

EXTENSIONS = [Extension("model",
                        ["cymodel.pyx"],
                        extra_compile_args=["-w",
                                            "-mtune=native",
                                            "-flto",
                                            "-march=native",
                                            "-O2",
                                            "-funroll-loops"],
                        extra_link_args=["-flto",
                                         "-O2"])]

setup(
    include_dirs=[numpy.get_include()],
    package_dir={"swacmod": ""},
)
