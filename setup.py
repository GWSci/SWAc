#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod setup file."""

import sys

try:
    from setuptools import setup
except ImportError:
    print 'Please install or upgrade setuptools or pip to continue'
    sys.exit(1)

import swacmod

setup(
    name='swacmod',
    packages=['swacmod'],
    package_data={'swacmod': ['input.yml']},
    description='Water accounting simulator.',
    long_description=open('README.rst').read(),
    download_url='https://github.com/AlastairBlack/swacmod/tarball/0.1',
    version=swacmod.__version__,
    url=swacmod.__url__,
    author=swacmod.__authors__,
    author_email=swacmod.__author_email__,
    license=swacmod.__license__,
    test_suite='tests.tests.EndToEndTests',
    keywords=['water management', 'water accounting'],
    install_requires=['Cython', 'numpy', 'psutil', 'python-dateutil', 'pytz',
                      'PyYAML'],
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Development Status :: 3 - Alpha',
                 'Topic :: Scientific/Engineering'])
