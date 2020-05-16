#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'nevin',
      version = __version__,
      python_requires='>3.5.2',
      description = 'k-Nearest Neighbor Evidence estimator',
      requires = ['numpy', 'matplotlib', 'scipy', 'emcee'],
      provides = ['nevin'],
      packages = ['nevin']
      )
