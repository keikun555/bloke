from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["bloke/optimize.pyx", "bloke/mcmc.pyx"]))
