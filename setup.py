#!/usr/bin/env/python

import numpy
import setuptools
from setuptools.command.build_py import build_py as _build_py
import os


conda_prefix = os.environ.get("CONDA_PREFIX", "/usr")  # fallback if not in conda


NLOP_LIB_DIRS = [
    '/opt/homebrew/lib'                     # MacOS BS
    #'/opt/homebrew/Cellar/nlopt/2.7.1/lib', # For ARM
    #'/usr/local/Cellar/nlopt/2.7.1/lib',    # For x86
    '/usr/lib',                             # system library path
    '/usr/local/lib',                       # custom lib path
    os.path.join(conda_prefix, 'lib'),
]

NLOP_INCLUDE_DIRS = [
    '/opt/homebrew/include'
    #'/opt/homebrew/Cellar/nlopt/2.7.1/include',    # For ARM
    #'/usr/local/Cellar/nlopt/2.7.1/include',       # For x86
    '/usr/include',                                # system includes (e.g. nlopt.hpp)
    '/usr/local/include',                          # custom install path
    'include',                                     # local project includes
    os.path.join(conda_prefix, 'include'),
]

INCLUDE_DIRS = NLOP_INCLUDE_DIRS + [numpy.get_include()]

spires = setuptools.Extension(  
    name='spires._core',
    sources=['spires/spires.i', 'spires/spires.cpp'],
    swig_opts=['-c++'],
    extra_compile_args=['-std=c++11'],
    library_dirs=NLOP_LIB_DIRS,    
    include_dirs=INCLUDE_DIRS,     
    libraries=['nlopt'],
    language='c++'
)


class build_py(_build_py):

    def run(self):
        """ 
        We need to overwrite run to make sure extension is built before getting copied
        """
        self.run_command("build_ext")
        return super().run()


setuptools.setup(        
    packages=setuptools.find_packages(),
    ext_modules=[spires],
)