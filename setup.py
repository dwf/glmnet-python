from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
import os, sys

fflags= '-fdefault-real-8 -ffixed-form -g'
config = Configuration(
    'glmnet',
    parent_package=None,
    top_path=None,
    f2py_options='--f77flags=\'%s\' --f90flags=\'%s\'' % (fflags, fflags)
)

f_sources = ['src/glmnet.pyf','src/glmnet.f']

config.add_extension(name='_glmnet',sources=f_sources)
config_dict = config.todict()
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(version='1.1-5',
          description='Python wrappers for the GLMNET package',
          author='David Warde-Farley',
          author_email='dwf@cs.toronto.edu',
          url='github.com/dwf/glmnet-python',
          license='GPL2',
          requires=['NumPy (>= 1.3)'],
          **config_dict)

