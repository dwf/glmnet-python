glmnet wrappers for Python
==========================

Very much a work in progress.

In order to get double precision working without modifying Friedman's code,
some compiler trickery is required. The wrappers have been written such that
everything returned is expected to be a ``real*8`` i.e. a double-precision
floating point number, and unfortunately the code is written in a way 
Fortran is often written with simply ``real`` specified, letting the compiler
decide on the appropriate width. ``f2py`` assumes ``real`` are always 4 byte/
single precision, hence the manual change in the wrappers to ``real*8``, but
that change requires the actual Fortran code to be compiled with 8-byte reals,
otherwise bad things will happen (the stack will be blown, program will hang 
or segfault, etc.).

AFAIK, this package requires  ``gfortran`` to build. ``g77`` will not work as
it does not support ``-fdefault-real-8``.

The way to get this to build properly is:

::

    python setup.py config_fc --fcompiler=gnu95 \
        --f77flags='-fdefault-real-8' \
        --f90flags='-fdefault-real-8' build

The ``--fcompiler=gnu95`` business may be omitted if gfortran is the only 
Fortran compiler you have installed, but the compiler flags are essential.
