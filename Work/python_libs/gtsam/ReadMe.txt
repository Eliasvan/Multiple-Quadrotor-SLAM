Python wrapper for GTSAM
========================

Dependencies
------------

- GTSAM
- Cython
- autowrap (https://github.com/uweschmitt/autowrap)

Tested with autowrap 0.6.1, Cython 0.21.1, Python 2.7.5 and GTSAM 3.2.1.


Installation
------------

The C Python "gtsam_ext.so" wrapper is compiled automatically
if it is not yet present and this "gtsam" package is imported.

To avoid compilation at runtime, or to recompile the library,
execute the "setup.py" script.
Alternatively, you can also run:
$ rm -f gtsam.so && autowrap --out gtsam.pyx gtsam_defs.pxd && python setup.py build_ext --inplace

After installation, run the "test.py" script to check for errors,
normally it should execute without errors.


Usage
-----

Take a look at "test.py" for some examples with Python.

I recommend to follow the original GTSAM tutorial, and use the GTSAM reference API,
and then try to use the corresponding function in Python.
The wrapped functions are defined in "gtsam_defs.pxd".

Tip: The interactive Python console can show the available wrapped functionality, e.g.:
$ python
>>> from gtsam import *    # assuming we're in the "python_libs" directory
>>> help(ISAM2)


TODO
----

- Proper conversion between Numpy arrays and Eigen matrices/vectors.

- Some functionality of GTSAM is unavailable due to autowrap bugs preventing to wrap certain things:
    most importantly: a reference variable as returnvalue is not supported:
    https://github.com/uweschmitt/autowrap/issues/31

- Convert some of the original GTSAM examples to Python.
