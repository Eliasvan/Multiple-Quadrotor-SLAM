#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function    # Python 3 compatibility

try:    # first, try to load the "test_ext" module if it already exists, otherwise build it
    import test_ext
except ImportError:
    from setup import main as setup_main
    setup_main()    # run "setup.py" to compile the "test_ext" module
    import test_ext    # retry import

import numpy as np



def main():
    # Setup arguments
    a = 1.0
    b = 3.0
    buf = np.array([[0, -1],    # note that we provide a bigger matrix: 4x2 instead of 3x2,
                    [0, +1],    # so only the first 3x2 matrix should be altered by func1()
                    [-1, 0],
                    [+1, 0]], dtype=np.int32)
    print ("a:", a)
    print ("b:", b)
    print ("buf:")
    print (buf)
    print ()

    # Run the functions defined in "test.c"
    print ("Running func2()")
    ret = test_ext.func2()
    print ("=> function returned:", ret)
    print ()
    print ("Running func1()")
    ret = test_ext.func1(a, b, buf)
    print ("=> function returned:", ret)
    print ()
    print ("Running func2()")
    ret = test_ext.func2()
    print ("=> function returned:", ret)
    print ()

    # Show modified buf
    print ("buf:")
    print (buf)
    print ()

    # Run again and notice the counter increment at func2()
    print ("Running func1()")
    ret = test_ext.func1(a, b, buf)
    print ("=> function returned:", ret)
    print ()
    print ("Running func2()")
    ret = test_ext.func2()
    print ("=> function returned:", ret)


if __name__ == "__main__":
    main()
