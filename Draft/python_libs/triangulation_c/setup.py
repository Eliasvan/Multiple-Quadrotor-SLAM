#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))    # path to "python_libs" directory

import convert_c_to_ext_lib



def main():
    # Create the "triangulation_ext" module from the "triangulation.c" source file;
    # disable openmp because cpu-usage x4 and perf only x1.5
    convert_c_to_ext_lib.create_ext_lib("triangulation.c", openmp=False)


if __name__ == "__main__":
    main()
