#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "python_libs"))

import convert_c_to_ext_lib



def main():
    # Create the "test_ext" module from the "test.c" source file
    convert_c_to_ext_lib.create_ext_lib("test.c")


if __name__ == "__main__":
    main()
