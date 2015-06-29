#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import pkg_resources

data_dir = pkg_resources.resource_filename("autowrap", "data_files")

module_name = "gtsam_ext"
pxd_file = "gtsam_defs.pxd"
ext = Extension(module_name,
                sources=["%s.cpp" % module_name],
                language="c++",
                include_dirs=[data_dir],
                libraries=["gtsam"]
               )

def main():
    # Remove pyx, cpp and library, to force recompilation
    for f in os.listdir('.'):
        filename, extension = os.path.splitext(f)
        if filename == module_name and extension in (".pyx", ".cpp", ".so", ".dylib", ".dll") and os.path.isfile(f):
            os.remove(f)
    
    # Generate cpp from pxd file
    from autowrap.Main import _main as autowrap_main
    args = [pxd_file, "--out", "%s.pyx" % module_name]
    autowrap_main(args)
    
    # Generate library from cpp file
    argv = list(sys.argv)    # backup
    sys.argv[1:] = ["build_ext", "--inplace"]
    setup(cmdclass={"build_ext": build_ext},
          name=module_name,
          ext_modules=[ext])
    sys.argv[:] = argv    # restore

if __name__ == "__main__":
    main()
