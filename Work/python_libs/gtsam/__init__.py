try:    # first, try to load the "gtsam_ext" module if it already exists, ...
    from gtsam_ext import *
except ImportError:    # ... otherwise build it
    from setup import main as setup_main
    setup_main()    # run "setup.py" to compile the "gtsam_ext" module
    from gtsam_ext import *    # retry import
