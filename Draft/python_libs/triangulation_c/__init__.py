loaded = True

try:    # first, try to load the "triangulation_ext" module if it already exists, ...
    import triangulation_ext
except ImportError:    # ... otherwise build it
    from setup import main as setup_main
    try:
        setup_main()    # run "setup.py" to compile the "triangulation_ext" module
        import triangulation_ext    # retry import
    except:
        loaded = False

import numpy as np



from triangulation_ext import linear_LS_triangulation as linear_LS_triangulation_c
def linear_LS_triangulation(u1, P1, u2, P2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 3.0
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    
    # Cast to double if needed
    if np.finfo(u1.dtype).dtype != np.float64: u1 = u1.astype(np.float64)
    if np.finfo(u2.dtype).dtype != np.float64: u2 = u2.astype(np.float64)
    
    # Align the data of all arrays
    u1 = u1.reshape(u1.size).reshape(u1.shape)
    P1 = P1.reshape(P1.size).reshape(P1.shape)
    u2 = u2.reshape(u2.size).reshape(u2.shape)
    P2 = P2.reshape(P2.size).reshape(P2.shape)
    
    # Create array of triangulated points
    x = np.empty((len(u1), 3), dtype=np.float64)
    
    # Call the C function
    linear_LS_triangulation_c(u1, P1, u2, P2, x)
    
    return x, np.ones(len(u1), dtype=bool)


from triangulation_ext import iterative_LS_triangulation as iterative_LS_triangulation_c
def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.67
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.
    
    Additionally returns a status-vector to indicate outliers:
        True:  inlier
        False: outlier
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    
    # Cast to double if needed
    if np.finfo(u1.dtype).dtype != np.float64: u1 = u1.astype(np.float64)
    if np.finfo(u2.dtype).dtype != np.float64: u2 = u2.astype(np.float64)
    
    # Align the data of all arrays
    u1 = u1.reshape(u1.size).reshape(u1.shape)
    P1 = P1.reshape(P1.size).reshape(P1.shape)
    u2 = u2.reshape(u2.size).reshape(u2.shape)
    P2 = P2.reshape(P2.size).reshape(P2.shape)
    
    # Create array of triangulated points
    x = np.empty((len(u1), 3), dtype=np.float64)
    x_status = np.empty(len(u1), dtype=np.int32)
    
    # Call the C function
    iterative_LS_triangulation_c(u1, P1, u2, P2, tolerance, x, x_status)
    
    return x, x_status
