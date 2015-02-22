import numpy as np
import cv2



# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)

def linear_LS_triangulation(u, P, u1, P1):
    """
    Linear Least Squares based triangulation.
    
    (u, P) is the reference pair containing homogenous image coordinates (x, y) and the corresponding camera matrix.
    (u1, P1) is the second pair.
    
    u and u1 are matrices: amount of points equals #rows and should be equal for u and u1.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u)))
    
    # Initialize C matrices
    C = np.array(linear_LS_triangulation_C)
    C1 = np.array(linear_LS_triangulation_C)
    
    for i in range(len(u)):
        # Build C matrices, to visualize calculation structure of A and b
        C[:, 2] = u[i, :]
        C1[:, 2] = u1[i, :]
        
        # Build A matrix
        A[0:2, :] = C.dot(P[0:3, 0:3])    # C * R
        A[2:4, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        
        # Build b vector
        b[0:2, :] = C.dot(P[0:3, 3:4])    # C * t
        b[2:4, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return x.T.astype(output_dtype)


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)
iterative_LS_triangulation_tolerance = 1.e-6

def iterative_LS_triangulation(u, P, u1, P1):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    
    (u, P) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u1, P1) is the second pair.
    
    Additionally returns a status-vector to indicate outliers:
        True:  inlier
        False: outlier
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
    
    u and u1 are matrices: amount of points equals #rows and should be equal for u and u1.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.empty((4, len(u))); x[3, :].fill(1)    # create empty array of homogenous 3D coordinates
    x_status = np.zeros(len(u), dtype=int)    # default: mark every point as an outlier
    
    # Initialize C matrices
    C = np.array(iterative_LS_triangulation_C)
    C1 = np.array(iterative_LS_triangulation_C)
    
    for xi in range(len(u)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C[:, 2] = u[xi, :]
        C1[:, 2] = u1[xi, :]
        
        # Build A matrix:
        # [
        #     [ u .x * P [2,0] - P [0,0],    u .x * P [2,1] - P [0,1],    u .x * P [2,2] - P [0,2] ],
        #     [ u .y * P [2,0] - P [1,0],    u .y * P [2,1] - P [1,1],    u .y * P [2,2] - P [1,2] ],
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ]
        # ]
        A[0:2, :] = C.dot(P[0:3, 0:3])     # C * R
        A[2:4, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        
        # Build b vector:
        # [
        #     [ -(u .x * P [2,3] - P [0,3]) ],
        #     [ -(u .y * P [2,3] - P [1,3]) ],
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ]
        # ]
        b[0:2, :] = C.dot(P[0:3, 3:4])    # C * t
        b[2:4, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b *= -1
        
        # Init depths
        d = d1 = 1.
        
        for i in range(10):    # Hartley suggests 10 iterations at most
            # Solve for x vector
            cv2.solve(A, b, x[0:3, xi:xi+1], cv2.DECOMP_SVD)
            
            # Calculate new depths
            d_new = P[2, :].dot(x[:, xi])
            d1_new = P1[2, :].dot(x[:, xi])
            
            # Convergence criterium
            #print i, d_new - d, d1_new - d1, (d_new > 0 and d1_new > 0)    # TODO: remove
            if abs(d_new - d) <= iterative_LS_triangulation_tolerance and \
                    abs(d1_new - d1) <= iterative_LS_triangulation_tolerance:
                x_status[xi] = (d_new > 0 and d1_new > 0)    # points should be in front of both cameras
                if d_new <= 0: x_status[xi] -= 1    # TODO: remove
                if d1_new <= 0: x_status[xi] -= 2    # TODO: remove
                break
            
            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d_new
            A[2:4, :] *= 1 / d1_new
            b[0:2, :] *= 1 / d_new
            b[2:4, :] *= 1 / d1_new
            
            # Update depths
            d = d_new
            d1 = d1_new
    
    return x[0:3, :].T.astype(output_dtype), x_status



output_dtype = float

def set_triangl_output_dtype(output_dtype_):
    """
    Set the datatype of the triangulated 3D point positions.
    (Default is set to "float".)
    """
    global output_dtype
    output_dtype = output_dtype_
