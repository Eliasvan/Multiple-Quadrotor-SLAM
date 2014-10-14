from math import sqrt, sin, cos, acos, pi
import numpy as np
import numpy.linalg as LA



### Quaternion transformations

def mult_quat(q2, q1):
    """
    Multiply two quaternions: q2 * q1.
    
    Equivalent of accumulating new rotation 'q2' to original 'q1' (in that order).
    """
    qwt = np.zeros((4, 1))
    
    qwt[0] = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]    # x component
    qwt[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0]    # y component
    qwt[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3]    # z component
    qwt[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]    # w component
    
    return qwt

def inv_quat(qwt):
    """
    Return inverse quaternion.
    """
    qwt_inv = np.array(qwt)
    
    qwt_inv[0:3] *= -1
    
    return qwt_inv / (qwt**2).sum()

def delta_quat(q2, q1):
    """
    Return the delta quaternion q = q1^-1 * q2.
    
    Equivalent of rotation 'q2' w.r.t. 'q1',
    thus accumulating 'q' to 'q1' yields 'q2'.
    """
    return mult_quat(q2, inv_quat(q1))


### Conversions between quaternions and other representations

def quat_from_rvec(rvec):
    """
    Convert axis-angle represented 'rvec' to a quaternion.
    """
    qwt = np.zeros((4, 1))

    angle = LA.norm(rvec)    # magnitude of 'angular velocity'
    if angle > 0:
        qwt[0:3] = rvec * sin(angle/2) / angle
        qwt[3] = cos(angle/2)
    else:    # to avoid illegal expressions
        qwt[3] = 1.
    
    return qwt

def rvec_from_quat(qwt):
    """
    Convert quaternion to axis-angle representation.
    
    Source: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
    """
    if qwt[3] > 1:
        qwt /= LA.norm(qwt)    # if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised
    
    angle = 2 * acos(qwt[3])
    
    s = sqrt(1 - qwt[3]**2)    # assuming quaternion normalised then w is less than 1, so term always positive.
    if s < 0.001:    # test to avoid divide by zero, s is always positive due to sqrt
        # If 's' close to zero then direction of axis not important
        rvec = np.zeros((3, 1))    # it is important that axis is normalised, so replace with x=1; y=z=0;
        rvec[0] = 1
    else:
        rvec = qwt[0:3] / s    # normalize axis
    
    return rvec * angle

def axis_and_angle_from_rvec(rvec):
    """
    Return the axis vector and angle of the axis-angle represented 'rvec'.
    """
    angle = LA.norm(rvec)
    
    sign = np.sign(rvec[abs(rvec).argmax()])
    axis = sign * rvec / angle    # make the dominant axis positive
    angle *= sign

    if abs(angle) > pi:    # abs(angle) should be <= pi
        angle -= np.sign(angle) * 2*pi

    return axis, angle


### Axis-angle transformations

def delta_rvec(r2, r1):
    """
    Return r = r2 '-' r1,
    where '-' denotes the difference between rotations.
    """
    return rvec_from_quat(delta_quat(
            quat_from_rvec(r2),
            quat_from_rvec(r1) ))


### Perspective transformations

def P_from_R_and_t(R, t):
    """
    Return the 4x4 P matrix from 3x3 R matrix and 3x1 t vector, as:
        [    R    | t ]
        [---------+---]
        [ 0  0  0 | 1 ]
    """
    P = np.eye(4)
    
    P[0:3, 0:3] = R
    P[0:3, 3:4] = t
    
    return P
