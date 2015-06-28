from math import sqrt, sin, cos, acos, pi
import numpy as np
import numpy.linalg as LA
try:
    import cv2
except ImportError:
    print ("Warning: can't load module \"cv2\", required for some functions of \"transforms\" module.")



""" Quaternion transformations """


def unit_quat():
    """
    Return a unit quaternion (qx, qy, qz, qw) = (0, 0, 0, 1).
    """
    return np.eye(4)[3]


def mult_quat(q2, q1):
    """
    Multiply two quaternions: q2 * q1.
    
    Equivalent of accumulating new rotation 'q2' to original 'q1' (in that order).
    """
    qwt = np.zeros((4, 1))
    
    qwt[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[2]*q2[1] - q1[1]*q2[2]    # x component
    qwt[1] = q1[1]*q2[3] - q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2]    # y component
    qwt[2] = q1[2]*q2[3] + q1[1]*q2[0] - q1[0]*q2[1] + q1[3]*q2[2]    # z component
    qwt[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]    # w component
    
    return qwt


def conj_quat(qwt):
    """
    Return the conjugate quaternion.
    """
    qwt_conj = np.array(qwt)
    
    qwt_conj[0:3] *= -1
    
    return qwt_conj


def inv_quat(qwt):
    """
    Return inverse quaternion.
    """
    return conj_quat(qwt) / (qwt**2).sum()


def delta_quat(q2, q1):
    """
    Return the delta quaternion q = q2 * q1^-1.
    
    Equivalent of rotation 'q2' w.r.t. 'q1',
    thus accumulating 'q' to 'q1' yields 'q2'.
    """
    return mult_quat(q2, inv_quat(q1))


""" Quaternions operating on points """


def apply_quat_on_point(qwt, point):
    """
    Apply quaternion 'qwt' rotation on 3D point 'point' and return resulting 3D point.
    """
    qp = np.zeros((4, 1))
    
    qp[0:3] = point.reshape(3, 1)
    qp_result = mult_quat(qwt, mult_quat(qp, conj_quat(qwt)))
    
    return qp_result[0:3]


""" Conversions between quaternions and other representations """


def quat_from_rvec(rvec):
    """
    Convert axis-angle represented 'rvec' to a quaternion,
    where 'rvec' is a 3x1 numpy array.
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
    axis = sign * rvec    # make the dominant axis positive
    if angle > 0:
        axis /= angle
    angle *= sign

    if abs(angle) > pi:    # abs(angle) should be <= pi
        angle -= np.sign(angle) * 2*pi

    return axis, angle


""" Axis-angle transformations """


def delta_rvec(r2, r1):
    """
    Return r = r2 '-' r1,
    where '-' denotes the difference between rotations.
    """
    return rvec_from_quat(delta_quat(
            quat_from_rvec(r2),
            quat_from_rvec(r1) ))


""" Perspective transformations """


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


def P_inv(P):
    """
    Return the inverse of a 4x4 P matrix (projection matrix).
    
    Only use if higher accuracy is needed, it's 4 times slower than cv2.invert(P)[0].
    """
    
    R = LA.inv(P[0:3, 0:3])
    t = -R.dot(P[0:3, 3:4])
    
    return P_from_R_and_t(R, t)


def delta_P(P2, P1):
    """
    Return P = P2 '-' P1,
    where '-' denotes the difference between perspective transformations.
    More accurately: P2 = P * P1, solved for "P".
    """
    P = np.empty((4, 4))
    
    cv2.solve(P1.T, P2.T, P, cv2.DECOMP_SVD)
    P = P.T
    P[3, 0:3] = 0    # make sure these are zero
    P[3, 3] = 1    # make sure this is one
    
    return P


def project_points(points, K, image_size, P, round=True):
    """
    Return the 2D projections of 3D points array via 4x4 P camera projection matrix using 3x3 K camera intrinsics matrix,
    additionally return a corresponding status vector:
        1 if point is in front of camera and inside view with size 'image_size' [height, width],
        otherwise 0.
    If 'round' is True, the projected points will become (nearest) integers.
    """
    points_nrm = np.empty((len(points), 4))
    points_nrm[:, 0:3] = points
    points_nrm[:, 3].fill(1)

    points_proj = points_nrm .dot (P[0:3, :].T) .dot (K.T)
    points_proj[:, 0:2] /= points_proj[:, 2:3]
    
    status = (points_proj[:, 2] > 0)
    if image_size != None:
        inside_image = np.logical_and(
                np.logical_and((0 <= points_proj[:, 0]), (points_proj[:, 0] < image_size[1])),
                np.logical_and((0 <= points_proj[:, 1]), (points_proj[:, 1] < image_size[0])) )
        status = np.logical_and(status, inside_image)
    
    points_proj = points_proj[:, 0:2]
    if round:
        points_proj = np.rint(points_proj).astype(int)
    
    return points_proj, status


def projection_depth(points, P):
    """
    Return the (Z) depth of the projections of 3D points array via 4x4 P camera projection matrix.
    """
    points_nrm = np.empty((len(points), 4))
    points_nrm[:, 0:3] = points
    points_nrm[:, 3].fill(1)

    points_depth = points_nrm .dot (P[2:3, :].T)

    return points_depth.reshape(-1)


""" Conversions between camera pose projection matrices P and other representations """


def P_from_rvec_and_tvec(rvec, tvec):
    """
    Return the 4x4 P camera projection matrix from OpenCV's camera's 'rvec' and 'tvec'.
    """
    return P_from_R_and_t(cv2.Rodrigues(rvec)[0], tvec)


def P_from_pose_TUM(q, l):
    """
    Return the 4x4 P camera projection matrix, converted from a camera pose in TUM format ('q', 'l').
    
    For more info on the TUM format, see: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    'q' and 'l' stand for "quaternion" and "location" respectively.
    """
    M = np.eye(4)
    
    rvec = rvec_from_quat(q)
    M[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    M[0:3, 3] = l
    
    # Take the inverse, to obtain the transformation matrix that projects points
    # from the world axis-system to the camera axis-system
    P = P_inv(M)
    
    return P


def pose_TUM_from_P(P):
    """
    Return the camera pose in TUM format, converted from 4x4 camera projection matrix 'P'.
    
    For more info on the TUM format, see: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    'q' and 'l' stand for "quaternion" and "location" respectively.
    """
    
    # Take the inverse, to obtain the transformation matrix that projects points
    # from the camera axis-system to the world axis-system
    M = P_inv(P)
    
    R = cv2.Rodrigues(M[0:3, 0:3])[0]
    q = quat_from_rvec(R)
    l = M[0:3, 3:4]
    
    return q, l
