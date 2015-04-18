from textwrap import dedent
import numpy as np
import cv2



def grid_objp(boardSize):
    """
    Generate 3D points on a grid, in order: (0,0,0), (0,1,0), (0,2,0),  ..., (5,7,0).
    This can be useful for points on chessboard-corners.
    
    "boardSize[0]" is used as Y-axis, "boardSize[1]" as X-axis.
    """
    objp = np.zeros((np.prod(boardSize), 3), dtype=np.float32)
    
    objp[:, :] = np.array([ map(float, [i, j, 0])
                            for i in range(boardSize[1])
                            for j in range(boardSize[0]) ])
    
    return objp


def save_camera_intrinsics(filename, cameraMatrix, distCoeffs, imageSize):
    """
    Save camera intrinsics (defined by "cameraMatrix", "distCoeffs", and "imageSize" (w, h))
    to "filename".
    
    See OpenCV's doc about the format of "cameraMatrix" and "distCoeffs".
    """
    
    out = """\
    # cameraMatrix, distCoeffs, imageSize =
    
    %s, \\
    \\
    %s, \\
    \\
    %s
    """
    out = dedent(out) % (repr(cameraMatrix), repr(distCoeffs), repr(imageSize))
    open(filename, 'w').write(out)


def load_camera_intrinsics(filename):
    """
    Load camera intrinsics ("cameraMatrix", "distCoeffs", and "imageSize" (w, h))
    from "filename".
    
    See OpenCV's doc about the format of "cameraMatrix" and "distCoeffs".
    """
    from numpy import array
    
    cameraMatrix, distCoeffs, imageSize = \
            eval(open(filename, 'r').read())
    
    return cameraMatrix, distCoeffs, imageSize


def undistort_image(img, cameraMatrix, distCoeffs, imageSize):
    """
    Undistort image "img",
    shot with a camera with intrinsics ("cameraMatrix", "distCoeffs", and "imageSize" (w, h)).
    Apart from the undistorted image, a region-of-interest will also be returned.
    
    See OpenCV's doc about the format of "cameraMatrix" and "distCoeffs".
    """
    
    # Refine cameraMatrix, and calculate RegionOfInterest
    cameraMatrix_new, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix, distCoeffs, imageSize,
            1 )    # all source image pixels retained in undistorted image

    # Undistort
    mapX, mapY = cv2.initUndistortRectifyMap(
            cameraMatrix, distCoeffs,
            None,    # optional rectification transformation
            cameraMatrix_new, imageSize,
            5 )    # type of the first output map (CV_32FC1)
    img_undistorted = cv2.remap(
            img, mapX, mapY, cv2.INTER_LINEAR )

    # Crop the image
    x,y, w,h = roi
    img_undistorted = img_undistorted[y:y+h, x:x+w]
    
    return img_undistorted, roi


def reprojection_error_ext(objp, imgp, cameraMatrix, distCoeffs, rvecs, tvecs):
    """
    Returns the mean absolute error, and the RMS error of the reprojection
    of 3D points "objp" on the images from a camera
    with intrinsics ("cameraMatrix", "distCoeffs") and poses ("rvecs", "tvecs").
    The original 2D points should be given by "imgp".
    
    See OpenCV's doc about the format of "cameraMatrix", "distCoeffs", "rvec" and "tvec".
    """
    
    mean_error = np.zeros((1, 2))
    square_error = np.zeros((1, 2))
    n_images = len(imgp)

    for i in xrange(n_images):
        imgp_reproj, jacob = cv2.projectPoints(
                objp[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs )
        error = imgp_reproj.reshape(-1, 2) - imgp[i]
        mean_error += abs(error).sum(axis=0) / len(imgp[i])
        square_error += (error**2).sum(axis=0) / len(imgp[i])

    mean_error = cv2.norm(mean_error / n_images)
    square_error = np.sqrt(square_error.sum() / n_images)
    
    return mean_error, square_error


def reprojection_error(objp, imgp, cameraMatrix, distCoeffs, rvec, tvec):
    """
    Minimalist version of "reprojection_error_ext()",
    only returns the RMS error of one image.
    """
    
    imgp_reproj, jacob = cv2.projectPoints(
            objp, rvec, tvec, cameraMatrix, distCoeffs )
    return np.sqrt(((imgp_reproj.reshape(-1, 2) - imgp)**2).sum() / float(len(imgp))), imgp_reproj
