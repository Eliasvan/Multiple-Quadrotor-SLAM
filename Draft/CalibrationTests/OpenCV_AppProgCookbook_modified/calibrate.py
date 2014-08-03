"""
    Code originates from:
        http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""
import numpy as np
import cv2
import glob



# prepare object points, like (0,0,0), (0,1,0), (0,2,0) ....,(5,7,0)
boardSize = (8, 6)
objp = np.zeros( (np.prod(boardSize), 3), np.float32 )
objp[:,:] = np.array([ map(float, [i, j, 0])
                        for i in range(boardSize[1])
                        for j in range(boardSize[0]) ])


# Arrays to store object points and image points from all the images.
objectPoints = [] # 3d point in real world space
imagePoints = [] # 2d points in image plane.

images = glob.glob('chessboards/chessboard*.jpg')
imageSize = (640, 480)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
            gray, boardSize )

    # If found, add object points, image points (after refining them)
    if ret == True:
        objectPoints.append(objp)

        cv2.cornerSubPix(
                gray, corners,
                (11,11), # window
                (-1,-1), # deadzone
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) ) # termination criteria
        imagePoints.append(corners.reshape(-1, 2))

        # Draw and display the corners
        cv2.drawChessboardCorners(
                img, boardSize, corners, ret )
        cv2.imshow('img', img)
        cv2.waitKey(100)


# Calibration
reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, imageSize )
print "cameraMatrix:\n", cameraMatrix
print "distCoeffs:\n", distCoeffs
print "reproj_error:\n", reproj_error


# Undistortion (note that we reuse images[6], this is only to get the 'roi')
img = cv2.imread(images[6])
cameraMatrix_new, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize,
        1 ) # all source image pixels retained in undistorted image

# undistort
mapX, mapY = cv2.initUndistortRectifyMap(
        cameraMatrix, distCoeffs,
        None, # optional rectification transformation
        cameraMatrix_new, imageSize,
        5 ) # type of the first output map
img_undistorted = cv2.remap(
        img, mapX, mapY, cv2.INTER_LINEAR )

# crop the image
x,y, w,h = roi
img_undistorted = img_undistorted[y:y+h, x:x+w]

cv2.imshow("Original Image", img);
cv2.imshow('Undistorted Image', img_undistorted)
cv2.waitKey()


# Re-projection Error
mean_error = np.zeros((1, 2))
square_error = np.zeros((1, 2))

for i in xrange(len(images)):
    imgp_reproj, jacob = cv2.projectPoints(
            objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs )
    error = imgp_reproj.reshape(-1, 2) - imagePoints[i]
    mean_error += abs(error).sum(axis=0) / np.prod(boardSize)
    square_error += (error**2).sum(axis=0) / np.prod(boardSize)

mean_error = cv2.norm(mean_error / len(images))
square_error = np.sqrt(square_error.sum() / len(images))

print "mean absolute error:", mean_error
print "square error:", square_error


# Close all windows
cv2.destroyAllWindows()
