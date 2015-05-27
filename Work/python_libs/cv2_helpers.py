import numpy as np
import cv2

import transforms as trfm



def imshow(title, img):
    """
    Use this combined with calling 'cv2.startWindowThread()' at the start of your program
    to allow to afterwards close windows in the middle of runtime
    (by calling 'cv2.destroyAllWindows()').
    
    Don't use this methodology if you want realtime key-captures.
    """
    cv2.namedWindow(title)
    cv2.imshow(title, img)

rgb = cv2.cv.CV_RGB

# Some wrapper functions to make some calls more concise
def line(img, p1, p2, col, *args, **kwargs):
    return cv2.line(img, tuple(p1), tuple(p2), col, *args, **kwargs)
def rectangle(img, p1, p2, col, *args, **kwargs):
    return cv2.rectangle(img, tuple(p1), tuple(p2), col, *args, **kwargs)
def circle(img, c, rad, col, *args, **kwargs):
    return cv2.circle(img, tuple(c), rad, col, *args, **kwargs)
def putText(img, txt, p, fF, fS, col, *args, **kwargs):
    return cv2.putText(img, txt, tuple(p), fF, fS, col, *args, **kwargs)
def Rodrigues(rvec_or_R):
    return cv2.Rodrigues(rvec_or_R)[0]    # only output R or rvec, not the jacobian
def invert(matrix):
    return cv2.invert(matrix)[1]    # only output the result, not the status; use with care
def goodFeaturesToTrack(img, to_add, corner_quality_level, corner_min_dist, *args, **kwargs):
    if to_add == 0:    # work around undocumented behavior of OpenCV, don't add new corners when to_add == 0
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.goodFeaturesToTrack(img, to_add, corner_quality_level, corner_min_dist, *args, **kwargs).reshape((-1, 2))

def format3DVector(v):
    return "[ %.3f  %.3f  %.3f ]" % tuple(v)


def drawKeypointsAndMotion(img2, points1, points2, color):
    """
    Returns a new image with vectors from "points1" to "points2", and keypoints on "img2".
    The motion is colored with "color".
    """
    img = cv2.drawKeypoints(img2, [cv2.KeyPoint(p[0],p[1], 7.) for p in points2], color=rgb(0,0,255))
    for p1, p2 in zip(points1, points2):
        line(img, p1, p2, color)
    return img


def drawAxisSystem(img, cameraMatrix, distCoeffs, rvec, tvec, scale=4.):
    """
    Draws an axis-system on image "img" of which the camera has
    the intrinsics ("cameraMatrix", "distCoeffs") and pose ("rvec", "tvec").
    The scale of the axis-system is set by "scale".
    
    See OpenCV's doc about the format of "cameraMatrix", "distCoeffs", "rvec" and "tvec".
    """
    
    # Define world object-points
    axis_system_objp = np.array([ [0., 0., 0.],      # Origin (black)
                                  [1., 0., 0.],      # X-axis (red)
                                  [0., 1., 0.],      # Y-axis (green)
                                  [0., 0., 1.] ])    # Z-axis (blue)
    axis_system_objp *= scale
    
    # Project the object-points on the camera
    imgp_reproj, jacob = cv2.projectPoints(
            axis_system_objp, rvec, tvec, cameraMatrix, distCoeffs )
    origin, xAxis, yAxis, zAxis = np.rint(imgp_reproj.reshape(-1, 2)).astype(np.int32)    # round to nearest int
    
    # If projected origin lays out of the image, don't draw axis-system
    if not (0 <= origin[0] < img.shape[1] and 0 <= origin[1] < img.shape[0]):
        return img
    
    # Draw the axis-system
    line(img, origin, xAxis, rgb(255,0,0), thickness=2, lineType=cv2.CV_AA)
    line(img, origin, yAxis, rgb(0,255,0), thickness=2, lineType=cv2.CV_AA)
    line(img, origin, zAxis, rgb(0,0,255), thickness=2, lineType=cv2.CV_AA)
    circle(img, origin, 4, rgb(0,0,0), thickness=-1)    # filled circle, radius 4
    circle(img, origin, 5, rgb(255,255,255), thickness=2)    # white 'O', radius 5
    
    return img


def drawCamera(img, cam_origin, cam_axes, K, P, neg_fy=False, scale_factor=0.07, draw_axes=True, draw_frustum=True):
    """
    Draws a camera (X right, Y down) with 3D position "cam_origin" and local axes "cam_axes"
    on "img" where the viewer cam has (3x3 matrix) intrinsics "K" and (3x4 matrix) pose "P".
    
    "neg_fy" : set to True if the drawn cam has negative Y focal length, otherwise False
    "scale_factor" : the size of the drawn cam
    "draw_axes" : set to True to draw the axes of the to-be-drawn cam, otherwise False
    "draw_frustum" : set to True to draw the frustrum of the to-be-drawn cam, otherwise False
    """
    
    # Define local object-points
    objp_to_project = np.array([ [ 0. ,  0. , 0.],      # cam origin
                                 [ 1. ,  0.,  0.],      # cam X-axis
                                 [ 0. ,  1.,  0.],      # cam Y-axis
                                 [ 0. ,  0.,  1.],      # cam Z-axis
                                 
                                 [-0.5, -0.3, 1.],      # frustum top-left
                                 [ 0.5, -0.3, 1.],      # frustum top-right
                                 [ 0.5,  0.3, 1.],      # frustum bottom-right
                                 [-0.5,  0.3, 1.],      # frustum bottom-left
                                 
                                 [-0.3, -0.3, 1.],      # cam up indication triangle left
                                 [ 0.3, -0.3, 1.],      # cam up indication triangle right
                                 [ 0. , -0.6, 1.] ])    # cam up indication triangle top
    
    # Keep size of camera constant by normalizing using the distance between cam origin and origin of visualizing cam P
    objp_to_project *= cv2.norm(cam_origin.T + P[0:3, 0:3].T.dot(P[0:3, 3:4])) * scale_factor
    
    # Negative focal length requires the cam's Y-axis to be flipped
    if neg_fy:
        objp_to_project[:, 1] *= -1
    
    # Transform points in cam coords to world coords
    objp_to_project = cam_origin + objp_to_project.dot(cam_axes)
    
    # Project world coords to the viewer cam defined by (P, K)
    objp_projected, cam_visible = trfm.project_points(objp_to_project, K, img.shape, P)
    
    # Only draw axis-system if it's entirely in sight
    if cam_visible.sum() == len(cam_visible):
        cam_origin = objp_projected[0]
        
        if draw_axes:
            cam_xAxis, cam_yAxis, cam_zAxis = objp_projected[1:4]
            line(img, cam_origin, cam_xAxis, rgb(255,0,0), lineType=cv2.CV_AA)      # X-axis (red)
            line(img, cam_origin, cam_yAxis, rgb(0,255,0), lineType=cv2.CV_AA)      # Y-axis (green)
            line(img, cam_origin, cam_zAxis, rgb(0,0,255), lineType=cv2.CV_AA)      # Z-axis (blue)
            circle(img, cam_zAxis, 3, rgb(0,0,255))    # small dot to highlight cam Z axis
        
        if draw_frustum:
            yellow = rgb(255,255,0)
            frustum_plane = objp_projected[4:8]
            for i, p in enumerate(frustum_plane):
                # Create frustum plane
                line(img, frustum_plane[i], frustum_plane[(i+1) % 4], yellow, lineType=cv2.CV_AA)
                # Connect frustum plane points with origin
                line(img, cam_origin, p, yellow, lineType=cv2.CV_AA)
            # Draw up-facing frustum triangle
            cv2.fillPoly(img, [objp_projected[8:11]], yellow, lineType=cv2.CV_AA)
    
    return img


class MultilineText:
    """A class that enables to draw richer text on images."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clears all text."""
        self._texts_params = []
        self._size_ys = []
        self._size_extra_ys = []
        self._size = [0, 0]
    
    def text(self, txt, fF, fS, col, thickness=1, **kwargs):
        """
        Adds new line(s) of text "txt" in
        fontface "fF", fontstyle "fS", color "col" and with tickness "tickness"
        to the text-buffer.
        For more arguments, see OpenCV's "putText()" documentation.
        """
        
        # Split multiple lines across multiple text-objects
        txts = txt.split("\n")
        if len(txts) > 1:
            for txt in txts:
                self.text(txt, fF, fS, col, thickness, **kwargs)
            return
        
        # Save text-object parameters and sizes
        textSize, baseLine = cv2.getTextSize(txt, fF, fS, thickness)
        self._texts_params.append(([txt, None, fF, fS, col, thickness], kwargs))
        self._size_ys.append(textSize[1])
        self._size_extra_ys.append(baseLine + thickness)
        self._size[0] = max(self._size[0], textSize[0])
        self._size[1] += self._size_ys[-1] + self._size_extra_ys[-1]
    
    def getTextSize(self):
        """Returns the size of the (to-be-)drawn area."""
        return tuple(self._size)
    
    def putText(self, img, p):
        """
        Draws all text supplied by "MultilineText.text()" on the image "img" at position "p",
        and returns the rectangle of the drawn area.
        
        "p" : bottom-left corner of text in the image
        """
        
        # Trim resulting text, to avoid falling outside the image, if possible.
        # 'start_point' is set to the top-left corner of the text
        start_point = np.array(p)
        start_point[0] = max(0, min(p[0], img.shape[1] - self._size[0]))
        start_point[1] = max(0, p[1] - self._size[1])
        
        # Draw all texts
        p = np.array(start_point)
        for tP, sY, seY in zip(self._texts_params, self._size_ys, self._size_extra_ys):
            p[1] += sY
            args, kwargs = tP
            args.insert(0, img)
            args[2] = p    # pass 'p' as argument: bottom-left corner of text in img
            putText(*args, **kwargs)    # OpenCV's putText()
            p[1] += seY
        
        # Return drawn area
        rectangle = start_point, start_point + self._size
        return rectangle


def wireframe3DGeometry(img, verts, edges, col,
                        rvec, tvec, cameraMatrix, distCoeffs):
    """Draws a 3D object in wireframe, returns the resulting projection imagepoints."""
    
    # Calculate image-projections
    verts_imgp, jacob = cv2.projectPoints(
            verts, rvec, tvec, cameraMatrix, distCoeffs )
    rounding = np.vectorize(lambda x: int(round(x)))
    verts_imgp = rounding(verts_imgp.reshape(-1, 2)) # round to nearest int
    
    # Draw edges and vertices, in that order to prioritize vertices' appearance
    for edge in edges:
        v1, v2 = verts_imgp[edge]
        line(img, v1, v2, col, thickness=2, lineType=cv2.CV_AA)
    for vert in verts_imgp:
        circle(img, vert, 4, rgb(0,0,0), thickness=-1)    # filled circle, radius 4
        circle(img, vert, 5, col, thickness=2)    # circle circumference, radius 5
    
    return verts_imgp


def extractChessboardFeatures(img, boardSize):
    """Extract subpixel chessboard corners as features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
            gray, boardSize )

    # Refine them, if found
    if ret == True:
        cv2.cornerSubPix(
                gray, corners,
                (11,11),    # window
                (-1,-1),    # deadzone
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )    # termination criteria
        corners = corners.reshape(-1, 2)
    
    return ret, corners


"""
The following code works around some OpenCV BUGs:
(written on 2014-08-18 11:38:56 AM)

BUG #1:
    opencv/modules/features2d/include/opencv2/features2d.hpp near line 749:
        "CV_WRAP" is omitted for the method "radiusMatch",
        which results in hiding this method for higher-level languages such as Python and Java.
BUG #2:
    opencv/modules/core/src/stat.cpp near line 3209:
        An assert gets triggered when argument "OutputArray _nidx" of function "cv::batchDistance()"
        is set to "None" (Python) while "int K" is set to "0".
        This means that (_nidx::needed() != NULL) when (_nidx == "None" (Python)),
        to fix this: _nidx::needed() should evaluate to "NULL".
"""
native_BFMatcher = cv2.BFMatcher()
has_native_radiusMatch = "radiusMatch" in dir(native_BFMatcher)

if has_native_radiusMatch:
    BFMatcher = cv2.BFMatcher
    
else:
    class BFMatcher:
        """
        Wrapper class + Python implementation of 'radiusMatch' to work around BUG #1.
        
        Python's overhead is negligible in this 'radiusMatch' implementation:
            ratio of runtime C++/Python: more than 50/1
        """
        
        def __init__(self, *args, **kwargs):
            self.this = cv2.BFMatcher(*args, **kwargs)
        
        def radiusMatch(self, query_points, train_points, max_radius, **kwargs):
            """
            kNN radius match with k=2.
            """
            dist_matrix, nidx_matrix = cv2.batchDistance(
                    query_points, train_points,
                    cv2.cv.CV_32F,    # FIXME: infer dtype from query_points.dtype
                    None, None,    # dist, nidx: output args
                    self.getInt("normType"),
                    train_points.shape[0],    # work around BUG #2
                    None, 0, False )    # called as implemented in C-impl of 'radiusMatch'
            
            matches = []
            for queryIdx, (dists_from_query_point, nidx_from_query_point) in \
                    enumerate(zip(dist_matrix, nidx_matrix)):
                # Initialize to 'no good matches'
                query_matches = []
                valid_idxs = np.where(dists_from_query_point <= max_radius)[0]
                
                # There is at least one good match
                if valid_idxs.size:
                    dist_valid = dists_from_query_point[valid_idxs]
                    nidx_valid = nidx_from_query_point[valid_idxs]
                    
                    # Append the best match, as fist match
                    dist_first_idx = dist_valid.argmin()
                    query_matches.append(cv2.DMatch(
                            queryIdx,
                            nidx_valid[dist_first_idx],    # trainIdx
                            dist_valid[dist_first_idx] ))    # distance
                    
                    # There is at least another one good match, append the best one, as second match
                    if dist_valid.size > 1:
                        dist_valid[dist_first_idx] = float("inf")    # exclude this one from next 'argmin' step
                        dist_second_idx = dist_valid.argmin()
                        query_matches.append(cv2.DMatch(
                                queryIdx,
                                nidx_valid[dist_second_idx],    # trainIdx
                                dist_valid[dist_second_idx] ))    # distance
                
                # Add list of matches corresponding with this queryIdx
                matches.append(query_matches)
            
            return matches
    
    # Append remaining methods of the native BFMatcher class
    for methodName in dir(native_BFMatcher):
        if not methodName.startswith('_'):
            exec("def method(self, *args, **kwargs): return self.this.%s(*args, **kwargs)" % methodName)
            setattr(BFMatcher, methodName, method)
