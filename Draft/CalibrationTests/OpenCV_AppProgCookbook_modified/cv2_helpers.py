import numpy as np
import cv2



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

def format3DVector(v):
    return "[ %.3f  %.3f  %.3f ]" % tuple(v)


def drawKeypointsAndMotion(img2, points1, points2, color):
    """Returns a new image with vectors from points1 to points2, and keypoints on img2."""
    img = cv2.drawKeypoints(img2, [cv2.KeyPoint(p[0],p[1], 7.) for p in points2], color=rgb(0,0,255))
    for p1, p2 in zip(points1, points2):
        line(img, p1, p2, color)
    return img


class MultilineText:
    """A class that enables to draw richer text on images."""
    def __init__(self):
        self.clear()
    
    def clear(self):
        self._texts_params = []
        self._size_ys = []
        self._size_extra_ys = []
        self._size = [0, 0]
    
    def text(self, txt, fF, fS, col, thickness=1, **kwargs):
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
        return tuple(self._size)
    
    def putText(self, img, p):    # 'p': bottom-left corner of text in image
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
            putText(*args, **kwargs)
            p[1] += seY
        
        # Return drawn area
        rectangle = start_point, start_point + self._size
        return rectangle


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
