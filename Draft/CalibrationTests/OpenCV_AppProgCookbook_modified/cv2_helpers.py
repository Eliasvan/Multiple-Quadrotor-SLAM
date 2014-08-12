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

def format3DVector(v):
    return "[ %.3f  %.3f  %.3f ]" % tuple(v)


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
