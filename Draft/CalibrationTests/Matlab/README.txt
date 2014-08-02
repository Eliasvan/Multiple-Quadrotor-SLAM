README
======

The "Matlab Calibration Toolbox" from http://www.vision.caltech.edu/bouguetj/calib_doc/index.html was used.


Instructions for use in Octave
______________________________

The GUI doesn't seem to be functional as of Octave 2.6.4,
so we'll use the commandline interface.


Setup
-----

- In the file 'click_ima_calib.m', comment line 282:
	% zoom on;


Usage
-----

>>> clear all

Replace <path to TOOLBOX_calib>:
>>> addpath("<path to TOOLBOX_calib>")

>>> cd "Draft/CalibrationTests/Matlab/calib_example"

>>> data_calib
.                 chessboard05.jpg  chessboard11.jpg  chessboard17.jpg
..                chessboard06.jpg  chessboard12.jpg  chessboard18.jpg
chessboard01.jpg  chessboard07.jpg  chessboard13.jpg  chessboard19.jpg
chessboard02.jpg  chessboard08.jpg  chessboard14.jpg  chessboard20.jpg
chessboard03.jpg  chessboard09.jpg  chessboard15.jpg
chessboard04.jpg  chessboard10.jpg  chessboard16.jpg

Basename camera calibration images (without number nor suffix):  chessboard
Image format: ([]='r'='ras', 'b'='bmp', 't'='tif', 'p'='pgm', 'j'='jpg', 'm'='ppm')  j
Loading image 1...2...3...4...5...6...7...8...9...10...11...12...13...14...15...16...17...18...19...20...
done

>>> click_calib
Extraction of the grid corners on the images
Number(s) of image(s) to process ([] = all images) =  
Window size for corner finder (wintx and winty):
wintx ([] = 5) =  
winty ([] = 5) =  
Window size = 11x11
Do you want to use the automatic square counting mechanism (0=[]=default)
  or do you always want to enter the number of squares manually (1,other)?

Processing image 1...
Using (wintx,winty)=(5,5) - Window size = 11x11      (Note: To reset the window size, run script clearwin)
Click on the four extreme corners of the rectangular complete pattern (the first clicked corner is the origin)...
Size dX of each square along the X direction ([]=100mm) =  29
Size dY of each square along the Y direction ([]=100mm) =  29
If the guessed grid corners (red crosses on the image) are not close to the actual corners, it is necessary to enter an initial guess for the radial distortion factor kc (useful for subpixel detection)
Need of an initial guess for distortion? ([]=no, other=yes)  
Corner extraction...

... (repeat for all 20 images)
done

>>> go_calib_optim
Aspect ratio optimized (est_aspect_ratio = 1) -> both components of fc are estimated (DEFAULT).
Principal point optimized (center_optim=1) - (DEFAULT). To reject principal point, set center_optim=0
Skew not optimized (est_alpha=0) - (DEFAULT)
Distortion not fully estimated (defined by the variable est_dist):
     Sixth order distortion not estimated (est_dist(5)=0) - (DEFAULT) .
Initialization of the principal point at the center of the image.
Initialization of the intrinsic parameters using the vanishing points of planar patterns.

Initialization of the intrinsic parameters - Number of images: 20


Calibration parameters after initialization:

Focal Length:          fc = [ 749.51098   749.51098 ]
Principal point:       cc = [ 319.50000   239.50000 ]
Skew:             alpha_c = [ 0.00000 ]   => angle of pixel = 90.00000 degrees
Distortion:            kc = [ 0.00000   0.00000   0.00000   0.00000   0.00000 ]

Main calibration optimization procedure - Number of images: 20
Gradient descent iterations: 1...2...3...4...5...6...7...8...9...10...11...12...13...14...15...16...17...18...19...20...done
Estimation of uncertainties...
done

Calibration results after optimization (with uncertainties):

Focal Length:          fc = [ 714.63414   718.09482 ] ± [ 8.09474   8.08007 ]
Principal point:       cc = [ 325.57555   211.17740 ] ± [ 2.33863   2.11411 ]
Skew:             alpha_c = [ 0.00000 ] ± [ 0.00000  ]   => angle of pixel axes = 90.00000 ± 0.00000 degrees
Distortion:            kc = [ 0.02363   -0.26528   -0.00397   0.00053  0.00000 ] ± [ 0.01382   0.08046   0.00091   0.00103  0.00000 ]
Pixel error:          err = [ 0.23998   0.20895 ]

Note: The numerical errors are approximately three times the standard deviations (for reference).

>>> recomp_corner_calib
Re-extraction of the grid corners on the images (after first calibration)
Window size for corner finder (wintx and winty):
wintx ([] = 5) =  
winty ([] = 5) =  
Window size = 11x11
Number(s) of image(s) to process ([] = all images) =  
Use the projection of 3D grid or manual click ([]=auto, other=manual):  
Processing image 1...2...3...4...5...6...7...8...9...10...11...12...13...14...15...16...17...18...19...20...
done

>>> go_calib_optim
Aspect ratio optimized (est_aspect_ratio = 1) -> both components of fc are estimated (DEFAULT).
Principal point optimized (center_optim=1) - (DEFAULT). To reject principal point, set center_optim=0
Skew not optimized (est_alpha=0) - (DEFAULT)
Distortion not fully estimated (defined by the variable est_dist):
     Sixth order distortion not estimated (est_dist(5)=0) - (DEFAULT) .

Main calibration optimization procedure - Number of images: 20
Gradient descent iterations: 1...2...3...4...5...6...7...8...9...10...11...12...13...14...15...done
Estimation of uncertainties...done

Calibration results after optimization (with uncertainties):

Focal Length:          fc = [ 714.61054   718.07119 ] ± [ 8.09003   8.07538 ]
Principal point:       cc = [ 325.57678   211.17560 ] ± [ 2.33706   2.11280 ]
Skew:             alpha_c = [ 0.00000 ] ± [ 0.00000  ]   => angle of pixel axes = 90.00000 ± 0.00000 degrees
Distortion:            kc = [ 0.02362   -0.26518   -0.00397   0.00053  0.00000 ] ± [ 0.01381   0.08040   0.00091   0.00103  0.00000 ]
Pixel error:          err = [ 0.23986   0.20882 ]

Note: The numerical errors are approximately three times the standard deviations (for reference).

>>> recomp_corner_calib
Re-extraction of the grid corners on the images (after first calibration)
Window size for corner finder (wintx and winty):
wintx ([] = 5) =  
winty ([] = 5) =  
Window size = 11x11
Number(s) of image(s) to process ([] = all images) =  
Use the projection of 3D grid or manual click ([]=auto, other=manual):  
Processing image 1...2...3...4...5...6...7...8...9...10...11...12...13...14...15...16...17...18...19...20...
done

>>> est_dist(5) = 1

>>> go_calib_optim
Aspect ratio optimized (est_aspect_ratio = 1) -> both components of fc are estimated (DEFAULT).
Principal point optimized (center_optim=1) - (DEFAULT). To reject principal point, set center_optim=0
Skew not optimized (est_alpha=0) - (DEFAULT)

Main calibration optimization procedure - Number of images: 20
Gradient descent iterations: 1...2...3...4...5...6...7...8...9...10...11...12...13...14...15...16...17...18...done
Estimation of uncertainties...done

Calibration results after optimization (with uncertainties):

Focal Length:          fc = [ 713.62323   717.08055 ] ± [ 8.03415   8.01983 ]
Principal point:       cc = [ 325.67232   211.13183 ] ± [ 2.32230   2.08834 ]
Skew:             alpha_c = [ 0.00000 ] ± [ 0.00000  ]   => angle of pixel axes = 90.00000 ± 0.00000 degrees
Distortion:            kc = [ 0.07761   -0.95816   -0.00395   0.00058  2.52464 ] ± [ 0.03303   0.39522   0.00090   0.00102  1.41062 ]
Pixel error:          err = [ 0.23774   0.20759 ]

Note: The numerical errors are approximately three times the standard deviations (for reference).

>>> saving_calib
Saving calibration results under Calib_Results.mat
Generating the matlab script file Calib_Results.m containing the intrinsic and extrinsic parameters...
done

>>> visualize_distortions
Select "Complete distortion model" window.

>>> print ("complete-distortion-model.png", "-color", "-FHelvetica:10", "-dpng", "-S2400,1600")
