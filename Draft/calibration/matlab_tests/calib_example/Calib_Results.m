% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 713.620691740159941 ; 717.078061529379283 ];

%-- Principal point:
cc = [ 325.671793186765740 ; 211.132116468066897 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.077628312024137 ; -0.958335259789436 ; -0.003950825405128 ; 0.000577695571861 ; 2.525130747741366 ];

%-- Focal length uncertainty:
fc_error = [ 8.034097904698342 ; 8.019776881202356 ];

%-- Principal point uncertainty:
cc_error = [ 2.322267357290329 ; 2.088310734481813 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.033031658302080 ; 0.395213806073564 ; 0.000902719530121 ; 0.001018536003926 ; 1.410608756161583 ];

%-- Image size:
nx = 640;
ny = 480;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 20;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 1 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 2 ; 2 ; 0 ];
Tc_1  = [ -104 ; -50 ; 403 ];
omc_error_1 = [ 0 ; 0 ; 0 ];
Tc_error_1  = [ 1 ; 1 ; 4 ];

%-- Image #2:
omc_2 = [ -2 ; -2 ; 0 ];
Tc_2  = [ -110 ; -50 ; 419 ];
omc_error_2 = [ 0 ; 0 ; 0 ];
Tc_error_2  = [ 1 ; 1 ; 4 ];

%-- Image #3:
omc_3 = [ 2 ; 2 ; 0 ];
Tc_3  = [ -110 ; -50 ; 385 ];
omc_error_3 = [ 0 ; 0 ; 0 ];
Tc_error_3  = [ 1 ; 1 ; 4 ];

%-- Image #4:
omc_4 = [ 2 ; 2 ; 0 ];
Tc_4  = [ -89 ; -56 ; 370 ];
omc_error_4 = [ 0 ; 0 ; 0 ];
Tc_error_4  = [ 1 ; 1 ; 4 ];

%-- Image #5:
omc_5 = [ -2 ; -1 ; 0 ];
Tc_5  = [ -93 ; -52 ; 378 ];
omc_error_5 = [ 0 ; 0 ; 0 ];
Tc_error_5  = [ 1 ; 1 ; 4 ];

%-- Image #6:
omc_6 = [ -2 ; -1 ; 0 ];
Tc_6  = [ -133 ; -27 ; 372 ];
omc_error_6 = [ 0 ; 0 ; 0 ];
Tc_error_6  = [ 1 ; 1 ; 4 ];

%-- Image #7:
omc_7 = [ 2 ; 2 ; 0 ];
Tc_7  = [ -122 ; -48 ; 378 ];
omc_error_7 = [ 0 ; 0 ; 0 ];
Tc_error_7  = [ 1 ; 1 ; 4 ];

%-- Image #8:
omc_8 = [ 2 ; 2 ; 0 ];
Tc_8  = [ -103 ; -57 ; 380 ];
omc_error_8 = [ 0 ; 0 ; 0 ];
Tc_error_8  = [ 1 ; 1 ; 4 ];

%-- Image #9:
omc_9 = [ 2 ; 2 ; 0 ];
Tc_9  = [ -96 ; -60 ; 358 ];
omc_error_9 = [ 0 ; 0 ; 0 ];
Tc_error_9  = [ 1 ; 1 ; 4 ];

%-- Image #10:
omc_10 = [ 2 ; 2 ; 0 ];
Tc_10  = [ -94 ; -65 ; 347 ];
omc_error_10 = [ 0 ; 0 ; 0 ];
Tc_error_10  = [ 1 ; 1 ; 3 ];

%-- Image #11:
omc_11 = [ 2 ; 2 ; 0 ];
Tc_11  = [ -83 ; -61 ; 345 ];
omc_error_11 = [ 0 ; 0 ; 0 ];
Tc_error_11  = [ 1 ; 1 ; 3 ];

%-- Image #12:
omc_12 = [ 2 ; 2 ; 0 ];
Tc_12  = [ -89 ; -50 ; 332 ];
omc_error_12 = [ 0 ; 0 ; 0 ];
Tc_error_12  = [ 1 ; 0 ; 3 ];

%-- Image #13:
omc_13 = [ -2 ; -2 ; 0 ];
Tc_13  = [ -111 ; -47 ; 331 ];
omc_error_13 = [ 0 ; 0 ; 0 ];
Tc_error_13  = [ 1 ; 0 ; 3 ];

%-- Image #14:
omc_14 = [ -2 ; -1 ; 0 ];
Tc_14  = [ -113 ; -49 ; 334 ];
omc_error_14 = [ 0 ; 0 ; 0 ];
Tc_error_14  = [ 1 ; 0 ; 3 ];

%-- Image #15:
omc_15 = [ -2 ; -2 ; 0 ];
Tc_15  = [ -120 ; -46 ; 343 ];
omc_error_15 = [ 0 ; 0 ; 0 ];
Tc_error_15  = [ 1 ; 1 ; 3 ];

%-- Image #16:
omc_16 = [ -2 ; -2 ; 0 ];
Tc_16  = [ -121 ; -45 ; 339 ];
omc_error_16 = [ 0 ; 0 ; 0 ];
Tc_error_16  = [ 1 ; 0 ; 3 ];

%-- Image #17:
omc_17 = [ -2 ; -2 ; 0 ];
Tc_17  = [ -115 ; -56 ; 342 ];
omc_error_17 = [ 0 ; 0 ; 0 ];
Tc_error_17  = [ 1 ; 1 ; 3 ];

%-- Image #18:
omc_18 = [ -2 ; -2 ; 0 ];
Tc_18  = [ -110 ; -62 ; 369 ];
omc_error_18 = [ 0 ; 0 ; 0 ];
Tc_error_18  = [ 1 ; 1 ; 4 ];

%-- Image #19:
omc_19 = [ -2 ; -2 ; 0 ];
Tc_19  = [ -97 ; -52 ; 386 ];
omc_error_19 = [ 0 ; 0 ; 0 ];
Tc_error_19  = [ 1 ; 1 ; 4 ];

%-- Image #20:
omc_20 = [ -2 ; -2 ; 0 ];
Tc_20  = [ -105 ; -38 ; 395 ];
omc_error_20 = [ 0 ; 0 ; 0 ];
Tc_error_20  = [ 1 ; 1 ; 4 ];

