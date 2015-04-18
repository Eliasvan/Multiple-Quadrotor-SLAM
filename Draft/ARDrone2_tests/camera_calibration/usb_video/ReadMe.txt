Warning, calibrationdata_bottom_orig contains the raw images of the bottom cam, but they need to be trimmed:
    using OpenCV: img = img_orig[:, 160 : 1280-160, :]
