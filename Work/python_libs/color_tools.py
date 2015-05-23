import numpy as np
import cv2

rgb = cv2.cv.CV_RGB



def color_palette(nc_L, nc_a, nc_b):
    """
    Generate color palette, colors are randomly distributed (but with a constant seed).
    
    "nc_L", "nc_a", "nc_b" give the number of colors in each dimension of Lab color-space.
    Returns the color palette and the total amount of colors.
    """
    
    # Define the limits of each color dimension, such that it is visible/distinguishable for the human eye
    L_min, L_max = 99, 230
    a_min, a_max = 26, 230
    b_min, b_max = 26, 230
    
    # Generate a 'grid' of Lab colors
    num_colors = nc_L * nc_a * nc_b
    colors = np.zeros((num_colors, 1, 3), dtype=np.uint8)
    for Li, L in enumerate(np.arange(L_min, L_max + 1, (L_max-L_min) / (nc_L-1))):
        for ai, a in enumerate(np.arange(a_min, a_max + 1, (a_max-a_min) / (nc_a-1))):
            for bi, b in enumerate(np.arange(b_min, b_max + 1, (b_max-b_min) / (nc_b-1))):
                colors[Li*nc_a*nc_b + ai*nc_b + bi, 0, :] = (L, a, b)
    
    # Convert to RGB and reshuffle
    color_palette = cv2.cvtColor(colors, cv2.COLOR_LAB2RGB).reshape(num_colors, 3)
    rstate = np.random.get_state()    # backup random state
    np.random.seed(1)
    color_palette = np.random.permutation(color_palette)
    np.random.set_state(rstate)    # restore original random state
    
    return np.array(map(rgb, *zip(*color_palette))), num_colors


def sample_colors(img, imgp):
    """
    Sample the colors of image "img" at points "imgp", and return the resulting list of colors.
    """
    return img[tuple(np.rint(imgp[:, ::-1]).astype(int).T)]
