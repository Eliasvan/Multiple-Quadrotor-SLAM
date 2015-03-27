import struct
import numpy as np



""" File- import/export functions """

def load_cam_trajectory_TUM(filename):
    """
    Load ground-truth camera trajectories from file "filename",
    the format is specified on "http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats".
    
    Returns the following lists: "timestps", "locations", and "quaternions".
    
    Note: some typical filenames of the ICL NUIM dataset compatible with this function, are:
    "livingRoom1.gt.freiburg", "traj1.gt.freiburg", ...
    """
    timestps, locations, quaternions = [], [], []
    
    lines = open(filename, 'r').read().replace(",", " ").replace("\t", " ").split('\n')
    for line in lines:
        line = line.strip()
        
        # Ignore empty lines or comments
        if not line or line[0] == '#':
            continue
        
        timestp, lx, ly, lz, qx, qy, qz, qw = map(float, line.split(' '))
        timestps.append(timestp)
        locations.append([lx, ly, lz])
        quaternions.append([qx, qy, qz, qw])
    
    return timestps, locations, quaternions

def save_cam_trajectory_TUM(filename, timestps, locations, quaternions):
    """
    Save ground-truth camera trajectories to file "filename",
    the format is specified on "http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats".
    """
    
    out = '\n'.join([
            ' '.join(map(str, [timestp] + l + q))
            for timestp, l, q in zip(timestps, locations, quaternions)
    ]) + '\n'
    
    open(filename, 'w').write(out)

def load_3D_points_from_pcd_file(filename, use_alpha=False):
    """
    Load the 3D points (numpy float32 array) from the .pcd-file "filename",
    the format is specified on "http://pointclouds.org/documentation/tutorials/pcd_file_format.php".
    
    If there is no color associated with each 3D point,
    the "colors" (numpy uint8 array) is set to None.
    Otherwise, each color is formatted as (B, G, R), or (B, G, R, A).
    Set "use_alpha" to False to force the (B, G, R) format.
    
    Note: the only supported header configs (for the listed entries) are:
        FIELDS x y z
        FIELDS x y z rgb
        SIZE 4 4 4
        SIZE 4 4 4 4
        TYPE F F F
        TYPE F F F F
        HEIGHT 1
        DATA ascii
    And it's not recommended to load huge files with this function.
    """
    
    def float2bgra(f):
        return map(ord, list(struct.pack('f', f)))
    
    lines = open(filename, 'r').read().split('\n')
    
    num_points = 0
    use_colors = False
    
    # Go over all interesting header entries ("FIELDS", "WIDTH", "HEIGHT", "DATA")
    entry = "FIELDS"
    for i, line in enumerate(lines):
        words = line.split(' ')
        
        if words[0] == entry == "FIELDS":
            entry = "WIDTH"
            if words[1:4] == ['x', 'y', 'z']:
                if len(words) == 1 + 3:
                    continue
                elif len(words) == 1 + 4 and words[4] == "rgb":
                    use_colors = True
                    continue
            raise ValueError("The following 'FIELDS' config in the .pcd-file is not supported: %s" % words[1:])
        
        elif words[0] == entry == "WIDTH":
            num_points = int(words[1])
            entry = "HEIGHT"
        
        elif words[0] == entry == "HEIGHT":
            if int(words[1]) != 1:
                raise ValueError("Organized point clouds in the .pcd-file are not supported.")
            entry = "DATA"
        
        elif words[0] == entry == "DATA":
            if words[1] != "ascii":
                raise ValueError("The following 'DATA' config in the .pcd-file is not supported: '%s'" % words[1])
            entry = ""
            break
    
    if entry:
        raise ValueError("The .pcd-file did not include all neccessary header entries.")
    
    lines = lines[i + 1: i + 1 + num_points]    # strip header and other non-data
    if len(lines) < num_points:
        raise ValueError("The .pcd-file did not include all advertised points. (%s instead of %s)" % 
                         len(lines), num_points)
    
    points = np.array([tuple(map(float, line.split(' '))) for line in lines], dtype=np.float32)
    
    if use_colors:
        colors = np.array(map(float2bgra, points[:, -1:]), dtype=np.uint8)    # split each point into color, ...
        points = points[:, :-1]    # ... and x, y, z coordinates
        if not use_alpha:
            colors = colors[:, 0:3]
    else:
        colors = None
    
    return points, colors

def save_3D_points_to_pcd_file(filename, points, colors=None):
    """
    Save the 3D points "points" (numpy array) to the .pcd-file "filename",
    the format is specified on "http://pointclouds.org/documentation/tutorials/pcd_file_format.php".
    
    To also save the color associated with each 3D point, supply "colors" (numpy uint8 array).
    The format of each color can be either (B, G, R, A), or (B, G, R).
    
    Note: the binary form is not supported, only the ascii form is supported.
    The two least-significant bits of the alpha values should be 0b01,
    hence the minimum value is 1, and the maximum is 253,
    and the resolution is divided by 4.
    """
    use_alpha = (colors != None and colors.shape[1] == 4)
    
    def bgra2float(bgra):
        return struct.unpack('f', "".join(map(chr, bgra)))[0]
    
    def float2string(f):
        return "%.8e" % f    # just enough precision to recover the color afterwards
    
    header = """
    # .PCD v.7 - Point Cloud Data file format
    VERSION .7
    FIELDS x y z%s
    SIZE 4 4 4%s
    TYPE F F F%s
    COUNT 1 1 1%s
    WIDTH %s
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %s
    DATA ascii
    """ % (" rgb" * (colors != None), " 4" * (colors != None), " F" * (colors != None), " 1" * (colors != None),
           len(points), len(points))
    from textwrap import dedent
    header = dedent(header[1:])    # removes first new-line and indents
    
    points = points.astype(np.float32)
    
    if colors != None:
        if use_alpha:
            # Float32 exponent 0xFF would create NaN/Inf values, while exponent 0x00 could cause denormal values,
            # so avoid them by ensuring that the last two least-significant bits of alpha is 0b01,
            # in this way the R, G and B values are not restricted.
            colors[:, 3] &= 0b11111100
            colors[:, 3] |=       0b01
        else:
            alpha = np.empty((len(colors), 1), dtype=np.uint8); alpha.fill(0xFD)
            colors = np.concatenate((colors, alpha), axis=1)    # add the maximum alpha value (= 0xFD)
        colors = np.array(map(bgra2float, colors), dtype=np.float32).reshape(len(colors), 1)    # convert to floats
        points = np.concatenate((points, colors), axis=1)
    
    data = '\n'.join([' '.join(map(float2string, point)) for point in points])
    
    open(filename, 'w').write("%s%s\n" % (header, data))


""" Transformation functions """

try:
    import cv2
    import transforms as trfm
except ImportError:
    print ("Warning: can't load modules \"cv2\" or \"transforms\" required for some functions of \"dataset_tools\" module.")

def convert_cam_poses_to_cam_trajectory_TUM(Ps, fps=30):
    """
    Convert camera pose projection matrices "Ps" to ground-truth camera trajectories,
    the result can be saved with "save_cam_trajectory_TUM()".
    """
    timestps, locations, quaternions = [], [], []
    
    for i, P in enumerate(Ps):
        # We take the inverse,
        # to obtain the trfm matrix that projects points from the camera axis-system to the world axis-system
        M = trfm.P_inv(P)
        
        R = cv2.Rodrigues(M[0:3, 0:3])[0]
        q = list(trfm.quat_from_rvec(R)[0:4, 0])
        t = list(M[0:3, 3])
        
        timestps.append(float(i) / fps)
        locations.append(t)
        quaternions.append(q)
    
    return timestps, locations, quaternions
