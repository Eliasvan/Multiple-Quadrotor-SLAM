import os
import struct
import numpy as np



""" Helper functions """


def _cam_trajectory_to_numpy(timestps, locations, quaternions, normalize_quaternions=False):
    """Convenience function to convert python lists to numpy arrays."""
    if timestps:
        quaternions = np.array(quaternions, dtype=float)
        if normalize_quaternions:
            quaternions /= np.linalg.norm(quaternions, axis=1).reshape(len(quaternions), 1)
        return np.array(timestps, dtype=float), np.array(locations, dtype=float), quaternions
    else:
        return np.empty((0), dtype=float), np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=float)


""" Filepath functions """


def image_filepaths_by_directory(img_dir):
    """
    Returns the filepaths to the images contained in directory "img_dir".
    
    Note: numbers are treated as numbers, as expected,
    e.g. "img-2.png" vs "img-10.png" are sorted correctly.
    """
    
    # Only allow certain image extensions
    images = [image for image in os.listdir(img_dir)
              if os.path.splitext(image)[1] in (".png", ".jpg", ".jpeg", ".tiff")]
    
    # Find the portions of the filename containing numbers, and retrieve the maximum length
    images_splitted = []
    max_number_length = 0
    for img in images:
        img_splitted = []
        img_splitted_idxs = []
        state = None
        for c in img:
            state_new = ("NaN", "Number")[c in tuple("0123456789")]
            if state_new != state:
                if state_new == "Number":
                    img_splitted_idxs.append(len(img_splitted))
                img_splitted.append("")
                state = state_new
            img_splitted[-1] += c
            if state == "Number":
                max_number_length = max(max_number_length, len(img_splitted[-1]))
        images_splitted.append((img_splitted, img_splitted_idxs))
    
    # Insert leading zeros to each number, to fill up to the maximum number length, then sort
    keys_and_images = []
    for image, (img_splitted, img_splitted_idxs) in zip(images, images_splitted):
        for i in img_splitted_idxs:
            img_splitted[i] = '0' * (max_number_length - len(img_splitted[i])) + img_splitted[i]
        key = ''.join(img_splitted)
        keys_and_images.append((key, image))
    keys_and_images.sort()
    
    # Append the directory to the sorted list of image filenames
    return [os.path.join(img_dir, image) for key, image in keys_and_images]


""" File- import/export functions """


def load_cam_trajectory_TUM(filename):
    """
    Load (e.g. ground-truth) camera trajectories from file "filename",
    the format is specified on "http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats".
    
    Returns the following numpy arrays: "timestps", "locations", and "quaternions".
    
    Note: some typical filenames of the ICL NUIM dataset compatible with this function, are:
    "livingRoom1.gt.freiburg", "traj1.gt.freiburg", ...
    """
    timestps, locations, quaternions = [], [], []
    
    lines = open(filename, 'r').read().replace(',', ' ').replace('\t', ' ').split('\n')
    for line in lines:
        line = line.strip()
        
        # Ignore empty lines or comments
        if not line or line[0] == '#':
            continue
        
        timestp, lx, ly, lz, qx, qy, qz, qw = map(float, line.split(' '))
        timestps.append(timestp)
        locations.append([lx, ly, lz])
        quaternions.append([qx, qy, qz, qw])
    
    return _cam_trajectory_to_numpy(timestps, locations, quaternions, normalize_quaternions=True)


def save_cam_trajectory_TUM(filename, cam_trajectory):
    """
    Save (e.g. ground-truth) camera trajectory "cam_trajectory" to file "filename",
    the format is specified on "http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats".
    
    "cam_trajectory" should consist of the following numpy arrays: "timestps", "locations", and "quaternions".
    """
    lines = []
    
    lines.append("# Format: timestamp tx ty tz qx qy qz qw")
    lines.append("# Where translations and quaternions are defined in world coordinates (=> inverse of pose)")
    lines += [
            ' '.join(map(str, (timestp,) + tuple(l) + tuple(q)))
            for timestp, l, q in zip(*cam_trajectory) ]
    lines.append("")    # empty line
    
    open(filename, 'w').write('\n'.join(lines))


def load_3D_points_from_pcd_file(filename, use_alpha=False):
    """
    Load the 3D points (numpy float32 array) from the .pcd-file "filename",
    the format is specified on "http://pointclouds.org/documentation/tutorials/pcd_file_format.php".
    
    If there is no color associated with each 3D point,
    the "colors" (numpy uint8 array) is set to None.
    Otherwise, each color is formatted as (B, G, R), or (B, G, R, A).
    Set "use_alpha" to False to force the (B, G, R) format.
    "found_alpha" is set to True if an alpha color channel was found.
    
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
        return bytearray(struct.pack('f', f))
    
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
        raise ValueError("The .pcd-file did not include all necessary header entries.")
    
    lines = lines[i + 1: i + 1 + num_points]    # strip header and other non-data
    if len(lines) < num_points:
        print (lines[i: i + 1 + num_points])
        print (i)
        raise ValueError("The .pcd-file did not include all advertised points. (%s instead of %s)" % 
                         (len(lines), num_points))
    
    points = np.array([tuple(map(float, line.split(' '))) for line in lines], dtype=np.float32)
    if not len(points):
        return np.zeros((0, 3), dtype=np.float32), None, False    # no points found
    
    found_alpha = False
    if use_colors:
        colors = np.array(tuple(map(float2bgra, points[:, -1:])))    # split each point into color, ...
        points = points[:, :-1]    # ... and x, y, z coordinates
        found_alpha = (colors.shape[1] > 3)
        if not use_alpha:
            colors = colors[:, 0:3]
    else:
        colors = None
    
    return points, colors, found_alpha


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
    
    Note 2: if you want to save to PLY format instead, convert the output file of this function
    to PLY format with the application "pcd2ply" or "pcl_pcd2ply", included in PointCloudLibrary (pcl):
    http://pointclouds.org/
    """
    use_alpha = (colors != None and colors.shape[1] == 4)
    
    def bgra2float(bgra):
        return struct.unpack('f', bytearray(bgra))[0]
    
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
        if colors.dtype != np.uint8:
            colors = colors.astype(np.uint8)
        if use_alpha:
            # Float32 exponent 0xFF would create NaN/Inf values, while exponent 0x00 could cause denormal values,
            # so avoid them by ensuring that the last two least-significant bits of alpha form 0b01,
            # in this way the R, G and B values are not restricted.
            colors[:, 3] &= 0b11111100
            colors[:, 3] |=       0b01
        else:
            alpha = np.empty((len(colors), 1), dtype=np.uint8); alpha.fill(0xFD)
            colors = np.concatenate((colors, alpha), axis=1)    # add the maximum alpha value (= 0xFD)
        colors = np.array(tuple(map(bgra2float, colors)), dtype=np.float32).reshape(len(colors), 1)    # convert to floats
        points = np.concatenate((points, colors), axis=1)
    
    data = '\n'.join([' '.join(map(float2string, point)) for point in points])
    
    open(filename, 'w').write("%s%s\n" % (header, data))


""" Transformation functions """

import transforms as trfm


def convert_cam_poses_to_cam_trajectory_TUM(Ps, fps=30):
    """
    Convert camera pose projection matrices "Ps" to camera trajectories in TUM format,
    the result can be saved with "save_cam_trajectory_TUM()".
    
    Timestamp of first pose starts at 1.0 / fps.
    """
    timestps, locations, quaternions = [], [], []
    
    for i, P in enumerate(Ps):
        if P == None:
            continue
        
        timestps.append(float(1 + i) / fps)
        
        q, t = trfm.pose_TUM_from_P(P)
        locations.append(list(t.reshape(-1)))
        quaternions.append(list(q.reshape(-1)))
    
    return _cam_trajectory_to_numpy(timestps, locations, quaternions)


def transform_between_cam_trajectories(cam_trajectory_from, cam_trajectory_to,
                                       at_frame=1, at_time=None,
                                       infer_scale=True, offset_frames=None, offset_time=float("inf")):
    """
    Returns the transformation ("delta_quaternion", "delta_scale", "delta_location") (apply from left to right)
    between two camera trajectories "cam_trajectory_from" and "cam_trajectory_to",
    of which the format is given by the output of "load_cam_trajectory_TUM()".
    
    "at_frame" or "at_time" : moment at which the translation and rotation are calculated
    "at_frame" : frame number, starting from 1; set to None if "at_time" is used instead
    "at_time" : timestamp in seconds; set to None if "at_frame" is used instead
    
    "infer_scale" : set to True if the scale should also be calculated;
                    requires one of following offsets to be set:
    "offset_frames" or "offset_time" : offset between first and second moment
    "offset_frames" : frames inbetween both moments; set to None if "offset_time" is used instead
    "offset_time" : seconds inbetween both moments; set to None if "offset_frames" is used instead
    
    Note: in case "at_frame" or "offset_frames" is used,
    the corresponding timestamps of the "to" trajectory are used.
    In case a timestamp at first or second moment of one of the trajectories
    is out of range of the other trajectory,
    the moments are adjusted such that the timestamps between trajectories match as good as possible.
    """
    ts_from, locs_from, quats_from = cam_trajectory_from
    ts_to, locs_to, quats_to = cam_trajectory_to
    
    # Return unit transformation, if one of the trajectories is empty
    if not len(ts_from) or not len(ts_to):
        return trfm.unit_quat().reshape(4), 1., np.zeros(3)
    
    def closest_element_index(array, element):
        if abs(element) != float("inf"):
            return (np.abs(array - element)).argmin()
        elif element == float("inf"):
            return len(array) - 1
        else:    # element == float("-inf")
            return 0
    
    # Get time and frame indices at first moment
    if at_frame != None:
        at_frame_to = max(0, min(at_frame - 1, len(ts_to) - 1))    # to Python indexing
    elif at_time != None:
        at_frame_to = closest_element_index(ts_to, at_time)
    at_frame_from = closest_element_index(ts_from, ts_to[at_frame_to])
    at_frame_to = closest_element_index(ts_to, ts_from[at_frame_from])    # make sure both are close to eachother
    at_time = ts_to[at_frame_to]    # trajectory "to" is considered as the groundtruth one, hence the preference
    
    # Calculate rotation and fetch location, at first moment
    delta_quaternion = trfm.delta_quat(
            quats_to[at_frame_to].reshape(4, 1), quats_from[at_frame_from].reshape(4, 1) )
    location_from = locs_from[at_frame_from]
    location_to = locs_to[at_frame_to]
    
    delta_scale = 1.
    if infer_scale:
        # Get frame indices at second moment; implementation analogue to first moment
        if offset_frames != None:
            scnd_frame_to = max(0, min(at_frame_to + offset_frames, len(ts_to) - 1))
        elif offset_time != None:
            scnd_frame_to = closest_element_index(ts_to, at_time + offset_time)
        scnd_frame_from = closest_element_index(ts_from, ts_to[scnd_frame_to])
        scnd_frame_to = closest_element_index(ts_to, ts_from[scnd_frame_from])
        
        # Calculate scale:
        # first the "from" trajectory is transformed such that
        # the cam poses of both trajectories are equal at the first moment,
        # then the translation-vectors between first and second moment are calculated for both trajectories, ...
        inbetween_location_from = trfm.apply_quat_on_point(
                delta_quaternion, locs_from[scnd_frame_from] - locs_from[at_frame_from] ).reshape(3)
        inbetween_location_to   = locs_to[scnd_frame_to] - locs_to[at_frame_to]
        # ... then the translation-vector of the "to" trajectory is projection on
        # the normalized translation-vector of the "from" trajectory,
        # and the delta_scale is defined by the ratio of the length of this projection-vector,
        # with the length of the translation-vector of the "to" trajectory.
        nominator   = inbetween_location_from.dot(inbetween_location_to)
        denominator = inbetween_location_from.dot(inbetween_location_from)
        if denominator != 0:
            delta_scale = nominator / denominator
    
    # Return transformation: translation, rotation and scale
    delta_location = location_to - delta_scale * trfm.apply_quat_on_point(delta_quaternion, location_from).reshape(3)
    return delta_quaternion.reshape(4), delta_scale, delta_location


def transformed_points(points, transformation):
    """
    Return the transformation "transformation" applied on 3D points "points",
    where "transformation" equals ("delta_quaternion", "delta_scale", "delta_location") (apply from left to right).
    """
    delta_quaternion, delta_scale, delta_location = transformation
    delta_quaternion = delta_quaternion.reshape(4, 1)
    
    return np.array([
            delta_location + delta_scale * trfm.apply_quat_on_point(delta_quaternion, p).reshape(3)
            for p in points ])


def transformed_cam_trajectory(cam_trajectory, transformation):
    """
    Return the transformation "transformation" applied on camera trajectory "cam_trajectory",
    where the format of "cam_trajectory" is given by the output of "load_cam_trajectory_TUM()",
    and "transformation" equals ("delta_quaternion", "delta_scale", "delta_location") (apply from left to right).
    """
    timestps, locations, quaternions = cam_trajectory
    delta_quaternion, delta_scale, delta_location = transformation
    delta_quaternion = delta_quaternion.reshape(4, 1)
    
    timestps = np.array(timestps)
    locations = transformed_points(locations, transformation)
    quaternions = np.array([trfm.mult_quat(delta_quaternion, q).reshape(4) for q in quaternions])
    
    return timestps, locations, quaternions
