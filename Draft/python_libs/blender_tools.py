import sys
import traceback
import os
from math import radians
import numpy as np

try:
    import bpy    # tested in Blender v2.69
    import bmesh
    from mathutils import Quaternion
except ImportError:
    print ("Warning: can't load Blender modules required for most functions of \"blender_tools\" module.")

import dataset_tools



""" Helper functions """

def get_objects(by_selection=False, name_starts_with="", name_ends_with=""):
    """
    Return a list of tuples, each containing the name of the object, and the object itself.
    The entries are sorted by object-name.
    
    "by_selection" :
        if True, only the selected objects are considered,
        otherwise the objects of the current scene are considered
    "name_starts_with" : prefix of object name
    "name_ends_with" : suffix of object name
    """
    
    if by_selection:
        objects = bpy.context.selected_objects
    else:
        objects = bpy.context.scene.objects
    
    return sorted([
            (ob.name, ob) for ob in objects
            if ob.name.startswith(name_starts_with) and ob.name.endswith(name_ends_with) and ob.type == "MESH" ])

def object_name_from_filename(filename, name_prefix="", strip_file_extension=True):
    """
    Create an object-name corresponding with the filename "filename" of the file containing
    the data to represent the object.
    
    "name_prefix" : prefix for the object-name
    "strip_file_extension" : if True, omit the file-extension in the object-name
    """
    name = bpy.path.basename(filename)
    
    if strip_file_extension:
        name = os.path.splitext(name)[0]
    
    return name_prefix + name

def backup_ob_selection():
    """
    Backup the current mode and selection.
    """
    mode_backup = bpy.context.mode
    selected_objects_backup = bpy.context.selected_objects
    active_object_backup = bpy.context.active_object
    return mode_backup, selected_objects_backup, active_object_backup

def restore_ob_selection(mode_backup, selected_objects_backup, active_object_backup):
    """
    Restore original mode and selection.
    Input should be "*backup_ob_selection()".
    """
    if bpy.context.mode != "OBJECT": bpy.ops.object.mode_set()    # switch to object mode
    bpy.ops.object.select_all(action="DESELECT")
    for ob in selected_objects_backup: ob.select = True
    bpy.context.scene.objects.active = active_object_backup
    bpy.ops.object.mode_set(mode_backup)

def backup_anim_state():
    """
    Backup the current framenr and active layers.
    """
    frame_current_backup = bpy.context.scene.frame_current
    layers_backup = tuple(bpy.context.scene.layers)
    return frame_current_backup, layers_backup

def restore_anim_state(frame_current_backup, layers_backup):
    """
    Restore the original framenr and active layers.
    Input should be "*backup_anim_state()".
    """
    bpy.context.scene.layers = layers_backup
    bpy.context.scene.frame_current = frame_current_backup


""" Functions related to cameras """

def print_pose(rvec, tvec):
    """
    Some debug printing of the camera pose projection matrix,
    the printed output can be used in Blender to visualize this camera pose,
    using the "create_pose_camera()" function.
    
    "rvec", "tvec" : defining the camera pose, compatible with OpenCV's output
    """
    import cv2
    import transforms as trfm
    
    ax, an = trfm.axis_and_angle_from_rvec(-rvec)
    
    print ("axis, angle = \\")
    print (list(ax.reshape(-1)), an)    # R^(-1)
    
    print ("pos = \\")
    print (list(-cv2.Rodrigues(-rvec)[0].dot(tvec).reshape(-1)))    # -R^(-1) * t

def create_camera_pose(name, axis, angle, pos):
    """
    Create a camera named "name" by providing information of the pose.
    Unit pose corresponds with a camera at the origin,
    with view-direction lying along the +Z-axis,
    and with the +Y-axis facing downwards in the camera frustrum.
    
    "pos" : the camera center with respect to the world origin
    "axis", "angle" : axis-angle representation of the orientation of the camera with respect to the world origin
    
    The "print_pose()" function can be used to print the input for this function,
    useful to visualize OpenCV's "rvec" and "tvec".
    """
    #name = name_camera + "_" + suffix
    
    if name in bpy.data.objects and bpy.data.objects[name].type == "CAMERA":
        ob = bpy.data.objects[name]
    else:
        if bpy.context.mode != "OBJECT": bpy.ops.object.mode_set()    # switch to object mode
        bpy.ops.object.camera_add()
        ob = bpy.context.object
        ob.name = name
    
    ob.rotation_mode = "AXIS_ANGLE"
    ob.rotation_axis_angle[0] = angle[0]
    ob.rotation_axis_angle[1] = axis[0]
    ob.rotation_axis_angle[2] = axis[1]
    ob.rotation_axis_angle[3] = axis[2]
    
    ob.location[0] = pos[0]
    ob.location[1] = pos[1]
    ob.location[2] = pos[2]
    
    ob.rotation_mode = "QUATERNION"    # rotate 180 deg around local X because a blender camera has Y and Z axes opposite to OpenCV's
    ob.rotation_quaternion *= Quaternion((1.0, 0.0, 0.0), radians(180.0))

def extract_current_pose():
    """
    Convert current object's pose to OpenCV's "rvec" and "tvec".
    """
    
    ob = bpy.context.object
    if ob.rotation_mode == "QUATERNION":
        q = ob.rotation_quaternion
    elif ob.rotation_mode == "AXIS_ANGLE":
        q = Quaternion(ob.rotation_axis_angle[1:4], ob.rotation_axis_angle[0])
    else:
        assert ob.rotation_mode in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
        q = ob.rotation_euler.to_quaternion()
    
    # Rotate 180 deg around local X because a blender camera has Y and Z axes opposite to OpenCV's
    q *= Quaternion((1.0, 0.0, 0.0), radians(180.0))
    
    aa = q.to_axis_angle()
    rvec = [c * -aa[1] for c in aa[0]]
    tvec = list(q.inverted() * (-ob.location))
    
    return rvec, tvec

def extract_pose_to_ASCII(filename, pose_ob_name=None):
    """
    Extract the 4x4 pose of object "pose_ob_name" (e.g. a cam)
    to ASCII file "filename" (MATLAB/Octave compatible).
    The extracted pose is returned.
    
    If "pose_ob_name" is set to None, the currently active object will be chosen.
    """
    
    if pose_ob_name:
        ob = bpy.data.objects[pose_ob_name]
    else:
        ob = bpy.context.object
    
    # Backup original scale and normalize scale
    scale = tuple(ob.scale)
    ob.scale = (1, 1, 1)
    
    # Rotate 180 deg around local X because a blender camera has Y and Z axes opposite to OpenCV's
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion *= Quaternion((1.0, 0.0, 0.0), radians(180.0))
    
    # Extract pose
    bpy.context.scene.update()
    pose = np.array([tuple(row) for row in ob.matrix_world.inverted()])
    
    # Restore original rotation and scale
    ob.rotation_quaternion *= Quaternion((1.0, 0.0, 0.0), radians(180.0))
    ob.scale = scale
    
    np.savetxt(filename, pose)
    return pose

def create_cam_trajectory(name,
                          locations, quaternions,
                          start_frame=1, framenrs=None,
                          no_keyframe_highlighting=False,
                          select_ob=True, goto_last_keyframe=False):
    """
    "name" : name of Camera to be created
    "locations" : list of cam center positions for each trajectory node
    "quaternions" : list of cam orientation (quaternion (qx, qy, qz, qw)) for each trajectory node
    "start_frame" : start frame of the trajectory
    "framenrs" : list of frame numbers for each trajectory node (should be in increasing order)
    "no_keyframe_highlighting" : if True, don't show framenumbers for each keyframe along trajectory
    "select_ob" : select the camera object (and trajectory)
    "goto_last_keyframe" : go to the last keyframe of the generated trajectory
    """
    
    if not select_ob:
        ob_selection_backup = backup_ob_selection()
    anim_state_backup = list(backup_anim_state())
    
    # Create the camera
    if bpy.context.mode != "OBJECT": bpy.ops.object.mode_set()    # switch to object mode
    if name in bpy.data.objects and bpy.data.objects[name].type == "CAMERA":
        ob = bpy.data.objects[name]
        bpy.ops.object.select_all(action="DESELECT")
        ob.select = True
        bpy.context.scene.objects.active = ob
        bpy.context.object.animation_data_clear()    # clear all previous keyframes
        bpy.context.scene.layers = bpy.context.object.layers    # only activate object's layers to insert keyframes
    else:
        bpy.ops.object.camera_add()
        ob = bpy.context.object
        ob.name = name
    
    # Unhide object
    ob_hide_backup = ob.hide
    ob.hide = False
    
    ob.rotation_mode = "QUATERNION"
    
    # Create path of the camera
    for i, (location, quaternion) in enumerate(zip(locations, quaternions)):
        bpy.context.scene.frame_current = framenrs[i] if framenrs != None else i + 1
        
        ob.location = list(location)
        
        qx, qy, qz, qw = quaternion
        ob.rotation_quaternion = [qw, qx, qy, qz]
        # We assume the TUM format uses the OpenCV cam convention (Z-axis in direction of view, Y-axis down)
        # so we'll have to convert, since Blender follows OpenGL convention
        ob.rotation_quaternion *= Quaternion((1.0, 0.0, 0.0), radians(180.0))
        
        bpy.ops.anim.keyframe_insert_menu(type="BUILTIN_KSI_LocRot")
    
    # Visualize path
    ob.animation_visualization.motion_path.show_keyframe_highlight = \
            (framenrs != None and not no_keyframe_highlighting)
    if framenrs != None:
        bpy.ops.object.paths_calculate(start_frame=framenrs[0], end_frame=framenrs[-1])
    else:
        bpy.ops.object.paths_calculate(start_frame=start_frame, end_frame=start_frame + len(locations))
    
    # Restore hide-state
    ob.hide = ob_hide_backup
    
    if goto_last_keyframe:
        anim_state_backup[0] = bpy.context.scene.frame_current
    restore_anim_state(*anim_state_backup)
    if not select_ob:
        restore_ob_selection(*ob_selection_backup)

def load_and_create_cam_trajectory(filename, name_prefix="", strip_file_extension=True,
                                   start_frame=1, start_time=None, fps="data",
                                   no_keyframe_highlighting=True, goto_last_keyframe=False):
    """
    Load a camera trajectory (in the TUM format, see "dataset_tools" module for more info)
    with filename "filename", and create it in Blender.
    
    "name_prefix", "strip_file_extension" : see documentation of "object_name_from_filename()"
    "start_frame" : Blender's start frame of the trajectory
    "start_time" : if not None, this float will be used as trajectory's timestamp offset to start from
    "fps" : should be one of the following:
            - "blender" : infer the fps from Blender's scene "fps" property
            - "data" : infer the fps from the data (using minimum delta time), Blender's "fps" will be adjusted
            - an integer indicating the data's fps
    "no_keyframe_highlighting" : if True, don't show framenumbers for each keyframe along trajectory
    "goto_last_keyframe" : go to the last keyframe of the generated trajectory
    
    Note: to synchronize multiple trajectories in time,
    it is advised to set:
            - "start_frame" to 0
            - "start_time" to 0.
            - "fps" to the exact fps integer
    """
    
    timestps, locations, quaternions = dataset_tools.load_cam_trajectory_TUM(filename)
    
    if len(timestps) == 0:
        return
    elif len(timestps) == 1:
        framenrs = [start_frame]
    else:
        if fps == "blender":
            fps = bpy.context.scene.render.fps
        elif fps == "data":
            fps = 1. / np.min(timestps[1:] - timestps[:-1])
            bpy.context.scene.render.fps = int(round(fps))
        if start_time == None:
            start_time = timestps[0]
        framenrs = np.rint(start_frame + (timestps - start_time) * float(fps)).astype(int)
    
    create_cam_trajectory(
            object_name_from_filename(filename, name_prefix, strip_file_extension),
            locations, quaternions, start_frame, framenrs, no_keyframe_highlighting,
            goto_last_keyframe=goto_last_keyframe )


""" Functions related to 3D geometry """

def create_mesh(name, coords, connect=False, edges=None):
    """
    Create a mesh with name "name" from a list of vertices "coords".
    Return the generated object, and "is_new_ob" which is True when a new object has been created.
    
    If "connect" is True, two successive (in order) vertices
    will be linked together by an edge.
    Otherwise, if "edges" list is specified, each element is a tuple of 2 vertex indices,
    to be linked together with an edge.
    """
    
    # Create a new mesh
    mesh_name = name + "Mesh"
    me = bpy.data.meshes.new(mesh_name)    
    
    is_new_ob = not (name in bpy.data.objects and bpy.data.objects[name].type == "MESH")
    if is_new_ob:
        ob = bpy.data.objects.new(name, me)    # create an object with that mesh
        bpy.context.scene.objects.link(ob)    # link object to scene
    else:
        if bpy.context.mode != "OBJECT": bpy.ops.object.mode_set()    # switch to object mode
        ob = bpy.data.objects[name]
        me_old = ob.data
        ob.data = me
        bpy.data.meshes.remove(me_old)
    
    ob.location = [0, 0, 0]    # position object at origin
    
    # Define the edge by index numbers of its vertices. Each edge is defined by a tuple of 2 integers.
    if connect:
        edges = list(zip(range(len(coords) - 1), range(1, len(coords))))
    elif not edges:
        edges = []
    
    # Define the faces by index numbers of its vertices. Each face is defined by a tuple of 3 or more integers.
    # N-gons would require a tuple of size N.
    faces = []
    
    # Fill the mesh with verts, edges, faces
    me.from_pydata(coords, edges, faces)    # edges or faces should be [], or you ask for problems
    me.update(calc_edges=True)    # update mesh with new data
    
    return ob, is_new_ob


""" File- import/export functions """

def extract_points_to_MATLAB(filename,
                             by_selection=False, name_starts_with="", name_ends_with="",
                             var_name="scene_3D_points"):
    """
    Extract 3D coordinates of vertices of meshes to a MATLAB/Octave compatible file.
    The resulting vertices are sorted by object-name.
    
    "filename" : .mat-file to save to
    "by_selection", "name_starts_with", "name_ends_with" : see documentation of "get_objects()"
    "var_name" : MATLAB variable name in which the data will be stored
    """
    import scipy.io as sio
    
    verts = []
    for ob_name, ob in get_objects(by_selection, name_starts_with, name_ends_with):
        verts += [tuple(ob.matrix_world * vertex.co) for vertex in ob.data.vertices]
    verts = np.array(verts)
    
    sio.savemat(filename, {var_name: verts})

def extract_points_to_ply_file(filename, by_selection=False, name_starts_with="", name_ends_with=""):
    """
    Extract 3D coordinates of vertices of meshes to a PointCloud (.ply) file.
    
    "filename" : .ply-file to save to
    "by_selection", "name_starts_with", "name_ends_with" : see documentation of "get_objects()"
    
    Note: at least 3 vertices should be extracted (in total).
    """
    ob_selection_backup = backup_ob_selection()
    
    # Select to-be-exported objects
    if bpy.context.mode != "OBJECT": bpy.ops.object.mode_set()    # switch to object mode
    bpy.ops.object.select_all(action="DESELECT")
    for ob_name, ob in get_objects(by_selection, name_starts_with, name_ends_with):
        ob.select = True
        bpy.context.scene.objects.active = ob
    
    # Join to-be-exported objects into one temporary mesh
    bpy.ops.object.duplicate()
    if len(bpy.context.selected_objects) > 1:
        bpy.ops.object.join()
    
    # Remove all edges and faces, and add dummy faces (required for ply-exporter)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete(type='EDGE_FACE')
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.edge_face_add()
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.editmode_toggle()
    
    bpy.ops.export_mesh.ply(filepath=filename, use_normals=False, use_uv_coords=False)    # export to ply-file
    bpy.ops.object.delete(use_global=True)    # remove temporary mesh
    
    restore_ob_selection(*ob_selection_backup)

def extract_points_to_pcd_file(filename, by_selection=False, name_starts_with="", name_ends_with=""):
    """
    Extract 3D coordinates of vertices of meshes to a PointCloud (.pcd) file.
    The resulting vertices are sorted by object-name.
    
    "filename" : .pcd-file to save to
    "by_selection", "name_starts_with", "name_ends_with" : see documentation of "get_objects()"
    
    Note: currently, colors are not exported.
    """
    
    verts = []
    for ob_name, ob in get_objects(by_selection, name_starts_with, name_ends_with):
        verts += [tuple(ob.matrix_world * vertex.co) for vertex in ob.data.vertices]
    verts = np.array(verts)
    
    dataset_tools.save_3D_points_to_pcd_file(filename, verts)

def import_points_from_pcd_file(filename, name_prefix="", select_ob=True):
    """
    Import 3D coordinates of vertices from a PointCloud (.pcd) file.
    
    "name_prefix" : see documentation of "object_name_from_filename()"
    "select_ob" : select the imported pointcloud object
    
    Note: currently, colors are not yet supported.
    """
    
    # Import point cloud
    verts, colors, found_alpha = dataset_tools.load_3D_points_from_pcd_file(filename, use_alpha=True)
    ob, is_new_ob = create_mesh(object_name_from_filename(filename, name_prefix), verts)
    
    # Mark object as a pointcloud
    ob["is_pointcloud"] = True
    
    # Mark object as a colored pointcloud, if applicable
    ob["pointcloud_has_rgba"] = (colors != None)
    if ob["pointcloud_has_rgba"]:
        if is_new_ob:    # only change transparency in this case
            ob.show_transparent = found_alpha    # render alpha channel
        
        # Open mesh for editing vertex custom data
        bm = bmesh.new()
        bm.from_mesh(ob.data)

        # Create custom data layers, for the vertex color
        r = bm.verts.layers.float.new('r')
        g = bm.verts.layers.float.new('g')
        b = bm.verts.layers.float.new('b')
        a = bm.verts.layers.float.new('a')

        # Color vertices
        for vert, color in zip(bm.verts, colors / 255.):
            vert[b], vert[g], vert[r] = color[0:3]
            vert[a] = color[3]
        
        # Save colors
        bm.to_mesh(ob.data)
        del bm
    
    if select_ob:
        # Select the generated object
        if bpy.context.mode != "OBJECT": bpy.ops.object.mode_set()    # switch to object mode
        bpy.ops.object.select_all(action="DESELECT")
        ob.select = True
        bpy.context.scene.objects.active = ob


""" Event handling functionality """

file_listener = []

def run_file_listener(filepaths, handler, notification="FileListener running", timer_timeout=1.):
    """
    Create a file listener, checking files in the "filepaths" list for change at period "timer_timeout".
    In the top header bar, the text defined in "notification" will be displayed to notify the user.
    Press ESCAPE to stop the file listener.
    
    When a change occurs (initially all files are assumed to not exist), function "handler" will be called
    with the following signature:
        def handler(self, changed_files, filepaths, filepath_statuses):
            ...
    where:
        "changed_files" : list of indices in "filepaths" for which their statuses changed
        "filepaths" : reference to the original "filepaths" list
        "filepath_statuses" : status for each file in "filepaths": file-size in bytes, or None if file doesn't exist
        "self" : instance of ModalFileListenerOperator, ignore this
    
    Note: a 'change' means either change in existence of the file, or change in file-size.
    """
    
    # Unregister previous ModalFileListenerOperator class, if present
    if file_listener:
        bpy.utils.unregister_class(file_listener[0])
        file_listener[:] = []
    
    # Notification class, to make the user aware ModalFileListenerOperator is running
    class INFO_HT_FileListenerNotification(bpy.types.Header):
        bl_label = "File Listener Notification"
        bl_space_type = "INFO"
        
        _notification = notification
        
        def draw(self, context):
            self.layout.label(text=self._notification, icon="TRIA_RIGHT")
    
    # Define new singleton class, using new arguments as parameters
    class ModalFileListenerOperator(bpy.types.Operator):
        """Operator which runs its self from a timer"""
        bl_idname = "wm.modal_file_listener_operator"
        bl_label = "Modal File Listener Operator"

        _timer = None
        _timer_timeout = timer_timeout
        _filepaths = filepaths
        _handler = handler
        
        _filepath_statuses = [None] * len(filepaths)    # initially assume file doesn't exist
        _notification_class = INFO_HT_FileListenerNotification

        def modal(self, context, event):
            if event.type == 'ESC':
                return self.cancel(context)

            if event.type == 'TIMER':
                changed_files = []
                
                for i, (filepath, filepath_status) in enumerate(zip(self._filepaths, self._filepath_statuses)):
                    filepath_status_new = None    # by default assume file doesn't exist
                    try:
                        filepath_status_new = os.stat(filepath).st_size
                    except OSError:
                        pass    # file probably doesn't exist
                    
                    if filepath_status_new != filepath_status:
                        changed_files.append(i)
                        self._filepath_statuses[i] = filepath_status_new
                
                if changed_files:
                    try:
                        self._handler(changed_files, self._filepaths, self._filepath_statuses)
                    except:
                        msg = "Exception in handler of %s" % self.__class__.__name__
                        self.report({"ERROR"}, "%s, check console" % msg)
                        print ("%s:" % msg)
                        print ("-" * 60)
                        traceback.print_exc(file=sys.stdout)
                        print ("-" * 60)

            return {'PASS_THROUGH'}

        def execute(self, context):
            self._timer = context.window_manager.event_timer_add(self._timer_timeout, context.window)
            context.window_manager.modal_handler_add(self)
            bpy.utils.register_class(self._notification_class)
            return {'RUNNING_MODAL'}

        def cancel(self, context):
            context.window_manager.event_timer_remove(self._timer)
            bpy.utils.unregister_class(self._notification_class)
            return {'CANCELLED'}
    
    # Register new ModalFileListenerOperator class
    file_listener.append(ModalFileListenerOperator)
    bpy.utils.register_class(file_listener[0])
    
    # Call the registered "Modal File Listener Operator"
    bpy.ops.wm.modal_file_listener_operator()
