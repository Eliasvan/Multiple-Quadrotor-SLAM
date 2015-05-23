"""
Blender addon to visualize pointclouds
======================================

Setup
-----
1.  Open Blender
2.  Go to File > User Preferences
3.  Go to the "Addons" tab and click "Install from File..."
4.  Search for this file ("blender_view3D_pointclouds.py")
5.  Check the checkbox right to the "3D View: Draw pointclouds" addon
5b. (optional) Expand the addon and modify the "Point Size" preference
6.  Click "Save User Settings" to enable this addon by default

Example
-------
1.  Select a Cube object
2.  In the Properties > Object > Display Panel, check the "Pointcloud" checkbox
3.  In the same panel, adjust the "Object Color" and watch the points' colors change
4.  In the same panel, check the "Transparent" checkbox and adjust the alpha value of "Object Color",
    and watch the transparency of points change

Note:
    It's also possible to load PCD files using
    the "blender_tools.import_points_from_pcd_file()" function and
    draw the colors of each individual point.
    If that pointcloud has stored individual color values,
    the RGB values of the "Object Color" property (see Example) will not have any effect,
    but the alpha value will have, at least when "Transparent" is checked.
"""

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Draw pointclouds",
    "author": "Elias Vanderstuyft",
    "version": (0, 1),
    "blender": (2, 69, 0),
    "location": "Properties > Object > Display Panel",
    "description": "Draw pointclouds' colors using bmesh vert custom layers 'r', 'g', 'b', 'a'",
    "note": "Based on code of 'Math Vis (Console)' by Campbell Barton",
    "category": "3D View" }

import bpy
import bmesh
import mathutils


callback_handle = []


class View3DPointcloudsPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    point_size = bpy.props.FloatProperty(
            name="Point Size",
            description="Size of points of a pointcloud - limited by glPointSize() implementation",
            default=5.0,
            min=1.0,
            max=20.0,
            step=1 )

    def draw(self, context):
        self.layout.prop(self, "point_size")


def object_display_panel(self, context):
    col = self.layout.split().column()
    ob = context.object
    
    # Add "is_pointcloud" boolean to the object display panel, in case of a mesh
    if ob.type == "MESH":
        col.prop(ob, "is_pointcloud")


def tag_redraw_all_view3d():
    # Py can't access notifers
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region.tag_redraw()


def draw_callback_view():
    pointclouds = [ob for ob in bpy.context.visible_objects if "is_pointcloud" in ob and ob["is_pointcloud"]]
    
    if not pointclouds:
        return
    
    from bgl import glPointSize, glEnable, GL_BLEND, glBegin, GL_POINTS, \
            glColor3f, glColor4f, glColor3ub, glColor4ub, glVertex3f, glEnd, glDisable
    
    point_size = bpy.context.user_preferences.addons[__name__].preferences.point_size
    glPointSize(point_size)
    glEnable(GL_BLEND)
    glBegin(GL_POINTS)
    
    for ob in pointclouds:
        matrix_world = ob.matrix_world
        color = tuple(ob.color)
        alpha = color[3]
        use_alpha = ob.show_transparent
        
        if "pointcloud_has_rgba" in ob and ob["pointcloud_has_rgba"]:    # draw individual vertex color
            bm = bmesh.new()
            bm.from_mesh(ob.data)
            
            # Get the custom data layer by its name
            r = bm.verts.layers.float['r']
            g = bm.verts.layers.float['g']
            b = bm.verts.layers.float['b']

            if use_alpha:
                a = bm.verts.layers.float['a']
                for vert in bm.verts:
                    glColor4f(vert[r], vert[g], vert[b], vert[a] * alpha)
                    glVertex3f(*(matrix_world * vert.co))
            else:
                for vert in bm.verts:
                    glColor3f(vert[r], vert[g], vert[b])
                    glVertex3f(*(matrix_world * vert.co))
            
            del bm
        
        else:    # draw same color for all vertices
            if use_alpha:
                glColor4f(*color)
            else:
                glColor3f(*color[0:3])
            for vert in ob.data.vertices:
                glVertex3f(*(matrix_world * vert.co))
    
    glEnd()
    glDisable(GL_BLEND)
    glPointSize(1.0)


def register():
    if callback_handle:
        return

    bpy.utils.register_class(View3DPointcloudsPreferences)
    
    bpy.types.Object.is_pointcloud = bpy.props.BoolProperty(
            name="Pointcloud",
            description="Draw vertices as a pointcloud",
            default=False )
    bpy.types.OBJECT_PT_display.append(object_display_panel)

    handle_view = bpy.types.SpaceView3D.draw_handler_add(draw_callback_view, (), 'WINDOW', 'POST_VIEW')
    callback_handle.append(handle_view)
    tag_redraw_all_view3d()


def unregister():
    if not callback_handle:
        return

    handle_view = callback_handle[0]
    bpy.types.SpaceView3D.draw_handler_remove(handle_view, 'WINDOW')
    callback_handle[:] = []
    tag_redraw_all_view3d()
    
    bpy.types.OBJECT_PT_display.remove(object_display_panel)
    del bpy.types.Object.is_pointcloud
    
    bpy.utils.unregister_class(View3DPointcloudsPreferences)
