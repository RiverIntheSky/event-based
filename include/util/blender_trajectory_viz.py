#!/usr/bin/env python

'''
Loads a camera trajectory from a file into Blender

Trajectory file should contain one line per frame containing the pose
(transformation from cam to world)

Line formatting is as follows:

timestamp tx ty tz qx qy qz qw

usage:

blender scene.blend -P blender_trajectory_viz.py


'''

import os
import sys
sys.path.append('/usr/lib/python3/dist-packages') # Add python3 packages to the environment
import mathutils
import bpy
import math


scene = bpy.data.scenes['Scene']

def draw_path(camera, rotation, translation):
    # Blender coordinate system is different than ours (z-axis is reversed)
    R_blender_cam = mathutils.Matrix(((1, 0, 0),
                                      (0, -1, 0),
                                      (0, 0, -1)))

    R_cam_blender = R_blender_cam

    # Parse trajectory file
    rotation_lines = [line.rstrip('\n') for line in open(rotation)]
    translation_lines = [line.rstrip('\n') for line in open(translation)]
    trajectory = {}
    for i in range(len(rotation_lines)):
        splitted = translation_lines[i].split(' ')
        # 60 fps
        frame_id = int(float(splitted[0])*60)

        t_world_cam = [float(i) * 1.1674 for i in splitted[1:4]] 

        splitted = rotation_lines[i].split(' ')

        w = [float(i) for i in splitted[1:4]]

        angle = math.sqrt(w[0]**2 + w[1]**2 + w[2]**2)

        if abs(angle)<1e-6:
            q = [1, 0, 0, 0]
        else:
            q = [w[0] * math.sin(angle/2) / angle, w[1] * math.sin(angle/2) / angle, w[2] * math.sin(angle/2) / angle, math.cos(angle/2)]

        T = (mathutils.Quaternion([q[3], q[0], q[1], q[2]]).to_matrix()*R_cam_blender).to_4x4()
        T.translation = R_blender_cam * mathutils.Vector([t_world_cam[0], t_world_cam[1], t_world_cam[2]])

        trajectory[frame_id] = T

    # Copy camera trajectory into current scene
    camera.animation_data_clear()
    scene.frame_start = min(trajectory.keys())
    scene.frame_end = max(trajectory.keys())

    for frame_id, T in trajectory.items():
        scene.frame_current = frame_id
        camera.matrix_world = T

        camera.keyframe_insert(data_path="location", frame=frame_id)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_id)

    bpy.context.scene.frame_set(scene.frame_start)


    # motion paths visualization
    scene.render.fps = 60
    bpy.context.scene.objects.active = camera
    camera.select = True
    bpy.ops.object.paths_calculate(start_frame=scene.frame_start,end_frame=scene.frame_end)
    camera.animation_visualization.motion_path.show_keyframe_numbers = False
    camera.animation_visualization.motion_path.show_keyframe_highlight = False
    camera.scale[2]=0.1


def main():
    cam = bpy.data.objects['Camera']

    cam2 = cam.copy()
    cam2.data = cam.data.copy()
    scene.objects.link(cam2)

    groundtruth_translation_path = '/home/weizhen/Dropbox/MA/thesis defense/data/shapes_6dof_4000/groundtruth_pose_translation.txt'
    groundtruth_rotation_path = '/home/weizhen/Dropbox/MA/thesis defense/data/shapes_6dof_4000/groundtruth_pose_rotation.txt'
    draw_path(cam, groundtruth_rotation_path, groundtruth_translation_path)
    estimated_translation_path = '/home/weizhen/Dropbox/MA/thesis defense/data/shapes_6dof_4000/estimated_pose_translation.txt'
    estimated_rotation_path = '/home/weizhen/Dropbox/MA/thesis defense/data/shapes_6dof_4000/estimated_pose_rotation.txt'
    draw_path(cam2, estimated_rotation_path, estimated_translation_path)


if __name__ == "__main__":
    main()
