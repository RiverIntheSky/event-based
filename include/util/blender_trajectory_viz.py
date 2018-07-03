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


scene = bpy.data.scenes['Scene']

def draw_path(camera, trajectory_path):
    # Blender coordinate system is different than ours (z-axis is reversed)
    R_blender_cam = mathutils.Matrix(((1, 0, 0),
                                      (0, -1, 0),
                                      (0, 0, -1)))

    R_cam_blender = R_blender_cam

    # Parse trajectory file
    lines = [line.rstrip('\n') for line in open(trajectory_path)]
    trajectory = {}
    for line in lines:
        splitted = line.split(' ')
        # 60 fps
        frame_id = int(float(splitted[0])*60)

        t_world_cam = [float(i) for i in splitted[1:4]]
        q = [float(i) for i in splitted[4:]]

        T = (mathutils.Quaternion([q[3], q[0], q[1], q[2]]).to_matrix()*R_cam_blender).to_4x4()
        T.translation = mathutils.Vector([t_world_cam[0], t_world_cam[1], t_world_cam[2]])

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

    groundtruth_path = '/home/weizhen/Documents/dataset/poster_6dof/groundtruth.txt'
    draw_path(cam, groundtruth_path)
    estimated_path = '/home/weizhen/Documents/dataset/poster_6dof/planar/50001/estimated.txt'
    draw_path(cam2, estimated_path)


if __name__ == "__main__":
    main()
