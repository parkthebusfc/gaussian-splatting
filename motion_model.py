
import numpy as np
import cv2

def project_pixel_to_frame_j(pixel_i, depth_i, K, pose_i, pose_j):
    ''' Project a pixel from frame i to frame j using depth and camera poses '''
    u, v = pixel_i
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    point_camera_i = np.array([x * depth_i, y * depth_i, depth_i, 1.0])

    T_i_inv = np.linalg.inv(pose_i)
    point_world = T_i_inv @ point_camera_i

    point_camera_j = pose_j @ point_world
    if point_camera_j[2] <= 0:
        return None  # Behind the camera

    point_proj = K @ (point_camera_j[:3] / point_camera_j[2])
    return int(point_proj[0]), int(point_proj[1])
