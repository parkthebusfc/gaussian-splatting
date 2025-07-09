import cv2
import yaml
import numpy as np

def load_ros_yaml(yaml_path):
    """
    Loads a ROS-style YAML calibration file and returns K, D, R, P.
    """
    with open(yaml_path, "r") as f:
        calib = yaml.safe_load(f)
    K = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
    D = np.array(calib["distortion_coefficients"]["data"])
    R = np.array(calib["rectification_matrix"]["data"]).reshape(3, 3)
    P = np.array(calib["projection_matrix"]["data"]).reshape(3, 4)
    return K, D, R, P

def load_stereo_calibration(left_yaml, right_yaml, image_size):
    """
    Loads stereo calibration from left/right YAML files, creates rectification maps,
    and computes stereo baseline from the projection matrices.
    """
    K1, D1, R1, P1 = load_ros_yaml(left_yaml)
    K2, D2, R2, P2 = load_ros_yaml(right_yaml)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1[:3, :3], image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2[:3, :3], image_size, cv2.CV_16SC2)

    # Compute baseline: Tx/fx
    baseline = abs(P2[0, 3]) / P2[0, 0]

    return K1, left_map1, left_map2, right_map1, right_map2, baseline
