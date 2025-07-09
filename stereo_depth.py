import cv2
import numpy as np

def create_stereo_matcher():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

def compute_depth(disparity, K, baseline):
    disparity[disparity <= 0] = 0.1
    return (K[0,0] * baseline) / disparity
