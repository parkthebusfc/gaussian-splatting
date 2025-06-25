
import cv2
import numpy as np
import glob
import os

# Settings
CHECKERBOARD = (9, 6)  # Number of inner corners in checkerboard 
SQUARE_SIZE = 0.025  # Size of one square in meters

def calibrate_stereo(left_folder, right_folder, output_file):
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    left_images = sorted(glob.glob(os.path.join(left_folder, "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))

    assert len(left_images) == len(right_images), "Number of left and right images must match!"

    for left_img_path, right_img_path in zip(left_images, right_images):
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

    # Calibrate individual cameras
    ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

#(x,y)=K⋅[R∣t]⋅P

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, K1, D1, K2, D2, gray_left.shape[::-1], flags=flags
    )

    np.savez(output_file, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
    print(f"Calibration saved to {output_file}")

if __name__ == "__main__":
    calibrate_stereo("calib_images/left", "calib_images/right", "stereo_calib.npz")
