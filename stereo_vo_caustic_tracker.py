
import cv2
import numpy as np
import os

# Load calibration parameters (stereo calibration must be precomputed)
calib = np.load("stereo_calib.npz")
K = calib["K1"]
D = calib["D1"]
R_stereo = calib["R"]
T_stereo = calib["T"]
baseline = np.linalg.norm(T_stereo)

# Parameters
HSV_V_THRESHOLD = 200
window_size = 5  # Block matching window size for disparity
min_disp = 0
num_disp = 64  # Must be divisible by 16

# Stereo matcher initialization (Semi-Global Block Matching)
stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Motion model projection function
def project_pixel_motion(u, v, z, T_i, T_j, K):
    point_camera_i = np.linalg.inv(K) @ np.array([u * z, v * z, z])
    point_camera_i_h = np.append(point_camera_i, 1)
    world_point = np.linalg.inv(T_i) @ point_camera_i_h
    point_camera_j = T_j @ world_point
    if point_camera_j[2] <= 0:
        return None
    pixel_proj = K @ (point_camera_j[:3] / point_camera_j[2])
    return int(pixel_proj[0]), int(pixel_proj[1])

# Simplified monocular visual odometry
class VisualOdometry:
    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_kp = None
        self.last_des = None
        self.pose = np.eye(4)

    def process(self, img_gray):
        kp, des = self.orb.detectAndCompute(img_gray, None)
        if self.last_des is None:
            self.last_kp, self.last_des = kp, des
            return self.pose

        matches = self.bf.match(self.last_des, des)
        pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        if len(pts1) < 8:
            return self.pose  # Not enough points

        E, _ = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.flatten()
        self.pose = self.pose @ np.linalg.inv(T)

        self.last_kp, self.last_des = kp, des
        return self.pose

def compute_depth_map(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.1  # Prevent division by zero
    depth = (K[0,0] * baseline) / disparity
    return depth

def run_stereo_vo_caustic(left_folder, right_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    left_images = sorted([os.path.join(left_folder, f) for f in os.listdir(left_folder)])
    right_images = sorted([os.path.join(right_folder, f) for f in os.listdir(right_folder)])
    vo = VisualOdometry(K)
    caustic_mask_prev = None
    pose_prev = np.eye(4)

    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

        # Visual odometry pose estimation (monocular VO)
        pose_now = vo.process(gray_left)

        # Compute stereo depth
        depth_map = compute_depth_map(img_left, img_right)

        # Caustic detection
        hsv = cv2.cvtColor(img_left, cv2.COLOR_BGR2HSV)
        _, caustic_mask_now = cv2.threshold(hsv[:,:,2], HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Propagate previous caustics using motion model + depth
        if caustic_mask_prev is not None:
            h, w = caustic_mask_prev.shape
            propagated_mask = np.zeros_like(caustic_mask_prev)
            for v in range(0, h, 5):
                for u in range(0, w, 5):
                    if caustic_mask_prev[v, u] == 255:
                        z = depth_map[v, u]
                        proj = project_pixel_motion(u, v, z, pose_prev, pose_now, K)
                        if proj is not None:
                            u_new, v_new = proj
                            if 0 <= u_new < w and 0 <= v_new < h:
                                if caustic_mask_now[v_new, u_new] == 255:
                                    propagated_mask[v_new, u_new] = 255

            combined_mask = cv2.bitwise_or(propagated_mask, caustic_mask_now)
            cv2.imwrite(os.path.join(output_folder, f"mask_{idx:04d}.png"), combined_mask)
            caustic_mask_prev = combined_mask.copy()
            pose_prev = pose_now.copy()
        else:
            cv2.imwrite(os.path.join(output_folder, f"mask_{idx:04d}.png"), caustic_mask_now)
            caustic_mask_prev = caustic_mask_now.copy()
            pose_prev = pose_now.copy()

if __name__ == "__main__":
    run_stereo_vo_caustic("dataset/left", "dataset/right", "output/masks")
