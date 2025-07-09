import cv2
import numpy as np

class StereoVisualOdometry:
    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_3D_points = None
        self.prev_des = None
        self.pose = np.eye(4)  # Pose initialized only once here

    def initialize_map(self, keypoints, depth_map, descriptors):
        points_3D = []
        valid_des = []
        for i, pt in enumerate(keypoints):
            u, v = map(int, pt.pt)
            if 0 <= u < depth_map.shape[1] and 0 <= v < depth_map.shape[0]:
                Z = depth_map[v, u]
                if np.isfinite(Z) and 0.1 < Z < 100.0:
                    X = (u - self.K[0,2]) * Z / self.K[0,0]
                    Y = (v - self.K[1,2]) * Z / self.K[1,1]
                    points_3D.append([X, Y, Z])
                    valid_des.append(descriptors[i])
        self.prev_3D_points = np.array(points_3D, dtype=np.float32)
        self.prev_des = np.array(valid_des)

    def process_frame(self, gray, depth_map):
        kp, des = self.orb.detectAndCompute(gray, None)
        if kp is None or des is None or len(kp) == 0:
            print("[WARN] No features detected in current frame.")
            return self.pose, kp, des

        if self.prev_3D_points is None or self.prev_des is None:
            self.initialize_map(kp, depth_map, des)
            return self.pose, kp, des

        # Match features
        matches = self.bf.match(self.prev_des, des)
        if len(matches) < 8:
            print(f"[WARN] Not enough matches: {len(matches)} (need >=8). Skipping pose update.")
            return self.pose, kp, des

        # Gather 2D-3D correspondences
        pts3D = np.array([self.prev_3D_points[m.queryIdx] for m in matches], dtype=np.float32)
        pts2D = np.array([kp[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Solve PnP
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts3D, pts2D, self.K, None)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        self.pose = self.pose @ np.linalg.inv(T)

        print(f"[POSE] Current translation: {self.pose[:3, 3]}")  # Debug print

        self.initialize_map(kp, depth_map, des)
        return self.pose, kp, des


# check for relative transformations for each pose, show matches after RANSAC. Also check for initialization. check for units, depth was at max 12-18ft.