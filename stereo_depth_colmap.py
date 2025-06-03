
import cv2
import numpy as np
import os
from colmap_utils import read_cameras_txt, read_images_txt

def load_intrinsics(cameras_path):
    cameras = read_cameras_txt(cameras_path)
    for cam_id, cam in cameras.items():
        fx = cam.params[0]
        fy = cam.params[1]
        cx = cam.params[2]
        cy = cam.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K

def get_camera_pose(image_data):
    qvec = image_data.qvec
    tvec = image_data.tvec
    R = qvec2rotmat(qvec)
    return R, tvec.reshape((3, 1))

def qvec2rotmat(qvec):
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def compute_stereo_depth(img1_path, img2_path, intrinsics, R1, t1, R2, t2):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    E, _ = cv2.findEssentialMat(pts1, pts2, intrinsics)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, intrinsics)

    proj1 = intrinsics @ np.hstack((R1, t1))
    proj2 = intrinsics @ np.hstack((R2, t2))

    points_4d = cv2.triangulatePoints(proj1, proj2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]
    depths = points_3d[2]

    return pts1, pts2, depths

if __name__ == "__main__":
    cameras_path = "Dataset/Caustics/Dept/sparse/0/cameras.txt"
    images_path = "Dataset/Caustics/Dept/sparse/0/images.txt"
    img_folder = "Dataset/Caustics/Dept/input"

    intrinsics = load_intrinsics(cameras_path)
    images = read_images_txt(images_path)

    # Use first two images as stereo pair
    ids = sorted(images.keys())
    img1_data = images[ids[0]]
    img2_data = images[ids[1]]

    R1, t1 = get_camera_pose(img1_data)
    R2, t2 = get_camera_pose(img2_data)

    img1_path = os.path.join(img_folder, img1_data.name)
    img2_path = os.path.join(img_folder, img2_data.name)

    pts1, pts2, depths = compute_stereo_depth(img1_path, img2_path, intrinsics, R1, t1, R2, t2)

    print(f"Computed {len(depths)} stereo depths.")
