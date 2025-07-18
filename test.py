# import cv2
# import numpy as np
# import glob
# import os

# image_dir = "Dataset/test"
# image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

# HSV_V_THRESHOLD = 190  
# DILATION_KERNEL_SIZE = 5 

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE))

# prev_frame = None
# prev_mask = None

# running_sum = None
# running_count = None

# for image_path in image_paths:
#     current_frame = cv2.imread(image_path)
#     if current_frame is None:
#         continue

#     cv2.imshow("Current Frame", current_frame)

#     if running_sum is None:
#         height, width, channels = current_frame.shape
#         running_sum = np.zeros((height, width, channels), dtype=np.float32)
#         running_count = np.zeros((height, width), dtype=np.uint16)

#     hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
#     _, mask = cv2.threshold(hsv[:, :, 2], HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)
#     mask_dilated = cv2.dilate(mask, kernel, iterations=1)

#     cv2.imshow('mask', mask)
#     cv2.imshow('mask_dilated', mask_dilated)

#     mask_curr_bin = (mask_dilated > 127).astype(np.uint8)  #A

#     if prev_mask is not None:
#         mask_prev_bin = (prev_mask > 127).astype(np.uint8)  #B

#         # Pixels where caustics have newly appeared: currently caustic & previously clear
#         newly_occluded_pixels = (mask_curr_bin == 1) & (mask_prev_bin == 0)

#         # Replace newly occluded pixels in the current frame with previous stable values
#         current_frame[newly_occluded_pixels] = prev_frame[newly_occluded_pixels]

#         #  newly occluded pixels
#         debug_newly_occluded_mask = np.zeros_like(mask_curr_bin, dtype=np.uint8)
#         debug_newly_occluded_mask[newly_occluded_pixels] = 255
#         cv2.imshow("Newly Occluded Pixels Mask (A ∩ (A - B))", debug_newly_occluded_mask)

#     # Stable pixels: currently not caustic
#     stable_pixels = mask_curr_bin == 0

    
#     running_sum[stable_pixels] += current_frame[stable_pixels].astype(np.float32)
#     running_count[stable_pixels] += 1

#     # Compute temporally averaged output from stable pixels
#     average_output = np.zeros_like(current_frame, dtype=np.uint8)
#     valid = running_count > 0
#     average_output[valid] = (running_sum[valid] / running_count[valid, None]).astype(np.uint8)

#     cv2.imshow("Caustic Removed Averaged Output", average_output)

#     prev_frame = current_frame.copy()  
#     prev_mask = mask_dilated.copy()
    
#     key = cv2.waitKey(0)  
#     if key & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

#### if caustics in last frame, not covered in any frame, then crop firs
#calibar
#generate an image by increasing v


# --- Same imports as before ---
# Updated version of your script with:
# 1. Confidence filtering for matches
# 2. Patch-based recovery using pose transformations
# 3. Debug visualizations to inspect patch replacements

import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from calibration import load_stereo_calibration
from stereo_depth import create_stereo_matcher, compute_depth
from stereo_vo import StereoVisualOdometry

# === Calibration and setup ===
K1, left_map1, left_map2, right_map1, right_map2, baseline = load_stereo_calibration(
    "gopro_calibration/left.yaml", "gopro_calibration/right.yaml", (1920, 1080))
stereo_matcher = create_stereo_matcher()
vo = StereoVisualOdometry(K1)
orb = cv2.ORB_create(3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

left_images = sorted(glob.glob("Dataset/gt_stereo/left/*.png"))
right_images = sorted(glob.glob("Dataset/gt_stereo/right/*.png"))

print(f"[INFO] Found {len(left_images)} left images, {len(right_images)} right images")
if len(left_images) == 0 or len(right_images) == 0:
    print("[ERROR] No images found")
    exit(1)

output_dir = "Dataset/gt_stereo/output"
os.makedirs(output_dir, exist_ok=True)

trajectory = []
patch_size = 35
corrected_pixels_per_frame = []
prev_kp, prev_des, prev_mask_updated, prev_frame_hsv, prev_pose, prev_depth = None, None, None, None, None, None
pose_prev = np.eye(4)
video_writer = None

for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    rect_left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)

    disparity = stereo_matcher.compute(rect_left, rect_right).astype(np.float32) / 16.0
    depth_map = compute_depth(disparity, K1, baseline)

    mean_disp, mean_depth = np.mean(disparity), np.mean(depth_map)
    print(f"[DEBUG] Frame {idx}: Mean disparity={mean_disp:.2f}, Mean depth={mean_depth:.2f}")

    pose_now, kp, des = vo.process_frame(gray_left, depth_map)
    trajectory.append(pose_now[:3, 3].copy())

    hsv = cv2.cvtColor(rect_left, cv2.COLOR_BGR2HSV)
    current_frame_hsv = hsv.copy()
    _, mask = cv2.threshold(hsv[:, :, 2], 200, 255, cv2.THRESH_BINARY)
    mask_dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask_curr_bin = (mask_dilated > 127).astype(np.uint8)

    corrected_pixel_count = 0

    if prev_kp is not None and prev_des is not None and prev_mask_updated is not None and prev_frame_hsv is not None and prev_pose is not None and prev_depth is not None:
        matches = bf.match(prev_des, des)

        distances = np.array([m.distance for m in matches])
        threshold = np.percentile(distances, 50)
        confident_matches = [m for m in matches if m.distance <= threshold]
        print(f"[CONFIDENCE] {len(confident_matches)}/{len(matches)} matches retained after filtering")

        if len(confident_matches) > 0:
            match_img = cv2.drawMatches(cv2.cvtColor(prev_frame_hsv, cv2.COLOR_HSV2BGR), prev_kp,
                                        cv2.cvtColor(current_frame_hsv, cv2.COLOR_HSV2BGR), kp,
                                        confident_matches[:50], None, flags=2)
            cv2.imshow("Feature Matches", match_img)

        T = np.linalg.inv(prev_pose) @ pose_now
        warped_prev_mask = np.zeros_like(mask_curr_bin)
        h, w = mask_curr_bin.shape
        ys, xs = np.nonzero(prev_mask_updated)
        for y, x in zip(ys, xs):
            z = prev_depth[y, x]
            if not np.isfinite(z) or z < 0.1 or z > 100.0:
                continue
            pt_cam = np.array([
                (x - K1[0, 2]) * z / K1[0, 0],
                (y - K1[1, 2]) * z / K1[1, 1],
                z,
                1.0
            ])
            pt_cam_curr = T @ pt_cam
            u = int(K1[0, 0] * pt_cam_curr[0] / pt_cam_curr[2] + K1[0, 2])
            v = int(K1[1, 1] * pt_cam_curr[1] / pt_cam_curr[2] + K1[1, 2])
            if 0 <= u < w and 0 <= v < h:
                warped_prev_mask[v, u] = 1

        newly_occluded = (mask_curr_bin == 1) & (warped_prev_mask == 0)
        debug_newly_occluded_mask = (newly_occluded * 255).astype(np.uint8)
        cv2.namedWindow("Newly Occluded Pixels Mask (B ∩ (¬Warp(A_updated) → B))", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Newly Occluded Pixels Mask (B ∩ (¬Warp(A_updated) → B))", 800, 600)
        cv2.imshow("Newly Occluded Pixels Mask (B ∩ (¬Warp(A_updated) → B))", debug_newly_occluded_mask)

        for m in confident_matches:
            prev_pt = np.round(prev_kp[m.queryIdx].pt).astype(int)
            curr_pt = np.round(kp[m.trainIdx].pt).astype(int)

            if not (0 <= curr_pt[0] < w and 0 <= curr_pt[1] < h): continue
            if not newly_occluded[curr_pt[1], curr_pt[0]]: continue

            z = depth_map[curr_pt[1], curr_pt[0]]
            if not np.isfinite(z) or z < 0.1 or z > 100.0:
                continue
            pt_cam = np.array([
                (curr_pt[0] - K1[0, 2]) * z / K1[0, 0],
                (curr_pt[1] - K1[1, 2]) * z / K1[1, 1],
                z,
                1.0
            ])
            pt_cam_prev = T @ pt_cam
            u = int(K1[0, 0] * pt_cam_prev[0] / pt_cam_prev[2] + K1[0, 2])
            v = int(K1[1, 1] * pt_cam_prev[1] / pt_cam_prev[2] + K1[1, 2])

            if 0 <= u < w - patch_size and 0 <= v < h - patch_size:
                patch = prev_frame_hsv[v:v+patch_size, u:u+patch_size].copy()
                current_frame_hsv[curr_pt[1]-patch_size//2:curr_pt[1]+patch_size//2+1,
                                  curr_pt[0]-patch_size//2:curr_pt[0]+patch_size//2+1] = patch

                cv2.rectangle(current_frame_hsv,
                              (curr_pt[0]-patch_size//2, curr_pt[1]-patch_size//2),
                              (curr_pt[0]+patch_size//2, curr_pt[1]+patch_size//2),
                              (30,255,255), 1)  # HSV yellow border

                mask_curr_bin[curr_pt[1]-patch_size//2:curr_pt[1]+patch_size//2+1,
                              curr_pt[0]-patch_size//2:curr_pt[0]+patch_size//2+1] = 0

                corrected_pixel_count += patch_size * patch_size

    corrected_pixels_per_frame.append(corrected_pixel_count)

    current_frame_bgr = cv2.cvtColor(current_frame_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Current Frame with Stable Replacements", current_frame_bgr)
    cv2.imshow("Current Caustic Mask", (mask_curr_bin*255).astype(np.uint8))

    output_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
    cv2.imwrite(output_path, current_frame_bgr)

    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            os.path.join(output_dir, "caustic_removed_patch_based.mp4"),
            fourcc, 10, (current_frame_bgr.shape[1], current_frame_bgr.shape[0]))
    video_writer.write(current_frame_bgr)

    prev_kp, prev_des, prev_mask_updated, prev_frame_hsv, prev_pose, prev_depth = kp, des, mask_curr_bin.copy(), current_frame_hsv.copy(), pose_now.copy(), depth_map.copy()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

if video_writer is not None:
    video_writer.release()
    print(f"[INFO] Saved output frames and video to {output_dir}")

cv2.destroyAllWindows()

trajectory = np.array(trajectory)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label="Camera Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Estimated Camera Trajectory")
ax.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(corrected_pixels_per_frame, marker='o')
plt.title("Number of Corrected Pixels per Frame")
plt.xlabel("Frame Index")
plt.ylabel("Corrected Pixels")
plt.grid(True)
plt.tight_layout()
plt.show()
