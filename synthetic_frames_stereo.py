import cv2
import numpy as np
import os
import yaml

# Parameters
output_base = "Dataset/gt_stereo/synthetic_stereo"
left_dir = os.path.join(output_base, "left")
right_dir = os.path.join(output_base, "right")
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

frame_height = 1080
frame_width = 1920
num_frames = 30
caustic_v_threshold = 200

# Camera intrinsics and stereo parameters
fx = fy = 1000
cx = frame_width // 2
cy = frame_height // 2
baseline_m = 0.05  # 5 cm

# Disparity between cameras based on depth
depth_m = 15.0
disparity_pixels = int(fx * baseline_m / depth_m)  # e.g. 3 pixels

# Video writers
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_left = cv2.VideoWriter(os.path.join(output_base, "left.mp4"), fourcc, fps, (frame_width, frame_height))
video_right = cv2.VideoWriter(os.path.join(output_base, "right.mp4"), fourcc, fps, (frame_width, frame_height))

def create_base_frame():
    hsv = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    hsv[..., 1] = 0
    hsv[..., 2] = 30
    for i in range(5):
        center = (200 + i * 250, frame_height // 3)
        cv2.circle(hsv, center, 40, (30, 255, 150), -1)
        top_left = (100 + i * 300, frame_height // 2)
        bottom_right = (top_left[0] + 80, top_left[1] + 50)
        cv2.rectangle(hsv, top_left, bottom_right, (120, 255, 120), -1)
        center_ellipse = (150 + i * 350, (2 * frame_height) // 3)
        axes = (60, 30)
        cv2.ellipse(hsv, center_ellipse, axes, 0, 0, 360, (150, 255, 100), -1)
    return hsv

def add_caustics(hsv, num_blocks=8, block_size=(60, 40)):
    for _ in range(num_blocks):
        x = np.random.randint(0, frame_width - block_size[0])
        y = np.random.randint(0, frame_height - block_size[1])
        hsv[y:y+block_size[1], x:x+block_size[0], 2] = np.random.randint(caustic_v_threshold + 20, 255)
    return hsv

for i in range(num_frames):
    base = create_base_frame()

    # Simulate camera motion: rightward movement (scene left shift)
    M = np.float32([[1, 0, i * 5], [0, 1, 0]])
    base_translated = cv2.warpAffine(base, M, (frame_width, frame_height), borderValue=(0, 0, 30))

    # Left and right views from stereo shift
    left_hsv = add_caustics(base_translated.copy())
    right_hsv = cv2.warpAffine(left_hsv, np.float32([[1, 0, -disparity_pixels], [0, 1, 0]]),
                               (frame_width, frame_height), borderValue=(0, 0, 30))

    left_bgr = cv2.cvtColor(left_hsv, cv2.COLOR_HSV2BGR)
    right_bgr = cv2.cvtColor(right_hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(os.path.join(left_dir, f"frame_{i:04d}.png"), left_bgr)
    cv2.imwrite(os.path.join(right_dir, f"frame_{i:04d}.png"), right_bgr)
    video_left.write(left_bgr)
    video_right.write(right_bgr)

video_left.release()
video_right.release()
print(f"[INFO] Stereo frames and videos saved to {output_base}")

# Save intrinsics YAML files
K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
D = [-0.1, 0.09, 0.001, 0.001, 0]
R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
P_left = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
P_right = [fx, 0, cx, -fx * baseline_m, 0, fy, cy, 0, 0, 0, 1, 0]

def save_yaml(path, K, D, R, P):
    calib = {
        'image_width': frame_width,
        'image_height': frame_height,
        'camera_name': os.path.basename(path).split('.')[0],
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': K},
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': D},
        'rectification_matrix': {'rows': 3, 'cols': 3, 'data': R},
        'projection_matrix': {'rows': 3, 'cols': 4, 'data': P}
    }
    with open(path, 'w') as f:
        yaml.dump(calib, f)

save_yaml(os.path.join(output_base, "synthetic_left.yaml"), K, D, R, P_left)
save_yaml(os.path.join(output_base, "synthetic_right.yaml"), K, D, R, P_right)
print("[INFO] Intrinsics YAML files saved")
