import cv2
import numpy as np
import os

# Parameters
output_dir = "Dataset/gt_stereo/synthetic_frames"
os.makedirs(output_dir, exist_ok=True)

frame_height = 1080
frame_width = 1920
num_frames = 30
caustic_v_threshold = 200


def create_base_frame():
    hsv = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    hsv[..., 0] = 60      # green hue
    hsv[..., 1] = 255     # full saturation
    hsv[..., 2] = 80      # low V (dark background)

    # Add non-caustic stable shapes
    for i in range(5):
        center = (200 + i * 250, frame_height // 3)
        cv2.circle(hsv, center, 40, (30, 255, 150), -1)  # dull yellow circle

        top_left = (100 + i * 300, frame_height // 2)
        bottom_right = (top_left[0] + 80, top_left[1] + 50)
        cv2.rectangle(hsv, top_left, bottom_right, (120, 255, 120), -1)  # dull cyan rectangle

        center_ellipse = (150 + i * 350, (2 * frame_height) // 3)
        axes = (60, 30)
        cv2.ellipse(hsv, center_ellipse, axes, 0, 0, 360, (150, 255, 100), -1)  # dull magenta ellipse

    return hsv


def add_caustics(hsv, num_blocks=8, block_size=(60, 40)):
    for _ in range(num_blocks):
        x = np.random.randint(0, frame_width - block_size[0])
        y = np.random.randint(0, frame_height - block_size[1])
        hsv[y:y+block_size[1], x:x+block_size[0], 2] = np.random.randint(caustic_v_threshold + 20, 255)
    return hsv


# Generate frames
for i in range(num_frames):
    base = create_base_frame()

    # Simulate rightward movement
    M = np.float32([[1, 0, i * 5], [0, 1, 0]])
    translated = cv2.warpAffine(base, M, (frame_width, frame_height), borderValue=(60, 255, 80))

    frame_with_caustics = add_caustics(translated.copy())

    bgr_frame = cv2.cvtColor(frame_with_caustics, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), bgr_frame)

print(f"[INFO] Synthetic frames saved to: {output_dir}")
