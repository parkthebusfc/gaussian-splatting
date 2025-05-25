import os
import cv2
import argparse
import numpy as np

from hsv import hsv_caustic_removal
from blur import selective_gaussian_blur
from bootup import BootupFrameAverager, bootup_based_approximation

# Argument parser to select which method to use
parser = argparse.ArgumentParser(description="Caustic Removal Preprocessing")
parser.add_argument("--method", type=str, choices=["hsv", "blur", "bootup"], required=True,
                    help="Choose the caustic removal method: hsv, blur, or bootup")
args = parser.parse_args()

# Input and output paths
input_folder = "Dataset/Caustics/Dept/input"
output_folder_clean = "Dataset/Caustics/Dept/caustic_free"
output_folder_caustics = "Dataset/Caustics/Dept/preprocessed"
mask_folder = "Dataset/Caustics/Dept/masks"
os.makedirs(mask_folder, exist_ok=True)


os.makedirs(output_folder_clean, exist_ok=True)
os.makedirs(output_folder_caustics, exist_ok=True)

# Initialize Bootup Averager (Only if using bootup-based method)
if args.method == "bootup":
    frame_averager = BootupFrameAverager(buffer_size=10)

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    gt_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Apply the selected method
    if args.method == "hsv":
        caustic_free_image, caustic_pattern_image, mask_image = hsv_caustic_removal(gt_image)
        mask_output_path = os.path.join("Dataset/Caustics/Dept/masks", img_name)
        cv2.imwrite(mask_output_path, mask_image)
    elif args.method == "blur":
        caustic_free_image, caustic_pattern_image = selective_gaussian_blur(gt_image)
    elif args.method == "bootup":
        caustic_free_image, caustic_pattern_image = bootup_based_approximation(gt_image)

    
    output_path_clean = os.path.join(output_folder_clean, img_name)
    output_path_caustics = os.path.join(output_folder_caustics, img_name)

    cv2.imwrite(output_path_clean, caustic_free_image)
    cv2.imwrite(output_path_caustics, caustic_pattern_image)

print(f"Caustic-free images  {args.method}.")
