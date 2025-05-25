import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# Parameters
hsv_threshold = 200
feature_detector = cv2.ORB_create(nfeatures=1000)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def generate_feature_mask(image_folder, output_mask_folder):
    os.makedirs(output_mask_folder, exist_ok=True)

    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    masks = []

    # Load and preprocess images
    images = [cv2.imread(path) for path in image_paths]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Step 1: Create caustic masks per frame (HSV thresholding)
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        _, mask = cv2.threshold(v, hsv_threshold, 255, cv2.THRESH_BINARY)
        masks.append(mask)

    # Step 2: Track consistent features across frames
    consistent_masks = [np.zeros_like(masks[0]) for _ in masks]

    for i in tqdm(range(len(images) - 1)):
        kp1, des1 = feature_detector.detectAndCompute(gray_images[i], None)
        kp2, des2 = feature_detector.detectAndCompute(gray_images[i + 1], None)

        if des1 is None or des2 is None:
            continue

        matches = bf_matcher.match(des1, des2)
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
            x2, y2 = int(round(pt2[0])), int(round(pt2[1]))

            if masks[i][y1, x1] == 255 and masks[i + 1][y2, x2] == 255:
                consistent_masks[i][y1, x1] = 255
                consistent_masks[i + 1][y2, x2] = 255

    # Save masks
    for path, mask in zip(image_paths, consistent_masks):
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(output_mask_folder, filename), mask)
