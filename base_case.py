import cv2
import numpy as np

# Load your original GT image
gt_image = cv2.imread("Dataset/test.jpg")  # Update path

# Create synthetic caustic frame: copy GT and add white blob
synthetic_frame = gt_image.copy()
cv2.circle(synthetic_frame, (150, 100), 50, (255, 255, 255), -1)  # draw white circle

# Show inputs
cv2.imshow("GT Image", gt_image)
cv2.imshow("Synthetic Caustic Frame", synthetic_frame)

HSV_V_THRESHOLD = 190
DILATION_KERNEL_SIZE = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE))

running_sum = gt_image.astype(np.float32)  # start with GT as initial stable frame
running_count = np.ones(gt_image.shape[:2], dtype=np.uint16)  # count=1 for all pixels initially

# ----------------------------
# Process synthetic caustic frame
# ----------------------------

hsv = cv2.cvtColor(synthetic_frame, cv2.COLOR_BGR2HSV)
_, mask = cv2.threshold(hsv[:, :, 2], HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)
mask_dilated = cv2.dilate(mask, kernel, iterations=1)

mask_curr_bin = (mask_dilated > 127).astype(np.uint8)  # 1=caustic, 0=stable

# Simulate prev_mask: from GT frame (all stable → so prev_mask is all zeros)
mask_prev_bin = np.zeros_like(mask_curr_bin)  # no caustics in GT

# According to logic: stable_pixels = (mask_prev_bin==1) & (mask_curr_bin==0) → but nothing was caustic before
# So to handle this 2-frame test, we should *keep* previous stable pixels where current pixels are caustic
caustic_pixels = mask_curr_bin == 1
stable_pixels = ~caustic_pixels  # pixels not in caustics

# Accumulate sums & counts for stable pixels
running_sum[stable_pixels] += synthetic_frame[stable_pixels].astype(np.float32)
running_count[stable_pixels] += 1

# Reset sums & counts for current caustics (simulate algorithm behavior)
running_sum[caustic_pixels] = running_sum[caustic_pixels]  # keep old sum unchanged
running_count[caustic_pixels] = running_count[caustic_pixels]  # keep old count unchanged

# Compute temporally averaged output
average_output = np.zeros_like(gt_image, dtype=np.uint8)
valid = running_count > 0
average_output[valid] = (running_sum[valid] / running_count[valid, None]).astype(np.uint8)

cv2.imshow("Caustic Removed Output", average_output)
cv2.waitKey(1)
cv2.destroyAllWindows()
