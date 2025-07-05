import cv2
import numpy as np
import glob
import os

image_dir = "Dataset/gt"
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

HSV_V_THRESHOLD = 190  
DILATION_KERNEL_SIZE = 5 

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE))

prev_frame = None
prev_mask = None

running_sum = None
running_count = None

for image_path in image_paths:
    current_frame = cv2.imread(image_path)
    if current_frame is None:
        continue

    cv2.imshow("Current Frame", current_frame)

    if running_sum is None:
        height, width, channels = current_frame.shape
        running_sum = np.zeros((height, width, channels), dtype=np.float32)
        running_count = np.zeros((height, width), dtype=np.uint16)

    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[:, :, 2], HSV_V_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    cv2.imshow('mask', mask)
    cv2.imshow('mask_dilated', mask_dilated)

    mask_curr_bin = (mask_dilated > 127).astype(np.uint8)  #A

    if prev_mask is not None:
        mask_prev_bin = (prev_mask > 127).astype(np.uint8)  #B

        # Pixels where caustics have newly appeared: currently caustic & previously clear
        newly_occluded_pixels = (mask_curr_bin == 1) & (mask_prev_bin == 0)

        # Replace newly occluded pixels in the current frame with previous stable values
        current_frame[newly_occluded_pixels] = prev_frame[newly_occluded_pixels]

        #  newly occluded pixels
        debug_newly_occluded_mask = np.zeros_like(mask_curr_bin, dtype=np.uint8)
        debug_newly_occluded_mask[newly_occluded_pixels] = 255
        cv2.imshow("Newly Occluded Pixels Mask (A ∩ (A - B))", debug_newly_occluded_mask)

    # Stable pixels: currently not caustic
    stable_pixels = mask_curr_bin == 0

    
    running_sum[stable_pixels] += current_frame[stable_pixels].astype(np.float32)
    running_count[stable_pixels] += 1

    # Compute temporally averaged output from stable pixels
    average_output = np.zeros_like(current_frame, dtype=np.uint8)
    valid = running_count > 0
    average_output[valid] = (running_sum[valid] / running_count[valid, None]).astype(np.uint8)

    cv2.imshow("Caustic Removed Averaged Output", average_output)

    prev_frame = current_frame.copy()  
    prev_mask = mask_dilated.copy()
    
    key = cv2.waitKey(0)  
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


#calibar
#generate an image by increasing v