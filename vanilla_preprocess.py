import os
import cv2
import numpy as np

def simulate_rendered_image(image):
    """Simulates a basic rendered image using Gaussian blur."""
    return cv2.GaussianBlur(image, (15, 15), 5)

def fourier_filter_decoupled(gt_image, k=30):
    cv2.imshow('inverted_gt', gt_image)

    rendered_image = simulate_rendered_image(gt_image)
    cv2.imshow('renderred',rendered_image)

    # Difference
    diff_image = gt_image.astype(np.float32) - rendered_image.astype(np.float32)
    cv2.imshow('difference',diff_image)
    b_diff, g_diff, r_diff = cv2.split(diff_image)

    def process_channel(diff_channel):
        f = np.fft.fft2(diff_channel)
        fshift = np.fft.fftshift(f)

        h, w = diff_channel.shape[:2]

        # Band-stop filter 
        mask = np.ones((h, w), np.float32)
        cv2.circle(mask, (w//2, h//2), k, 0, -1)  

        f_filtered = fshift * mask  
        f_ishift = np.fft.ifftshift(f_filtered)

        # fdiff (removed caustic patterns)
        fdiff_channel = np.real(np.fft.ifft2(fshift - f_filtered))

        filtered_diff_channel = np.real(np.fft.ifft2(f_ishift))

        # Normalize
        fdiff_channel -= fdiff_channel.min()
        fdiff_channel = (fdiff_channel / fdiff_channel.max()) * 255

        filtered_diff_channel -= filtered_diff_channel.min()
        filtered_diff_channel = (filtered_diff_channel / filtered_diff_channel.max()) * 255

        return np.clip(filtered_diff_channel, 0, 255).astype(np.uint8), \
               np.clip(fdiff_channel, 0, 255).astype(np.uint8)

    r_residual, r_fdiff = process_channel(r_diff)
    g_residual, g_fdiff = process_channel(g_diff)
    b_residual, b_fdiff = process_channel(b_diff)

    caustic_pattern_image = cv2.merge([b_fdiff, g_fdiff, r_fdiff])  # fdiff
    residual_image = cv2.merge([r_residual,b_residual,g_residual])  # residual
    cv2.imshow("residual",residual_image)

    # GT - fdiff to get images without caustics
    gt_float = gt_image.astype(np.float32)
    fdiff_float = caustic_pattern_image.astype(np.float32)
    caustic_free_image = gt_float - fdiff_float  
    caustic_free_image = np.clip(caustic_free_image, 0, 255).astype(np.uint8)

    caustic_free_image = cv2.cvtColor(caustic_free_image, cv2.COLOR_RGB2BGR)
    caustic_pattern_image = cv2.cvtColor(caustic_pattern_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('compen',caustic_free_image)
    cv2.imshow('fdiff',caustic_pattern_image)
    cv2.waitKey()
    return caustic_free_image, caustic_pattern_image

input_folder = "Dataset/Caustics/Dept/input"
output_folder_clean = "Dataset/Caustics/Dept/caustic_free"
output_folder_caustics = "Dataset/Caustics/Dept/preprocessed"

os.makedirs(output_folder_clean, exist_ok=True)
os.makedirs(output_folder_caustics, exist_ok=True)

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    gt_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    caustic_free_image, caustic_pattern_image = fourier_filter_decoupled(gt_image)

    output_path_clean = os.path.join(output_folder_clean, img_name)
    output_path_caustics = os.path.join(output_folder_caustics, img_name)

    cv2.imwrite(output_path_clean, caustic_free_image)
    cv2.imwrite(output_path_caustics, caustic_pattern_image)

print("Caustic-free images and extracted caustic patterns")





