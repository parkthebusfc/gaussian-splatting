import os
import cv2
import numpy as np

def hsv_caustic_removal(gt_image, k=30):
    
    hsv_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]  # Extract Value (Brightness) channel

    # Caustics are thresholded to white and rest to black. 
    _, mask = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to refine the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    #cv2.imshow('mask',mask)

    # Convert mask to 3 channels
    mask_3d = np.stack([mask]*3, axis=-1)

    # Fourier filtering only on masked regions
    def process_channel(channel):
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        h, w = channel.shape[:2]
        band_mask = np.ones((h, w), np.float32)
        cv2.circle(band_mask, (w//2, h//2), k, 0, -1)

        f_filtered = fshift * band_mask  
        f_ishift = np.fft.ifftshift(f_filtered)
        fdiff_channel = np.real(np.fft.ifft2(fshift - f_filtered)) 

        fdiff_channel -= fdiff_channel.min()
        fdiff_channel = (fdiff_channel / fdiff_channel.max()) * 255
        return np.clip(fdiff_channel, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(gt_image)
    b_fdiff, g_fdiff, r_fdiff = process_channel(b), process_channel(g), process_channel(r)

    caustic_pattern = cv2.merge([b_fdiff, g_fdiff, r_fdiff])

    #cv2.imshow('caustic_pattern',caustic_pattern)
    
    caustic_free_image = np.where(mask_3d == 255, gt_image - caustic_pattern, gt_image)
    caustic_free_image = np.clip(caustic_free_image, 0, 255).astype(np.uint8)
    #cv2.imshow('caustic_free_image',caustic_free_image)
    #cv2.waitKey()

    return caustic_free_image, caustic_pattern, mask
