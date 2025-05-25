import os
import cv2
import numpy as np

def selective_gaussian_blur(gt_image, ksize=(15, 15), sigma=5):
    """Applies Gaussian blur only around detected caustic regions."""
    
    # Convert to grayscale
    gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find caustics
    edges = cv2.Canny(gray, 100, 200)
    cv2.imshow('Detected Edges', edges)

    # Expand edges to make a blur mask
    mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    cv2.imshow('Dilated Mask', mask)

    # Smooth the mask for softer blending
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    cv2.imshow('Blurred Mask', mask)

    # Convert mask to 3 channels
    mask_3d = np.stack([mask] * 3, axis=-1)  

    # Apply Gaussian blur only on masked areas
    blurred = cv2.GaussianBlur(gt_image, ksize, sigma)
    cv2.imshow('Fully Blurred Image', blurred)

    # Selectively blend blurred regions into the original image
    caustic_free_image = np.where(mask_3d > 10, blurred, gt_image)
    cv2.imshow('Caustic-Free Image', caustic_free_image)

    cv2.waitKey(0)  
      

    return caustic_free_image, mask
