import os
import cv2
import numpy as np
from collections import deque

class BootupFrameAverager:
    def __init__(self, buffer_size=10):
        """Initializes a frame buffer for computing an averaged image over time."""
        self.buffer_size = buffer_size
        self.frames = deque(maxlen=buffer_size)

    def update_and_get_average(self, new_frame):
        """Updates the buffer and returns the averaged frame."""
        self.frames.append(new_frame.astype(np.float32))
        avg_frame = np.mean(self.frames, axis=0)
        return np.clip(avg_frame, 0, 255).astype(np.uint8)

# Initialize the averager (persistent across frames)
frame_averager = BootupFrameAverager(buffer_size=10)

def bootup_based_approximation(gt_image):
    """Uses averaged bootup images as a reference for vanilla 3DGS rendering."""
    
    avg_image = frame_averager.update_and_get_average(gt_image)

    # Compute difference
    diff_image = gt_image.astype(np.float32) - avg_image.astype(np.float32)
    b_diff, g_diff, r_diff = cv2.split(diff_image)

    def process_channel(diff_channel):
        f = np.fft.fft2(diff_channel)
        fshift = np.fft.fftshift(f)

        h, w = diff_channel.shape[:2]
        band_mask = np.ones((h, w), np.float32)
        cv2.circle(band_mask, (w//2, h//2), 30, 0, -1)

        f_filtered = fshift * band_mask  
        f_ishift = np.fft.ifftshift(f_filtered)
        fdiff_channel = np.real(np.fft.ifft2(fshift - f_filtered))

        fdiff_channel -= fdiff_channel.min()
        fdiff_channel = (fdiff_channel / fdiff_channel.max()) * 255
        return np.clip(fdiff_channel, 0, 255).astype(np.uint8)

    r_fdiff, g_fdiff, b_fdiff = process_channel(r_diff), process_channel(g_diff), process_channel(b_diff)
    caustic_pattern = cv2.merge([b_fdiff, g_fdiff, r_fdiff])

    caustic_free_image = gt_image.astype(np.float32) - caustic_pattern.astype(np.float32)
    caustic_free_image = np.clip(caustic_free_image, 0, 255).astype(np.uint8)


    cv2.waitKey()
    return caustic_free_image, caustic_pattern
