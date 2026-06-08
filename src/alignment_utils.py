"""Image alignment helper functions for ECC translation recovery."""

import numpy as np
import cv2


def normalise_for_ecc(image):
    """Replace NaNs and scale an image to the 0-1 range for ECC alignment."""
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    if amax == amin:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)


def run_ecc_translation(reference, moving, max_iter=5000, eps=1e-7):
    """Estimate a translation between two images using OpenCV ECC."""
    ref = normalise_for_ecc(reference)
    mov = normalise_for_ecc(moving)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    cc, warp_matrix = cv2.findTransformECC(ref, mov, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
    return cc, warp_matrix


def pixel_shift_to_metres(warp_matrix, x_min, x_max, y_min, y_max, nx, ny):
    """Convert ECC pixel shift to projected-coordinate metres."""
    dx_pixels = float(warp_matrix[0, 2])
    dy_pixels = float(warp_matrix[1, 2])
    dx_metres = dx_pixels * (x_max - x_min) / (nx - 1)
    dy_metres = dy_pixels * (y_max - y_min) / (ny - 1)
    return dx_metres, dy_metres
