import cv2
import numpy as np 

from .convert_uint8 import to_uint8_gray
from .grayscale_norm import normalize_grayscale

def is_grayscale(image: np.ndarray, tolerance: float = 5.0) -> bool:
    """Return whether the image is grayscale."""
    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 2:
        return True

    if image.ndim == 3 and image.shape[2] == 1:
        return True

    if image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[:, :, :3]
        min_channel = np.min(bgr, axis=2)
        max_channel = np.max(bgr, axis=2)

        av_spread = np.mean(max_channel.astype(np.float32) - min_channel.astype(np.float32))

        return float(av_spread) < tolerance

    return False

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the image based on the image type."""
    if image is None:
        raise ValueError("Input image is None")

    if is_grayscale(image):
        gray = to_uint8_gray(image)
        return normalize_grayscale(gray)

        