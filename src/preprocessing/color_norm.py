import cv2
import numpy as np
from .convert_uint8 import to_uint8_bgr

def normalize_non_grayscale(image: np.ndarray) -> np.ndarray:
    """Normalize a non-grayscale image."""
    bgr = to_uint8_bgr(image)

