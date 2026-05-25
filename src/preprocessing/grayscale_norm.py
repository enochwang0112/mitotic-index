import cv2
import numpy as np
from .convert_uint8 import to_uint8_gray

def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Normalize a grayscale image."""
    gray = to_uint8_gray(image)

