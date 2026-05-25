import cv2
import numpy as np

def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Returns a standardized 1-channel uint8 grayscale matrix."""
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 3:
        gray = image[:, :, 0]
    else:
        gray = image

    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating):
            gray = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)

    return gray

def to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    """Returns a standardized 3-channel uint8 BGR image."""
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 2:
        return cv2.cvtColor(to_uint8_gray(image), cv2.COLOR_GRAY2BGR)

    bgr = image[:, :, :3]

    if bgr.dtype != np.uint8:
        if np.issubdtype(bgr.dtype, np.floating):
            bgr = (np.clip(bgr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            bgr = bgr.astype(np.uint8)

    return bgr