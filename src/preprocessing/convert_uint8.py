import cv2
import numpy as np


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 3 and image.shape[2] >= 3:
        bgr = _ensure_uint8(image[:, :, :3])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3:
        gray = _ensure_uint8(image[:, :, 0])
    else:
        gray = _ensure_uint8(image)

    return gray


def to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 2:
        return cv2.cvtColor(to_uint8_gray(image), cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(to_uint8_gray(image), cv2.COLOR_GRAY2BGR)

    return _ensure_uint8(image[:, :, :3])
