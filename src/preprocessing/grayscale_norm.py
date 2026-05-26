import cv2
import numpy as np
from .convert_uint8 import to_uint8_gray


def normalize_grayscale(
    image: np.ndarray,
    *,
    output: str = "uint8",  # "uint8" or "zscore"
) -> np.ndarray:
    """
    Gaussian background subtraction + auto percentile stretch + CLAHE for grayscale images.
    """
    if image is None:
        raise ValueError("image is None")

    gray = to_uint8_gray(image)

    pct_98 = int(np.percentile(gray, 98))
    mean_intensity = float(gray.mean())

    white_background = False
    if (pct_98 >= 245) or (mean_intensity > 200):
        proc_img = 255 - gray
        white_background = True
    else:
        proc_img = gray.copy()

    blurred = cv2.GaussianBlur(proc_img, (151, 151), 0)
    corrected = cv2.subtract(proc_img, blurred)

    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    low_pct, high_pct = 1.0, 99.0
    low_val, high_val = np.percentile(corrected, (low_pct, high_pct))
    pct_span = float(high_val - low_val)
    span_threshold = 0.5 * 255.0
    if pct_span < span_threshold and high_val > low_val:
        stretched = (corrected.astype(np.float32) - float(low_val)) * (255.0 / float(high_val - low_val))
        corrected = np.clip(stretched, 0, 255).astype(np.uint8)

    corrected = cv2.medianBlur(corrected, 3)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(24, 24))
    processed = clahe.apply(corrected)

    blended = cv2.addWeighted(processed, 0.6, corrected, 0.4, 0.0)

    if white_background:
        blended = 255 - blended

    if output == "uint8":
        return blended
    
    img_float = blended.astype(np.float32) / 255.0
    if output == "zscore":
        mean_val = img_float.mean()
        std_val = img_float.std()
        if std_val < 1e-6:
            std_val = 1.0
        return (img_float - mean_val) / std_val

    raise ValueError("unknown output format: " + str(output))