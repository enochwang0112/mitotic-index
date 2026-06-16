import numpy as np

from .convert_uint8 import to_uint8_gray
from .grayscale_norm import normalize_fluorescence
from .color_norm import normalize_brightfield

FLUORESCENCE_BG_MAX = 80.0


def detect_modality(image: np.ndarray) -> str:
    if image is None:
        raise ValueError("Input image is None")

    gray = to_uint8_gray(image)
    background = float(np.median(gray))

    if background < FLUORESCENCE_BG_MAX:
        return "fluorescence"
    return "brightfield"


def to_model_tensor(arr: np.ndarray, *, output: str = "float01") -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if output == "float01":
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    if output == "uint8":
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    if output == "zscore":
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        if std_val < 1e-6:
            std_val = 1.0
        return ((arr - mean_val) / std_val).astype(np.float32)

    raise ValueError(f"unknown output format: {output}")


def normalize_image(image: np.ndarray, *, output: str = "float01") -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")

    modality = detect_modality(image)

    if modality == "fluorescence":
        nuclei = normalize_fluorescence(image)
    else:
        nuclei = normalize_brightfield(image, return_hematoxylin=True)

    return to_model_tensor(nuclei, output=output)
