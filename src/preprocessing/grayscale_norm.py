import cv2
import numpy as np


def normalize_fluorescence(
    image: np.ndarray,
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 3 and image.shape[2] >= 3:
        chan = image[:, :, :3].max(axis=2)
    elif image.ndim == 3:
        chan = image[:, :, 0]
    else:
        chan = image

    chan = chan.astype(np.float32)

    low_val = float(np.percentile(chan, p_low))

    chan_u8 = cv2.normalize(chan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu_t, _ = cv2.threshold(chan_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mask = chan_u8 > otsu_t

    if int(fg_mask.sum()) >= 32:
        high_val = float(np.percentile(chan[fg_mask], p_high))
    else:
        high_val = float(np.percentile(chan, p_high))

    if high_val <= low_val:
        high_val = low_val + 1.0

    stretched = (chan - low_val) / (high_val - low_val)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)
