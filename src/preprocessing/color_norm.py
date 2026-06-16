import cv2
import numpy as np
from .convert_uint8 import to_uint8_bgr, to_uint8_gray

_IO = 240.0
_ALPHA = 1.0
_BETA = 0.15

_HE_REF = np.array(
    [[0.5626, 0.2159],
     [0.7201, 0.8012],
     [0.4062, 0.5581]],
    dtype=np.float64,
)
_MAXC_REF = np.array([1.9705, 1.0308], dtype=np.float64)

_HE_FIXED = np.array(
    [[0.650, 0.704, 0.286],
     [0.072, 0.990, 0.105]],
    dtype=np.float64,
)
_HEMA_REF_MAX = 1.5


def _rescale01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _macenko_stains(rgb: np.ndarray, Io: float = _IO, alpha: float = _ALPHA, beta: float = _BETA):
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.float64)

    OD = -np.log((flat + 1.0) / Io)

    mask = np.all(OD >= beta, axis=1)
    ODhat = OD[mask]
    if ODhat.shape[0] < 10:
        raise ValueError("not enough tissue pixels for stain estimation")

    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    V = eigvecs[:, 1:3]

    proj = ODhat.dot(V)
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v1 = V.dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
    v2 = V.dot(np.array([np.cos(max_phi), np.sin(max_phi)]))

    if v1[0] > v2[0]:
        HE = np.array([v1, v2]).T
    else:
        HE = np.array([v2, v1]).T

    C, *_ = np.linalg.lstsq(HE, OD.T, rcond=None)
    return HE, C, (h, w)


def _hematoxylin_map(rgb: np.ndarray, Io: float = _IO) -> np.ndarray:
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.float64)
    OD = -np.log((flat + 1.0) / Io)

    stains = _HE_FIXED / np.linalg.norm(_HE_FIXED, axis=1, keepdims=True)
    third = np.cross(stains[0], stains[1])
    n = np.linalg.norm(third)
    third = third / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    M = np.vstack([stains, third])

    conc = OD.dot(np.linalg.inv(M))
    hema = conc[:, 0].reshape(h, w)

    hema = np.clip(hema, 0.0, _HEMA_REF_MAX) / _HEMA_REF_MAX
    return hema.astype(np.float32)


def normalize_brightfield(image: np.ndarray, *, return_hematoxylin: bool = True) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    bgr = to_uint8_bgr(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if return_hematoxylin:
        return _hematoxylin_map(rgb)

    try:
        _, C, (h, w) = _macenko_stains(rgb)
    except Exception:
        gray = to_uint8_gray(image).astype(np.float32) / 255.0
        nuclei = _rescale01(1.0 - gray)
        rgb01 = cv2.cvtColor((nuclei * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return (rgb01.astype(np.float32) / 255.0)

    maxC = np.percentile(C, 99, axis=1)
    maxC = np.where(maxC <= 0, 1.0, maxC)
    C_norm = C / maxC[:, None] * _MAXC_REF[:, None]

    OD_norm = _HE_REF.dot(C_norm)
    I_norm = _IO * np.exp(-OD_norm)
    I_norm = np.clip(I_norm, 0, 255).T.reshape(h, w, 3)
    return (I_norm.astype(np.float32) / 255.0)


def normalize_non_grayscale(image: np.ndarray) -> np.ndarray:
    rgb01 = normalize_brightfield(image, return_hematoxylin=False)
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
