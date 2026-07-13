import cv2
import numpy as np
import torch
import torch.nn.functional as F

from preprocessing.normalize import detect_modality, normalize_image
from .targets import INTERIOR, BORDER

_DECODE_PARAMS = {
    "fluorescence": dict(min_area=25, close_ksize=3),
    "brightfield": dict(min_area=10, close_ksize=3, min_area_frac=0.30),
}

CANON_DIAM = 22.0


def _pad_to_multiple(x: np.ndarray, multiple: int = 16):
    h, w = x.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    padded = np.pad(x, ((0, ph), (0, pw)), mode="reflect") if (ph or pw) else x
    return padded, (h, w)


def predict_maps(model, image: np.ndarray, device=None) -> np.ndarray:
    model.eval()
    padded, (h, w) = _pad_to_multiple(image.astype(np.float32), getattr(model, "size_divisor", 16))
    t = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()
    if device is not None:
        t = t.to(device)
    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    return probs[:, :h, :w]


def predict_maps_tta(model, image: np.ndarray, device=None) -> np.ndarray:
    acc = None
    for flip in (False, True):
        base = image[:, ::-1] if flip else image
        for k in range(4):
            probs = predict_maps(model, np.ascontiguousarray(np.rot90(base, k)), device=device)
            probs = np.rot90(probs, -k, axes=(1, 2))
            if flip:
                probs = probs[:, :, ::-1]
            acc = probs if acc is None else acc + probs
    return np.ascontiguousarray(acc / 8.0)


def instances_from_probs(
    probs: np.ndarray,
    *,
    seed_threshold: float = 0.5,
    fg_threshold: float = 0.5,
    min_area: int = 25,
    close_ksize: int = 3,
    min_area_frac: float = 0.0,
) -> np.ndarray:
    interior = probs[INTERIOR]
    fg_prob = probs[INTERIOR] + probs[BORDER]

    foreground = fg_prob > fg_threshold
    seeds = (interior > seed_threshold).astype(np.uint8)
    if close_ksize:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, kernel)
    if int(seeds.sum()) == 0:
        return np.zeros(probs.shape[1:], dtype=np.int32)

    src = np.where(seeds > 0, 0, 255).astype(np.uint8)
    _, seed_labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
    )
    seed_labels = np.where(foreground, seed_labels.astype(np.int32), 0)

    areas = np.bincount(seed_labels.ravel())
    threshold = min_area
    if min_area_frac:
        candidates = areas[1:][areas[1:] > 0]
        if candidates.size:
            threshold = max(min_area, min_area_frac * float(np.median(candidates)))
    keep = np.nonzero(areas >= threshold)[0]
    keep = keep[keep != 0]
    remap = np.zeros(len(areas), dtype=np.int32)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return remap[seed_labels]


def _median_diameter(labels: np.ndarray) -> float:
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    if areas.size == 0:
        return 0.0
    return float(np.median(2.0 * np.sqrt(areas / np.pi)))


def _canonical_scale(labels: np.ndarray, target_diam: float, lo: float = 0.4, hi: float = 2.5) -> float:
    median = _median_diameter(labels)
    if median <= 0:
        return 1.0
    return float(np.clip(target_diam / median, lo, hi))


def segment_image(image: np.ndarray, model, *, device=None, preprocess: bool = True, modality=None,
                  tta: bool = False, canonical_diam: float = CANON_DIAM):
    if preprocess and modality is None:
        modality = detect_modality(image)
    predict = predict_maps_tta if tta else predict_maps
    params = _DECODE_PARAMS.get(modality, {})

    def run(arr):
        canonical = normalize_image(arr, output="zscore", modality=modality) if preprocess else arr
        return instances_from_probs(predict(model, canonical, device=device), **params)

    labels = run(image)
    if canonical_diam:
        scale = _canonical_scale(labels, canonical_diam)
        if abs(scale - 1.0) > 0.02:
            h, w = image.shape[:2]
            scaled = cv2.resize(image, (max(8, round(w * scale)), max(8, round(h * scale))),
                                interpolation=cv2.INTER_LINEAR)
            labels_s = run(scaled)
            resized = labels_s if labels_s.shape == (h, w) else \
                cv2.resize(labels_s.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            return resized, int(labels_s.max())
    return labels, int(labels.max())
