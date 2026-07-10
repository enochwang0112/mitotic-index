from pathlib import Path

import cv2
import numpy as np

BACKGROUND = 0
INTERIOR = 1
BORDER = 2


def load_instance_masks(masks_dir) -> list[np.ndarray]:
    masks_dir = Path(masks_dir)

    label_png = masks_dir / "instances.png"
    if label_png.exists():
        lab = cv2.imread(str(label_png), cv2.IMREAD_UNCHANGED)
        if lab is not None:
            if lab.ndim == 3:
                lab = lab[:, :, 0]
            return [lab == i for i in np.unique(lab) if i != 0]

    masks = []
    for p in sorted(masks_dir.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        masks.append(m > 0)
    return masks


def build_instance_map(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("no masks provided")

    h, w = masks[0].shape
    inst = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        inst[m] = i
    return inst


_CONTACT_FILL = np.float32(1e7)

TARGET_MODES = ("ring", "contact")


def _ring_interior(labels: np.ndarray, foreground: np.ndarray, border_width: int) -> np.ndarray:
    k = 2 * border_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lab = labels.astype(np.float32)
    neigh_max = cv2.dilate(lab, kernel)
    neigh_min = cv2.erode(lab, kernel)
    return foreground & (neigh_max == lab) & (neigh_min == lab)


def _contact_interior(labels: np.ndarray, foreground: np.ndarray, border_width: int) -> np.ndarray:
    k = 2 * border_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    lab = labels.astype(np.float32)
    filled = np.where(foreground, lab, _CONTACT_FILL).astype(np.float32)
    neigh_max = cv2.dilate(lab, kernel)
    neigh_min = cv2.erode(filled, kernel)
    contact = (neigh_max > 0) & (neigh_min < _CONTACT_FILL) & (neigh_max != neigh_min)
    return foreground & ~contact


def build_3class_target_from_labels(
    labels: np.ndarray, border_width: int = 2, mode: str = "ring"
) -> np.ndarray:
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode {mode!r}; choose from {TARGET_MODES}")

    foreground = labels > 0
    target = np.full(labels.shape, BACKGROUND, dtype=np.uint8)
    if not foreground.any():
        return target

    build = _ring_interior if mode == "ring" else _contact_interior
    interior = build(labels, foreground, border_width)

    ids = np.unique(labels[foreground])
    kept = np.unique(labels[interior]) if interior.any() else np.empty(0, dtype=labels.dtype)
    vanished = np.setdiff1d(ids, kept, assume_unique=True)
    if vanished.size:
        interior |= np.isin(labels, vanished)

    target[foreground] = BORDER
    target[interior] = INTERIOR
    return target


def build_3class_target(masks: list[np.ndarray], border_width: int = 2, mode: str = "ring") -> np.ndarray:
    if not masks:
        raise ValueError("no masks provided")
    return build_3class_target_from_labels(build_instance_map(masks), border_width=border_width, mode=mode)


def masks_to_targets(masks_dir, border_width: int = 2, mode: str = "ring"):
    masks = load_instance_masks(masks_dir)
    target = build_3class_target(masks, border_width=border_width, mode=mode)
    inst = build_instance_map(masks)
    return target, inst
