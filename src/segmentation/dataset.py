from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.normalize import normalize_image
from .targets import build_3class_target_from_labels, build_instance_map, load_instance_masks

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_samples(root) -> list[dict]:
    root = Path(root)
    samples = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images_dir = sample_dir / "images"
        masks_dir = sample_dir / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue
        imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        if not imgs:
            continue
        modality_file = sample_dir / "modality.txt"
        modality = modality_file.read_text().strip() if modality_file.exists() else None
        samples.append(
            {"id": sample_dir.name, "image": imgs[0], "masks": masks_dir, "modality": modality}
        )
    return samples


def split_samples(samples: list[dict], val_frac: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(samples) * val_frac)))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def _random_crop_pad(image: np.ndarray, target: np.ndarray, tile: int, rng):
    h, w = image.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        target = np.pad(target, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w = image.shape[:2]

    top = int(rng.integers(0, h - tile + 1))
    left = int(rng.integers(0, w - tile + 1))
    image = image[top : top + tile, left : left + tile]
    target = target[top : top + tile, left : left + tile]
    return image, target


def _random_scaled_crop(image: np.ndarray, labels: np.ndarray, tile: int, rng, scale_range):
    lo, hi = scale_range
    scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    window = max(8, int(round(tile / scale)))
    image, labels = _random_crop_pad(image, labels.astype(np.float32), window, rng)
    if window != tile:
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_LINEAR)
        labels = cv2.resize(labels, (tile, tile), interpolation=cv2.INTER_NEAREST)
    return image, labels.astype(np.int32)


def _canonical_scale(labels: np.ndarray, target_diam: float,
                     lo: float = 0.4, hi: float = 2.5) -> float:
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    if areas.size == 0:
        return 1.0
    median_diam = float(np.median(2.0 * np.sqrt(areas / np.pi)))
    if median_diam <= 0:
        return 1.0
    return float(np.clip(target_diam / median_diam, lo, hi))


def _augment(image: np.ndarray, target: np.ndarray, rng):
    if rng.random() < 0.5:
        image, target = image[:, ::-1], target[:, ::-1]
    if rng.random() < 0.5:
        image, target = image[::-1, :], target[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        target = np.rot90(target, k)
    if rng.random() < 0.5:
        image = image + rng.normal(0.0, 0.05)
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng([info.seed % (2**32), worker_id])


class NucleiSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        *,
        tile_size: int = 256,
        augment: bool = True,
        border_width: int = 2,
        output: str = "zscore",
        seed: int = 0,
        scale_range: tuple[float, float] | None = None,
        target_mode: str = "ring",
        adaptive_border: bool = False,
        canonical_diam: float = 0.0,
    ):
        self.samples = samples
        self.tile_size = tile_size
        self.augment = augment
        self.border_width = border_width
        self.output = output
        self.scale_range = scale_range
        self.target_mode = target_mode
        self.adaptive_border = adaptive_border
        self.canonical_diam = canonical_diam
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        raw = cv2.imread(str(s["image"]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"cannot read image: {s['image']}")

        image = normalize_image(raw, output=self.output, modality=s.get("modality")).astype(np.float32)
        labels = build_instance_map(load_instance_masks(s["masks"]))

        canon = _canonical_scale(labels, self.canonical_diam) if self.canonical_diam else 1.0
        if self.augment and self.scale_range:
            lo, hi = self.scale_range
            image, labels = _random_scaled_crop(image, labels, self.tile_size, self._rng, (lo * canon, hi * canon))
        elif canon != 1.0:
            image, labels = _random_scaled_crop(image, labels, self.tile_size, self._rng, (canon, canon))
        else:
            image, labels = _random_crop_pad(image, labels.astype(np.float32), self.tile_size, self._rng)
            labels = labels.astype(np.int32)
        target = build_3class_target_from_labels(
            labels, border_width=self.border_width, mode=self.target_mode,
            adaptive_border=self.adaptive_border,
        )

        if self.augment:
            image, target = _augment(image, target, self._rng)

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        target_t = torch.from_numpy(target.astype(np.int64)).long()
        return image_t, target_t
