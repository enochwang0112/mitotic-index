from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABELS = {"imposter": 0, "mitotic": 1}


def list_crops(root) -> list[dict]:
    root = Path(root)
    items = []
    for lab, y in LABELS.items():
        for p in sorted((root / lab).glob("*.png")):
            items.append({"path": p, "label": y, "image": p.stem.rsplit("_", 1)[0]})
    return items


def split_crops(items: list[dict], val_frac: float = 0.15, seed: int = 0):
    stems = sorted({it["image"] for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val = set(stems[:n_val])
    train = [it for it in items if it["image"] not in val]
    held = [it for it in items if it["image"] in val]
    return train, held


def _augment(img: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
    img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
    return np.ascontiguousarray(np.clip(img, 0.0, 1.0))


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng([info.seed % (2**32), worker_id])


class MitosisCropDataset(Dataset):
    def __init__(self, items: list[dict], *, train: bool = True, size: int = 128, seed: int = 0):
        self.items = items
        self.train = train
        self.size = size
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = cv2.imread(str(it["path"]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.train:
            img = _augment(img, self._rng)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), it["label"]
