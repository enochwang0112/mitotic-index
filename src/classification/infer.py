import cv2
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def _prep(crop: np.ndarray, size: int) -> np.ndarray:
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    img = crop.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def _crop_rgb(image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch = image[y0:y1, x0:x1]
    if px0 or py0 or px1 or py1:
        patch = np.pad(patch, ((py0, py1), (px0, px1), (0, 0)), mode="reflect")
    return patch


def classify_crops(model, crops, *, device=None, size: int = 128, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(crops), batch_size):
        batch = np.stack([_prep(c, size) for c in crops[i : i + batch_size]])
        t = torch.from_numpy(batch).float()
        if device is not None:
            t = t.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[:, 1].cpu().numpy()
        out.append(probs)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def mitotic_from_labels(rgb_image, seg_labels, model, *, device=None, size: int = 128, threshold: float = 0.5):
    ids = [int(i) for i in np.unique(seg_labels) if i != 0]
    if not ids:
        return {"ids": [], "probs": np.zeros(0), "mitotic": [], "n_total": 0, "n_mitotic": 0}
    crops = []
    for i in ids:
        ys, xs = np.where(seg_labels == i)
        crops.append(_crop_rgb(rgb_image, int(xs.mean()), int(ys.mean()), size))
    probs = classify_crops(model, crops, device=device, size=size)
    mitotic = [i for i, pr in zip(ids, probs) if pr >= threshold]
    return {"ids": ids, "probs": probs, "mitotic": mitotic, "n_total": len(ids), "n_mitotic": len(mitotic)}
