import numpy as np

DEFAULT_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def _foreground_intersection(pred: np.ndarray, gt: np.ndarray):
    p_ids = np.unique(pred)
    g_ids = np.unique(gt)
    p_ids = p_ids[p_ids != 0]
    g_ids = g_ids[g_ids != 0]

    p_index = {v: i for i, v in enumerate(p_ids)}
    g_index = {v: i for i, v in enumerate(g_ids)}
    n_p, n_g = len(p_ids), len(g_ids)

    inter = np.zeros((n_p, n_g), dtype=np.int64)
    both = (pred != 0) & (gt != 0)
    if both.any():
        pv = pred[both].ravel()
        gv = gt[both].ravel()
        pi = np.array([p_index[v] for v in pv])
        gi = np.array([g_index[v] for v in gv])
        flat = pi * n_g + gi
        counts = np.bincount(flat, minlength=n_p * n_g).reshape(n_p, n_g)
        inter = counts

    p_area = np.array([(pred == v).sum() for v in p_ids], dtype=np.int64)
    g_area = np.array([(gt == v).sum() for v in g_ids], dtype=np.int64)
    return inter, p_area, g_area


def average_precision(pred: np.ndarray, gt: np.ndarray, thresholds=None) -> float:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    inter, p_area, g_area = _foreground_intersection(pred, gt)
    n_p, n_g = len(p_area), len(g_area)

    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0

    union = p_area[:, None] + g_area[None, :] - inter
    iou = inter / np.maximum(union, 1)

    scores = []
    for t in thresholds:
        matches = iou > t
        tp = int((matches.any(axis=0)).sum())
        fp = int(n_p - matches.any(axis=1).sum())
        fn = int(n_g - tp)
        scores.append(tp / (tp + fp + fn))
    return float(np.mean(scores))


def count_error(pred: np.ndarray, gt: np.ndarray) -> int:
    n_p = len(np.unique(pred)) - (1 if (pred == 0).any() else 0)
    n_g = len(np.unique(gt)) - (1 if (gt == 0).any() else 0)
    return int(n_p - n_g)
