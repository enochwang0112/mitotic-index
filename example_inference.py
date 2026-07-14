"""Minimal end-to-end mitotic-index inference on a single H&E image.

Run from the repository root:

    PYTHONPATH=src python example_inference.py path/to/he_image.png
    PYTHONPATH=src python example_inference.py image.png --overlay out.png --threshold 0.6

Pipeline: segment nuclei (denominator, two-pass self-calibrated) ->
crop each nucleus -> classify mitotic/interphase (numerator) ->
mitotic index = mitotic_count / total_count.

Requires: torch, torchvision, opencv-python, numpy, and the two checkpoints
(see MODEL_CARD.md for where to download them).
"""
import sys
sys.path.insert(0, "src")

import argparse
from pathlib import Path

import cv2
import numpy as np

from segmentation.model import load_model
from segmentation.infer import segment_image
from classification.model import load_classifier
from classification.infer import mitotic_from_labels
from segmentation.train import pick_device


def to_rgb_uint8(raw: np.ndarray) -> np.ndarray:
    img = raw
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (img / (float(img.max()) or 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute the mitotic index of one H&E image")
    p.add_argument("image", type=Path, help="Path to an H&E image (png/jpg/tif)")
    p.add_argument("--segmenter", type=Path, default=Path("models/segmenter_contact.pt"))
    p.add_argument("--classifier", type=Path, default=Path("models/mitosis.pt"))
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Mitotic probability cutoff (raise to reduce false positives)")
    p.add_argument("--overlay", type=Path, default=None,
                   help="Optional path to save a visualization (mitotic=red, other=green)")
    args = p.parse_args(argv)

    for path in (args.image, args.segmenter, args.classifier):
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 1

    raw = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"ERROR: could not read image {args.image}", file=sys.stderr)
        return 2

    device = pick_device()
    segmenter = load_model(args.segmenter, device=device)
    classifier = load_classifier(args.classifier, device=device)

    labels, n_total = segment_image(raw, segmenter, device=device)
    rgb = to_rgb_uint8(raw)
    res = mitotic_from_labels(rgb, labels, classifier, device=device, threshold=args.threshold)
    n_mitotic = res["n_mitotic"]
    index = n_mitotic / max(n_total, 1)

    print(f"Image: {args.image.name}  ({device})")
    print(f"Total nuclei (denominator): {n_total}")
    print(f"Mitotic figures (numerator, p>={args.threshold:.2f}): {n_mitotic}")
    print(f"Mitotic index: {index:.4f}  ({index * 100:.2f}%)")

    if args.overlay is not None:
        is_mit = np.zeros(int(labels.max()) + 1, dtype=bool)
        for i in res["mitotic"]:
            is_mit[i] = True
        bg = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw[:, :, :3].copy()
        mit_px = (labels > 0) & is_mit[labels]
        oth_px = (labels > 0) & ~is_mit[labels]
        bg[oth_px] = (0.6 * bg[oth_px] + 0.4 * np.array((0, 180, 0))).astype(np.uint8)
        bg[mit_px] = (0.4 * bg[mit_px] + 0.6 * np.array((0, 0, 255))).astype(np.uint8)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.overlay), bg)
        print(f"Wrote overlay to {args.overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
