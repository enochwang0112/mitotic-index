# Mitotic Index

Given an H&E tissue-slide image, count the total nuclei, identify the cells undergoing mitosis, and
compute the **mitotic index** = mitotic figures / total nuclei.

Two decoupled stages: a **segmenter** counts every nucleus (denominator) and a per-nucleus
**classifier** counts the dividing ones (numerator).

## Quickstart

```bash
pip install torch torchvision opencv-python numpy

gh release download v1.0 -D models/                              # get the trained weights
PYTHONPATH=src python example_inference.py image.png --overlay out.png
```

```
Total nuclei (denominator): 524
Mitotic figures (numerator, p>=0.50): 1
Mitotic index: 0.0019  (0.19%)
```

See **[MODEL_CARD.md](MODEL_CARD.md)** for architectures, metrics, and licensing.

## Hardware

Training and inference auto-select the device: **Apple Silicon (MPS)**, **NVIDIA (CUDA)**, or CPU —
no flags needed. Just install the matching PyTorch build:

- **Apple Silicon / CPU:** `pip install torch torchvision` (the default wheel is MPS-capable).
- **NVIDIA GPU:** install the CUDA build, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

## Training

Weights are shipped via Releases, but to reproduce:

```bash
# segmenter (denominator)
PYTHONPATH=src python -m segmentation.train \
  --encoder resnet34 --target-mode contact --adaptive-border --border-weight 2.5 \
  --scale-min 0.5 --scale-max 2.0 --lr 3e-4 --epochs 30 --workers 4 \
  --data data/raw/segmentation/stage1_train data/raw/segmentation/monuseg data/raw/segmentation/pannuke \
  --out models/segmenter_contact.pt

# mitosis classifier (numerator)
PYTHONPATH=src python -m classification.train --out models/mitosis.pt --workers 4
```

Rough training time (30 / 20 epochs): ~1 hr / ~40 min on Apple M-series (MPS); noticeably faster on a
CUDA GPU. Raise `--workers` if data loading is the bottleneck.

## Status

Each stage is validated on its own held-out set; the end-to-end index is not yet calibrated against
mitosis ground truth. **Research use only — not a medical device.**

## License

Code is MIT (see `LICENSE`). Model weights differ: `mitosis.pt` is permissive (MIDOG++, MIT);
`segmenter_contact.pt` is **research / non-commercial** (trained on CC BY-NC-SA data). See
[MODEL_CARD.md](MODEL_CARD.md#licensing-and-attribution).
