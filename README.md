# Mitotic Index

Given an image of a microscopic tissue slide, count the total number of cells, identify and count the
cells undergoing mitosis, and calculate the **mitotic index** for the slide — a key proliferation
marker in tumor grading.

```
mitotic index = mitotic figures / total nuclei
              = numerator (classifier) / denominator (segmenter)
```

The two stages are decoupled: a **nuclei segmenter** counts every cell (the denominator), and a
per-nucleus **mitosis classifier** counts the dividing ones (the numerator).

## Quickstart

```bash
pip install torch torchvision opencv-python numpy

# 1. get the trained weights into ./models/ (from the GitHub Release):
gh release download v1.0 -D models/

# 2. run the full pipeline on one H&E image:
PYTHONPATH=src python example_inference.py path/to/he_image.png --overlay out.png
```

```
Total nuclei (denominator): 524
Mitotic figures (numerator, p>=0.50): 1
Mitotic index: 0.0019  (0.19%)
```

See **[MODEL_CARD.md](MODEL_CARD.md)** for architectures, metrics, the training recipe, limitations,
and licensing.

## How it works

1. **Segment (denominator).** A 3-class semantic model (background / interior / border) with a
   distance-transform decode, plus **two-pass self-calibrated inference**: a first pass estimates the
   median nucleus size and resamples the image to a canonical scale, so counting is robust to unknown
   magnification. On by default.
2. **Classify (numerator).** Each nucleus is cropped (128×128, centroid-centered) and classified
   mitotic vs interphase.
3. **Index.** `mitotic / total`. `--threshold` (default 0.5) trades precision vs recall on the numerator.

## Repository structure

```
src/
  preprocessing/   modality detection + normalization (H&E / brightfield / fluorescence)
  segmentation/    denominator: dataset, model, targets, loss, train, infer, metrics
  classification/  numerator: crop extraction, dataset, model, train, infer
example_inference.py   end-to-end mitotic index on one image
MODEL_CARD.md          model details, metrics, license
data/raw/
  segmentation/    nuclei-mask datasets (DSB2018, MoNuSeg, PanNuke, ...)
  classification/  MIDOG++ mitosis crops
```

Datasets and model checkpoints are not tracked in git (see `.gitignore`); weights are distributed via
GitHub Releases.

## Training

Weights are pre-trained and shipped via Releases, but to reproduce:

```bash
# segmenter (denominator) — H&E-focused recipe, ~1 hr on Apple MPS
PYTHONPATH=src python -m segmentation.train \
  --encoder resnet34 --target-mode contact --adaptive-border --border-weight 2.5 \
  --scale-min 0.5 --scale-max 2.0 --lr 3e-4 --epochs 30 --workers 4 \
  --data data/raw/segmentation/stage1_train data/raw/segmentation/monuseg data/raw/segmentation/pannuke \
  --out models/segmenter_contact.pt

# mitosis classifier (numerator) — MIDOG++ crops, ~40 min on Apple MPS
PYTHONPATH=src python -m classification.train --out models/mitosis.pt --workers 4
```

## Status

Each stage is validated independently (segmenter on held-out H&E, classifier on held-out MIDOG++
slides). The **end-to-end index is not yet calibrated against mitosis ground truth** — treat the
output as indicative, not diagnostic. **Research use only; not a medical device.**

## License

Code: see `LICENSE`. Note that `models/segmenter_contact.pt` is trained on CC BY-NC-SA datasets and is
**research / non-commercial only**; `models/mitosis.pt` (MIDOG++, MIT) is permissive. Details in
[MODEL_CARD.md](MODEL_CARD.md#licensing-and-attribution).
