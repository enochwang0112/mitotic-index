# Mitotic-Index Models

A two-stage pipeline for computing the **mitotic index** of H&E histology images:

```
mitotic index = mitotic figures / total nuclei
              = numerator (classifier) / denominator (segmenter)
```

| File | Task | Arch | Size | Training data | Headline metric |
|------|------|------|------|---------------|-----------------|
| `segmenter_contact.pt` | Nuclei segmentation (denominator) | ResNet34 U-Net | 93 MB | DSB2018 + MoNuSeg + PanNuke | H&E recall 0.938, count bias +19.9/img |
| `mitosis.pt` | Mitosis classification (numerator) | ResNet18 | 43 MB | MIDOG++ | val F1 0.82 (P 0.82 / R 0.83) |

## Usage

```bash
pip install torch torchvision opencv-python numpy
# download the two .pt files into ./models/ (see Releases), then:
PYTHONPATH=src python example_inference.py path/to/he_image.png --overlay out.png
```

Output:
```
Total nuclei (denominator): 524
Mitotic figures (numerator, p>=0.50): 1
Mitotic index: 0.0019  (0.19%)
```

The checkpoints are self-describing (they store `arch`, `target_mode`, `adaptive_border`, etc.), so
`segmentation.model.load_model` and `classification.model.load_classifier` rebuild the architecture
automatically — you only need this repository's `src/` plus the two files.

## How it works

1. **Segment (denominator).** 3-class semantic model (background / interior / border) with a
   distance-transform Voronoi decode. Inference is **two-pass self-calibrated**: a first pass
   estimates the median nucleus diameter, the image is resampled so that median hits ~22 px, then a
   second pass segments. This is on by default and removes most scale-driven count bias — no pixel
   size needs to be known per slide.
2. **Classify (numerator).** Each nucleus centroid is cropped to a 128×128 window and classified
   mitotic vs interphase. Crop preprocessing: RGB, `/255`, ImageNet mean/std normalization.
3. **Index.** `mitotic / total`. The classifier threshold (`--threshold`, default 0.5) is the knob
   for trading precision vs recall on the numerator; raise it to reduce false-positive mitoses.

## Training recipe (for reproduction)

- **Segmenter:** ResNet34 encoder (ImageNet-pretrained), `contact` border target, adaptive border,
  border-class weight 2.5, scale augmentation 0.5–2.0×, **lr 3e-4** (1e-3 diverges — too hot for the
  pretrained encoder), 30 epochs, modality-balanced sampling. PanNuke is **essential** — without it
  the model over-fragments dense H&E badly (count bias +276/img).
- **Classifier:** ResNet18 (ImageNet-pretrained), 128 px crops, class-balanced sampling, lr 3e-4,
  Adam + cosine schedule, 20 epochs, split **by slide** (no slide leakage), best-**val-F1** checkpoint.

## Metrics

**Segmenter** (`segmenter_contact.pt`), two-pass inference:
- Held-out H&E (MoNuSeg test, 14 imgs / 6697 nuclei): recall **0.938**, count bias **+19.9/img**.
- DSB2018 held-out: all-recall 0.824 (fluorescence 0.879, brightfield 0.733), count bias −6.6/img.

**Classifier** (`mitosis.pt`), MIDOG++ validation (split by slide):
- best **F1 0.82**, precision 0.82, recall 0.83 (balanced → near-unbiased mitotic count).

## Limitations

- **The end-to-end index is not yet validated against mitosis ground truth.** Each stage was
  validated separately; the compounded count bias of segment→classify on whole images, and the
  optimal threshold, are not yet calibrated. Treat the index as indicative, not diagnostic.
- **The segmenter must find a mitotic nucleus before it can be classified.** Mitotic figures look
  atypical (condensed chromatin); segmentation misses are lost from the numerator before the
  classifier runs.
- **Domain:** tuned for **H&E** histology. Brightfield-general and fluorescence performance is
  weaker; scale is self-calibrated but staining/scanner shifts are not normalized.
- Not a medical device. Research use only.

## Licensing and attribution

Because the two models were trained on datasets with different licenses, they inherit different terms:

- **`mitosis.pt`** — trained only on **MIDOG++** (MIT license). Permissive; may be used and
  redistributed with attribution.
- **`segmenter_contact.pt`** — trained on DSB2018, **MoNuSeg**, and **PanNuke**. MoNuSeg and PanNuke
  are released under **CC BY-NC-SA 4.0 (non-commercial, share-alike)**. Accordingly this checkpoint
  is provided for **research / non-commercial use only**, with attribution.

Please cite the underlying datasets if you use these models:

- **MIDOG++** — Aubreville et al., *MIDOG++: A Comprehensive Multi-Domain Dataset for Mitotic Figure
  Detection* (MIT).
- **PanNuke** — Gamper et al., *PanNuke: An Open Pan-Cancer Histology Dataset for Nuclei Instance
  Segmentation and Classification* (CC BY-NC-SA).
- **MoNuSeg** — Kumar et al., *A Multi-Organ Nucleus Segmentation Challenge* (CC BY-NC-SA).
- **DSB2018** — 2018 Data Science Bowl nuclei segmentation dataset.
