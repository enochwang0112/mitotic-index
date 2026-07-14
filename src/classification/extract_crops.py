import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

_FIGSHARE = "https://ndownloader.figshare.com/files/{}"
_LABELS = {1: "mitotic", 2: "imposter"}


def _load(root: Path):
    coco = json.loads((root / "MIDOGpp.json").read_text())
    figshare = json.loads((root / "figshare_files.json").read_text())
    images = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    return images, anns, figshare


def _crop(image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
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


def _fetch(name: str, fid: str, images_dir: Path | None):
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        path = images_dir / name
        if not path.exists():
            urlretrieve(_FIGSHARE.format(fid), str(path))
        return cv2.imread(str(path), cv2.IMREAD_COLOR), None
    td = tempfile.mkdtemp()
    tiff = os.path.join(td, name)
    urlretrieve(_FIGSHARE.format(fid), tiff)
    return cv2.imread(tiff, cv2.IMREAD_COLOR), td


def extract(root: Path, out: Path, crop_size: int = 128, limit: int = 0, tumor_type: str = "",
            images_dir: Path | None = None) -> dict:
    images, anns, figshare = _load(root)
    for lab in _LABELS.values():
        (out / lab).mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.txt"
    done = set(manifest.read_text().split()) if manifest.exists() else set()

    order = [im for im in images.values() if not tumor_type or im["tumor_type"] == tumor_type]
    order.sort(key=lambda im: im["file_name"])
    counts = defaultdict(int)
    processed = 0
    skipped_nofile = 0

    for im in order:
        name = im["file_name"]
        if name in done:
            for a in anns[im["id"]]:
                counts[_LABELS[a["category_id"]]] += 1
            continue
        fid = figshare.get(name)
        if fid is None:
            skipped_nofile += 1
            continue

        try:
            img, td = _fetch(name, fid, images_dir)
        except Exception as e:
            print(f"  ! {name}: download/read failed ({e}); skipping")
            continue
        if img is None:
            if td is not None:
                shutil.rmtree(td, ignore_errors=True)
            print(f"  ! {name}: cv2 could not decode; skipping")
            continue

        n = 0
        for a in anns[im["id"]]:
            x0, y0, x1, y1 = a["bbox"]
            cx, cy = int((x0 + x1) // 2), int((y0 + y1) // 2)
            patch = _crop(img, cx, cy, crop_size)
            lab = _LABELS[a["category_id"]]
            cv2.imwrite(str(out / lab / f"{Path(name).stem}_{a['id']}.png"), patch)
            counts[lab] += 1
            n += 1

        if td is not None:
            shutil.rmtree(td, ignore_errors=True)

        with manifest.open("a") as fh:
            fh.write(name + "\n")
        processed += 1
        print(f"  [{processed}] {name} ({im['tumor_type']}): {n} crops  "
              f"[total mit {counts['mitotic']}, imp {counts['imposter']}]", flush=True)
        if limit and processed >= limit:
            break

    return {"processed": processed, "skipped_nofile": skipped_nofile, **counts}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download MIDOG++ tiffs from figshare, extract mitosis/imposter crops")
    p.add_argument("--root", type=Path, default=Path("data/raw/classification/midogpp"))
    p.add_argument("--out", type=Path, default=Path("data/raw/classification/midogpp/crops"))
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tumor-type", default="")
    p.add_argument("--keep-images", action="store_true",
                   help="Save full tiffs to <root>/images/ (~65 GB) instead of streaming and deleting; enables instant re-cropping later")
    p.add_argument("--images-dir", type=Path, default=None,
                   help="Where to keep full tiffs when --keep-images (default <root>/images)")
    args = p.parse_args(argv)

    images_dir = None
    if args.keep_images:
        images_dir = args.images_dir or (args.root / "images")

    stats = extract(args.root, args.out, args.crop_size, args.limit, args.tumor_type, images_dir)
    print(f"\ndone: processed {stats['processed']} images, "
          f"{stats.get('mitotic', 0)} mitotic + {stats.get('imposter', 0)} imposter crops; "
          f"skipped {stats['skipped_nofile']} images with no figshare file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
