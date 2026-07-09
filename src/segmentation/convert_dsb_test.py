import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_MAX_LABEL = 65535


def _instance_map(rles: list[str], h: int, w: int) -> np.ndarray:
    flat = np.zeros(h * w, dtype=np.int32)
    for label, rle in enumerate(rles, start=1):
        s = np.array(rle.split(), dtype=np.int64)
        for start, length in zip(s[0::2] - 1, s[1::2]):
            flat[start : start + length] = label
    return flat.reshape((h, w), order="F")


def _read_solution(csv_path: Path) -> dict[str, dict]:
    rows = defaultdict(lambda: {"rles": [], "h": 0, "w": 0})
    skipped = 0
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("Usage", "").strip() == "Ignored":
                skipped += 1
                continue
            e = rows[r["ImageId"]]
            e["rles"].append(r["EncodedPixels"])
            e["h"] = int(r["Height"])
            e["w"] = int(r["Width"])
    if skipped:
        print(f"  skipped {skipped} rows marked Usage=Ignored (unscored decoy images)")
    return rows


def convert_dsb_test(images_root: Path, csv_path: Path, out: Path, limit: int = 0) -> int:
    solution = _read_solution(csv_path)
    if not solution:
        raise SystemExit(f"no rows parsed from {csv_path}")
    print(f"  {len(solution)} scored images in {csv_path.name}")

    written = 0
    for image_id, entry in solution.items():
        src_dir = images_root / image_id / "images"
        imgs = sorted(p for p in src_dir.glob("*.png")) if src_dir.is_dir() else []
        if not imgs:
            print(f"  skip {image_id[:12]}: no image under {src_dir}")
            continue

        img = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  skip {image_id[:12]}: cannot read {imgs[0].name}")
            continue
        h, w = img.shape[:2]
        if (h, w) != (entry["h"], entry["w"]):
            print(f"  skip {image_id[:12]}: image {h}x{w} != csv {entry['h']}x{entry['w']}")
            continue

        inst = _instance_map(entry["rles"], h, w)
        n = int(inst.max())
        if n == 0:
            print(f"  skip {image_id[:12]}: no instances decoded")
            continue
        if n > _MAX_LABEL:
            raise ValueError(f"{image_id}: {n} instances exceed 16-bit label map")

        d = out / image_id
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(parents=True, exist_ok=True)
        shutil.copy2(imgs[0], d / "images" / imgs[0].name)
        cv2.imwrite(str(d / "masks" / "instances.png"), inst.astype(np.uint16))

        written += 1
        if limit and written >= limit:
            break
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="stage1_test root (<id>/images/<id>.png)")
    p.add_argument("--csv", type=Path, required=True, help="stage1_solution.csv with RLE ground truth")
    p.add_argument("--out", type=Path, default=Path("data/raw/stage1_test_labeled"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    n = convert_dsb_test(args.src, args.csv, args.out, limit=args.limit)
    print(f"wrote {n} samples to {args.out}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
