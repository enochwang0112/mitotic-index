import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_MAX_LABEL = 65535


def _write_sample(out_root: Path, sample_id: str, image_bgr: np.ndarray, inst: np.ndarray) -> None:
    if int(inst.max()) > _MAX_LABEL:
        raise ValueError(f"{sample_id}: {inst.max()} instances exceed 16-bit label map")
    d = out_root / sample_id
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"{sample_id}.png"), image_bgr)
    cv2.imwrite(str(d / "masks" / "instances.png"), inst.astype(np.uint16))
    (d / "modality.txt").write_text("brightfield\n")


def _index_images(src: Path) -> dict[str, Path]:
    index = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in _IMAGE_EXTS:
            index.setdefault(p.stem, p)
    return index


def _polygons_from_xml(xml_path: Path) -> list[np.ndarray]:
    root = ET.parse(xml_path).getroot()
    polys = []
    for region in root.findall(".//Region"):
        pts = []
        for v in region.findall(".//Vertex"):
            x = v.get("X", v.get("x"))
            y = v.get("Y", v.get("y"))
            if x is None or y is None:
                continue
            pts.append((float(x), float(y)))
        if len(pts) >= 3:
            polys.append(np.round(np.array(pts)).astype(np.int32))
    return polys


def convert_monuseg(src: Path, out: Path, limit: int = 0) -> int:
    images = _index_images(src)
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"no .xml annotations found under {src}")

    written = 0
    for xml_path in xmls:
        img_path = images.get(xml_path.stem)
        if img_path is None:
            print(f"  skip {xml_path.name}: no matching image")
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skip {xml_path.name}: cannot read {img_path.name}")
            continue

        h, w = image.shape[:2]
        inst = np.zeros((h, w), dtype=np.int32)
        for label, poly in enumerate(_polygons_from_xml(xml_path), start=1):
            cv2.fillPoly(inst, [poly], color=int(label))

        n = int(inst.max())
        if n == 0:
            print(f"  skip {xml_path.name}: no polygons parsed")
            continue
        _write_sample(out, f"monuseg_{xml_path.stem}", image, inst)
        written += 1
        print(f"  {xml_path.stem}: {n} nuclei")
        if limit and written >= limit:
            break
    return written


def _pannuke_instance_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    inst = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for c in range(min(5, mask.shape[2])):
        chan = mask[:, :, c]
        for v in np.unique(chan):
            if v == 0:
                continue
            inst[chan == v] = next_id
            next_id += 1
    return inst


def convert_pannuke(src: Path, out: Path, limit: int = 0) -> int:
    img_files = sorted(src.rglob("images.npy"))
    mask_files = sorted(src.rglob("masks.npy"))
    if not img_files:
        raise SystemExit(f"no images.npy found under {src}")
    if not mask_files:
        raise SystemExit(f"no masks.npy found under {src}")

    masks_by_n = {}
    for m in mask_files:
        n = int(np.load(m, mmap_mode="r").shape[0])
        masks_by_n.setdefault(n, m)

    written = 0
    for img_file in img_files:
        images = np.load(img_file, mmap_mode="r")
        n = int(images.shape[0])
        mpath = masks_by_n.get(n)
        if mpath is None:
            print(f"  skip {img_file}: no masks.npy with {n} items")
            continue
        masks = np.load(mpath, mmap_mode="r")
        token = f"n{n}"
        print(f"  {img_file.parent.name}: {n} images (paired with {mpath.name})")

        for i in range(n):
            rgb = np.asarray(images[i]).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            inst = _pannuke_instance_map(np.asarray(masks[i]))
            if inst.max() == 0:
                continue
            _write_sample(out, f"pannuke_{token}_{i:05d}", bgr, inst)
            written += 1
            if limit and written >= limit:
                return written
    return written


def convert_lizard(src: Path, out: Path, limit: int = 0) -> int:
    pairs = []
    for img_file in sorted(src.rglob("*_img.npy")):
        lab_file = img_file.with_name(img_file.name.replace("_img.npy", "_lab.npy"))
        if lab_file.exists():
            pairs.append((img_file, lab_file))
    if not pairs:
        raise SystemExit(f"no *_img.npy / *_lab.npy pairs found under {src}")

    arrays = {}
    index = []
    for img_file, lab_file in pairs:
        split = img_file.stem.replace("_img", "")
        arrays[split] = (np.load(img_file, mmap_mode="r"), np.load(lab_file, mmap_mode="r"))
        n = int(arrays[split][0].shape[0])
        print(f"  {split}: {n} patches")
        index += [(split, i) for i in range(n)]

    if limit and limit < len(index):
        rng = np.random.default_rng(0)
        index = [index[j] for j in rng.permutation(len(index))[:limit]]

    written = 0
    for split, i in index:
        images, labels = arrays[split]
        inst = np.asarray(labels[i])
        if inst.ndim == 3:
            inst = inst[:, :, 0]
        inst = inst.astype(np.int32)
        if int(inst.max()) == 0:
            continue
        bgr = cv2.cvtColor(np.asarray(images[i]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _write_sample(out, f"lizard_{split}_{i:05d}", bgr, inst)
        written += 1
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["monuseg", "pannuke", "lizard"], required=True)
    p.add_argument("--src", type=Path, required=True, help="Downloaded dataset directory")
    p.add_argument("--out", type=Path, default=None, help="Output root (default data/raw/segmentation/<source>)")
    p.add_argument("--limit", type=int, default=0, help="Cap samples (0 = all); for a quick trial")
    args = p.parse_args(argv)

    out = args.out or Path("data/raw/segmentation") / args.source
    out.mkdir(parents=True, exist_ok=True)
    convert = {"monuseg": convert_monuseg, "pannuke": convert_pannuke, "lizard": convert_lizard}[args.source]
    n = convert(args.src, out, limit=args.limit)
    print(f"wrote {n} samples to {out}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
