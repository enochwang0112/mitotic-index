import argparse
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from preprocessing.normalize import detect_modality
from .dataset import NucleiSegmentationDataset, list_samples, split_samples, worker_init
from .loss import SegLoss
from .model import build_model


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_modalities(samples) -> list[str]:
    mods = []
    for s in samples:
        tag = s.get("modality")
        if tag:
            mods.append(tag)
            continue
        raw = cv2.imread(str(s["image"]), cv2.IMREAD_UNCHANGED)
        mods.append(detect_modality(raw) if raw is not None else "unknown")
    return mods


def modality_sampler(samples, power: float = 0.5):
    mods = sample_modalities(samples)
    counts = {m: mods.count(m) for m in sorted(set(mods))}
    weights = [(1.0 / counts[m]) ** power for m in mods]
    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    return sampler, counts


def run_epoch(model, loader, loss_fn, device, optimizer=None, *, desc="", log_every=25) -> float:
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    nb = len(loader)
    t0 = time.time()
    for i, (image, target) in enumerate(loader, 1):
        image, target = image.to(device), target.to(device)
        with torch.set_grad_enabled(train):
            logits = model(image)
            loss = loss_fn(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * image.size(0)
        n += image.size(0)
        if desc and (i % log_every == 0 or i == nb):
            elapsed = time.time() - t0
            rate = n / max(elapsed, 1e-6)
            eta = (nb - i) * elapsed / max(i, 1)
            print(f"\r  {desc} {i:>4}/{nb}  loss {total / max(n, 1):.4f}  {rate:4.0f} img/s  eta {eta:4.0f}s    ",
                  end="", flush=True)
    return total / max(n, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the U-Net nuclei segmenter on DSB2018")
    p.add_argument("--data", type=Path, nargs="+", default=[Path("data/raw/stage1_train")],
                   help="One or more sample roots (DSB2018 layout); extra H&E roots are merged in")
    p.add_argument("--out", type=Path, default=Path("models/segmenter.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Cap number of samples (0 = all); useful for a quick trial run")
    p.add_argument("--balance-modality", action=argparse.BooleanOptionalAction, default=True,
                   help="Oversample the rarer modality (brightfield) during training")
    p.add_argument("--balance-power", type=float, default=0.5,
                   help="Oversampling strength: 0=natural freq, 0.5=sqrt (~30%% brightfield, default), 1=fully balanced")
    p.add_argument("--encoder", choices=["unet", "resnet34"], default="unet",
                   help="unet = from scratch; resnet34 = ImageNet-pretrained encoder")
    p.add_argument("--scale-min", type=float, default=1.0, help="Min random scale (1.0 = no scale aug)")
    p.add_argument("--scale-max", type=float, default=1.0, help="Max random scale (e.g. 2.0)")
    p.add_argument("--target-mode", choices=["ring", "contact"], default="ring",
                   help="ring = border around every nucleus; contact = border only where nuclei touch")
    p.add_argument("--border-weight", type=float, default=2.0,
                   help="Cross-entropy weight for the border class (raise it for --target-mode contact)")
    p.add_argument("--adaptive-border", action=argparse.BooleanOptionalAction, default=False,
                   help="Thin the border for small cells so they keep interior signal (helps small-cell recall)")
    p.add_argument("--canonical-diam", type=float, default=0.0,
                   help="Resample each image so its median nucleus diameter hits this px target (0 = off)")
    args = p.parse_args(argv)

    samples = []
    for root in args.data:
        found = list_samples(root)
        print(f"  {root}: {len(found)} samples")
        samples += found
    if not samples:
        print(f"ERROR: no samples found under {args.data}")
        return 1
    if args.limit:
        samples = samples[: args.limit]
    train_s, val_s = split_samples(samples)
    print(f"samples: {len(samples)} (train {len(train_s)}, val {len(val_s)})")

    scale_range = None
    if (args.scale_min, args.scale_max) != (1.0, 1.0):
        scale_range = (args.scale_min, args.scale_max)
        print(f"scale augmentation: {scale_range[0]}x - {scale_range[1]}x (log-uniform)")

    train_ds = NucleiSegmentationDataset(train_s, tile_size=args.tile, augment=True,
                                         scale_range=scale_range, target_mode=args.target_mode,
                                         adaptive_border=args.adaptive_border,
                                         canonical_diam=args.canonical_diam)
    val_ds = NucleiSegmentationDataset(val_s, tile_size=args.tile, augment=False,
                                       target_mode=args.target_mode, adaptive_border=args.adaptive_border,
                                       canonical_diam=args.canonical_diam)
    if args.balance_modality:
        sampler, counts = modality_sampler(train_s, power=args.balance_power)
        eff = {m: counts[m] ** (1 - args.balance_power) for m in counts}
        tot = sum(eff.values())
        frac = {m: f"{eff[m] / tot:.0%}" for m in counts}
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, worker_init_fn=worker_init)
        print(f"train modality counts: {counts}  -> sampled ~{frac} (power={args.balance_power})")
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, worker_init_fn=worker_init)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = pick_device()
    model = build_model(args.encoder).to(device)
    loss_fn = SegLoss(class_weights=(1.0, 1.0, args.border_weight)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}  encoder: {args.encoder}  params: {params / 1e6:.1f}M  lr: {args.lr}")
    print(f"target mode: {args.target_mode}  border class weight: {args.border_weight}  adaptive border: {args.adaptive_border}")
    if args.canonical_diam:
        print(f"canonical resampling: median nucleus -> {args.canonical_diam:.0f}px diameter")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_dl, loss_fn, device, optimizer, desc=f"epoch {epoch:3d} train")
        val_loss = run_epoch(model, val_dl, loss_fn, device, desc=f"epoch {epoch:3d} val  ")
        scheduler.step()
        flag = ""
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "arch": args.encoder,
                        "target_mode": args.target_mode,
                        "adaptive_border": args.adaptive_border,
                        "canonical_diam": args.canonical_diam}, args.out)
            flag = "  <- saved"
        print(f"\r{' ' * 80}\repoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}{flag}")

    print(f"best val loss {best:.4f}; checkpoint at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
