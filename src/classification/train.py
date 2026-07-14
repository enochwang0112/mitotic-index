import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import MitosisCropDataset, list_crops, split_crops, worker_init
from .model import build_classifier


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None, *, desc="", log_every=25):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    tp = fp = fn = tn = 0
    nb = len(loader)
    t0 = time.time()
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1)
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        if desc and (i % log_every == 0 or i == nb):
            elapsed = time.time() - t0
            rate = n / max(elapsed, 1e-6)
            eta = (nb - i) * elapsed / max(i, 1)
            print(f"\r  {desc} {i:>4}/{nb}  loss {total / max(n, 1):.4f}  {rate:4.0f} img/s  eta {eta:4.0f}s    ",
                  end="", flush=True)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return total / max(n, 1), {"prec": prec, "rec": rec, "f1": f1, "acc": (tp + tn) / max(n, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the mitosis crop classifier on MIDOG++")
    p.add_argument("--crops", type=Path, default=Path("data/raw/classification/midogpp/crops"))
    p.add_argument("--out", type=Path, default=Path("models/mitosis.pt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                   help="Class-balanced sampling of mitotic vs imposter crops")
    args = p.parse_args(argv)

    items = list_crops(args.crops)
    if not items:
        print(f"ERROR: no crops under {args.crops}; run classification.extract_crops first")
        return 1
    if args.limit:
        items = items[: args.limit]
    train_i, val_i = split_crops(items)
    n_pos = sum(it["label"] for it in items)
    print(f"crops: {len(items)} ({n_pos} mitotic, {len(items) - n_pos} imposter); "
          f"train {len(train_i)} / val {len(val_i)} (split by slide)")

    train_ds = MitosisCropDataset(train_i, train=True, size=args.size)
    val_ds = MitosisCropDataset(val_i, train=False, size=args.size)

    if args.balance:
        labels = np.array([it["label"] for it in train_i])
        freq = np.bincount(labels, minlength=2)
        weights = 1.0 / np.maximum(freq[labels], 1)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_i), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, worker_init_fn=worker_init)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, worker_init_fn=worker_init)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = pick_device()
    model = build_classifier(n_classes=2).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"device: {device}  params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M  lr: {args.lr}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(model, train_dl, loss_fn, device, optimizer, desc=f"epoch {epoch:3d} train")
        va_loss, m = run_epoch(model, val_dl, loss_fn, device, desc=f"epoch {epoch:3d} val  ")
        scheduler.step()
        flag = ""
        if m["f1"] > best:
            best = m["f1"]
            torch.save({"model": model.state_dict(), "arch": "resnet18",
                        "n_classes": 2, "size": args.size}, args.out)
            flag = "  <- saved"
        print(f"\r{' ' * 90}\repoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"P {m['prec']:.3f} R {m['rec']:.3f} F1 {m['f1']:.3f} acc {m['acc']:.3f}{flag}")

    print(f"best val F1 {best:.4f}; checkpoint at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
