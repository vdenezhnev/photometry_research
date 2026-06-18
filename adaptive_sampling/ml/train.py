"""Fine-tune pair quality classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common.config import load_yaml
from ..common.paths import PROJECT_ROOT, resolve_path
from .build_dataset import build_dataset, load_ml_config
from .dataset import build_pair_dataset, load_training_pairs, split_by_video
from .model import PairQualityModel
from .progress import batch_progress, update_postfix


def _run_loader(
    loader,
    *,
    model,
    device,
    criterion,
    optimizer=None,
    desc: str,
    show_progress: bool,
) -> tuple[float, float]:
    import torch

    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    n = 0
    n_batches = len(loader)
    postfix_every = max(1, n_batches // 40)

    batches = batch_progress(loader, desc=desc, total=n_batches, enabled=show_progress)
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch_idx, (img_a, img_b, y) in enumerate(batches, start=1):
            img_a, img_b, y = img_a.to(device), img_b.to(device), y.float().to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(img_a, img_b)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y.long()).sum().item()
            n += len(y)
            if show_progress:
                update_postfix(
                    batches,
                    loss=total_loss / max(n, 1),
                    acc=correct / max(n, 1),
                    every=postfix_every,
                    step=batch_idx,
                )

    return total_loss / max(n, 1), correct / max(n, 1)


def train(config_path: Path | None = None, *, show_progress: bool = True) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms

    cfg = load_ml_config(config_path)
    paths = cfg.get("paths") or {}
    tr = cfg.get("training") or {}

    build_dataset(config_path)

    frames_root = resolve_path(paths.get("frames_root", "data/frames"))
    manual = resolve_path(paths.get("manual_labels_xlsx", "data/labels/manual/pairs.xlsx"))
    dataset_csv = resolve_path(paths.get("dataset_csv", "data/labels/pairs_dataset.csv"))
    ckpt_dir = resolve_path(paths.get("checkpoints_dir", "models/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_training_pairs(
        manual_xlsx=manual,
        dataset_csv=dataset_csv,
        frames_root=frames_root,
    )
    if len(pairs) < 8:
        raise RuntimeError(f"Need at least 8 labeled pairs, got {len(pairs)}")

    train_pairs, val_pairs = split_by_video(
        pairs,
        val_ratio=float(tr.get("val_ratio", 0.2)),
        seed=int(tr.get("seed", 42)),
    )

    image_size = int(tr.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = build_pair_dataset(train_pairs, frames_root, transform)
    val_ds = build_pair_dataset(val_pairs, frames_root, transform) if val_pairs else None

    batch_size = int(tr.get("batch_size", 16))
    num_workers = int(tr.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if show_progress:
        print(f"Device: {device}  train pairs: {len(train_pairs)}  val pairs: {len(val_pairs)}")

    model = PairQualityModel(backbone=str(tr.get("backbone", "resnet18")), pretrained=True).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr.get("learning_rate", 1e-4)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
    )
    criterion = nn.BCEWithLogitsLoss()

    epochs = int(tr.get("epochs", 15))
    best_val = -1.0
    best_path = ckpt_dir / "best.pt"
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_loader(
            train_loader,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=opt,
            desc=f"Epoch {epoch}/{epochs} train",
            show_progress=show_progress,
        )

        metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }

        if val_loader:
            val_loss, val_acc = _run_loader(
                val_loader,
                model=model,
                device=device,
                criterion=criterion,
                desc=f"Epoch {epoch}/{epochs} val",
                show_progress=show_progress,
            )
            metrics["val_loss"] = val_loss
            metrics["val_acc"] = val_acc
            score = val_acc
        else:
            score = train_acc

        history.append(metrics)

        line = f"Epoch {epoch}/{epochs}  train_acc={train_acc:.3f}  train_loss={train_loss:.4f}"
        if val_loader:
            line += f"  val_acc={metrics['val_acc']:.3f}  val_loss={metrics['val_loss']:.4f}"
        print(line)

        if score >= best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": tr.get("backbone", "resnet18"),
                    "image_size": image_size,
                    "threshold": float(cfg.get("inference", {}).get("threshold", 0.5)),
                },
                best_path,
            )

    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (score={best_val:.3f})")
    return best_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train pair quality classifier")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/ml_training.yaml")
    p.add_argument("--quiet", action="store_true", help="Disable progress bars")
    args = p.parse_args(argv)
    train(args.config, show_progress=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
