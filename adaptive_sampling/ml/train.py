"""Fine-tune pair quality classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_yaml
from ..paths import PROJECT_ROOT, resolve_path
from .build_dataset import build_dataset, load_ml_config
from .dataset import build_pair_dataset, load_training_pairs, split_by_video
from .model import PairQualityModel


def train(config_path: Path | None = None) -> Path:
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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(tr.get("num_workers", 2)),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(tr.get("num_workers", 2)))
        if val_ds
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        for img_a, img_b, y in train_loader:
            img_a, img_b, y = img_a.to(device), img_b.to(device), y.float().to(device)
            opt.zero_grad()
            logits = model(img_a, img_b)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == y.long()).sum().item()
            train_n += len(y)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_n, 1),
            "train_acc": train_correct / max(train_n, 1),
        }

        if val_loader:
            model.eval()
            val_correct = val_n = 0
            val_loss = 0.0
            with torch.no_grad():
                for img_a, img_b, y in val_loader:
                    img_a, img_b, y = img_a.to(device), img_b.to(device), y.float().to(device)
                    logits = model(img_a, img_b)
                    val_loss += criterion(logits, y).item() * len(y)
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    val_correct += (preds == y.long()).sum().item()
                    val_n += len(y)
            metrics["val_loss"] = val_loss / max(val_n, 1)
            metrics["val_acc"] = val_correct / max(val_n, 1)
            score = metrics["val_acc"]
        else:
            score = metrics["train_acc"]

        history.append(metrics)
        print(
            f"epoch {epoch}/{epochs}  train_acc={metrics['train_acc']:.3f}"
            + (f"  val_acc={metrics.get('val_acc', 0):.3f}" if val_loader else "")
        )

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
    args = p.parse_args(argv)
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
