"""Inference: score adjacent frame pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..config import load_yaml
from ..paths import PROJECT_ROOT, resolve_path
from .build_dataset import load_ml_config
from .labels import adjacent_pairs


def predict_frames_dir(
    frames_dir: Path,
    checkpoint: Path,
    *,
    config_path: Path | None = None,
    output_csv: Path | None = None,
) -> list[dict]:
    import torch
    from PIL import Image
    from torchvision import transforms

    from .model import PairQualityModel

    cfg = load_ml_config(config_path)
    inf = cfg.get("inference") or {}
    threshold = float(inf.get("threshold", 0.5))

    ckpt = torch.load(resolve_path(checkpoint), map_location="cpu", weights_only=False)
    image_size = int(ckpt.get("image_size", 224))
    backbone = str(ckpt.get("backbone", "resnet18"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairQualityModel(backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames_dir = resolve_path(frames_dir)
    rows: list[dict] = []

    with torch.no_grad():
        for fa, fb in adjacent_pairs(frames_dir):
            pa, pb = frames_dir / fa, frames_dir / fb
            img_a = transform(Image.open(pa).convert("RGB")).unsqueeze(0).to(device)
            img_b = transform(Image.open(pb).convert("RGB")).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(img_a, img_b)).item()
            rows.append(
                {
                    "frame_a": fa,
                    "frame_b": fb,
                    "prob_good": round(prob, 4),
                    "label": "good" if prob >= threshold else "bad",
                }
            )

    if output_csv:
        output_csv = resolve_path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame_a", "frame_b", "prob_good", "label"])
            w.writeheader()
            w.writerows(rows)

    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict good/bad frame pairs")
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/best.pt"))
    p.add_argument("--output", type=Path, help="CSV output path")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/ml_training.yaml")
    args = p.parse_args(argv)

    rows = predict_frames_dir(
        args.frames_dir,
        args.checkpoint,
        config_path=args.config,
        output_csv=args.output,
    )
    summary = {
        "frames_dir": str(args.frames_dir),
        "pairs": len(rows),
        "good": sum(1 for r in rows if r["label"] == "good"),
        "bad": sum(1 for r in rows if r["label"] == "bad"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
