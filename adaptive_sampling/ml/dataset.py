"""PyTorch dataset for frame pairs."""

from __future__ import annotations

import csv
from pathlib import Path

from ..common.paths import resolve_path
from .labels import PairLabel, pair_image_paths, read_manual_xlsx, validate_pairs_exist


def load_pairs_from_csv(path: Path) -> list[PairLabel]:
    path = resolve_path(path)
    pairs: list[PairLabel] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append(
                PairLabel(
                    video_slug=row["video_slug"],
                    fps_label=row["fps_label"],
                    frame_a=row["frame_a"],
                    frame_b=row["frame_b"],
                    label=int(row["label"]),
                    source=row.get("source") or "csv",
                    notes=row.get("notes") or None,
                )
            )
    return pairs


def load_training_pairs(
    *,
    manual_xlsx: Path,
    dataset_csv: Path | None,
    frames_root: Path,
) -> list[PairLabel]:
    pairs = read_manual_xlsx(manual_xlsx)
    if not pairs and dataset_csv and resolve_path(dataset_csv).is_file():
        pairs = load_pairs_from_csv(dataset_csv)
    return validate_pairs_exist(pairs, frames_root)


def split_by_video(pairs: list[PairLabel], val_ratio: float, seed: int) -> tuple[list[PairLabel], list[PairLabel]]:
    import random

    by_video: dict[str, list[PairLabel]] = {}
    for p in pairs:
        by_video.setdefault(p.video_slug, []).append(p)

    videos = sorted(by_video)
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_ratio)) if len(videos) > 1 else 0
    val_set = set(videos[:n_val])

    train, val = [], []
    for v, items in by_video.items():
        (val if v in val_set else train).extend(items)
    return train, val


def build_pair_dataset(pairs: list[PairLabel], frames_root: Path, transform):
    from torch.utils.data import Dataset

    frames_root = resolve_path(frames_root)

    class _PairDS(Dataset):
        def __len__(self) -> int:
            return len(pairs)

        def __getitem__(self, idx: int):
            from PIL import Image

            p = pairs[idx]
            pa, pb = pair_image_paths(p, frames_root)
            img_a = transform(Image.open(pa).convert("RGB"))
            img_b = transform(Image.open(pb).convert("RGB"))
            return img_a, img_b, p.label

    return _PairDS()
