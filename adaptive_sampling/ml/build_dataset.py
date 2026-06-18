"""Сборка датасета и экспорт шаблона Excel для разметки."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.config import load_yaml
from ..common.paths import PROJECT_ROOT, resolve_path
from .labels import (
    export_template_xlsx,
    iter_pairs_for_template,
    read_manual_xlsx,
    validate_pairs_exist,
    write_dataset_csv,
)


def load_ml_config(path: Path | None = None) -> dict:
    return load_yaml(path, "configs/ml_training.yaml")


def export_template(config_path: Path | None = None) -> Path:
    cfg = load_ml_config(config_path)
    paths = cfg.get("paths") or {}
    sparse_cfg = cfg.get("sparse_auto") or {}
    tpl = cfg.get("template") or {}

    rows = iter_pairs_for_template(
        paths.get("frames_root", "data/frames"),
        paths.get("sparse_results_root", "results/task2_sparse_eval"),
        pair_stride=int(tpl.get("pair_stride", 5)),
        include_all_on_failed=bool(tpl.get("include_all_pairs_on_failed_fps", True)),
        good_min_ratio=float(sparse_cfg.get("good_min_registered_ratio", 0.6)),
        require_passes=bool(sparse_cfg.get("require_passes_criteria", True)),
    )
    out = resolve_path(paths.get("manual_labels_xlsx", "data/labels/manual/pairs.xlsx"))
    export_template_xlsx(rows, out)
    return out


def build_dataset(config_path: Path | None = None, *, source: str = "manual") -> int:
    cfg = load_ml_config(config_path)
    paths = cfg.get("paths") or {}
    frames_root = resolve_path(paths.get("frames_root", "data/frames"))
    dataset_csv = resolve_path(paths.get("dataset_csv", "data/labels/pairs_dataset.csv"))

    if source == "pair_quality":
        from ..frame_pair_quality.dataset import load_results_from_dir, results_to_pair_labels

        pq_root = resolve_path(
            paths.get("pair_quality_results_root", "results/frame_pair_quality")
        )
        labels = results_to_pair_labels(load_results_from_dir(pq_root))
        labels = validate_pairs_exist(labels, frames_root)
        if not labels:
            raise RuntimeError(
                f"No pair quality labels in {pq_root}. "
                "Run: python -m adaptive_sampling.frame_pair_quality"
            )
    else:
        manual = resolve_path(paths.get("manual_labels_xlsx", "data/labels/manual/pairs.xlsx"))
        labels = read_manual_xlsx(manual)
        labels = validate_pairs_exist(labels, frames_root)
        if not labels:
            raise RuntimeError(
                f"No labeled pairs in {manual}. Fill column 'label' (good/bad). "
                "Run: python -m adaptive_sampling.ml.build_dataset --export-template"
            )

    write_dataset_csv(labels, dataset_csv)
    return len(labels)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build ML pair dataset / export Excel template")
    p.add_argument("--export-template", action="store_true", help="Create pairs.xlsx for manual labeling")
    p.add_argument(
        "--from-pair-quality",
        action="store_true",
        help="Build dataset from frame_pair_quality results (auto labels)",
    )
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/ml_training.yaml")
    args = p.parse_args(argv)

    if args.export_template:
        out = export_template(args.config)
        print(f"Template written: {out}")
        print("Fill column 'label' with good or bad, then run without --export-template")
        return 0

    source = "pair_quality" if args.from_pair_quality else "manual"
    n = build_dataset(args.config, source=source)
    print(f"Dataset built: {n} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
