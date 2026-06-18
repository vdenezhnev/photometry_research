#!/usr/bin/env python3
"""CLI: быстрая проверка пригодности пар кадров для SfM + сборка размеченного датасета."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.config import load_pair_quality_config
from ..common.paths import PROJECT_ROOT, resolve_path
from .dataset import build_labeled_dataset
from .run import run_batch, run_for_video, run_frames_dir


def _print_fps_summary(results) -> None:
    for r in results:
        print(
            f"  {r.fps_label}: {r.good_pairs}/{r.total_pairs} good, "
            f"{r.bad_pairs} problematic (bad_ratio={(1 - r.good_ratio):.1%})"
        )


def _build_dataset(cfg, *, frames_root, output_root) -> int:
    paths = cfg.get("paths") or {}
    n = build_labeled_dataset(
        results_root=output_root,
        frames_root=frames_root,
        dataset_csv=resolve_path(paths.get("dataset_csv", "data/labels/pair_quality_dataset.csv")),
        dataset_xlsx=resolve_path(paths.get("dataset_xlsx", "data/labels/pair_quality/pairs_labeled.xlsx")),
    )
    dataset_csv = resolve_path(paths.get("dataset_csv", "data/labels/pair_quality_dataset.csv"))
    print(f"Labeled dataset: {n} pairs -> {dataset_csv}")
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fast frame pair quality check for SfM (similarity, feature matches, scene cut)"
    )
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/frame_pair_quality.yaml")
    p.add_argument("--frames-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--video", type=str, default=None, help="Single video slug")
    p.add_argument("--frames-dir", type=Path, default=None, help="Single fps_* directory")
    p.add_argument("--build-dataset", action="store_true", help="Build labeled dataset from results")
    p.add_argument("--run-and-build", action="store_true", help="Run check and build labeled dataset")
    args = p.parse_args(argv)

    cfg = load_pair_quality_config(args.config)
    paths = cfg.get("paths") or {}
    frames_root = resolve_path(args.frames_root or paths.get("frames_root", "data/frames"))
    output_root = resolve_path(args.output_root or paths.get("results_root", "results/frame_pair_quality"))

    if args.build_dataset and not args.frames_dir and not args.video and not args.run_and_build:
        _build_dataset(cfg, frames_root=frames_root, output_root=output_root)
        return 0

    if args.frames_dir:
        result = run_frames_dir(args.frames_dir, output_dir=args.output_root, config=cfg)
        print(f"{result.video_slug}/{result.fps_label}:")
        _print_fps_summary([result])
    elif args.video:
        print(f"{args.video}:")
        _print_fps_summary(run_for_video(args.video, frames_root=frames_root, output_root=output_root, config=cfg))
    else:
        all_results = run_batch(frames_root=frames_root, output_root=output_root, config=cfg)
        for slug, results in all_results.items():
            if not results:
                continue
            print(f"{slug}:")
            _print_fps_summary(results)

    if args.run_and_build or args.build_dataset:
        _build_dataset(cfg, frames_root=frames_root, output_root=output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
