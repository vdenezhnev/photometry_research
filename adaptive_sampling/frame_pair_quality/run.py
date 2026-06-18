"""Запуск проверки пригодности пар кадров по видео и FPS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.config import load_pair_quality_config
from ..common.paths import resolve_path
from .evaluate import FpsPairQualityResult, evaluate_frames_dir
from .export import (
    write_comparison_csv,
    write_comparison_xlsx,
    write_pair_metrics_json,
    write_problematic_pairs_csv,
    write_report_xlsx,
    write_summary_json,
)
from .progress import log_stage


def _comparison_row(result: FpsPairQualityResult) -> dict[str, Any]:
    return {
        "video_slug": result.video_slug,
        "fps_label": result.fps_label,
        "total_pairs": result.total_pairs,
        "good_pairs": result.good_pairs,
        "bad_pairs": result.bad_pairs,
        "good_ratio": result.good_ratio,
        "bad_ratio": round(1.0 - result.good_ratio, 4) if result.total_pairs else 0.0,
        "mean_similarity": result.mean_similarity,
        "mean_feature_matches": result.mean_feature_matches,
        "mean_scene_cut_score": result.mean_scene_cut_score,
        "problematic_count": len(result.problematic_pairs),
    }


def save_fps_result(result: FpsPairQualityResult, output_dir: Path, *, top_k: int) -> Path:
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_pair_metrics_json(result, output_dir / "pair_metrics.json")
    write_summary_json(result, output_dir / "summary.json")
    write_problematic_pairs_csv(result, output_dir / "problematic_pairs.csv", top_k=top_k)
    write_report_xlsx(result, output_dir / "report.xlsx", top_k=top_k)
    return output_dir


def _resolve_run_paths(
    *,
    frames_root: Path | str | None,
    output_root: Path | str | None,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    paths = config.get("paths") or {}
    return (
        resolve_path(frames_root or paths.get("frames_root", "data/frames")),
        resolve_path(output_root or paths.get("results_root", "results/frame_pair_quality")),
    )


def run_for_video(
    video_slug: str,
    *,
    frames_root: Path | str = "data/frames",
    output_root: Path | str = "results/frame_pair_quality",
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
    show_progress: bool = False,
) -> list[FpsPairQualityResult]:
    cfg = config or load_pair_quality_config(config_path)
    frames_root, output_root = _resolve_run_paths(
        frames_root=frames_root,
        output_root=output_root,
        config=cfg,
    )
    report_cfg = cfg.get("report") or {}
    top_k = int(report_cfg.get("top_k_problematic", 50))

    video_dir = frames_root / video_slug
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video frames not found: {video_dir}")

    results: list[FpsPairQualityResult] = []
    fps_dirs = sorted(
        d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("fps_")
    )
    for fps_idx, fps_dir in enumerate(fps_dirs, start=1):
        if show_progress:
            log_stage(f"  [{fps_idx}/{len(fps_dirs)}] {fps_dir.name}")
        result = evaluate_frames_dir(
            fps_dir,
            config=cfg,
            video_slug=video_slug,
            show_progress=show_progress,
        )
        out = output_root / video_slug / fps_dir.name
        save_fps_result(result, out, top_k=top_k)
        results.append(result)
        if show_progress:
            log_stage(
                f"    -> {result.good_pairs}/{result.total_pairs} good, "
                f"{result.bad_pairs} bad"
            )

    if results:
        rows = [_comparison_row(r) for r in results]
        batch_dir = output_root / "_batch" / video_slug
        write_comparison_csv(rows, batch_dir / "comparison_table.csv")
        write_comparison_xlsx(video_slug=video_slug, rows=rows, path=batch_dir / "comparison_table.xlsx")
        (batch_dir / "comparison_table.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return results


def run_batch(
    *,
    frames_root: Path | str = "data/frames",
    output_root: Path | str | None = None,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
    show_progress: bool = False,
) -> dict[str, list[FpsPairQualityResult]]:
    cfg = config or load_pair_quality_config(config_path)
    frames_root, output_root = _resolve_run_paths(
        frames_root=frames_root,
        output_root=output_root,
        config=cfg,
    )

    all_results: dict[str, list[FpsPairQualityResult]] = {}
    video_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
    for video_idx, video_dir in enumerate(video_dirs, start=1):
        if show_progress:
            log_stage(f"\n[{video_idx}/{len(video_dirs)}] {video_dir.name}")
        all_results[video_dir.name] = run_for_video(
            video_dir.name,
            frames_root=frames_root,
            output_root=output_root,
            config=cfg,
            show_progress=show_progress,
        )
    return all_results


def run_frames_dir(
    frames_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
    show_progress: bool = False,
) -> FpsPairQualityResult:
    cfg = config or load_pair_quality_config(config_path)
    frames_dir = resolve_path(frames_dir)
    paths = cfg.get("paths") or {}
    report_cfg = cfg.get("report") or {}

    if output_dir is None:
        video_slug, fps_label = frames_dir.parent.name, frames_dir.name
        output_dir = resolve_path(paths.get("results_root", "results/frame_pair_quality")) / video_slug / fps_label
    else:
        output_dir = resolve_path(output_dir)

    result = evaluate_frames_dir(frames_dir, config=cfg, show_progress=show_progress)
    save_fps_result(result, output_dir, top_k=int(report_cfg.get("top_k_problematic", 50)))
    return result
