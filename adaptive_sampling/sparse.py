"""Sparse SfM evaluation with PyCOLMAP."""

from __future__ import annotations

import csv
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_sparse_config
from .export import write_comparison_xlsx, write_sparse_run_xlsx
from .frames import copy_frames_to_workspace
from .metrics import empty_metrics, metrics_from_reconstruction
from .paths import resolve_path


@dataclass
class EvalRunResult:
    video_slug: str
    fps_label: str
    frames_dir: Path
    output_dir: Path
    metrics: Any
    stage_durations_sec: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_slug": self.video_slug,
            "fps_label": self.fps_label,
            "frames_dir": str(self.frames_dir),
            "output_dir": str(self.output_dir),
            "metrics": self.metrics.to_dict(),
            "stage_durations_sec": self.stage_durations_sec,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }


_FPS_DIR_RE = re.compile(r"^fps_(?P<fps>.+)$")


def parse_frames_dir(frames_dir: Path) -> tuple[str, str]:
    frames_dir = frames_dir.resolve()
    if not _FPS_DIR_RE.match(frames_dir.name):
        raise ValueError(f"Expected fps_<N> directory, got: {frames_dir.name}")
    return frames_dir.parent.name, frames_dir.name


def _pycolmap_device(name: str) -> Any:
    import pycolmap

    return getattr(pycolmap.Device, str(name).lower(), pycolmap.Device.auto)


def _count_database_images(database_path: Path) -> int:
    import sqlite3

    if not database_path.is_file():
        return 0
    con = sqlite3.connect(str(database_path))
    try:
        row = con.execute("SELECT COUNT(*) FROM images").fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def run_pycolmap_sparse(
    image_dir: Path,
    workspace: Path,
    pycolmap_cfg: dict[str, Any],
) -> tuple[Any | None, dict[str, float]]:
    import pycolmap

    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    device = _pycolmap_device(str(pycolmap_cfg.get("device") or "auto"))
    extraction_options = pycolmap_cfg.get("extraction_options") or {}
    matching_options = pycolmap_cfg.get("matching_options") or {}
    mapper_options = pycolmap_cfg.get("mapper_options") or {}

    durations: dict[str, float] = {}

    t0 = time.perf_counter()
    pycolmap.extract_features(
        database_path,
        image_dir,
        extraction_options=extraction_options,
        device=device,
    )
    durations["feature_extractor"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pycolmap.match_exhaustive(
        database_path,
        matching_options=matching_options,
        device=device,
    )
    durations["feature_matcher"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sparse_dir,
        options=mapper_options,
    )
    durations["mapper"] = time.perf_counter() - t0

    if not reconstructions:
        return None, durations

    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    best.write(sparse_dir / "0")
    return best, durations


def run_sparse_eval(
    frames_dir: Path,
    *,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    clean_workspace: bool = True,
) -> EvalRunResult:
    config = load_sparse_config(config_path)
    video_slug, fps_label = parse_frames_dir(frames_dir)
    frames_dir = resolve_path(frames_dir)

    if output_dir is None:
        output_dir = resolve_path(f"results/sparse_eval/{video_slug}/{fps_label}")
    else:
        output_dir = resolve_path(output_dir)

    workspace_root = resolve_path(config.get("workspace_root") or "results/workspaces")
    workspace = workspace_root / f"{video_slug}__{fps_label}"
    images_dir = workspace / "images"

    input_count = copy_frames_to_workspace(frames_dir, images_dir)
    pycolmap_cfg = config.get("pycolmap") or {}
    reconstruction, durations = run_pycolmap_sparse(images_dir, workspace, pycolmap_cfg)

    db_path = workspace / "database.db"
    db_images = _count_database_images(db_path)
    criteria = config.get("success_criteria") or {}

    if reconstruction is None:
        metrics = empty_metrics(input_images=input_count, database_images=db_images, criteria=criteria)
    else:
        metrics = metrics_from_reconstruction(
            reconstruction,
            input_images=input_count,
            database_images=db_images,
            criteria=criteria,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = EvalRunResult(
        video_slug=video_slug,
        fps_label=fps_label,
        frames_dir=frames_dir,
        output_dir=output_dir,
        metrics=metrics,
        stage_durations_sec=durations,
    )
    (output_dir / "sparse_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_sparse_run_xlsx(result.to_dict(), output_dir / "sparse_metrics.xlsx")

    if clean_workspace:
        shutil.rmtree(workspace, ignore_errors=True)

    return result


def _result_row(result: EvalRunResult) -> dict[str, Any]:
    m = result.metrics
    return {
        "video_slug": result.video_slug,
        "fps_label": result.fps_label,
        "input_images": m.input_images,
        "registered_images": m.registered_images,
        "registered_ratio": m.registered_ratio,
        "sparse_points": m.sparse_points,
        "mean_track_length": m.mean_track_length,
        "composite_score": m.composite_score,
        "passes_criteria": m.passes_criteria,
        "mapper_success": m.mapper_success,
    }


def select_top_fps_modes(results: list[EvalRunResult], top_k: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        results,
        key=lambda r: (r.metrics.composite_score, r.metrics.registered_images),
        reverse=True,
    )
    return [
        {
            "fps_label": r.fps_label,
            "composite_score": r.metrics.composite_score,
            "registered_images": r.metrics.registered_images,
            "registered_ratio": r.metrics.registered_ratio,
            "sparse_points": r.metrics.sparse_points,
            "passes_criteria": r.metrics.passes_criteria,
        }
        for r in ranked[:top_k]
    ]


def run_batch_for_video(
    video_slug: str,
    *,
    frames_root: Path | None = None,
    results_root: Path | None = None,
    config_path: Path | None = None,
    top_k: int | None = None,
) -> list[EvalRunResult]:
    config = load_sparse_config(config_path)
    frames_root = resolve_path(frames_root or "data/frames")
    results_root = resolve_path(results_root or "results/sparse_eval")
    top_k = int(top_k if top_k is not None else config.get("top_k_fps_modes", 3))

    video_dir = frames_root / video_slug
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Not found: {video_dir}")

    fps_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir() and p.name.startswith("fps_"))
    if not fps_dirs:
        raise ValueError(f"No fps_* under {video_dir}")

    results = [
        run_sparse_eval(fps_dir, output_dir=results_root / video_slug / fps_dir.name, config_path=config_path)
        for fps_dir in fps_dirs
    ]

    batch_dir = results_root / "_batch" / video_slug
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(r) for r in results]
    top = select_top_fps_modes(results, top_k=top_k)

    with (batch_dir / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_comparison_xlsx(
        video_slug=video_slug,
        comparison_rows=rows,
        top_fps_modes=top,
        xlsx_path=batch_dir / "comparison_table.xlsx",
    )
    (batch_dir / "top_fps_modes.json").write_text(
        json.dumps({"video_slug": video_slug, "top_fps_modes": top}, indent=2),
        encoding="utf-8",
    )
    return results
