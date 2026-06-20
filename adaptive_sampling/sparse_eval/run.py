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
from typing import Any, Callable

from ..common.config import load_sparse_config
from ..common.gpu_monitor import GpuMonitor
from ..common.paths import resolve_path
from ..frame_extraction import copy_frames_to_workspace
from .export import write_comparison_xlsx, write_sparse_run_xlsx
from .export_glb import export_ply_pointcloud_glb, export_sparse_pointcloud_glb
from .dense import dense_api_available, run_dense_fusion
from .metrics import empty_metrics, metrics_from_reconstruction
from .run_log import MASTER_RUN_COLUMNS, append_master_run_log

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class EvalRunResult:
    video_slug: str
    fps_label: str
    frames_dir: Path
    output_dir: Path
    metrics: Any
    stage_durations_sec: dict[str, float]
    source_frame_count: int
    eval_frame_count: int
    glb_path: Path | None = None
    fused_ply_path: Path | None = None
    fused_glb_path: Path | None = None
    fused_point_count: int = 0
    gpu_stats: dict[str, Any] | None = None
    evaluated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_slug": self.video_slug,
            "fps_label": self.fps_label,
            "frames_dir": str(self.frames_dir),
            "output_dir": str(self.output_dir),
            "source_frame_count": self.source_frame_count,
            "eval_frame_count": self.eval_frame_count,
            "glb_path": str(self.glb_path) if self.glb_path else None,
            "fused_ply_path": str(self.fused_ply_path) if self.fused_ply_path else None,
            "fused_glb_path": str(self.fused_glb_path) if self.fused_glb_path else None,
            "fused_point_count": self.fused_point_count,
            "gpu_stats": self.gpu_stats or {},
            "metrics": self.metrics.to_dict(),
            "stage_durations_sec": self.stage_durations_sec,
            "evaluated_at": self.evaluated_at,
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


def _run_matching(
    database_path: Path,
    *,
    matcher: str,
    matching_options: dict[str, Any],
    device: Any,
    log: LogFn,
) -> None:
    import pycolmap

    name = matcher.strip().lower()
    if name == "exhaustive":
        log("  matching: exhaustive (медленно при >80 кадрах)")
        pycolmap.match_exhaustive(database_path, matching_options=matching_options, device=device)
    elif name == "spatial":
        log("  matching: spatial")
        pycolmap.match_spatial(database_path, matching_options=matching_options, device=device)
    else:
        log("  matching: sequential (рекомендуется для видео)")
        pycolmap.match_sequential(database_path, matching_options=matching_options, device=device)


def run_pycolmap_sparse(
    image_dir: Path,
    workspace: Path,
    pycolmap_cfg: dict[str, Any],
    *,
    log: LogFn = _default_log,
) -> tuple[Any | None, dict[str, float]]:
    import pycolmap

    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    device = _pycolmap_device(str(pycolmap_cfg.get("device") or "auto"))
    matcher = str(pycolmap_cfg.get("matcher") or "sequential")
    extraction_options = pycolmap_cfg.get("extraction_options") or {}
    matching_options = pycolmap_cfg.get("matching_options") or {}
    mapper_options = pycolmap_cfg.get("mapper_options") or {}

    durations: dict[str, float] = {}

    log(f"  device={device}, matcher={matcher}, images={len(list(image_dir.iterdir()))}")

    t0 = time.perf_counter()
    log("  extract_features...")
    pycolmap.extract_features(
        database_path,
        image_dir,
        extraction_options=extraction_options,
        device=device,
    )
    durations["feature_extractor"] = time.perf_counter() - t0
    log(f"  extract_features done in {durations['feature_extractor']:.1f}s")

    t0 = time.perf_counter()
    _run_matching(
        database_path,
        matcher=matcher,
        matching_options=matching_options,
        device=device,
        log=log,
    )
    durations["feature_matcher"] = time.perf_counter() - t0
    log(f"  matching done in {durations['feature_matcher']:.1f}s")

    t0 = time.perf_counter()
    log("  incremental_mapping...")
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sparse_dir,
        options=mapper_options,
    )
    durations["mapper"] = time.perf_counter() - t0
    log(f"  mapper done in {durations['mapper']:.1f}s")

    if not reconstructions:
        log("  mapper: no reconstruction")
        return None, durations

    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    best.write(sparse_dir / "0")
    log(f"  registered {best.num_reg_images()} images, {best.num_points3D()} points")
    return best, durations


def run_sparse_eval(
    frames_dir: Path,
    *,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    clean_workspace: bool = True,
    fused: bool = False,
    log: LogFn = _default_log,
) -> EvalRunResult:
    config = load_sparse_config(config_path)
    video_slug, fps_label = parse_frames_dir(frames_dir)
    frames_dir = resolve_path(frames_dir)
    dense_cfg = config.get("dense") or {}
    run_fused = fused or bool(dense_cfg.get("enabled", False))

    if output_dir is None:
        results_subdir = "task2_sparse_eval"
        if run_fused:
            results_subdir = str(dense_cfg.get("results_subdir") or "task2_fused_eval")
        output_dir = resolve_path(f"results/{results_subdir}/{video_slug}/{fps_label}")
    else:
        output_dir = resolve_path(output_dir)

    workspace_root = resolve_path(config.get("workspace_root") or "results/workspaces")
    workspace = workspace_root / f"{video_slug}__{fps_label}"
    images_dir = workspace / "images"

    pycolmap_cfg = config.get("pycolmap") or {}
    max_images = pycolmap_cfg.get("max_images")
    max_images_int = int(max_images) if max_images is not None else None

    eval_count, source_count = copy_frames_to_workspace(
        frames_dir,
        images_dir,
        max_images=max_images_int,
    )
    if source_count > eval_count:
        log(f"[{video_slug}/{fps_label}] subsampled {source_count} -> {eval_count} frames (max_images)")

    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_cfg = config.get("gpu_monitor") or {}
    device_requested = str(pycolmap_cfg.get("device") or "auto")
    fused_ply_path: Path | None = None
    fused_glb_path: Path | None = None
    fused_point_count = 0

    with GpuMonitor(
        device_requested=device_requested,
        sample_interval_sec=float(gpu_cfg.get("sample_interval_sec", 1.0)),
        active_util_threshold=float(gpu_cfg.get("active_util_threshold", 5.0)),
        enabled=bool(gpu_cfg.get("enabled", True)),
    ) as gpu_monitor:
        reconstruction, durations = run_pycolmap_sparse(images_dir, workspace, pycolmap_cfg, log=log)

        if run_fused and reconstruction is not None:
            if not dense_api_available():
                log("  fused: CUDA dense API unavailable in pycolmap, skipping")
            else:
                try:
                    sparse_model_dir = workspace / "sparse" / "0"
                    workspace_fused, dense_durations, fused_point_count = run_dense_fusion(
                        workspace=workspace,
                        image_dir=images_dir,
                        sparse_model_dir=sparse_model_dir,
                        dense_cfg=dense_cfg,
                        log=log,
                    )
                    durations.update(dense_durations)
                    if workspace_fused is not None:
                        dest_ply = output_dir / str(dense_cfg.get("fused_ply_filename", "fused.ply"))
                        shutil.copy2(workspace_fused, dest_ply)
                        fused_ply_path = dest_ply
                        if dense_cfg.get("export_glb", True):
                            glb_name = str(dense_cfg.get("fused_glb_filename", "fused.glb"))
                            fused_glb_path = export_ply_pointcloud_glb(dest_ply, output_dir / glb_name)
                            if fused_glb_path:
                                log(f"  exported fused GLB: {fused_glb_path} ({fused_point_count} points)")
                except Exception as exc:
                    log(f"  fused pipeline failed: {exc}")

        gpu_stats = gpu_monitor.collect_stats().to_dict()

    if gpu_stats.get("gpu_available"):
        log(
            f"  GPU {gpu_stats.get('gpu_name')}: "
            f"active {gpu_stats.get('gpu_active_duration_sec')}s / "
            f"{gpu_stats.get('monitor_duration_sec')}s "
            f"(util avg={gpu_stats.get('utilization_gpu_avg')}%, "
            f"max={gpu_stats.get('utilization_gpu_max')}%, "
            f"mem max={gpu_stats.get('memory_used_mb_max')} MB)"
        )
    else:
        log("  GPU monitor: nvidia-smi unavailable or disabled")

    db_path = workspace / "database.db"
    db_images = _count_database_images(db_path)
    criteria = config.get("success_criteria") or {}

    if reconstruction is None:
        metrics = empty_metrics(input_images=eval_count, database_images=db_images, criteria=criteria)
    else:
        metrics = metrics_from_reconstruction(
            reconstruction,
            input_images=eval_count,
            database_images=db_images,
            criteria=criteria,
        )

    glb_path: Path | None = None
    export_cfg = config.get("export") or {}
    if export_cfg.get("glb", True) and reconstruction is not None:
        glb_name = str(export_cfg.get("glb_filename", "sparse_pointcloud.glb"))
        try:
            glb_path = export_sparse_pointcloud_glb(reconstruction, output_dir / glb_name)
            if glb_path:
                log(f"  exported GLB: {glb_path} ({reconstruction.num_points3D()} points)")
        except Exception as exc:
            log(f"  GLB export failed: {exc}")

    result = EvalRunResult(
        video_slug=video_slug,
        fps_label=fps_label,
        frames_dir=frames_dir,
        output_dir=output_dir,
        metrics=metrics,
        stage_durations_sec=durations,
        source_frame_count=source_count,
        eval_frame_count=eval_count,
        glb_path=glb_path,
        fused_ply_path=fused_ply_path,
        fused_glb_path=fused_glb_path,
        fused_point_count=fused_point_count,
        gpu_stats=gpu_stats,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
    result_dict = result.to_dict()
    (output_dir / "sparse_metrics.json").write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_sparse_run_xlsx(result_dict, output_dir / "sparse_metrics.xlsx")

    master_log_path = resolve_path(
        export_cfg.get("master_log_xlsx", "results/task2_sparse_eval/all_runs_log.xlsx")
    )
    if export_cfg.get("master_log", True):
        append_master_run_log(_build_master_run_row(result, gpu_stats), master_log_path)
        log(f"  master log updated: {master_log_path}")

    if clean_workspace:
        shutil.rmtree(workspace, ignore_errors=True)

    return result


def _build_master_run_row(result: EvalRunResult, gpu_stats: dict[str, Any]) -> dict[str, Any]:
    m = result.metrics
    durations = result.stage_durations_sec or {}
    total_duration = round(sum(durations.values()), 3) if durations else 0.0
    row = {
        "evaluated_at": result.evaluated_at,
        "video_slug": result.video_slug,
        "fps_label": result.fps_label,
        "output_dir": str(result.output_dir),
        "device_requested": gpu_stats.get("device_requested"),
        "gpu_available": gpu_stats.get("gpu_available"),
        "gpu_name": gpu_stats.get("gpu_name"),
        "monitor_duration_sec": gpu_stats.get("monitor_duration_sec"),
        "gpu_active_duration_sec": gpu_stats.get("gpu_active_duration_sec"),
        "gpu_active_ratio": gpu_stats.get("gpu_active_ratio"),
        "gpu_util_avg": gpu_stats.get("utilization_gpu_avg"),
        "gpu_util_max": gpu_stats.get("utilization_gpu_max"),
        "gpu_mem_util_avg": gpu_stats.get("utilization_mem_avg"),
        "gpu_mem_util_max": gpu_stats.get("utilization_mem_max"),
        "gpu_memory_used_mb_avg": gpu_stats.get("memory_used_mb_avg"),
        "gpu_memory_used_mb_max": gpu_stats.get("memory_used_mb_max"),
        "gpu_memory_total_mb": gpu_stats.get("memory_total_mb"),
        "gpu_temperature_c_max": gpu_stats.get("temperature_c_max"),
        "gpu_samples_count": gpu_stats.get("samples_count"),
        "total_duration_sec": total_duration,
        "stage_feature_extractor_sec": durations.get("feature_extractor"),
        "stage_feature_matcher_sec": durations.get("feature_matcher"),
        "stage_mapper_sec": durations.get("mapper"),
        "stage_undistort_sec": durations.get("undistort"),
        "stage_patch_match_stereo_sec": durations.get("patch_match_stereo"),
        "stage_stereo_fusion_sec": durations.get("stereo_fusion"),
        "fused_ply_path": str(result.fused_ply_path) if result.fused_ply_path else "",
        "fused_glb_path": str(result.fused_glb_path) if result.fused_glb_path else "",
        "fused_point_count": result.fused_point_count,
        "source_frames": result.source_frame_count,
        "eval_frames": result.eval_frame_count,
        "input_images": m.input_images,
        "registered_images": m.registered_images,
        "registered_ratio": m.registered_ratio,
        "sparse_points": m.sparse_points,
        "mean_track_length": m.mean_track_length,
        "composite_score": m.composite_score,
        "passes_criteria": m.passes_criteria,
        "mapper_success": m.mapper_success,
        "glb_path": str(result.glb_path) if result.glb_path else "",
    }
    return {key: row.get(key, "") for key in MASTER_RUN_COLUMNS}


def _result_row(result: EvalRunResult) -> dict[str, Any]:
    m = result.metrics
    return {
        "video_slug": result.video_slug,
        "fps_label": result.fps_label,
        "source_frames": result.source_frame_count,
        "eval_frames": result.eval_frame_count,
        "input_images": m.input_images,
        "registered_images": m.registered_images,
        "registered_ratio": m.registered_ratio,
        "sparse_points": m.sparse_points,
        "mean_track_length": m.mean_track_length,
        "composite_score": m.composite_score,
        "passes_criteria": m.passes_criteria,
        "mapper_success": m.mapper_success,
        "glb_path": str(result.glb_path) if result.glb_path else "",
        "fused_ply_path": str(result.fused_ply_path) if result.fused_ply_path else "",
        "fused_glb_path": str(result.fused_glb_path) if result.fused_glb_path else "",
        "fused_point_count": result.fused_point_count,
        "gpu_util_avg": (result.gpu_stats or {}).get("utilization_gpu_avg"),
        "gpu_active_duration_sec": (result.gpu_stats or {}).get("gpu_active_duration_sec"),
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
    log: LogFn = _default_log,
) -> list[EvalRunResult]:
    config = load_sparse_config(config_path)
    frames_root = resolve_path(frames_root or "data/frames")
    results_root = resolve_path(results_root or "results/task2_sparse_eval")
    top_k = int(top_k if top_k is not None else config.get("top_k_fps_modes", 3))

    video_dir = frames_root / video_slug
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Not found: {video_dir}")

    fps_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir() and p.name.startswith("fps_"))
    if not fps_dirs:
        raise ValueError(f"No fps_* under {video_dir}")

    results: list[EvalRunResult] = []
    for i, fps_dir in enumerate(fps_dirs, start=1):
        log(f"\n[{video_slug}] ({i}/{len(fps_dirs)}) {fps_dir.name}")
        results.append(
            run_sparse_eval(
                fps_dir,
                output_dir=results_root / video_slug / fps_dir.name,
                config_path=config_path,
                log=log,
            )
        )

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


def run_fused_eval(
    frames_dir: Path,
    *,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    clean_workspace: bool = True,
    log: LogFn = _default_log,
) -> EvalRunResult:
    """Sparse SfM + dense stereo fusion for one fps_* directory."""
    return run_sparse_eval(
        frames_dir,
        output_dir=output_dir,
        config_path=config_path,
        clean_workspace=clean_workspace,
        fused=True,
        log=log,
    )


def run_fused_batch_for_video(
    video_slug: str,
    *,
    frames_root: Path | None = None,
    results_root: Path | None = None,
    config_path: Path | None = None,
    top_k: int | None = None,
    log: LogFn = _default_log,
) -> list[EvalRunResult]:
    """Run fused reconstruction for every fps_* under data/frames/<video>."""
    config = load_sparse_config(config_path)
    frames_root = resolve_path(frames_root or "data/frames")
    dense_cfg = config.get("dense") or {}
    results_subdir = str(dense_cfg.get("results_subdir") or "task2_sparse_eval")
    results_root = resolve_path(results_root or f"results/{results_subdir}")
    top_k = int(top_k if top_k is not None else config.get("top_k_fps_modes", 3))

    video_dir = frames_root / video_slug
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Not found: {video_dir}")

    fps_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir() and p.name.startswith("fps_"))
    if not fps_dirs:
        raise ValueError(f"No fps_* under {video_dir}")

    log(f"Fused batch: {video_slug} — {len(fps_dirs)} FPS mode(s)")
    if not dense_api_available():
        log("Warning: pycolmap dense API not available — fused stage will be skipped per run")

    results: list[EvalRunResult] = []
    for i, fps_dir in enumerate(fps_dirs, start=1):
        log(f"\n[{video_slug}] fused ({i}/{len(fps_dirs)}) {fps_dir.name}")
        results.append(
            run_fused_eval(
                fps_dir,
                output_dir=results_root / video_slug / fps_dir.name,
                config_path=config_path,
                log=log,
            )
        )

    batch_dir = results_root / "_batch" / video_slug
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(r) for r in results]
    top = select_top_fps_modes(results, top_k=top_k)

    with (batch_dir / "fused_comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_comparison_xlsx(
        video_slug=video_slug,
        comparison_rows=rows,
        top_fps_modes=top,
        xlsx_path=batch_dir / "fused_comparison_table.xlsx",
    )
    (batch_dir / "fused_top_fps_modes.json").write_text(
        json.dumps({"video_slug": video_slug, "top_fps_modes": top}, indent=2),
        encoding="utf-8",
    )
    return results
