"""Fused (dense MVS) reconstruction for a single video/fps mode."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..common.config import load_fused_config
from ..common.gpu_monitor import GpuMonitor
from ..common.paths import resolve_path
from ..frame_extraction import copy_frames_to_workspace
from .dense import run_pycolmap_dense
from .export_glb import export_fused_ply_glb
from .run import _default_log, run_pycolmap_sparse

LogFn = Callable[[str], None]


@dataclass
class FusedRunResult:
    video_slug: str
    fps_label: str
    frames_dir: Path
    output_dir: Path
    fused_ply_path: Path | None
    fused_glb_path: Path | None
    sparse_registered_images: int
    sparse_points: int
    source_frame_count: int
    eval_frame_count: int
    stage_durations_sec: dict[str, float]
    gpu_stats: dict[str, Any] | None = None
    evaluated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_slug": self.video_slug,
            "fps_label": self.fps_label,
            "frames_dir": str(self.frames_dir),
            "output_dir": str(self.output_dir),
            "fused_ply_path": str(self.fused_ply_path) if self.fused_ply_path else None,
            "fused_glb_path": str(self.fused_glb_path) if self.fused_glb_path else None,
            "sparse_registered_images": self.sparse_registered_images,
            "sparse_points": self.sparse_points,
            "source_frame_count": self.source_frame_count,
            "eval_frame_count": self.eval_frame_count,
            "stage_durations_sec": self.stage_durations_sec,
            "gpu_stats": self.gpu_stats or {},
            "evaluated_at": self.evaluated_at,
        }


def run_fused_eval(
    *,
    video_slug: str | None = None,
    fps_label: str | None = None,
    frames_dir: Path | None = None,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    log: LogFn = _default_log,
) -> FusedRunResult:
    config = load_fused_config(config_path)
    paths_cfg = config.get("paths") or {}

    video_slug = video_slug or str(config.get("video_slug") or "")
    fps_label = fps_label or str(config.get("fps_label") or "")
    if not video_slug or not fps_label:
        raise ValueError("video_slug and fps_label are required")

    frames_root = resolve_path(paths_cfg.get("frames_root") or "data/frames")
    if frames_dir is None:
        frames_dir = frames_root / video_slug / fps_label
    else:
        frames_dir = resolve_path(frames_dir)

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    results_root = resolve_path(paths_cfg.get("results_root") or "results/fused_eval")
    if output_dir is None:
        output_dir = results_root / video_slug / fps_label
    else:
        output_dir = resolve_path(output_dir)

    workspace_root = resolve_path(paths_cfg.get("workspace_root") or "results/workspaces")
    workspace = workspace_root / f"{video_slug}__{fps_label}__fused"
    images_dir = workspace / "images"

    pycolmap_cfg = config.get("pycolmap") or {}
    max_images = pycolmap_cfg.get("max_images")
    max_images_int = int(max_images) if max_images is not None else None

    log(f"[{video_slug}/{fps_label}] fused reconstruction")
    eval_count, source_count = copy_frames_to_workspace(
        frames_dir,
        images_dir,
        max_images=max_images_int,
    )
    if source_count > eval_count:
        log(f"  subsampled {source_count} -> {eval_count} frames (max_images)")

    gpu_cfg = config.get("gpu_monitor") or {}
    device_requested = str(pycolmap_cfg.get("device") or "auto")
    durations: dict[str, float] = {}

    with GpuMonitor(
        device_requested=device_requested,
        sample_interval_sec=float(gpu_cfg.get("sample_interval_sec", 1.0)),
        active_util_threshold=float(gpu_cfg.get("active_util_threshold", 5.0)),
        enabled=bool(gpu_cfg.get("enabled", True)),
    ) as gpu_monitor:
        reconstruction, sparse_durations = run_pycolmap_sparse(
            images_dir,
            workspace,
            pycolmap_cfg,
            log=log,
        )
        if reconstruction is None:
            raise RuntimeError("Sparse reconstruction failed; cannot run dense MVS")

        dense_cfg = dict(config.get("dense") or {})
        export_cfg = config.get("export") or {}
        fused_ply_name = str(export_cfg.get("fused_ply") or "fused.ply")
        output_dir.mkdir(parents=True, exist_ok=True)
        fused_ply_path, dense_durations = run_pycolmap_dense(
            workspace=workspace,
            images_dir=images_dir,
            fused_output_path=output_dir / fused_ply_name,
            dense_cfg=dense_cfg,
            log=log,
        )
        gpu_stats = gpu_monitor.collect_stats().to_dict()

    durations.update(sparse_durations)
    durations.update(dense_durations)

    fused_glb_path: Path | None = None
    if export_cfg.get("glb", True):
        glb_name = str(export_cfg.get("fused_glb") or "fused.glb")
        try:
            fused_glb_path = export_fused_ply_glb(fused_ply_path, output_dir / glb_name)
            if fused_glb_path:
                log(f"  exported GLB: {fused_glb_path}")
        except Exception as exc:
            log(f"  GLB export failed: {exc}")

    if gpu_stats.get("gpu_available"):
        log(
            f"  GPU {gpu_stats.get('gpu_name')}: "
            f"active {gpu_stats.get('gpu_active_duration_sec')}s / "
            f"{gpu_stats.get('monitor_duration_sec')}s"
        )

    result = FusedRunResult(
        video_slug=video_slug,
        fps_label=fps_label,
        frames_dir=frames_dir,
        output_dir=output_dir,
        fused_ply_path=fused_ply_path,
        fused_glb_path=fused_glb_path,
        sparse_registered_images=int(reconstruction.num_reg_images()),
        sparse_points=int(reconstruction.num_points3D()),
        source_frame_count=source_count,
        eval_frame_count=eval_count,
        stage_durations_sec=durations,
        gpu_stats=gpu_stats,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "fused_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not export_cfg.get("keep_workspace", True):
        shutil.rmtree(workspace, ignore_errors=True)

    log(f"  done: {fused_ply_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fused (dense MVS) model for one video/fps.")
    parser.add_argument("--config", type=Path, default=None, help="Path to fused_eval.yaml")
    parser.add_argument("--video", default=None, help="Video slug (overrides config)")
    parser.add_argument("--fps", default=None, help="FPS label, e.g. fps_30")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Override frames directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args(argv)

    try:
        run_fused_eval(
            video_slug=args.video,
            fps_label=args.fps,
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
