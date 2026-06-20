"""Dense MVS stages (undistort → patch match → stereo fusion)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def require_mvs_api() -> Any:
    import pycolmap

    if not all(
        hasattr(pycolmap, name)
        for name in ("undistort_images", "patch_match_stereo", "stereo_fusion")
    ):
        raise RuntimeError(
            "PyCOLMAP MVS API недоступен в этой сборке. "
            "Нужен pycolmap с CUDA (например pycolmap-cuda12) или COLMAP CLI."
        )
    return pycolmap


def _merge_options(options_cls: Any, overrides: dict[str, Any]) -> Any:
    options = options_cls()
    if overrides:
        options.mergedict(overrides)
    return options


def run_pycolmap_dense(
    *,
    workspace: Path,
    images_dir: Path,
    fused_output_path: Path,
    dense_cfg: dict[str, Any],
    log: LogFn = _default_log,
) -> tuple[Path, dict[str, float]]:
    """Run COLMAP dense pipeline after sparse model exists under workspace/sparse/0."""
    pycolmap = require_mvs_api()

    sparse_model = workspace / "sparse" / "0"
    if not sparse_model.is_dir():
        raise FileNotFoundError(f"Sparse model not found: {sparse_model}")

    dense_path = workspace / "dense"
    if dense_path.exists():
        shutil.rmtree(dense_path)

    fused_output_path = Path(fused_output_path)
    fused_output_path.parent.mkdir(parents=True, exist_ok=True)
    if fused_output_path.exists():
        fused_output_path.unlink()

    undistort_options = _merge_options(
        pycolmap.UndistortCameraOptions,
        dense_cfg.get("undistort_options") or {},
    )
    patch_match_options = _merge_options(
        pycolmap.PatchMatchOptions,
        dense_cfg.get("patch_match_options") or {},
    )
    stereo_fusion_options = _merge_options(
        pycolmap.StereoFusionOptions,
        dense_cfg.get("stereo_fusion_options") or {},
    )
    input_type = str(dense_cfg.get("input_type") or "geometric")

    durations: dict[str, float] = {}

    t0 = time.perf_counter()
    log("  undistort_images...")
    pycolmap.undistort_images(
        dense_path,
        sparse_model,
        images_dir,
        undistort_options=undistort_options,
    )
    durations["image_undistorter"] = time.perf_counter() - t0
    log(f"  undistort_images done in {durations['image_undistorter']:.1f}s")

    t0 = time.perf_counter()
    log("  patch_match_stereo...")
    pycolmap.patch_match_stereo(dense_path, options=patch_match_options)
    durations["patch_match_stereo"] = time.perf_counter() - t0
    log(f"  patch_match_stereo done in {durations['patch_match_stereo']:.1f}s")

    t0 = time.perf_counter()
    log("  stereo_fusion...")
    pycolmap.stereo_fusion(
        fused_output_path,
        dense_path,
        input_type=input_type,
        options=stereo_fusion_options,
        output_type="ply",
    )
    durations["stereo_fusion"] = time.perf_counter() - t0
    log(f"  stereo_fusion done in {durations['stereo_fusion']:.1f}s")

    if not fused_output_path.is_file():
        raise RuntimeError(f"stereo_fusion did not produce {fused_output_path}")

    return fused_output_path, durations
