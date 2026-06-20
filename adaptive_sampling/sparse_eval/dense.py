"""Dense MVS: undistort → patch match → stereo fusion (CUDA)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[str], None]


def dense_api_available() -> bool:
    try:
        import pycolmap
    except ImportError:
        return False
    return all(
        hasattr(pycolmap, name)
        for name in ("undistort_images", "patch_match_stereo", "stereo_fusion")
    )


def _build_patch_match_options(cfg: dict[str, Any]) -> Any:
    import pycolmap

    opts = pycolmap.PatchMatchOptions()
    pm_cfg = dict(cfg.get("patch_match_options") or {})
    if pm_cfg.pop("filter", True) is not False:
        opts.filter = True
    for key, val in pm_cfg.items():
        if hasattr(opts, key):
            setattr(opts, key, val)
    return opts


def _count_ply_vertices(ply_path: Path) -> int:
    try:
        import trimesh

        loaded = trimesh.load(str(ply_path))
        if isinstance(loaded, trimesh.Scene):
            count = 0
            for geom in loaded.geometry.values():
                count += len(getattr(geom, "vertices", []))
            return count
        return len(getattr(loaded, "vertices", []))
    except Exception:
        return 0


def run_dense_fusion(
    *,
    workspace: Path,
    image_dir: Path,
    sparse_model_dir: Path,
    dense_cfg: dict[str, Any],
    log: LogFn,
) -> tuple[Path | None, dict[str, float], int]:
    """Run COLMAP dense pipeline; returns (fused_ply, stage_durations, point_count)."""
    if not dense_api_available():
        raise RuntimeError(
            "Dense reconstruction requires pycolmap with CUDA (undistort_images, "
            "patch_match_stereo, stereo_fusion). Install a CUDA build on Colab."
        )

    import pycolmap

    dense_dir = workspace / str(dense_cfg.get("workspace_subdir", "dense"))
    dense_dir.mkdir(parents=True, exist_ok=True)

    fused_name = str(dense_cfg.get("fused_ply_filename", "fused.ply"))
    fused_path = dense_dir / fused_name
    if fused_path.is_file():
        fused_path.unlink()

    durations: dict[str, float] = {}

    t0 = time.perf_counter()
    log("  undistort_images...")
    pycolmap.undistort_images(dense_dir, sparse_model_dir, image_dir)
    durations["undistort"] = time.perf_counter() - t0
    log(f"  undistort done in {durations['undistort']:.1f}s")

    t0 = time.perf_counter()
    log("  patch_match_stereo...")
    pm_options = _build_patch_match_options(dense_cfg)
    pycolmap.patch_match_stereo(dense_dir, patch_match_options=pm_options)
    durations["patch_match_stereo"] = time.perf_counter() - t0
    log(f"  patch_match_stereo done in {durations['patch_match_stereo']:.1f}s")

    t0 = time.perf_counter()
    log("  stereo_fusion...")
    output_type = str(dense_cfg.get("output_type", "ply"))
    pycolmap.stereo_fusion(str(fused_path), str(dense_dir), output_type=output_type)
    durations["stereo_fusion"] = time.perf_counter() - t0
    log(f"  stereo_fusion done in {durations['stereo_fusion']:.1f}s")

    if not fused_path.is_file():
        log("  stereo_fusion: output file missing")
        return None, durations, 0

    point_count = _count_ply_vertices(fused_path)
    log(f"  fused point cloud: {point_count} points")
    return fused_path, durations, point_count
