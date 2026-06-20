"""Export sparse COLMAP reconstruction to GLB (point cloud)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_sparse_pointcloud_glb(reconstruction: Any, output_path: Path) -> Path | None:
    """Export sparse 3D points from COLMAP reconstruction as GLB point cloud."""
    if reconstruction is None:
        return None

    num_points = int(reconstruction.num_points3D()) if hasattr(reconstruction, "num_points3D") else 0
    points3d = getattr(reconstruction, "points3D", None) or {}
    if num_points <= 0 and not points3d:
        return None

    import numpy as np
    import trimesh

    if not points3d and num_points > 0:
        points3d = reconstruction.points3D

    vertices = np.array([p.xyz for p in points3d.values()], dtype=np.float64)
    if len(vertices) == 0:
        return None

    colors = np.array([p.color for p in points3d.values()], dtype=np.uint8)
    if colors.ndim == 1:
        colors = np.stack(colors)
    if colors.shape[1] != 3:
        colors = np.broadcast_to(np.array([200, 200, 200], dtype=np.uint8), (len(vertices), 3))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = trimesh.points.PointCloud(vertices=vertices, colors=colors)
    cloud.export(str(output_path))
    return output_path


def export_fused_ply_glb(ply_path: Path, output_path: Path) -> Path | None:
    """Export dense fused.ply from COLMAP stereo_fusion as GLB point cloud."""
    ply_path = Path(ply_path)
    if not ply_path.is_file():
        return None

    import trimesh

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(str(ply_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return None
        loaded.export(str(output_path))
        return output_path

    if getattr(loaded, "vertices", None) is None or len(loaded.vertices) == 0:
        return None

    loaded.export(str(output_path))
    return output_path
