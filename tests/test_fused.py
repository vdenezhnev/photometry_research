"""Tests for fused PLY → GLB export and MVS guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_sampling.sparse_eval.dense import require_mvs_api
from adaptive_sampling.sparse_eval.export_glb import export_fused_ply_glb


def test_export_fused_ply_glb(tmp_path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    ply_path = tmp_path / "fused.ply"
    cloud = trimesh.points.PointCloud(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        colors=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
    )
    cloud.export(str(ply_path))

    out = tmp_path / "fused.glb"
    path = export_fused_ply_glb(ply_path, out)
    assert path == out
    assert out.is_file() and out.stat().st_size > 0


def test_export_fused_ply_glb_missing_file() -> None:
    assert export_fused_ply_glb(Path("missing.ply"), Path("out.glb")) is None


def test_require_mvs_api_or_skip() -> None:
    pytest.importorskip("pycolmap")
    try:
        require_mvs_api()
    except RuntimeError as exc:
        assert "MVS" in str(exc)
