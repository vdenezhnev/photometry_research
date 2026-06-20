"""Tests for sparse GLB export."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_sampling.sparse_eval.export_glb import export_sparse_pointcloud_glb


class _Point:
    def __init__(self, xyz, color) -> None:
        self.xyz = xyz
        self.color = color


class _Reconstruction:
    def __init__(self, points) -> None:
        self.points3D = points

    def num_points3D(self) -> int:
        return len(self.points3D)


def test_export_sparse_pointcloud_glb(tmp_path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    recon = _Reconstruction(
        {
            0: _Point([0.0, 0.0, 0.0], [255, 0, 0]),
            1: _Point([1.0, 0.0, 0.0], [0, 255, 0]),
            2: _Point([0.0, 1.0, 0.0], [0, 0, 255]),
        }
    )
    out = tmp_path / "sparse_pointcloud.glb"
    path = export_sparse_pointcloud_glb(recon, out)
    assert path == out
    assert out.is_file()
    assert out.stat().st_size > 0

    loaded = trimesh.load(str(out))
    if isinstance(loaded, trimesh.Scene):
        geom = next(iter(loaded.geometry.values()))
        assert len(geom.vertices) == 3
    else:
        assert len(loaded.vertices) == 3


def test_export_returns_none_without_points() -> None:
    recon = _Reconstruction({})
    assert export_sparse_pointcloud_glb(recon, Path("x.glb")) is None
    assert export_sparse_pointcloud_glb(None, Path("x.glb")) is None
