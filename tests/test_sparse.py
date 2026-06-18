"""Tests for sparse eval (no real PyCOLMAP run)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adaptive_sampling.metrics import composite_score, evaluate_success, empty_metrics
from adaptive_sampling.sparse import parse_frames_dir, select_top_fps_modes


def test_parse_frames_dir(tmp_path: Path) -> None:
    d = tmp_path / "video_a" / "fps_10"
    d.mkdir(parents=True)
    assert parse_frames_dir(d) == ("video_a", "fps_10")


def test_composite_score() -> None:
    assert composite_score(registered_ratio=0.9, sparse_points=3000, mean_track_length=5.0) > composite_score(
        registered_ratio=0.3, sparse_points=100, mean_track_length=1.0
    )


def test_evaluate_success() -> None:
    ok, checks = evaluate_success(
        registered_images=8,
        input_images=10,
        sparse_points=1200,
        mean_track_length=4.0,
        criteria={"min_registered_images_ratio": 0.6, "min_sparse_points": 500, "min_mean_track_length": 3.0},
    )
    assert ok and all(checks.values())


def test_empty_metrics() -> None:
    m = empty_metrics(input_images=5, criteria={})
    assert m.mapper_success is False
    assert m.registered_images == 0


class _M:
    def __init__(self, fps: str, score: float, reg: int) -> None:
        self.composite_score = score
        self.registered_images = reg
        self.registered_ratio = score
        self.sparse_points = 100
        self.passes_criteria = True


class _R:
    def __init__(self, fps: str, score: float, reg: int) -> None:
        self.fps_label = fps
        self.metrics = _M(fps, score, reg)


def test_select_top_fps_modes() -> None:
    results = [_R("fps_2", 0.9, 9), _R("fps_5", 0.7, 7), _R("fps_10", 0.5, 5)]
    top = select_top_fps_modes(results, top_k=2)
    assert top[0]["fps_label"] == "fps_2"


def test_subsample_files(tmp_path: Path) -> None:
    from adaptive_sampling.frames import _subsample_files

    files = [tmp_path / f"f{i:03d}.jpg" for i in range(10)]
    picked = _subsample_files(files, 4)
    assert len(picked) == 4
    assert picked[0] == files[0]
    assert picked[-1] == files[-1]
