"""Tests for frame pair quality check."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from adaptive_sampling.frame_pair_quality.metrics import (
    classify_pair,
    compute_pair_metrics_from_gray,
)
from adaptive_sampling.frame_pair_quality.evaluate import evaluate_frames_dir, FpsPairQualityResult
from adaptive_sampling.frame_pair_quality.dataset import results_to_pair_labels
from adaptive_sampling.frame_pair_quality.metrics import PairQualityMetrics


def _similar_frame(seed: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 200, (size, size), dtype=np.uint8)


def test_identical_frames_are_good() -> None:
    img = _similar_frame(42)
    m = compute_pair_metrics_from_gray(
        img,
        img.copy(),
        thresholds={"min_similarity": 0.35, "min_feature_matches": 25, "max_scene_cut_score": 0.55},
    )
    assert m.suitable is True
    assert m.status == "good"
    assert m.similarity > 0.9
    assert m.feature_matches > 50


def test_different_frames_are_bad() -> None:
    m = compute_pair_metrics_from_gray(
        np.zeros((128, 128), dtype=np.uint8),
        np.full((128, 128), 255, dtype=np.uint8),
    )
    assert m.suitable is False
    assert m.status == "bad"
    assert m.reasons


def test_classify_pair_thresholds() -> None:
    good = classify_pair(
        frame_a="a.jpg",
        frame_b="b.jpg",
        similarity=0.8,
        feature_matches=100,
        scene_cut_score=0.1,
        thresholds={"min_similarity": 0.35, "min_feature_matches": 25, "max_scene_cut_score": 0.55},
    )
    assert good.suitable

    bad = classify_pair(
        frame_a="a.jpg",
        frame_b="b.jpg",
        similarity=0.2,
        feature_matches=5,
        scene_cut_score=0.9,
        thresholds={"min_similarity": 0.35, "min_feature_matches": 25, "max_scene_cut_score": 0.55},
    )
    assert not bad.suitable
    assert len(bad.reasons) == 3


def test_low_fps_simulation_more_bad_pairs() -> None:
    """Больший интервал между кадрами → больше непригодных пар (симуляция низкого FPS)."""
    base = _similar_frame(1)
    thresholds = {"min_similarity": 0.5, "min_feature_matches": 30, "max_scene_cut_score": 0.45}

    high_bad = 0
    low_bad = 0
    for i in range(5):
        ga = np.roll(base, i * 3, axis=1)
        gb = np.roll(base, (i + 1) * 3, axis=1)
        if not compute_pair_metrics_from_gray(ga, gb, orb_features=500, thresholds=thresholds).suitable:
            high_bad += 1

        la = np.roll(base, i * 18, axis=1)
        lb = np.roll(base, (i + 1) * 18, axis=1)
        if not compute_pair_metrics_from_gray(la, lb, orb_features=500, thresholds=thresholds).suitable:
            low_bad += 1

    assert low_bad >= high_bad


def test_evaluate_frames_dir_summary(tmp_path: Path) -> None:
    frames_dir = tmp_path / "vid" / "fps_5"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000.png").write_bytes(b"x")
    (frames_dir / "frame_001.png").write_bytes(b"y")
    (frames_dir / "frame_002.png").write_bytes(b"z")

    def _fake_metrics(path_a, path_b, **kwargs):
        return PairQualityMetrics(
            frame_a=path_a.name,
            frame_b=path_b.name,
            similarity=0.8,
            feature_matches=50,
            scene_cut_score=0.1,
            suitable=True,
            status="good",
            reasons=[],
        )

    cfg = {"processing": {"max_image_size": 128, "orb_features": 300}, "thresholds": {}}
    with patch(
        "adaptive_sampling.frame_pair_quality.evaluate.compute_pair_metrics",
        side_effect=_fake_metrics,
    ):
        result = evaluate_frames_dir(frames_dir, config=cfg)

    assert result.total_pairs == 2
    assert result.video_slug == "vid"
    assert result.good_pairs == 2


def test_results_to_pair_labels() -> None:
    pair = PairQualityMetrics(
        frame_a="a.jpg",
        frame_b="b.jpg",
        similarity=0.9,
        feature_matches=50,
        scene_cut_score=0.1,
        suitable=True,
        status="good",
        reasons=[],
    )
    result = FpsPairQualityResult(
        video_slug="v",
        fps_label="fps_5",
        frames_dir="/x",
        evaluated_at="t",
        total_pairs=1,
        good_pairs=1,
        bad_pairs=0,
        good_ratio=1.0,
        mean_similarity=0.9,
        mean_feature_matches=50,
        mean_scene_cut_score=0.1,
        pairs=[pair],
        problematic_pairs=[],
    )
    labels = results_to_pair_labels([result])
    assert len(labels) == 1
    assert labels[0].label == 1
    assert labels[0].source == "pair_quality"
