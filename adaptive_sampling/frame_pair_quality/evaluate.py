"""Оценка всех соседних пар в каталоге кадров."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.paths import resolve_path
from ..ml.labels import list_frame_files
from .metrics import PairQualityMetrics, ProcessingContext
from .progress import log_pair_progress


@dataclass
class FpsPairQualityResult:
    video_slug: str
    fps_label: str
    frames_dir: str
    evaluated_at: str
    total_pairs: int
    good_pairs: int
    bad_pairs: int
    good_ratio: float
    mean_similarity: float
    mean_feature_matches: float
    mean_scene_cut_score: float
    pairs: list[PairQualityMetrics]
    problematic_pairs: list[PairQualityMetrics]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "video_slug": self.video_slug,
            "fps_label": self.fps_label,
            "frames_dir": self.frames_dir,
            "evaluated_at": self.evaluated_at,
            "total_pairs": self.total_pairs,
            "good_pairs": self.good_pairs,
            "bad_pairs": self.bad_pairs,
            "good_ratio": self.good_ratio,
            "mean_similarity": self.mean_similarity,
            "mean_feature_matches": self.mean_feature_matches,
            "mean_scene_cut_score": self.mean_scene_cut_score,
            "problematic_count": len(self.problematic_pairs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "pairs": [p.to_dict() for p in self.pairs],
            "problematic_pairs": [p.to_dict() for p in self.problematic_pairs],
        }


def parse_frames_dir(frames_dir: Path) -> tuple[str, str]:
    frames_dir = resolve_path(frames_dir)
    return frames_dir.parent.name, frames_dir.name


def _empty_result(
    *,
    video_slug: str,
    fps_label: str,
    frames_dir: Path,
) -> FpsPairQualityResult:
    return FpsPairQualityResult(
        video_slug=video_slug,
        fps_label=fps_label,
        frames_dir=str(frames_dir),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        total_pairs=0,
        good_pairs=0,
        bad_pairs=0,
        good_ratio=0.0,
        mean_similarity=0.0,
        mean_feature_matches=0.0,
        mean_scene_cut_score=0.0,
        pairs=[],
        problematic_pairs=[],
    )


def evaluate_frames_dir(
    frames_dir: Path,
    *,
    config: dict[str, Any],
    video_slug: str | None = None,
    fps_label: str | None = None,
    show_progress: bool = False,
) -> FpsPairQualityResult:
    frames_dir = resolve_path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    slug, fps = parse_frames_dir(frames_dir)
    video_slug = video_slug or slug
    fps_label = fps_label or fps

    files = list_frame_files(frames_dir)
    if len(files) < 2:
        return _empty_result(video_slug=video_slug, fps_label=fps_label, frames_dir=frames_dir)

    ctx = ProcessingContext.from_config(config)
    pairs: list[PairQualityMetrics] = []
    problematic: list[PairQualityMetrics] = []
    good_count = 0
    sim_sum = 0.0
    match_sum = 0.0
    cut_sum = 0.0
    total_pairs = len(files) - 1
    progress_label = f"{video_slug}/{fps_label}"

    for idx, (path_a, path_b) in enumerate(zip(files, files[1:]), start=1):
        metrics = ctx.eval_pair(path_a, path_b)
        pairs.append(metrics)
        sim_sum += metrics.similarity
        match_sum += metrics.feature_matches
        cut_sum += metrics.scene_cut_score
        if metrics.suitable:
            good_count += 1
        else:
            problematic.append(metrics)
        if show_progress:
            log_pair_progress(progress_label, idx, total_pairs)

    return FpsPairQualityResult(
        video_slug=video_slug,
        fps_label=fps_label,
        frames_dir=str(frames_dir),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        total_pairs=total_pairs,
        good_pairs=good_count,
        bad_pairs=total_pairs - good_count,
        good_ratio=round(good_count / total_pairs, 4),
        mean_similarity=round(sim_sum / total_pairs, 4),
        mean_feature_matches=round(match_sum / total_pairs, 2),
        mean_scene_cut_score=round(cut_sum / total_pairs, 4),
        pairs=pairs,
        problematic_pairs=problematic,
    )
