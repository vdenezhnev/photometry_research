"""Оценка всех соседних пар в каталоге кадров."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.paths import resolve_path
from ..ml.labels import adjacent_pairs, list_frame_files
from .metrics import PairQualityMetrics, compute_pair_metrics


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


def evaluate_frames_dir(
    frames_dir: Path,
    *,
    config: dict[str, Any],
    video_slug: str | None = None,
    fps_label: str | None = None,
) -> FpsPairQualityResult:
    frames_dir = resolve_path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    slug, fps = parse_frames_dir(frames_dir)
    video_slug = video_slug or slug
    fps_label = fps_label or fps

    proc = config.get("processing") or {}
    thresholds = config.get("thresholds") or {}
    max_size = int(proc.get("max_image_size", 960))
    orb_features = int(proc.get("orb_features", 2000))

    files = list_frame_files(frames_dir)
    if len(files) < 2:
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

    pairs: list[PairQualityMetrics] = []
    for fa, fb in adjacent_pairs(frames_dir):
        pa, pb = frames_dir / fa, frames_dir / fb
        metrics = compute_pair_metrics(
            pa,
            pb,
            max_image_size=max_size,
            orb_features=orb_features,
            thresholds=thresholds,
        )
        pairs.append(
            PairQualityMetrics(
                frame_a=fa,
                frame_b=fb,
                similarity=metrics.similarity,
                feature_matches=metrics.feature_matches,
                scene_cut_score=metrics.scene_cut_score,
                suitable=metrics.suitable,
                status=metrics.status,
                reasons=metrics.reasons,
            )
        )

    good = [p for p in pairs if p.suitable]
    bad = [p for p in pairs if not p.suitable]
    n = len(pairs)

    return FpsPairQualityResult(
        video_slug=video_slug,
        fps_label=fps_label,
        frames_dir=str(frames_dir),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        total_pairs=n,
        good_pairs=len(good),
        bad_pairs=len(bad),
        good_ratio=round(len(good) / n, 4) if n else 0.0,
        mean_similarity=round(sum(p.similarity for p in pairs) / n, 4) if n else 0.0,
        mean_feature_matches=round(sum(p.feature_matches for p in pairs) / n, 2) if n else 0.0,
        mean_scene_cut_score=round(sum(p.scene_cut_score for p in pairs) / n, 4) if n else 0.0,
        pairs=pairs,
        problematic_pairs=bad,
    )
