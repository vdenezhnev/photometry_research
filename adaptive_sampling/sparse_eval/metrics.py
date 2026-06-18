"""Sparse reconstruction metrics from PyCOLMAP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SparseMetrics:
    input_images: int
    database_images: int
    registered_images: int
    registered_ratio: float
    sparse_points: int
    observations: int
    mean_track_length: float
    mean_observations_per_image: float
    mean_reprojection_error_px: float | None
    mapper_success: bool
    passes_criteria: bool
    criteria_details: dict[str, bool]
    composite_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def composite_score(
    *,
    registered_ratio: float,
    sparse_points: int,
    mean_track_length: float,
) -> float:
    points_term = min(sparse_points / 5000.0, 1.0)
    track_term = min(mean_track_length / 10.0, 1.0)
    return 0.55 * registered_ratio + 0.30 * points_term + 0.15 * track_term


def evaluate_success(
    *,
    registered_images: int,
    input_images: int,
    sparse_points: int,
    mean_track_length: float,
    criteria: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    ratio = registered_images / input_images if input_images > 0 else 0.0
    checks = {
        "registered_ratio": ratio >= float(criteria.get("min_registered_images_ratio", 0.6)),
        "sparse_points": sparse_points >= int(criteria.get("min_sparse_points", 500)),
        "mean_track_length": mean_track_length >= float(criteria.get("min_mean_track_length", 3.0)),
    }
    return all(checks.values()), checks


def metrics_from_reconstruction(
    reconstruction: Any,
    *,
    input_images: int,
    database_images: int,
    criteria: dict[str, Any],
) -> SparseMetrics:
    registered = int(reconstruction.num_reg_images())
    ratio = registered / input_images if input_images > 0 else 0.0
    points = int(reconstruction.num_points3D())
    track_len = float(reconstruction.compute_mean_track_length())
    obs_per_img = float(reconstruction.compute_mean_observations_per_reg_image())
    reproj = float(reconstruction.compute_mean_reprojection_error())
    observations = sum(len(p.track.elements) for p in reconstruction.points3D.values())

    passes, checks = evaluate_success(
        registered_images=registered,
        input_images=input_images,
        sparse_points=points,
        mean_track_length=track_len,
        criteria=criteria,
    )
    score = composite_score(
        registered_ratio=ratio,
        sparse_points=points,
        mean_track_length=track_len,
    )
    return SparseMetrics(
        input_images=input_images,
        database_images=database_images,
        registered_images=registered,
        registered_ratio=round(ratio, 4),
        sparse_points=points,
        observations=observations,
        mean_track_length=track_len,
        mean_observations_per_image=obs_per_img,
        mean_reprojection_error_px=reproj,
        mapper_success=True,
        passes_criteria=passes,
        criteria_details=checks,
        composite_score=round(score, 4),
    )


def empty_metrics(
    *,
    input_images: int,
    database_images: int = 0,
    criteria: dict[str, Any],
) -> SparseMetrics:
    checks = {
        "registered_ratio": False,
        "sparse_points": False,
        "mean_track_length": False,
    }
    return SparseMetrics(
        input_images=input_images,
        database_images=database_images,
        registered_images=0,
        registered_ratio=0.0,
        sparse_points=0,
        observations=0,
        mean_track_length=0.0,
        mean_observations_per_image=0.0,
        mean_reprojection_error_px=None,
        mapper_success=False,
        passes_criteria=False,
        criteria_details=checks,
        composite_score=0.0,
    )
