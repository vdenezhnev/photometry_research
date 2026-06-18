"""Сборка размеченного датасета из результатов проверки пар."""

from __future__ import annotations

import json
from pathlib import Path

from ..common.paths import resolve_path
from ..ml.labels import PairLabel, export_template_xlsx, validate_pairs_exist, write_dataset_csv
from .evaluate import FpsPairQualityResult
from .metrics import PairQualityMetrics


def _pair_from_dict(data: dict) -> PairQualityMetrics:
    return PairQualityMetrics(
        frame_a=data["frame_a"],
        frame_b=data["frame_b"],
        similarity=float(data["similarity"]),
        feature_matches=int(data["feature_matches"]),
        scene_cut_score=float(data["scene_cut_score"]),
        suitable=bool(data["suitable"]),
        status=str(data["status"]),
        reasons=list(data.get("reasons") or []),
    )


def _result_from_dict(data: dict) -> FpsPairQualityResult:
    pairs = [_pair_from_dict(p) for p in data.get("pairs") or []]
    problematic = [p for p in pairs if not p.suitable]
    total = len(pairs)
    good = total - len(problematic)
    return FpsPairQualityResult(
        video_slug=str(data["video_slug"]),
        fps_label=str(data["fps_label"]),
        frames_dir=str(data.get("frames_dir", "")),
        evaluated_at=str(data.get("evaluated_at", "")),
        total_pairs=int(data.get("total_pairs", total)),
        good_pairs=int(data.get("good_pairs", good)),
        bad_pairs=int(data.get("bad_pairs", len(problematic))),
        good_ratio=float(data.get("good_ratio", round(good / total, 4) if total else 0.0)),
        mean_similarity=float(data.get("mean_similarity", 0)),
        mean_feature_matches=float(data.get("mean_feature_matches", 0)),
        mean_scene_cut_score=float(data.get("mean_scene_cut_score", 0)),
        pairs=pairs,
        problematic_pairs=problematic,
    )


def load_results_from_dir(results_root: Path) -> list[FpsPairQualityResult]:
    results_root = resolve_path(results_root)
    loaded: list[FpsPairQualityResult] = []
    for metrics_path in sorted(results_root.glob("*/fps_*/pair_metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        loaded.append(_result_from_dict(data))
    return loaded


def results_to_pair_labels(results: list[FpsPairQualityResult]) -> list[PairLabel]:
    return [
        PairLabel(
            video_slug=result.video_slug,
            fps_label=result.fps_label,
            frame_a=pair.frame_a,
            frame_b=pair.frame_b,
            label=1 if pair.suitable else 0,
            suggested_label=pair.status,
            notes="; ".join(pair.reasons) if pair.reasons else None,
            source="pair_quality",
        )
        for result in results
        for pair in result.pairs
    ]


def build_labeled_dataset(
    *,
    results_root: Path,
    frames_root: Path,
    dataset_csv: Path,
    dataset_xlsx: Path | None = None,
) -> int:
    results = load_results_from_dir(results_root)
    if not results:
        raise RuntimeError(
            f"No pair quality results in {results_root}. "
            "Run: python -m adaptive_sampling.frame_pair_quality"
        )

    labels = validate_pairs_exist(results_to_pair_labels(results), frames_root)
    if not labels:
        raise RuntimeError("No valid labeled pairs (frame files missing?)")

    write_dataset_csv(labels, dataset_csv)

    if dataset_xlsx:
        export_template_xlsx(
            [
                {
                    "video_slug": p.video_slug,
                    "fps_label": p.fps_label,
                    "frame_a": p.frame_a,
                    "frame_b": p.frame_b,
                    "suggested_label": "good" if p.label == 1 else "bad",
                    "label": "good" if p.label == 1 else "bad",
                    "notes": p.notes or "",
                }
                for p in labels
            ],
            dataset_xlsx,
        )

    return len(labels)
