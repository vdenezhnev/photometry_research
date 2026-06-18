"""Сборка размеченного датасета из результатов проверки пар."""

from __future__ import annotations

import json
from pathlib import Path

from ..common.paths import resolve_path
from ..ml.labels import PairLabel, export_template_xlsx, validate_pairs_exist, write_dataset_csv
from .evaluate import FpsPairQualityResult


def load_results_from_dir(results_root: Path) -> list[FpsPairQualityResult]:
    results_root = resolve_path(results_root)
    loaded: list[FpsPairQualityResult] = []
    for summary_path in sorted(results_root.glob("*/fps_*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        pairs_path = summary_path.parent / "pair_metrics.json"
        if pairs_path.is_file():
            full = json.loads(pairs_path.read_text(encoding="utf-8"))
            pairs_data = full.get("pairs") or []
        else:
            pairs_data = []

        from .metrics import PairQualityMetrics

        pairs = [
            PairQualityMetrics(
                frame_a=p["frame_a"],
                frame_b=p["frame_b"],
                similarity=float(p["similarity"]),
                feature_matches=int(p["feature_matches"]),
                scene_cut_score=float(p["scene_cut_score"]),
                suitable=bool(p["suitable"]),
                status=str(p["status"]),
                reasons=list(p.get("reasons") or []),
            )
            for p in pairs_data
        ]
        bad = [p for p in pairs if not p.suitable]
        loaded.append(
            FpsPairQualityResult(
                video_slug=str(data["video_slug"]),
                fps_label=str(data["fps_label"]),
                frames_dir=str(data.get("frames_dir", "")),
                evaluated_at=str(data.get("evaluated_at", "")),
                total_pairs=int(data.get("total_pairs", len(pairs))),
                good_pairs=int(data.get("good_pairs", sum(1 for p in pairs if p.suitable))),
                bad_pairs=int(data.get("bad_pairs", len(bad))),
                good_ratio=float(data.get("good_ratio", 0)),
                mean_similarity=float(data.get("mean_similarity", 0)),
                mean_feature_matches=float(data.get("mean_feature_matches", 0)),
                mean_scene_cut_score=float(data.get("mean_scene_cut_score", 0)),
                pairs=pairs,
                problematic_pairs=bad,
            )
        )
    return loaded


def results_to_pair_labels(results: list[FpsPairQualityResult]) -> list[PairLabel]:
    labels: list[PairLabel] = []
    for result in results:
        for pair in result.pairs:
            labels.append(
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
            )
    return labels


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

    labels = results_to_pair_labels(results)
    labels = validate_pairs_exist(labels, frames_root)
    if not labels:
        raise RuntimeError("No valid labeled pairs (frame files missing?)")

    write_dataset_csv(labels, dataset_csv)

    if dataset_xlsx:
        rows = [
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
        ]
        export_template_xlsx(rows, dataset_xlsx)

    return len(labels)
