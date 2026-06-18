"""Метод: быстрая проверка пригодности соседних пар кадров для SfM."""

from .dataset import build_labeled_dataset, load_results_from_dir, results_to_pair_labels
from .evaluate import FpsPairQualityResult, evaluate_frames_dir
from .metrics import PairQualityMetrics, ProcessingContext, classify_pair, compute_pair_metrics, compute_pair_metrics_from_gray
from .run import run_batch, run_for_video, run_frames_dir

__all__ = [
    "PairQualityMetrics",
    "ProcessingContext",
    "FpsPairQualityResult",
    "compute_pair_metrics",
    "compute_pair_metrics_from_gray",
    "classify_pair",
    "evaluate_frames_dir",
    "run_frames_dir",
    "run_for_video",
    "run_batch",
    "build_labeled_dataset",
    "load_results_from_dir",
    "results_to_pair_labels",
]
