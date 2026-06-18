"""Метрики соседних пар кадров: сходство, совпадения признаков, смена сцены."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class PairQualityMetrics:
    frame_a: str
    frame_b: str
    similarity: float
    feature_matches: int
    scene_cut_score: float
    suitable: bool
    status: str  # good | bad
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _resize_gray(path, max_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def histogram_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, corr))


def ssim_simple(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 6.5025, 58.5225
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if den == 0:
        return 0.0
    return float(max(0.0, min(1.0, num / den)))


def combined_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    hist_sim = histogram_similarity(gray_a, gray_b)
    ssim = ssim_simple(gray_a, gray_b)
    return round(0.5 * hist_sim + 0.5 * ssim, 4)


def scene_cut_score(gray_a: np.ndarray, gray_b: np.ndarray, *, hist_similarity: float) -> float:
    diff = cv2.absdiff(gray_a, gray_b)
    pixel_diff = float(np.mean(diff)) / 255.0
    hist_cut = 1.0 - hist_similarity
    return round(max(pixel_diff, hist_cut), 4)


def count_feature_matches(gray_a: np.ndarray, gray_b: np.ndarray, *, orb_features: int) -> int:
    orb = cv2.ORB_create(nfeatures=int(orb_features))
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return len(bf.match(des_a, des_b))


def classify_pair(
    *,
    frame_a: str,
    frame_b: str,
    similarity: float,
    feature_matches: int,
    scene_cut_score: float,
    thresholds: dict[str, Any],
) -> PairQualityMetrics:
    min_sim = float(thresholds.get("min_similarity", 0.35))
    min_matches = int(thresholds.get("min_feature_matches", 25))
    max_cut = float(thresholds.get("max_scene_cut_score", 0.55))

    reasons: list[str] = []
    if similarity < min_sim:
        reasons.append(f"low_similarity ({similarity:.3f} < {min_sim})")
    if feature_matches < min_matches:
        reasons.append(f"few_feature_matches ({feature_matches} < {min_matches})")
    if scene_cut_score > max_cut:
        reasons.append(f"scene_cut ({scene_cut_score:.3f} > {max_cut})")

    suitable = not reasons
    return PairQualityMetrics(
        frame_a=frame_a,
        frame_b=frame_b,
        similarity=similarity,
        feature_matches=feature_matches,
        scene_cut_score=scene_cut_score,
        suitable=suitable,
        status="good" if suitable else "bad",
        reasons=reasons,
    )


def compute_pair_metrics_from_gray(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    *,
    frame_a: str = "a",
    frame_b: str = "b",
    orb_features: int = 2000,
    thresholds: dict[str, Any] | None = None,
) -> PairQualityMetrics:
    hist_sim = histogram_similarity(gray_a, gray_b)
    similarity = combined_similarity(gray_a, gray_b)
    cut = scene_cut_score(gray_a, gray_b, hist_similarity=hist_sim)
    matches = count_feature_matches(gray_a, gray_b, orb_features=orb_features)
    return classify_pair(
        frame_a=frame_a,
        frame_b=frame_b,
        similarity=similarity,
        feature_matches=matches,
        scene_cut_score=cut,
        thresholds=thresholds or {},
    )


def compute_pair_metrics(
    path_a,
    path_b,
    *,
    max_image_size: int = 960,
    orb_features: int = 2000,
    thresholds: dict[str, Any] | None = None,
) -> PairQualityMetrics:
    gray_a = _resize_gray(path_a, max_image_size)
    gray_b = _resize_gray(path_b, max_image_size)
    return compute_pair_metrics_from_gray(
        gray_a,
        gray_b,
        frame_a=path_a.name if hasattr(path_a, "name") else str(path_a),
        frame_b=path_b.name if hasattr(path_b, "name") else str(path_b),
        orb_features=orb_features,
        thresholds=thresholds,
    )
