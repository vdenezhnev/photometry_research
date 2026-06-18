"""Метрики соседних пар кадров: сходство, совпадения признаков, смена сцены."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
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


@dataclass
class ProcessingContext:
    """Переиспользуемый контекст: ORB, matcher, кэш кадров."""

    max_image_size: int
    orb_features: int
    thresholds: dict[str, Any]
    _orb: Any = field(repr=False)
    _matcher: Any = field(repr=False)
    _frame_cache: dict[Path, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProcessingContext:
        proc = config.get("processing") or {}
        orb_features = int(proc.get("orb_features", 2000))
        orb = cv2.ORB_create(nfeatures=orb_features)
        return cls(
            max_image_size=int(proc.get("max_image_size", 960)),
            orb_features=orb_features,
            thresholds=config.get("thresholds") or {},
            _orb=orb,
            _matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
        )

    def load_gray(self, path: Path) -> np.ndarray:
        path = Path(path)
        cached = self._frame_cache.get(path)
        if cached is not None:
            return cached
        gray = _resize_gray(path, self.max_image_size)
        self._frame_cache[path] = gray
        return gray

    def eval_pair(self, path_a: Path, path_b: Path) -> PairQualityMetrics:
        return compute_pair_metrics_from_gray(
            self.load_gray(path_a),
            self.load_gray(path_b),
            frame_a=path_a.name,
            frame_b=path_b.name,
            orb=self._orb,
            matcher=self._matcher,
            thresholds=self.thresholds,
        )


def _resize_gray(path: Path, max_size: int) -> np.ndarray:
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
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 6.5025, 58.5225
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if den == 0:
        return 0.0
    return float(max(0.0, min(1.0, num / den)))


def _visual_scores(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, float]:
    hist_sim = histogram_similarity(gray_a, gray_b)
    similarity = round(0.5 * hist_sim + 0.5 * ssim_simple(gray_a, gray_b), 4)
    pixel_diff = float(cv2.absdiff(gray_a, gray_b).mean()) / 255.0
    scene_cut = round(max(pixel_diff, 1.0 - hist_sim), 4)
    return similarity, scene_cut


def count_feature_matches(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    *,
    orb_features: int = 2000,
    orb: Any | None = None,
    matcher: Any | None = None,
) -> int:
    detector = orb or cv2.ORB_create(nfeatures=int(orb_features))
    kp_a, des_a = detector.detectAndCompute(gray_a, None)
    kp_b, des_b = detector.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return 0
    matcher_impl = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return len(matcher_impl.match(des_a, des_b))


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
    orb: Any | None = None,
    matcher: Any | None = None,
    thresholds: dict[str, Any] | None = None,
) -> PairQualityMetrics:
    similarity, scene_cut = _visual_scores(gray_a, gray_b)
    matches = count_feature_matches(
        gray_a,
        gray_b,
        orb_features=orb_features,
        orb=orb,
        matcher=matcher,
    )
    return classify_pair(
        frame_a=frame_a,
        frame_b=frame_b,
        similarity=similarity,
        feature_matches=matches,
        scene_cut_score=scene_cut,
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
    gray_a = _resize_gray(Path(path_a), max_image_size)
    gray_b = _resize_gray(Path(path_b), max_image_size)
    return compute_pair_metrics_from_gray(
        gray_a,
        gray_b,
        frame_a=Path(path_a).name,
        frame_b=Path(path_b).name,
        orb_features=orb_features,
        thresholds=thresholds,
    )
