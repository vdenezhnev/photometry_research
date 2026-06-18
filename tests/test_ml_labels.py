"""Tests for ML label parsing."""

import pytest

from adaptive_sampling.ml.labels import normalize_label, suggest_label_from_sparse


def test_normalize_label_good_bad() -> None:
    assert normalize_label("good") == 1
    assert normalize_label("bad") == 0
    assert normalize_label("хорошая") == 1
    assert normalize_label("плохая") == 0


def test_normalize_label_empty() -> None:
    assert normalize_label("") is None
    assert normalize_label(None) is None


def test_suggest_from_sparse() -> None:
    assert suggest_label_from_sparse(
        {"registered_ratio": 0.9, "passes_criteria": True},
        good_min_ratio=0.6,
        require_passes=True,
    ) == "good"
    assert suggest_label_from_sparse(
        {"registered_ratio": 0.3, "passes_criteria": False},
        good_min_ratio=0.6,
        require_passes=True,
    ) == "bad"


def test_normalize_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_label("maybe")
