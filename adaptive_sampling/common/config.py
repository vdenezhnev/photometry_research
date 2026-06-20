"""Load YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT


def load_yaml(path: Path | None, default_relative: str) -> dict[str, Any]:
    p = path or (PROJECT_ROOT / default_relative)
    data = yaml.safe_load(p.read_bytes())
    return data if isinstance(data, dict) else {}


def load_extraction_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path, "configs/extraction_fps.yaml")


def load_sparse_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path, "configs/sparse_eval.yaml")


def load_fused_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path, "configs/fused_eval.yaml")


def load_pair_quality_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path, "configs/frame_pair_quality.yaml")


def fps_values_from_config(config: dict[str, Any]) -> list[float]:
    values = config.get("fps_values")
    if not isinstance(values, list) or not values:
        raise ValueError("Config must define non-empty fps_values list")
    return [float(v) for v in values]
