"""Общие пути и загрузка конфигов."""

from .config import (
    fps_values_from_config,
    load_extraction_config,
    load_sparse_config,
    load_yaml,
)
from .paths import PROJECT_ROOT, frames_dir_for, resolve_path, video_slug_from_path

__all__ = [
    "PROJECT_ROOT",
    "resolve_path",
    "video_slug_from_path",
    "frames_dir_for",
    "load_yaml",
    "load_extraction_config",
    "load_sparse_config",
    "fps_values_from_config",
]
