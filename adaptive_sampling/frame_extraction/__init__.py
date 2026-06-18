"""Метод 1: нарезка видео на кадры."""

from .extract import (
    ExtractionResult,
    copy_frames_to_workspace,
    extract_all_fps_for_video,
    extract_all_fps_in_directory,
    extract_frames,
    list_videos,
    sample_timestamps,
)

__all__ = [
    "ExtractionResult",
    "extract_frames",
    "extract_all_fps_for_video",
    "extract_all_fps_in_directory",
    "list_videos",
    "sample_timestamps",
    "copy_frames_to_workspace",
]
