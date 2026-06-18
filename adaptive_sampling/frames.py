"""Video frame extraction (OpenCV)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from .config import fps_values_from_config, load_extraction_config
from .paths import frames_dir_for, resolve_path, video_slug_from_path


@dataclass
class ExtractionResult:
    video_path: str
    video_slug: str
    output_dir: str
    extraction_fps: float
    frame_count: int
    duration_sec: float | None
    source_avg_fps: float | None
    extracted_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def probe_video(video_path: Path) -> tuple[float | None, float | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        avg_fps = fps if fps > 0 else None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / avg_fps if avg_fps and frame_count > 0 else None
        return duration, avg_fps
    finally:
        cap.release()


def sample_timestamps(
    duration_sec: float,
    extraction_fps: float,
    *,
    max_duration_sec: float | None = None,
) -> list[float]:
    if duration_sec <= 0 or extraction_fps <= 0:
        return []
    limit = min(duration_sec, max_duration_sec) if max_duration_sec and max_duration_sec > 0 else duration_sec
    interval = 1.0 / extraction_fps
    times: list[float] = []
    t = 0.0
    while t < limit - 1e-9:
        times.append(t)
        t += interval
    return times


def list_videos(videos_dir: Path) -> list[Path]:
    videos_dir = resolve_path(videos_dir)
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    )
    if not videos:
        raise ValueError(f"No videos in {videos_dir}")
    return videos


def extract_frames(
    video_path: Path,
    *,
    extraction_fps: float,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    max_duration_sec: float | None = None,
    overwrite: bool = True,
) -> ExtractionResult:
    if extraction_fps <= 0:
        raise ValueError("extraction_fps must be > 0")

    config = load_extraction_config(config_path)
    video_path = resolve_path(video_path)
    slug = video_slug_from_path(video_path)

    if output_dir is None:
        pattern = str(config.get("default_output_pattern") or "data/frames/{video_slug}/fps_{fps}")
        output_dir = frames_dir_for(slug, extraction_fps, pattern)
    else:
        output_dir = resolve_path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if max_duration_sec is None and config.get("max_duration_sec") is not None:
        max_duration_sec = float(config["max_duration_sec"])

    duration, source_fps = probe_video(video_path)
    if duration is None or duration <= 0:
        raise RuntimeError(f"Cannot determine duration: {video_path}")

    quality = int(config.get("jpeg_quality") or 95)
    ext = str(config.get("image_format") or "jpg").lstrip(".").lower()
    timestamps = sample_timestamps(duration, extraction_fps, max_duration_sec=max_duration_sec)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    written = 0
    try:
        for index, t_sec in enumerate(timestamps, start=1):
            out_path = output_dir / f"frame_{index:06d}.{ext}"
            if not overwrite and out_path.is_file():
                written += 1
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {"jpg", "jpeg"} else []
            if not cv2.imwrite(str(out_path), frame, params):
                raise RuntimeError(f"Failed to write {out_path}")
            written += 1
    finally:
        cap.release()

    result = ExtractionResult(
        video_path=str(video_path),
        video_slug=slug,
        output_dir=str(output_dir),
        extraction_fps=float(extraction_fps),
        frame_count=written,
        duration_sec=duration,
        source_avg_fps=source_fps,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "extraction_manifest.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def extract_all_fps_for_video(
    video_path: Path,
    *,
    config_path: Path | None = None,
    fps_values: list[float] | None = None,
    max_duration_sec: float | None = None,
) -> list[ExtractionResult]:
    config = load_extraction_config(config_path)
    values = fps_values or fps_values_from_config(config)
    return [
        extract_frames(video_path, extraction_fps=fps, config_path=config_path, max_duration_sec=max_duration_sec)
        for fps in values
    ]


def extract_all_fps_in_directory(
    videos_dir: Path,
    *,
    config_path: Path | None = None,
    fps_values: list[float] | None = None,
    max_duration_sec: float | None = None,
) -> list[ExtractionResult]:
    config = load_extraction_config(config_path)
    values = fps_values or fps_values_from_config(config)
    results: list[ExtractionResult] = []
    for video in list_videos(videos_dir):
        results.extend(
            extract_all_fps_for_video(
                video,
                config_path=config_path,
                fps_values=values,
                max_duration_sec=max_duration_sec,
            )
        )
    return results


def _subsample_files(files: list[Path], max_images: int) -> list[Path]:
    if max_images <= 0 or len(files) <= max_images:
        return files
    if max_images == 1:
        return [files[0]]
    last = len(files) - 1
    step = last / (max_images - 1)
    return [files[round(i * step)] for i in range(max_images)]


def copy_frames_to_workspace(
    frames_dir: Path,
    images_dir: Path,
    *,
    max_images: int | None = None,
) -> tuple[int, int]:
    """Copy frames into PyCOLMAP workspace. Returns (used_count, source_count)."""
    frames_dir = resolve_path(frames_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    files = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No images in {frames_dir}")

    source_count = len(files)
    if max_images is not None and source_count > max_images:
        files = _subsample_files(files, max_images)

    for src in files:
        shutil.copy2(src, images_dir / src.name)
    return len(files), source_count
