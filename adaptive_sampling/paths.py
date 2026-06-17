"""Project paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def video_slug_from_path(video_path: Path) -> str:
    return video_path.stem


def frames_dir_for(video_slug: str, fps: float, pattern: str) -> Path:
    fps_token = str(int(fps)) if fps == int(fps) else str(fps).replace(".", "_")
    rel = pattern.format(video_slug=video_slug, fps=fps_token)
    return resolve_path(rel)
