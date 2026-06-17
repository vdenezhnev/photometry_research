"""Tests for frame extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from adaptive_sampling.frames import extract_frames, sample_timestamps
from adaptive_sampling.paths import frames_dir_for, video_slug_from_path


def test_video_slug() -> None:
    assert video_slug_from_path(Path("room_01.mp4")) == "room_01"


def test_frames_dir_for() -> None:
    p = frames_dir_for("room_01", 2.0, "data/frames/{video_slug}/fps_{fps}")
    assert p.name == "fps_2"


def test_sample_timestamps() -> None:
    assert sample_timestamps(10.0, 2.0) == [i * 0.5 for i in range(20)]


@patch("adaptive_sampling.frames.cv2")
def test_extract_frames(mock_cv2: MagicMock, tmp_path: Path) -> None:
    cap = MagicMock()
    cap.isOpened.return_value = True
    reads = {"n": 0}

    def _read():
        if reads["n"] < 2:
            reads["n"] += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    cap.read.side_effect = _read
    cap.get.side_effect = lambda prop: {5: 30.0, 7: 60}.get(prop, 0)  # FPS, FRAME_COUNT
    mock_cv2.VideoCapture.return_value = cap
    mock_cv2.imwrite.return_value = True
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_POS_MSEC = 0
    mock_cv2.IMWRITE_JPEG_QUALITY = 1

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()

    with patch(
        "adaptive_sampling.frames.load_extraction_config",
        return_value={"jpeg_quality": 95, "image_format": "jpg"},
    ):
        with patch("adaptive_sampling.frames.probe_video", return_value=(2.0, 30.0)):
            result = extract_frames(video, extraction_fps=2.0, output_dir=out)

    assert result.frame_count == 2
    assert (out / "extraction_manifest.json").is_file()
