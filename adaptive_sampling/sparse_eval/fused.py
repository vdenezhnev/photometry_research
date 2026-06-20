"""CLI: dense fused reconstruction for all FPS of a video."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run import run_fused_batch_for_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused (dense) COLMAP reconstruction for all FPS modes")
    parser.add_argument(
        "--video",
        required=True,
        help="Video slug under data/frames/ (e.g. video_2026-04-16_11-31-49)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to sparse_eval.yaml")
    parser.add_argument("--frames-root", type=Path, default=None, help="Override data/frames root")
    parser.add_argument("--results-root", type=Path, default=None, help="Override results root")
    args = parser.parse_args()

    run_fused_batch_for_video(
        args.video,
        frames_root=args.frames_root,
        results_root=args.results_root,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
