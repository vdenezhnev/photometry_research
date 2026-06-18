"""Простой вывод прогресса без внешних зависимостей."""

from __future__ import annotations

import sys
from typing import TextIO


def log_stage(message: str, *, stream: TextIO | None = None) -> None:
    (stream or sys.stdout).write(f"{message}\n")
    (stream or sys.stdout).flush()


def log_pair_progress(
    label: str,
    current: int,
    total: int,
    *,
    stream: TextIO | None = None,
) -> None:
    out = stream or sys.stderr
    if total <= 0:
        return
    pct = 100.0 * current / total
    out.write(f"\r{label}: pairs {current}/{total} ({pct:.0f}%)")
    out.flush()
    if current >= total:
        out.write("\n")
        out.flush()
