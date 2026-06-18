"""Progress bar helpers for ML training."""

from __future__ import annotations

import sys
from typing import Any, Iterator, TypeVar

T = TypeVar("T")


def batch_progress(
    loader: Iterator[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterator[T]:
    if not enabled:
        yield from loader
        return

    from tqdm import tqdm

    bar = tqdm(
        loader,
        total=total,
        desc=desc,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.3,
        miniters=max(1, (total or 1) // 40),
        leave=False,
        unit="batch",
    )
    try:
        yield from bar
    finally:
        bar.close()


def update_postfix(bar: Any, *, loss: float, acc: float, every: int, step: int) -> None:
    if step == 1 or step % every == 0 or step == getattr(bar, "total", step):
        bar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}", refresh=False)
        bar.refresh()
