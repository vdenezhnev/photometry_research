"""Progress bar helpers for ML training."""

from __future__ import annotations

import sys
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def batch_progress(
    loader: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterable[T]:
    if not enabled:
        return loader

    from tqdm import tqdm

    return tqdm(
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


def update_postfix(bar: Any, *, loss: float, acc: float, every: int, step: int) -> None:
    if not hasattr(bar, "set_postfix"):
        return
    if step == 1 or step % every == 0 or step == getattr(bar, "total", step):
        bar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}", refresh=False)
        bar.refresh()
