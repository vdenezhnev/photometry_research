"""Экспорт отчётов по проверке пар кадров."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from .evaluate import FpsPairQualityResult
from .metrics import PairQualityMetrics

_PAIR_FIELDS = (
    "frame_a",
    "frame_b",
    "status",
    "similarity",
    "feature_matches",
    "scene_cut_score",
    "reasons",
)


def _autosize(ws) -> None:
    for col_idx, cells in enumerate(ws.columns, start=1):
        max_len = max(len("" if c.value is None else str(c.value)) for c in cells)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 48)


def _pair_row(pair: PairQualityMetrics) -> list[Any]:
    return [
        pair.frame_a,
        pair.frame_b,
        pair.status,
        pair.similarity,
        pair.feature_matches,
        pair.scene_cut_score,
        "; ".join(pair.reasons),
    ]


def _problematic_sorted(pairs: list[PairQualityMetrics], *, top_k: int) -> list[PairQualityMetrics]:
    return sorted(
        pairs,
        key=lambda p: (p.scene_cut_score, -p.similarity, -p.feature_matches),
        reverse=True,
    )[:top_k]


def write_pair_metrics_json(result: FpsPairQualityResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_json(result: FpsPairQualityResult, path: Path) -> None:
    path.write_text(json.dumps(result.summary_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def write_problematic_pairs_csv(result: FpsPairQualityResult, path: Path, *, top_k: int = 50) -> None:
    rows = _problematic_sorted(result.problematic_pairs, top_k=top_k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_PAIR_FIELDS))
        w.writeheader()
        for p in rows:
            w.writerow(dict(zip(_PAIR_FIELDS, _pair_row(p))))


def write_report_xlsx(result: FpsPairQualityResult, path: Path, *, top_k: int = 50) -> None:
    wb = Workbook()
    headers = list(_PAIR_FIELDS)

    ws_sum = wb.active
    ws_sum.title = "summary"
    ws_sum.append(["key", "value"])
    for k, v in result.summary_dict().items():
        ws_sum.append([k, v])
    _autosize(ws_sum)

    ws_pairs = wb.create_sheet("all_pairs")
    ws_pairs.append(headers)
    for cell in ws_pairs[1]:
        cell.font = Font(bold=True)
    for pair in result.pairs:
        ws_pairs.append(_pair_row(pair))
    _autosize(ws_pairs)

    ws_bad = wb.create_sheet("problematic")
    ws_bad.append(headers)
    for cell in ws_bad[1]:
        cell.font = Font(bold=True)
    for pair in _problematic_sorted(result.problematic_pairs, top_k=top_k):
        ws_bad.append(_pair_row(pair))
    _autosize(ws_bad)

    wb.save(path)


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def write_comparison_xlsx(*, video_slug: str, rows: list[dict[str, Any]], path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "comparison"
    if rows:
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row[h] for h in headers])
        _autosize(ws)
    ws_info = wb.create_sheet("info")
    ws_info.append(["video_slug", video_slug])
    ws_info.append(["fps_modes", len(rows)])
    _autosize(ws_info)
    wb.save(path)
