"""
report.py — reporting utilities for imgshape v2.2.0.

This module provides a legacy-compatible generate_markdown_report function used by tests.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json
import datetime


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _pretty_json(obj: Any, indent: int = 2) -> str:
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception:
        return str(obj)


def generate_markdown_report(out_path: str, dataset_summary: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None, preprocessing: Optional[Dict[str, Any]] = None, augmentations: Optional[Dict[str, Any]] = None) -> str:
    """
    Legacy-compatible API used by tests.

    Tests call this as:
        generate_markdown_report(md, stats, {}, {}, {"order":[], "augmentations":[]})

    To remain compatible we accept positional args:
      - out_path (path to write)
      - dataset_summary (or stats)
      - preprocessing
      - augmentations

    Writes a simple markdown file and returns the path (str).
    """
    # Allow tests to pass stats in the second positional arg
    dataset_summary = dataset_summary or stats or {}
    preprocessing = preprocessing or {}
    augmentations = augmentations or {}

    lines = []
    # Match test expectation exactly
    lines.append("# imgshape dataset report")
    lines.append(f"- Generated: {datetime.datetime.utcnow().isoformat()}Z")
    lines.append("")

    lines.append("## Dataset Summary")
    if dataset_summary:
        for k, v in dataset_summary.items():
            lines.append(f"- **{k}**: `{v}`")
    else:
        lines.append("_No dataset summary available._")

    lines.append("")
    lines.append("## Preprocessing")
    lines.append("```json")
    lines.append(_pretty_json(preprocessing))
    lines.append("```")

    lines.append("")
    lines.append("## Augmentations")
    lines.append("```json")
    lines.append(_pretty_json(augmentations))
    lines.append("```")

    out = Path(out_path)
    _ensure_dir(out)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
