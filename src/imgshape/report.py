# src/imgshape/report.py
"""
report.py — improved reporting utilities for imgshape v2.2.0.

Generates Markdown/HTML/PDF dataset reports from recommend_dataset output.
- Dataset summary rendered as a Markdown table.
- Representative preprocessing shown as JSON (single place).
- Augmentation plan extracted from representative_preprocessing (no duplication).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import datetime

from imgshape.recommender import recommend_dataset


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _md_table_from_dict(d: Dict[str, Any]) -> str:
    """Render a small 2-column markdown table from a dict of scalars/lists."""
    lines = ["| Field | Value |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, dict):
            val = "`json`"
        else:
            try:
                val = json.dumps(v)
            except Exception:
                val = str(v)
        lines.append(f"| `{k}` | {val} |")
    return "\n".join(lines)


def _pretty_json(obj: Any, indent: int = 2) -> str:
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception:
        return str(obj)


def generate_markdown_report(dataset_path: str, out_path: str = "report.md") -> str:
    rec = recommend_dataset(dataset_path)
    ds = rec.get("dataset_summary", {}) if isinstance(rec, dict) else {}
    pre = rec.get("representative_preprocessing", {}) if isinstance(rec, dict) else {}
    aug = pre.get("augmentation_plan") or rec.get("augmentation_plan", {})

    lines = []
    lines.append("# 📊 imgshape Report\n")
    lines.append(f"- Generated: {datetime.datetime.utcnow().isoformat()}Z")
    lines.append(f"- Dataset: `{dataset_path}`\n")

    lines.append("## Dataset Summary")
    if ds:
        lines.append(_md_table_from_dict(ds))
    else:
        lines.append("_No dataset summary available._")
    lines.append("")

    lines.append("## Representative Preprocessing")
    if pre:
        lines.append("<details><summary>Show JSON</summary>\n")
        lines.append("```json")
        lines.append(_pretty_json(pre))
        lines.append("```")
        lines.append("</details>")
    else:
        lines.append("_No representative preprocessing available._")
    lines.append("")

    lines.append("## Augmentation Plan")
    if aug:
        lines.append("<details><summary>Show augmentation plan</summary>\n")
        lines.append("```json")
        lines.append(_pretty_json(aug))
        lines.append("```")
        lines.append("</details>")
    else:
        lines.append("_No augmentation plan available._")
    lines.append("")

    out = Path(out_path)
    _ensure_dir(out)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def generate_html_report(dataset_path: str, out_path: str = "report.html") -> str:
    rec = recommend_dataset(dataset_path)
    ds = rec.get("dataset_summary", {}) if isinstance(rec, dict) else {}
    pre = rec.get("representative_preprocessing", {}) if isinstance(rec, dict) else {}
    aug = pre.get("augmentation_plan") or rec.get("augmentation_plan", {})

    html_parts = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html><head><meta charset='utf-8'><title>imgshape Report</title>")
    html_parts.append("<style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:18px}table{border-collapse:collapse}th,td{padding:6px 10px;border:1px solid #ddd}</style>")
    html_parts.append("</head><body>")
    html_parts.append("<h1>📊 imgshape Report</h1>")
    html_parts.append(f"<p><strong>Generated:</strong> {datetime.datetime.utcnow().isoformat()}Z</p>")
    html_parts.append(f"<p><strong>Dataset:</strong> <code>{dataset_path}</code></p>")

    html_parts.append("<h2>Dataset Summary</h2>")
    if ds:
        html_parts.append("<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>")
        for k, v in ds.items():
            val = "<code>json</code>" if isinstance(v, dict) else f"<code>{json.dumps(v)}</code>"
            html_parts.append(f"<tr><td><code>{k}</code></td><td>{val}</td></tr>")
        html_parts.append("</tbody></table>")
    else:
        html_parts.append("<p><em>No dataset summary available.</em></p>")

    html_parts.append("<h2>Representative Preprocessing</h2>")
    if pre:
        html_parts.append("<details><summary>Show JSON</summary>")
        html_parts.append(f"<pre>{_pretty_json(pre)}</pre>")
        html_parts.append("</details>")
    else:
        html_parts.append("<p><em>No representative preprocessing available.</em></p>")

    html_parts.append("<h2>Augmentation Plan</h2>")
    if aug:
        html_parts.append("<details><summary>Show augmentation plan</summary>")
        html_parts.append(f"<pre>{_pretty_json(aug)}</pre>")
        html_parts.append("</details>")
    else:
        html_parts.append("<p><em>No augmentation plan available.</em></p>")

    html_parts.append("</body></html>")

    out = Path(out_path)
    _ensure_dir(out)
    out.write_text("\n".join(html_parts), encoding="utf-8")
    return str(out)


def generate_pdf_report(dataset_path: str, out_path: str = "report.pdf") -> str:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        md_path = Path(out_path).with_suffix(".md")
        return generate_markdown_report(dataset_path, str(md_path))

    rec = recommend_dataset(dataset_path)
    ds = rec.get("dataset_summary", {}) if isinstance(rec, dict) else {}
    pre = rec.get("representative_preprocessing", {}) if isinstance(rec, dict) else {}
    aug = pre.get("augmentation_plan") or rec.get("augmentation_plan", {})

    out = Path(out_path)
    _ensure_dir(out)

    c = canvas.Canvas(str(out), pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(50, y, f"imgshape Report — {dataset_path}")
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.datetime.utcnow().isoformat()}Z")
    y -= 20

    def _write_section(title: str, text_lines: str):
        nonlocal y, c
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, title)
        y -= 14
        c.setFont("Courier", 8)
        for line in text_lines.splitlines():
            if y < 60:
                c.showPage()
                y = 750
            c.drawString(50, y, line[:200])
            y -= 10
        y -= 6

    _write_section("Dataset Summary", _pretty_json(ds))
    _write_section("Representative Preprocessing", _pretty_json(pre))
    _write_section("Augmentation Plan", _pretty_json(aug))

    c.save()
    return str(out)
