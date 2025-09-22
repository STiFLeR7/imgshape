# src/imgshape/report.py
"""
report.py — defensive reporting utilities for imgshape v2.2.0

This module intentionally avoids importing heavy optional dependencies at module import time.
All optional/large imports (weasyprint/reportlab, recommend_dataset) are imported lazily
inside the functions that need them. This prevents import-time crashes in environments
where optional packages are not installed (for example Streamlit Cloud without extras).

Public functions:
- generate_markdown_report(dataset_path: str, out_path: str = "report.md") -> str
- generate_html_report(dataset_path: str, out_path: str = "report.html") -> str
- generate_pdf_report(dataset_path: str, out_path: str = "report.pdf") -> str

Each function returns the path (string) of the generated file on success, or raises an
informative exception on failure. They try to fall back gracefully where possible.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import datetime
import logging
import traceback

logger = logging.getLogger("imgshape.report")
if not logger.handlers:
    import sys

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _pretty_json(obj: Any, indent: int = 2) -> str:
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception:
        return str(obj)


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


# ---------------------------
# Helper: lazy import recommend_dataset
# ---------------------------
def _get_recommend_dataset():
    """
    Lazily import recommend_dataset from imgshape.recommender.
    Returns (func, None) on success or (None, error) on failure.
    """
    try:
        from imgshape.recommender import recommend_dataset  # local import
        return recommend_dataset, None
    except Exception as exc:
        logger.debug("Lazy import recommend_dataset failed: %s", exc, exc_info=True)
        return None, exc


# ---------------------------
# Public API
# ---------------------------


def generate_markdown_report(dataset_path: str, out_path: str = "report.md") -> str:
    """
    Generate a markdown dataset report. Returns path to the written file.

    This function will try to obtain recommendations via recommend_dataset().
    If recommend_dataset cannot be imported or fails, the report will contain a helpful
    placeholder and minimal metadata.
    """
    recommend_dataset, err = _get_recommend_dataset()
    if recommend_dataset is None:
        logger.warning("recommend_dataset unavailable; generating a minimal markdown report.")
        rec = {"error": "recommend_dataset_unavailable", "detail": str(err)}
    else:
        try:
            rec = recommend_dataset(dataset_path)
        except Exception as e:
            logger.warning("recommend_dataset failed: %s", e, exc_info=True)
            rec = {"error": "recommendation_failed", "detail": str(e)}

    ds = rec.get("dataset_summary", {}) if isinstance(rec, dict) else {}
    pre = rec.get("representative_preprocessing", {}) if isinstance(rec, dict) else {}
    aug = pre.get("augmentation_plan") or rec.get("augmentation_plan", {})

    lines = []
    lines.append("# 📊 imgshape Report")
    lines.append(f"- Generated: {datetime.datetime.utcnow().isoformat()}Z")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append("")
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
    logger.info("Markdown report written to %s", out)
    return str(out)


def generate_html_report(dataset_path: str, out_path: str = "report.html") -> str:
    """
    Generate an HTML report (fallbacks to markdown if recommend_dataset isn't available).
    Returns path to the written file.
    """
    recommend_dataset, err = _get_recommend_dataset()
    if recommend_dataset is None:
        logger.warning("recommend_dataset unavailable; generating an HTML report with placeholders.")
        rec = {"error": "recommend_dataset_unavailable", "detail": str(err)}
    else:
        try:
            rec = recommend_dataset(dataset_path)
        except Exception as e:
            logger.warning("recommend_dataset failed: %s", e, exc_info=True)
            rec = {"error": "recommendation_failed", "detail": str(e)}

    ds = rec.get("dataset_summary", {}) if isinstance(rec, dict) else {}
    pre = rec.get("representative_preprocessing", {}) if isinstance(rec, dict) else {}
    aug = pre.get("augmentation_plan") or rec.get("augmentation_plan", {})

    html_parts = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html><head><meta charset='utf-8'><title>imgshape Report</title>")
    html_parts.append(
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:18px}table{border-collapse:collapse}th,td{padding:6px 10px;border:1px solid #ddd}</style>"
    )
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
    logger.info("HTML report written to %s", out)
    return str(out)


def generate_pdf_report(dataset_path: str, out_path: str = "report.pdf") -> str:
    """
    Generate a PDF report. Tries to use reportlab first; if not available, uses markdown fallback.
    This import is lazy and happens only when PDF generation is requested.
    """
    # Try reportlab first (lightweight PDF)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        # Try weasyprint (can convert HTML -> PDF) as a second option
        try:
            # Use our HTML generator then render to PDF via weasyprint
            from weasyprint import HTML  # type: ignore
        except Exception:
            # No PDF backends available — fallback to markdown report
            md_path = Path(out_path).with_suffix(".md")
            logger.warning("No PDF backend available; generating markdown fallback at %s", md_path)
            return generate_markdown_report(dataset_path, str(md_path))

        # weasyprint path:
        html_path = Path(out_path).with_suffix(".html")
        generate_html_report(dataset_path, str(html_path))
        try:
            HTML(filename=str(html_path)).write_pdf(str(out_path))
            logger.info("PDF generated via weasyprint at %s", out_path)
            return str(out_path)
        except Exception as e:
            logger.exception("weasyprint failed to write PDF: %s", e)
            raise

    # If we reach here we have reportlab
    rec_func, err = _get_recommend_dataset()
    if rec_func is None:
        rec = {"error": "recommend_dataset_unavailable", "detail": str(err)}
    else:
        try:
            rec = rec_func(dataset_path)
        except Exception as e:
            logger.warning("recommend_dataset failed: %s", e, exc_info=True)
            rec = {"error": "recommendation_failed", "detail": str(e)}

    ds = rec.get("dataset_summary", {}) if isinstance(rec, dict) else {}
    pre = rec.get("representative_preprocessing", {}) if isinstance(rec, dict) else {}
    aug = pre.get("augmentation_plan") or rec.get("augmentation_plan", {})

    out_p = Path(out_path)
    _ensure_dir(out_p)

    c = canvas.Canvas(str(out_p), pagesize=letter)
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
    logger.info("PDF report written to %s", out_p)
    return str(out_p)
