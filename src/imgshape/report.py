"""
report.py — lightweight reporting utilities for imgshape v2.2.0.
Generates Markdown/HTML/PDF dataset reports from recommendations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

from imgshape.recommender import recommend_dataset


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def generate_markdown_report(dataset_path: str, out_path: str = "report.md") -> str:
    rec = recommend_dataset(dataset_path)
    content = ["# 📊 imgshape Report", ""]
    content.append(f"**Dataset Path:** {dataset_path}")
    content.append(f"**Image Count:** {rec.get('dataset_summary', {}).get('image_count', 'N/A')}")
    content.append("## Representative Preprocessing")
    content.append("```json")
    content.append(json.dumps(rec.get("representative_preprocessing", {}), indent=2))
    content.append("```")
    out = Path(out_path)
    _ensure_dir(out)
    out.write_text("\n".join(content), encoding="utf-8")
    return str(out)


def generate_html_report(dataset_path: str, out_path: str = "report.html") -> str:
    rec = recommend_dataset(dataset_path)
    html = f"""
    <html>
    <head><title>imgshape Report</title></head>
    <body>
    <h1>📊 imgshape Report</h1>
    <p><b>Dataset Path:</b> {dataset_path}</p>
    <p><b>Image Count:</b> {rec.get('dataset_summary', {}).get('image_count', 'N/A')}</p>
    <h2>Representative Preprocessing</h2>
    <pre>{json.dumps(rec.get("representative_preprocessing", {}), indent=2)}</pre>
    </body>
    </html>
    """
    out = Path(out_path)
    _ensure_dir(out)
    out.write_text(html, encoding="utf-8")
    return str(out)


def generate_pdf_report(dataset_path: str, out_path: str = "report.pdf") -> str:
    """
    Minimal PDF generation — requires reportlab.
    If not available, falls back to writing a .txt file.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        # fallback
        txt_fallback = Path(out_path).with_suffix(".txt")
        return generate_markdown_report(dataset_path, str(txt_fallback))

    rec = recommend_dataset(dataset_path)
    c = canvas.Canvas(out_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"imgshape Report — {dataset_path}")
    c.drawString(50, 730, f"Image Count: {rec.get('dataset_summary', {}).get('image_count', 'N/A')}")
    c.drawString(50, 710, "Representative Preprocessing:")
    c.setFont("Courier", 8)
    y = 690
    for line in json.dumps(rec.get("representative_preprocessing", {}), indent=2).splitlines():
        c.drawString(50, y, line)
        y -= 10
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    return str(out_path)
