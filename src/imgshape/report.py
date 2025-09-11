# src/imgshape/report.py
from pathlib import Path
from typing import Dict, Any, Optional, Mapping
import json
import shutil

from PIL import Image

# try to import viz plotting helpers; if not available, we'll skip plot embedding
try:
    from imgshape.viz import plot_dataset_distribution
except Exception:
    plot_dataset_distribution = None


def _ensure_path(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_write_text(path: Path, content: str) -> None:
    _ensure_path(path)
    path.write_text(content, encoding="utf-8")


def _copy_and_thumb(src: Path, dst_dir: Path, thumb_size=(240, 240)) -> Path:
    """
    Copy image to dst_dir and create a thumbnail. Return relative dst path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # copy original for provenance
    shutil.copy2(src, dst)
    # create thumbnail alongside with suffix
    try:
        im = Image.open(src)
        im.thumbnail(thumb_size)
        thumb_name = f"{src.stem}_thumb{src.suffix}"
        thumb_path = dst_dir / thumb_name
        im.save(thumb_path)
        return thumb_path
    except Exception:
        # if thumbnailing fails, still return the copied file
        return dst


def generate_markdown_report(
    out_path: Path,
    stats: Dict[str, Any],
    compatibility: Dict[str, Any],
    preprocessing: Dict[str, Any],
    augmentation_plan: Dict[str, Any],
    samples: Optional[Mapping[str, Path]] = None,
    plots_from: Optional[str] = None,
    title: str = "imgshape dataset report",
) -> Path:
    """
    Generate a Markdown report and a companion JSON summary.

    Args:
      out_path: Path to write the markdown file (e.g. ./reports/report.md)
      stats: dataset stats dict (may include 'image_count', 'source_dir', etc.)
      compatibility: model compatibility results
      preprocessing: preprocessing recommendation dict
      augmentation_plan: augmentation plan dict or dataclass-serializable
      samples: optional mapping of label -> Path(image) to embed as thumbnails
      plots_from: optional folder path to generate plots from (overrides stats['source_dir'])
      title: report title

    Returns:
      Path to the generated markdown file.
    """
    out = Path(out_path)
    _ensure_path(out)

    out_dir = out.parent
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Image count: {stats.get('image_count', 'unknown')}")
    if stats.get("source_dir"):
        lines.append(f"- Source directory: `{stats.get('source_dir')}`")
    lines.append("")

    # Insert plots (if possible)
    plot_paths = {}
    plot_source = plots_from or stats.get("source_dir")
    if plot_source and plot_dataset_distribution is not None:
        try:
            generated = plot_dataset_distribution(plot_source, save=True, out_dir=str(assets_dir))
            # generated is dict of {name: path}
            for k, p in generated.items():
                if p:
                    rel = Path(p).relative_to(out_dir)
                    plot_paths[k] = rel.as_posix()
        except Exception:
            # don't fail report generation
            pass

    if plot_paths:
        lines.append("## Visualizations")
        lines.append("")
        for name, relpath in plot_paths.items():
            lines.append(f"### {name.replace('_',' ').title()}")
            lines.append("")
            lines.append(f"![{name}]({relpath})")
            lines.append("")
    else:
        lines.append("## Visualizations")
        lines.append("")
        lines.append("_No visualizations available (missing source_dir or plotting dependencies)._")
        lines.append("")

    # Stats block
    lines.append("## Stats (raw)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats, indent=2, default=str))
    lines.append("```")
    lines.append("")

    # Compatibility
    if compatibility:
        lines.append("## Compatibility")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(compatibility, indent=2, default=str))
        lines.append("```")
        lines.append("")

    # Preprocessing
    lines.append("## Preprocessing Recommendation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(preprocessing or {}, indent=2, default=str))
    lines.append("```")
    lines.append("")

    # Augmentation Plan
    lines.append("## Augmentation Plan")
    lines.append("")
    lines.append("```json")
    # augmentation_plan may be dataclasses or contain objects - ensure serializable
    def _serial(x):
        try:
            return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            return str(x)

    lines.append(json.dumps(_serial(augmentation_plan or {}), indent=2, default=str))
    lines.append("```")
    lines.append("")

    # Samples (before / after thumbnails)
    if samples:
        lines.append("## Sample Images")
        lines.append("")
        sample_dir = assets_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for label, img_path in samples.items():
            try:
                src = Path(img_path)
                thumb = _copy_and_thumb(src, sample_dir)
                rel_thumb = thumb.relative_to(out_dir).as_posix()
                lines.append(f"### {label}")
                lines.append("")
                lines.append(f"![{label}]({rel_thumb})")
                lines.append("")
            except Exception:
                lines.append(f"- Failed to embed sample for `{label}`")
                lines.append("")

    # Footer / metadata
    lines.append("---")
    lines.append("")
    lines.append(f"_Report generated by imgshape_ v2.1.0_")

    content = "\n".join(lines)
    out.write_text(content, encoding="utf-8")

    # write companion JSON summary
    summary = {
        "title": title,
        "md": str(out.resolve()),
        "stats": stats,
        "compatibility": compatibility,
        "preprocessing": preprocessing,
        "augmentation_plan": _serial(augmentation_plan or {}),
        "plots": plot_paths,
        "samples": {k: str((assets_dir / 'samples' / Path(v).name).resolve()) for k, v in (samples or {}).items()},
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return out


def generate_html_report(markdown_path: Path, out_path: Path, css: Optional[str] = None) -> Path:
    """
    Convert Markdown to HTML. If 'markdown' package is available, use it; otherwise
    wrap the markdown in a <pre> tag.
    """
    try:
        import markdown as md
    except Exception:
        md = None

    md_text = Path(markdown_path).read_text(encoding="utf-8")

    if md is not None:
        html_body = md.markdown(md_text, extensions=["tables", "fenced_code"])
        if css:
            head = f"<head><style>{css}</style></head>"
        else:
            head = "<head><meta charset='utf-8'></head>"
        html = f"<!doctype html><html>{head}<body>{html_body}</body></html>"
    else:
        # naive fallback
        safe = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = f"<!doctype html><html><head><meta charset='utf-8'></head><body><pre>{safe}</pre></body></html>"

    out = Path(out_path)
    _ensure_path(out)
    out.write_text(html, encoding="utf-8")
    return out


def generate_pdf_report(html_path: Path, out_path: Path) -> Path:
    """
    Generate PDF from HTML. Requires 'weasyprint' to be installed.
    """
    try:
        from weasyprint import HTML
    except Exception as e:
        raise ImportError("PDF generation requires 'weasyprint'. Install it to enable PDF output.") from e

    HTML(filename=str(html_path)).write_pdf(str(out_path))
    return Path(out_path)
