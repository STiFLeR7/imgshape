# tests/test_report.py
from pathlib import Path
import json

import pytest

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from imgshape.report import generate_markdown_report


def _maybe_create_sample_image(dirpath: Path, name: str = "sample.png"):
    """
    Create a tiny RGB PNG using Pillow if available; otherwise create a placeholder text file
    (report generator is resilient and will still run).
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    out = dirpath / name
    if PIL_AVAILABLE:
        img = Image.new("RGB", (32, 32), color=(73, 109, 137))
        img.save(out, format="PNG")
    else:
        out.write_text("no-image", encoding="utf-8")
    return out


def test_generate_markdown(tmp_path: Path):
    # Arrange: dataset folder + output path
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _maybe_create_sample_image(dataset_dir)

    out_md = tmp_path / "report.md"
    analysis = {"image_count": 1, "entropy_mean": 4.0, "resolution_mean": [32, 32]}
    pipeline = {"preprocessing": [{"name": "resize", "spec": {"resize": [256, 256]}}], "augmentations": []}

    # Act: generate markdown report
    md_path = generate_markdown_report(str(dataset_dir), str(out_md), analysis=analysis, pipeline=pipeline)

    # Assert: file exists and contains expected markers
    assert out_md.exists(), "Markdown report was not written"
    content = out_md.read_text(encoding="utf-8")
    assert "imgshape report" in content.lower(), "Report header missing"
    # entropy value should appear in JSON analysis block
    assert '"entropy_mean": 4.0' in content or "4.0" in content, "Entropy value not present in report"
