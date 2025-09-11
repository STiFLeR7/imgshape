from pathlib import Path
from src.imgshape.report import generate_markdown_report
def test_generate_markdown(tmp_path):
    md = tmp_path / "report.md"
    stats = {"entropy_mean":4.0}
    generate_markdown_report(md, stats, {}, {}, {"order":[], "augmentations":[]})
    assert md.exists()
    content = md.read_text()
    assert "imgshape dataset report" in content
