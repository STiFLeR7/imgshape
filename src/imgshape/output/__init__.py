from typing import Any, Optional
from imgshape.output import text, json as json_renderer, markdown


def render_validation_report(
    report: Any, output_format: str = "text", save_path: Optional[str] = None, **kwargs
) -> str:
    """Render a validation report in the specified format ('text', 'json', 'markdown')."""
    fmt = kwargs.get("format", output_format)
    if fmt == "text":
        out = text.render_validation_report(report)
    elif fmt == "json":
        return json_renderer.render_validation_report(report, save_path=save_path)
    elif fmt == "markdown":
        out = markdown.render_validation_report(report)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(out)
    return out


def render_test_report(
    report: Any, output_format: str = "text", save_path: Optional[str] = None, **kwargs
) -> str:
    """Render a test report in the specified format ('text', 'json', 'markdown')."""
    fmt = kwargs.get("format", output_format)
    if fmt == "text":
        out = text.render_test_report(report)
    elif fmt == "json":
        return json_renderer.render_test_report(report, save_path=save_path)
    elif fmt == "markdown":
        out = markdown.render_test_report(report)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(out)
    return out


def render_diff_report(
    report: Any, output_format: str = "text", save_path: Optional[str] = None, **kwargs
) -> str:
    """Render a diff report in the specified format ('text', 'json', 'markdown')."""
    fmt = kwargs.get("format", output_format)
    if fmt == "text":
        out = text.render_diff_report(report)
    elif fmt == "json":
        return json_renderer.render_diff_report(report, save_path=save_path)
    elif fmt == "markdown":
        out = markdown.render_diff_report(report)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(out)
    return out


__all__ = [
    "render_validation_report",
    "render_test_report",
    "render_diff_report",
]
