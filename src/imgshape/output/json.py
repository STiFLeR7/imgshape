import json
import dataclasses
from typing import Any, Optional


def _to_serializable(report: Any) -> Any:
    """Recursively convert custom objects/dataclasses to serializable dicts."""
    if hasattr(report, "to_dict") and callable(getattr(report, "to_dict")):
        return report.to_dict()
    if dataclasses.is_dataclass(report):
        return dataclasses.asdict(report)
    if isinstance(report, dict):
        return {k: _to_serializable(v) for k, v in report.items()}
    if isinstance(report, list):
        return [_to_serializable(item) for item in report]
    return report


def render_validation_report(report: Any, save_path: Optional[str] = None) -> str:
    """Serialize a validation report to formatted JSON, optionally saving to a file."""
    data = _to_serializable(report)
    json_str = json.dumps(data, indent=2)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str


def render_test_report(report: Any, save_path: Optional[str] = None) -> str:
    """Serialize a test report to formatted JSON, optionally saving to a file."""
    data = _to_serializable(report)
    json_str = json.dumps(data, indent=2)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str


def render_diff_report(report: Any, save_path: Optional[str] = None) -> str:
    """Serialize a diff report to formatted JSON, optionally saving to a file."""
    data = _to_serializable(report)
    json_str = json.dumps(data, indent=2)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str
