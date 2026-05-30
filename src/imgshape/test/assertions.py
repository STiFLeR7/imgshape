from typing import Optional, List, Tuple, Any


def _get_val(obj: Any, attr_path: str, default: Any = None) -> Any:
    current = obj
    for part in attr_path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


def assert_corruption(fp: Any, max_rate: float = 0.01) -> Tuple[str, float, float, str]:
    rate = _get_val(fp, "profiles.quality.corruption_rate", 0.0)
    actual = float(rate)
    expected = float(max_rate)
    if actual <= expected:
        return (
            "PASS",
            actual,
            expected,
            f"Corruption rate is {actual:.4f} (expected <= {expected})",
        )
    else:
        return (
            "FAIL",
            actual,
            expected,
            f"Corruption rate is {actual:.4f} (expected <= {expected})",
        )


def assert_duplicates(fp: Any, max_rate: float = 0.05) -> Tuple[str, float, float, str]:
    rate = _get_val(fp, "profiles.quality.duplication_rate", 0.0)
    actual = float(rate)
    expected = float(max_rate)
    if actual <= expected:
        return (
            "PASS",
            actual,
            expected,
            f"Duplication rate is {actual:.4f} (expected <= {expected})",
        )
    else:
        return (
            "FAIL",
            actual,
            expected,
            f"Duplication rate is {actual:.4f} (expected <= {expected})",
        )


def assert_resolution_min(
    fp: Any, min_w: int = 32, min_h: int = 32
) -> Tuple[str, Tuple[int, int], Tuple[int, int], str]:
    min_res = _get_val(fp, "profiles.spatial.resolution_range.min")
    if min_res is None:
        res_range = _get_val(fp, "profiles.spatial.resolution_range")
        if isinstance(res_range, dict):
            min_res = res_range.get("min")

    w = _get_val(min_res, "width", 0)
    h = _get_val(min_res, "height", 0)

    actual = (int(w), int(h))
    expected = (int(min_w), int(min_h))

    if w >= min_w and h >= min_h:
        return (
            "PASS",
            actual,
            expected,
            f"Minimum resolution is {w}x{h} (expected >= {min_w}x{min_h})",
        )
    else:
        return (
            "FAIL",
            actual,
            expected,
            f"Minimum resolution is {w}x{h} (expected >= {min_w}x{min_h})",
        )


def assert_entropy_min(fp: Any, min_val: float = 2.5) -> Tuple[str, float, float, str]:
    entropy_mean = _get_val(fp, "profiles.signal.entropy.mean", 0.0)
    actual = float(entropy_mean)
    expected = float(min_val)
    if actual >= expected:
        return (
            "PASS",
            actual,
            expected,
            f"Mean entropy is {actual:.4f} (expected >= {expected})",
        )
    else:
        return (
            "FAIL",
            actual,
            expected,
            f"Mean entropy is {actual:.4f} (expected >= {expected})",
        )


def assert_channel_consistency(
    fp: Any, strict: bool = True
) -> Tuple[str, float, str, str]:
    consistency = _get_val(fp, "metadata.grayscale_consistency", True)
    ratio = _get_val(fp, "metadata.grayscale_ratio", 0.0)

    actual = float(ratio)
    expected = "consistent (0.0 or 1.0)"

    if consistency:
        return (
            "PASS",
            actual,
            expected,
            f"Channel consistency passed (grayscale ratio: {actual:.4f})",
        )
    else:
        status = "FAIL" if strict else "WARN"
        return (
            status,
            actual,
            expected,
            f"Channel consistency failed with mixed channels (grayscale ratio: {actual:.4f})",
        )


def assert_formats(
    fp: Any, allowed_formats: Optional[List[str]] = None
) -> Tuple[str, List[str], List[str], str]:
    if allowed_formats is None:
        allowed_formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]

    allowed_formats = [fmt.lower() for fmt in allowed_formats]

    formats = _get_val(fp, "metadata.formats", [])
    actual = [fmt.lower() for fmt in formats]
    expected = allowed_formats

    disallowed = [fmt for fmt in actual if fmt not in expected]
    if not disallowed:
        return ("PASS", actual, expected, f"All formats are allowed: {actual}")
    else:
        return ("FAIL", actual, expected, f"Disallowed formats found: {disallowed}")
