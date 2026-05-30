import datetime
import uuid
from typing import Union, Any
from pathlib import Path

from imgshape.atlas.orchestrator import DatasetFingerprint
from imgshape.contract.clauses import ContractClauses
from imgshape.contract.violations import Violation, ViolationReport
from imgshape.contract.loader import ContractLoader


def get_nested_val(obj, *keys):
    """Safely navigate nested dictionaries/objects"""
    curr = obj
    for k in keys:
        if curr is None:
            return None
        if isinstance(curr, dict):
            curr = curr.get(k)
        elif hasattr(curr, k):
            curr = getattr(curr, k)
        else:
            try:
                curr = getattr(curr, k)
            except AttributeError:
                return None
    return curr


class Contract:
    def __init__(self, clauses: ContractClauses):
        self.clauses = clauses

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Contract":
        clauses = ContractLoader.load(path)
        return cls(clauses)

    def validate(self, fingerprint: DatasetFingerprint) -> ViolationReport:
        violations = []

        def add_violation(clause: str, expected: Any, actual: Any, detail: str):
            severity = self.clauses.severities.get(clause, "error")
            violations.append(
                Violation(
                    clause=clause,
                    expected=expected,
                    actual=actual,
                    severity=severity,
                    detail=detail,
                )
            )

        profiles = getattr(fingerprint, "profiles", {})
        metadata = getattr(fingerprint, "metadata", {}) or {}

        # 1. dataset.expected_channels
        expected_channels = self.clauses.dataset.expected_channels
        if expected_channels is not None:
            actual_channels = metadata.get("channel_count") or metadata.get(
                "channel_counts"
            )
            if actual_channels is None:
                # Derive from semantic characteristics
                characteristics = (
                    get_nested_val(profiles, "semantic", "characteristics") or []
                )
                if "grayscale" in characteristics:
                    actual_channels = 1
                else:
                    actual_channels = 3
            if actual_channels != expected_channels:
                add_violation(
                    clause="dataset.expected_channels",
                    expected=expected_channels,
                    actual=actual_channels,
                    detail=f"Expected {expected_channels} channels, but found {actual_channels}.",
                )

        # 2. dataset.allowed_formats
        allowed_formats = self.clauses.dataset.allowed_formats
        if allowed_formats is not None:
            actual_formats = metadata.get("formats") or []
            allowed_set = {f.lower().lstrip(".") for f in allowed_formats}
            actual_set = {f.lower().lstrip(".") for f in actual_formats}

            invalid_formats = actual_set - allowed_set
            if invalid_formats:
                add_violation(
                    clause="dataset.allowed_formats",
                    expected=sorted(list(allowed_set)),
                    actual=sorted(list(actual_set)),
                    detail=(
                        f"Dataset contains forbidden format(s): "
                        f"{sorted(list(invalid_formats))}."
                    ),
                )

        # 3. dataset.resolution_min
        resolution_min = self.clauses.dataset.resolution_min
        if resolution_min is not None:
            expected_min_w, expected_min_h = resolution_min
            actual_min_w = get_nested_val(
                profiles, "spatial", "resolution_range", "min", "width"
            )
            actual_min_h = get_nested_val(
                profiles, "spatial", "resolution_range", "min", "height"
            )

            if actual_min_w is None or actual_min_h is None:
                add_violation(
                    clause="dataset.resolution_min",
                    expected=(expected_min_w, expected_min_h),
                    actual=None,
                    detail="Could not retrieve minimum resolution from spatial profile.",
                )
            elif actual_min_w < expected_min_w or actual_min_h < expected_min_h:
                add_violation(
                    clause="dataset.resolution_min",
                    expected=(expected_min_w, expected_min_h),
                    actual=(actual_min_w, actual_min_h),
                    detail=(
                        f"Minimum resolution {actual_min_w}x{actual_min_h} is "
                        f"below the allowed boundary {expected_min_w}x{expected_min_h}."
                    ),
                )

        # 4. dataset.resolution_max
        resolution_max = self.clauses.dataset.resolution_max
        if resolution_max is not None:
            expected_max_w, expected_max_h = resolution_max
            actual_max_w = get_nested_val(
                profiles, "spatial", "resolution_range", "max", "width"
            )
            actual_max_h = get_nested_val(
                profiles, "spatial", "resolution_range", "max", "height"
            )

            if actual_max_w is None or actual_max_h is None:
                add_violation(
                    clause="dataset.resolution_max",
                    expected=(expected_max_w, expected_max_h),
                    actual=None,
                    detail="Could not retrieve maximum resolution from spatial profile.",
                )
            elif actual_max_w > expected_max_w or actual_max_h > expected_max_h:
                add_violation(
                    clause="dataset.resolution_max",
                    expected=(expected_max_w, expected_max_h),
                    actual=(actual_max_w, actual_max_h),
                    detail=(
                        f"Maximum resolution {actual_max_w}x{actual_max_h} exceeds "
                        f"the allowed boundary {expected_max_w}x{expected_max_h}."
                    ),
                )

        # 5. quality.corruption_max
        corruption_max = self.clauses.quality.corruption_max
        if corruption_max is not None:
            actual_corruption = get_nested_val(profiles, "quality", "corruption_rate")
            if actual_corruption is None:
                add_violation(
                    clause="quality.corruption_max",
                    expected=corruption_max,
                    actual=None,
                    detail="Could not retrieve corruption rate from quality profile.",
                )
            elif actual_corruption > corruption_max:
                add_violation(
                    clause="quality.corruption_max",
                    expected=corruption_max,
                    actual=actual_corruption,
                    detail=(
                        f"Corruption rate {actual_corruption} exceeds "
                        f"the maximum allowed {corruption_max}."
                    ),
                )

        # 6. quality.duplicate_max
        duplicate_max = self.clauses.quality.duplicate_max
        if duplicate_max is not None:
            actual_duplicate = get_nested_val(profiles, "quality", "duplication_rate")
            if actual_duplicate is None:
                add_violation(
                    clause="quality.duplicate_max",
                    expected=duplicate_max,
                    actual=None,
                    detail="Could not retrieve duplication rate from quality profile.",
                )
            elif actual_duplicate > duplicate_max:
                add_violation(
                    clause="quality.duplicate_max",
                    expected=duplicate_max,
                    actual=actual_duplicate,
                    detail=(
                        f"Duplication rate {actual_duplicate} exceeds "
                        f"the maximum allowed {duplicate_max}."
                    ),
                )

        # 7. quality.blur_threshold
        blur_threshold = self.clauses.quality.blur_threshold
        if blur_threshold is not None:
            actual_blur = get_nested_val(profiles, "quality", "mean_blur")
            if actual_blur is None:
                add_violation(
                    clause="quality.blur_threshold",
                    expected=blur_threshold,
                    actual=None,
                    detail="Could not retrieve mean blur score from quality profile.",
                )
            elif actual_blur < blur_threshold:
                add_violation(
                    clause="quality.blur_threshold",
                    expected=blur_threshold,
                    actual=actual_blur,
                    detail=(
                        f"Mean blur score {actual_blur} is below the "
                        f"threshold {blur_threshold}."
                    ),
                )

        # 8. distribution.entropy_min
        entropy_min = self.clauses.distribution.entropy_min
        if entropy_min is not None:
            actual_entropy_mean = get_nested_val(profiles, "signal", "entropy", "mean")
            if actual_entropy_mean is None:
                add_violation(
                    clause="distribution.entropy_min",
                    expected=entropy_min,
                    actual=None,
                    detail="Could not retrieve mean entropy from signal profile.",
                )
            elif actual_entropy_mean < entropy_min:
                add_violation(
                    clause="distribution.entropy_min",
                    expected=entropy_min,
                    actual=actual_entropy_mean,
                    detail=(
                        f"Mean entropy {actual_entropy_mean} is below the "
                        f"minimum allowed {entropy_min}."
                    ),
                )

        # 9. distribution.entropy_max
        entropy_max = self.clauses.distribution.entropy_max
        if entropy_max is not None:
            actual_entropy_max = get_nested_val(
                profiles, "signal", "entropy", "range", "max"
            )

            if actual_entropy_max is None:
                add_violation(
                    clause="distribution.entropy_max",
                    expected=entropy_max,
                    actual=None,
                    detail="Could not retrieve max entropy from signal profile.",
                )
            elif actual_entropy_max > entropy_max:
                add_violation(
                    clause="distribution.entropy_max",
                    expected=entropy_max,
                    actual=actual_entropy_max,
                    detail=(
                        f"Max entropy {actual_entropy_max} exceeds the "
                        f"maximum allowed {entropy_max}."
                    ),
                )

        # 10. distribution.channel_consistency
        channel_consistency = self.clauses.distribution.channel_consistency
        if channel_consistency == "strict":
            consistency = metadata.get("grayscale_consistency")
            if consistency is None:
                ratio = metadata.get("grayscale_ratio")
                if ratio is not None:
                    consistency = ratio == 0.0 or ratio == 1.0

            if consistency is None:
                add_violation(
                    clause="distribution.channel_consistency",
                    expected="strict",
                    actual=None,
                    detail="Could not retrieve grayscale consistency metadata from fingerprint.",
                )
            elif not consistency:
                ratio = metadata.get("grayscale_ratio")
                add_violation(
                    clause="distribution.channel_consistency",
                    expected="strict",
                    actual=f"grayscale_ratio={ratio}",
                    detail=(
                        f"Channels are inconsistent across the dataset "
                        f"(grayscale ratio is {ratio}, expected 0.0 or 1.0)."
                    ),
                )

        # Generate report IDs and timestamp
        fingerprint_id = getattr(fingerprint, "dataset_uri", "") or ""
        provenance_id = metadata.get("provenance_id") or f"prov_{uuid.uuid4().hex[:8]}"
        timestamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        passed = len(violations) == 0

        return ViolationReport(
            passed=passed,
            violations=violations,
            fingerprint_id=fingerprint_id,
            provenance_id=provenance_id,
            timestamp=timestamp,
        )
