from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any


class QualityLevel(Enum):
    """Overall quality assessment"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class WarningSeverity(Enum):
    """Warning severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class WarningCategory(Enum):
    """Warning categories"""

    CORRUPTION = "corruption"
    DUPLICATION = "duplication"
    RESOLUTION = "resolution"
    FORMAT = "format"
    MISSING_DATA = "missing_data"


@dataclass
class QualityWarning:
    """Quality warning"""

    severity: WarningSeverity
    category: WarningCategory
    message: str
    affected_count: int = 0


@dataclass
class QualityProfile:
    """Trustworthiness and data quality assessment"""

    overall_quality: QualityLevel
    warnings: List[QualityWarning]
    corruption_rate: float
    duplication_rate: float
    mean_blur: float


class QualityProfiler:
    """
    Implements quality profiling logic.
    """

    @staticmethod
    def profile(images: List[Dict[str, Any]], failed_count: int = 0) -> QualityProfile:
        if not images and failed_count == 0:
            raise ValueError("No images provided for quality profiling")

        warnings = []

        # Check for resolution issues
        sizes = [img["size"] for img in images]
        if sizes:
            min_dimension = min([min(s) for s in sizes])
            if min_dimension < 32:
                warnings.append(
                    QualityWarning(
                        severity=WarningSeverity.WARNING,
                        category=WarningCategory.RESOLUTION,
                        message="Very small images detected (< 32px)",
                        affected_count=sum(1 for s in sizes if min(s) < 32),
                    )
                )

        # Corruption rate
        total_attempted = len(images) + failed_count
        corruption_rate = (
            (failed_count / total_attempted) if total_attempted > 0 else 0.0
        )

        if failed_count > 0:
            warnings.append(
                QualityWarning(
                    severity=WarningSeverity.WARNING,
                    category=WarningCategory.CORRUPTION,
                    message=f"Corrupted images detected: {failed_count} files failed to load",
                    affected_count=failed_count,
                )
            )

        # Duplication rate
        hashes = [img.get("hash") for img in images if img.get("hash")]
        if hashes:
            unique_hashes = set(hashes)
            duplicate_count = len(hashes) - len(unique_hashes)
            duplication_rate = duplicate_count / len(images)

            if duplicate_count > 0:
                warnings.append(
                    QualityWarning(
                        severity=WarningSeverity.WARNING,
                        category=WarningCategory.DUPLICATION,
                        message=f"Duplicate images detected: {duplication_rate:.1%}",
                        affected_count=duplicate_count,
                    )
                )
        else:
            duplication_rate = 0.0

        # Mean blur
        mean_blur = (
            (sum(img.get("blur_score", 0.0) for img in images) / len(images))
            if images
            else 0.0
        )

        # Overall quality assessment
        if len(warnings) == 0:
            overall_quality = QualityLevel.EXCELLENT
        elif len(warnings) <= 2:
            overall_quality = QualityLevel.GOOD
        else:
            overall_quality = QualityLevel.ACCEPTABLE

        return QualityProfile(
            overall_quality=overall_quality,
            warnings=warnings,
            corruption_rate=corruption_rate,
            duplication_rate=duplication_rate,
            mean_blur=mean_blur,
        )
