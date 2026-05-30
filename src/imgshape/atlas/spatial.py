from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


class ResizeRisk(Enum):
    """Risk level for resizing operations"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GeometryClass(Enum):
    """Classification of geometric diversity"""

    UNIFORM = "uniform"
    DIVERSE = "diverse"
    EXTREME = "extreme"


@dataclass
class Resolution:
    """Resolution specification"""

    width: int
    height: int

    def area(self) -> int:
        return self.width * self.height

    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


@dataclass
class AspectRatioCluster:
    """Clustered aspect ratios found in the dataset"""

    ratio: float
    frequency: float


@dataclass
class ResolutionRange:
    """Range of resolutions found in the dataset with dict-like compatibility"""

    min: Resolution
    max: Resolution
    median: Resolution

    def __getitem__(self, key: str) -> Resolution:
        if key == "min":
            return self.min
        elif key == "max":
            return self.max
        elif key == "median":
            return self.median
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return ["min", "max", "median"]

    def values(self) -> List[Resolution]:
        return [self.min, self.max, self.median]

    def items(self) -> List[tuple]:
        return [("min", self.min), ("max", self.max), ("median", self.median)]

    def __contains__(self, key: str) -> bool:
        return key in ("min", "max", "median")


@dataclass
class SpatialProfile:
    """Geometric constraints and resizing risk assessment"""

    resolution_range: ResolutionRange
    aspect_ratio_variance: float
    resize_risk: ResizeRisk
    geometry_class: GeometryClass
    aspect_ratio_clusters: List[AspectRatioCluster] = field(default_factory=list)


class SpatialProfiler:
    """
    Implements the spatial profiling logic.
    """

    @staticmethod
    def profile(images: List[Dict[str, Any]]) -> SpatialProfile:
        if not images:
            raise ValueError("No images provided for spatial profiling")

        sizes = [img["size"] for img in images]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        aspect_ratios = [w / h for w, h in sizes]

        # Resolution range
        min_res = Resolution(min(widths), min(heights))
        max_res = Resolution(max(widths), max(heights))
        median_res = Resolution(int(np.median(widths)), int(np.median(heights)))

        # Aspect ratio variance
        ar_variance = float(np.var(aspect_ratios))

        # Determine geometry class
        if ar_variance < 0.01:
            geometry_class = GeometryClass.UNIFORM
        elif ar_variance < 0.1:
            geometry_class = GeometryClass.DIVERSE
        else:
            geometry_class = GeometryClass.EXTREME

        # Determine resize risk
        area_variance = np.var([w * h for w, h in sizes])
        if area_variance < 10000 and ar_variance < 0.01:
            resize_risk = ResizeRisk.LOW
        elif area_variance < 100000 and ar_variance < 0.1:
            resize_risk = ResizeRisk.MEDIUM
        else:
            resize_risk = ResizeRisk.HIGH

        # Aspect Ratio Clustering
        ars = [round(w / h, 2) for w, h in sizes]
        unique_ars, ar_counts = np.unique(ars, return_counts=True)
        ar_clusters = [
            AspectRatioCluster(ratio=float(r), frequency=float(f / len(images)))
            for r, f in zip(unique_ars, ar_counts)
        ]
        # Sort by frequency
        ar_clusters = sorted(ar_clusters, key=lambda x: x.frequency, reverse=True)[:5]

        return SpatialProfile(
            resolution_range=ResolutionRange(
                min=min_res, max=max_res, median=median_res
            ),
            aspect_ratio_variance=ar_variance,
            resize_risk=resize_risk,
            geometry_class=geometry_class,
            aspect_ratio_clusters=ar_clusters,
        )
