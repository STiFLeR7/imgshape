from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


class CapacityCeiling(Enum):
    """Maximum recommended model capacity"""

    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    HEAVY = "heavy"
    UNLIMITED = "unlimited"


class InformationDensity(Enum):
    """Information density classification"""

    SPARSE = "sparse"
    MODERATE = "moderate"
    DENSE = "dense"
    VERY_DENSE = "very_dense"


@dataclass
class EntropyStats:
    """Entropy statistics"""

    mean: float
    std: float
    range: Dict[str, float]  # min, max


@dataclass
class EntropyStabilityIndex:
    """Measures how stable/consistent entropy is across the dataset"""

    score: float  # 0.0 to 1.0
    is_stable: bool


@dataclass
class ColorHistogram:
    """Compact color distribution signature"""

    bins: List[int]
    channels: List[str]


@dataclass
class SignalProfile:
    """Information density and noise characteristics"""

    entropy: EntropyStats
    noise_estimate: float
    capacity_ceiling: CapacityCeiling
    information_density: InformationDensity
    channel_variance: Dict[str, float] = field(default_factory=dict)
    color_histogram: Optional[ColorHistogram] = None
    entropy_stability: Optional[EntropyStabilityIndex] = None


class SignalProfiler:
    """
    Implements the signal profiling logic.
    """

    @staticmethod
    def profile(images: List[Dict[str, Any]]) -> SignalProfile:
        if not images:
            raise ValueError("No images provided for signal profiling")

        entropies = [img["entropy"] for img in images]

        entropy_stats = EntropyStats(
            mean=float(np.mean(entropies)),
            std=float(np.std(entropies)),
            range={"min": float(min(entropies)), "max": float(max(entropies))},
        )

        # Estimate noise from variance
        variances = [np.mean(img["variance"]) for img in images]
        noise_estimate = min(1.0, float(np.mean(variances)) / 10000.0)

        # Determine information density
        mean_entropy = entropy_stats.mean
        if mean_entropy < 5:
            info_density = InformationDensity.SPARSE
        elif mean_entropy < 7:
            info_density = InformationDensity.MODERATE
        elif mean_entropy < 9:
            info_density = InformationDensity.DENSE
        else:
            info_density = InformationDensity.VERY_DENSE

        # Determine capacity ceiling
        if mean_entropy < 6 and noise_estimate < 0.3:
            capacity = CapacityCeiling.LIGHTWEIGHT
        elif mean_entropy < 8:
            capacity = CapacityCeiling.STANDARD
        elif mean_entropy < 10:
            capacity = CapacityCeiling.HEAVY
        else:
            capacity = CapacityCeiling.UNLIMITED

        # Entropy Stability
        entropy_score = 1.0 - min(1.0, entropy_stats.std / (entropy_stats.mean + 1e-6))
        entropy_stability = EntropyStabilityIndex(
            score=float(entropy_score), is_stable=entropy_score > 0.8
        )

        # Channel Variance
        avg_channel_variances = np.mean([img["variance"] for img in images], axis=0)
        r_val = (
            float(avg_channel_variances[0]) if len(avg_channel_variances) > 0 else 0.0
        )
        g_val = (
            float(avg_channel_variances[1]) if len(avg_channel_variances) > 1 else r_val
        )
        b_val = (
            float(avg_channel_variances[2]) if len(avg_channel_variances) > 2 else r_val
        )

        channel_var_dict = {"R": r_val, "G": g_val, "B": b_val}

        # Color Histogram
        all_histograms = [img["histogram"] for img in images]
        avg_hist = np.mean(all_histograms, axis=0)

        # Downsample to 16 bins per channel (compact signature)
        compact_bins = []
        num_channels = len(avg_hist) // 256
        if num_channels == 0:
            num_channels = 1

        for i in range(min(num_channels, 3)):
            c_hist = avg_hist[i * 256 : (i + 1) * 256]
            resampled = [int(sum(c_hist[j * 16 : (j + 1) * 16])) for j in range(16)]
            compact_bins.extend(resampled)

        # Duplicate for standard signature if fewer than 3 channels
        if num_channels < 3:
            single_bins = compact_bins[:16]
            while len(compact_bins) < 48:
                compact_bins.extend(single_bins)

        color_hist = ColorHistogram(bins=compact_bins, channels=["R", "G", "B"])

        return SignalProfile(
            entropy=entropy_stats,
            noise_estimate=noise_estimate,
            capacity_ceiling=capacity,
            information_density=info_density,
            channel_variance=channel_var_dict,
            color_histogram=color_hist,
            entropy_stability=entropy_stability,
        )
