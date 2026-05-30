from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any


class ClassBalanceType(Enum):
    """Class balance classification"""

    BALANCED = "balanced"
    SLIGHTLY_IMBALANCED = "slightly_imbalanced"
    HIGHLY_IMBALANCED = "highly_imbalanced"
    EXTREME_IMBALANCE = "extreme_imbalance"
    UNKNOWN = "unknown"


class SamplingStrategy(Enum):
    """Recommended sampling strategy"""

    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    OVERSAMPLING = "oversampling"
    UNDERSAMPLING = "undersampling"


class LossRecommendation(Enum):
    """Recommended loss function"""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    WEIGHTED_CE = "weighted_ce"
    BALANCED_CE = "balanced_ce"


@dataclass
class ClassBalance:
    """Class balance information"""

    type: ClassBalanceType
    imbalance_ratio: float
    class_count: int


@dataclass
class DistributionProfile:
    """Dataset structure and imbalance characteristics"""

    class_balance: ClassBalance
    sampling_strategy: SamplingStrategy
    loss_recommendation: LossRecommendation


class DistributionProfiler:
    """
    Implements distribution profiling based on dataset folder structure.
    """

    @staticmethod
    def profile(images: List[Dict[str, Any]]) -> DistributionProfile:
        if not images:
            raise ValueError("No images provided for distribution profiling")

        # Extract class labels from path structure
        import os
        from pathlib import Path
        from collections import Counter

        paths = [img["path"] for img in images if "path" in img]
        classes = []

        if len(paths) > 0:
            try:
                # Find common root
                common_root = Path(
                    os.path.commonpath([str(p.resolve()) for p in paths])
                )
                for p in paths:
                    resolved = p.resolve()
                    if resolved.parent == common_root:
                        classes.append("unlabeled")
                    else:
                        relative = resolved.relative_to(common_root)
                        class_name = relative.parts[0]
                        classes.append(class_name)
            except Exception:
                classes = ["unlabeled"] * len(paths)
        else:
            classes = []

        class_counts = Counter(classes)
        unique_classes = [c for c in class_counts if c != "unlabeled"]

        if unique_classes:
            counts = [class_counts[c] for c in unique_classes]
            class_count = len(unique_classes)
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = float(max_count / max(min_count, 1))

            if imbalance_ratio <= 1.2:
                balance_type = ClassBalanceType.BALANCED
                sampling_strategy = SamplingStrategy.UNIFORM
                loss_recommendation = LossRecommendation.CROSS_ENTROPY
            elif imbalance_ratio <= 2.0:
                balance_type = ClassBalanceType.SLIGHTLY_IMBALANCED
                sampling_strategy = SamplingStrategy.WEIGHTED
                loss_recommendation = LossRecommendation.WEIGHTED_CE
            elif imbalance_ratio <= 5.0:
                balance_type = ClassBalanceType.HIGHLY_IMBALANCED
                sampling_strategy = SamplingStrategy.OVERSAMPLING
                loss_recommendation = LossRecommendation.FOCAL_LOSS
            else:
                balance_type = ClassBalanceType.EXTREME_IMBALANCE
                sampling_strategy = SamplingStrategy.UNDERSAMPLING
                loss_recommendation = LossRecommendation.FOCAL_LOSS
        else:
            class_count = 0
            imbalance_ratio = 1.0
            balance_type = ClassBalanceType.UNKNOWN
            sampling_strategy = SamplingStrategy.UNIFORM
            loss_recommendation = LossRecommendation.CROSS_ENTROPY

        class_balance = ClassBalance(
            type=balance_type, imbalance_ratio=imbalance_ratio, class_count=class_count
        )

        return DistributionProfile(
            class_balance=class_balance,
            sampling_strategy=sampling_strategy,
            loss_recommendation=loss_recommendation,
        )
