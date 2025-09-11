# src/imgshape/augmentations.py
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Augmentation:
    name: str
    params: Dict[str, Any]
    reason: str
    score: float


@dataclass
class AugmentationPlan:
    augmentations: List[Augmentation]
    recommended_order: List[str]
    seed: Optional[int]


class AugmentationRecommender:
    """
    Minimal deterministic recommender implementing a few simple heuristics.

    Heuristics implemented:
      - If entropy_mean < 3.5 -> recommend ColorJitter
      - Always include RandomHorizontalFlip (common, safe)
      - If severe imbalance (max/min > 5) -> recommend ClassWiseOversample placeholder
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def recommend_for_dataset(self, dataset_stats: Dict[str, Any]) -> AugmentationPlan:
        ds = dataset_stats or {}
        entropy = ds.get("entropy_mean", None)
        colorfulness = ds.get("colorfulness_mean", None)
        shape_dist = ds.get("shape_distribution", {})
        class_balance = ds.get("class_balance", {})

        aug_list: List[Augmentation] = []

        # Simple geometric augmentation: safe default
        aug_list.append(
            Augmentation(
                name="RandomHorizontalFlip",
                params={"p": 0.5},
                reason="Common orientation variance; usually safe for many datasets",
                score=0.7,
            )
        )

        # Low entropy => color augmentation to increase variance
        if entropy is not None and entropy < 3.5:
            aug_list.append(
                Augmentation(
                    name="ColorJitter",
                    params={"brightness": [0.1, 0.3], "contrast": [0.1, 0.3], "saturation": [0.1, 0.3], "p": 0.6},
                    reason="Low entropy images -> increase color/contrast variation",
                    score=0.9,
                )
            )

        # If colorfulness high, down-weight color jitter (we just adjust score)
        if colorfulness is not None and colorfulness > 40:
            for a in aug_list:
                if a.name == "ColorJitter":
                    a.score = max(0.2, a.score - 0.3)
                    a.reason += " (colorfulness high; reduce intensity)"

        # Class imbalance heuristic (simple)
        if isinstance(class_balance, dict) and class_balance:
            counts = list(class_balance.values())
            try:
                mx, mn = max(counts), min(counts)
                if mn > 0 and (mx / mn) > 5:
                    aug_list.append(
                        Augmentation(
                            name="ClassWiseOversample",
                            params={"method": "augment-minority", "target_ratio": "balanced"},
                            reason="Strong class imbalance -> oversample or augment minority classes",
                            score=0.95,
                        )
                    )
            except Exception:
                pass

        # Build recommended order (geometric first, then color, then class-wise)
        order = [a.name for a in aug_list]

        return AugmentationPlan(augmentations=aug_list, recommended_order=order, seed=self.seed)
