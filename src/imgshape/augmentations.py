# src/imgshape/augmentations.py
"""
augmentations.py — lightweight augmentation heuristics for imgshape v2.2.0

Provides:
- Augmentation dataclass
- AugmentationPlan dataclass
- AugmentationRecommender with deterministic, explainable heuristics

Heuristics include:
- Always: RandomHorizontalFlip
- Entropy-based: ColorJitter, Sharpness adjustments
- Resolution-based: RandomResizedCrop, Rotation
- Class imbalance: ClassWiseOversample
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger("imgshape.augmentations")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "augmentations": [asdict(a) for a in self.augmentations],
            "recommended_order": self.recommended_order,
            "seed": self.seed,
        }


class AugmentationRecommender:
    """
    Deterministic augmentation recommender with simple interpretable heuristics.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def recommend_for_dataset(self, dataset_stats: Dict[str, Any]) -> AugmentationPlan:
        ds = dataset_stats or {}
        entropy = ds.get("entropy_mean", None)
        avg_w, avg_h = ds.get("avg_width"), ds.get("avg_height")
        image_count = ds.get("image_count", 0)
        class_balance = ds.get("class_balance", {})

        aug_list: List[Augmentation] = []

        # Always include safe geometric flip
        aug_list.append(
            Augmentation(
                name="RandomHorizontalFlip",
                params={"p": 0.5},
                reason="Common orientation variance; usually safe for most datasets",
                score=0.7,
            )
        )

        # Entropy-driven augmentations
        if entropy is not None:
            if entropy < 3.0:
                aug_list.append(
                    Augmentation(
                        name="ColorJitter",
                        params={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
                        reason="Low entropy -> add color and contrast variance",
                        score=0.85,
                    )
                )
                aug_list.append(
                    Augmentation(
                        name="RandomAdjustSharpness",
                        params={"sharpness_factor": 1.2},
                        reason="Low entropy -> sharpen details",
                        score=0.6,
                    )
                )
            elif entropy > 7.0:
                aug_list.append(
                    Augmentation(
                        name="GaussianBlur",
                        params={"kernel_size": 3, "sigma": (0.1, 2.0)},
                        reason="High entropy -> blur noise",
                        score=0.55,
                    )
                )

        # Resolution-driven augmentations
        try:
            min_side = min(int(avg_w), int(avg_h)) if avg_w and avg_h else None
        except Exception:
            min_side = None

        if min_side:
            if min_side >= 224:
                aug_list.append(
                    Augmentation(
                        name="RandomResizedCrop",
                        params={"size": 224, "scale": [0.8, 1.0]},
                        reason="Large images -> random resized crops improve robustness",
                        score=0.65,
                    )
                )
            elif min_side < 64:
                aug_list.append(
                    Augmentation(
                        name="Resize",
                        params={"size": 128, "interpolation": "bilinear"},
                        reason="Very small images -> upsample for stable training",
                        score=0.5,
                    )
                )

        # Rotation for larger datasets
        if image_count and image_count > 10:
            aug_list.append(
                Augmentation(
                    name="RandomRotation",
                    params={"degrees": 15},
                    reason="Moderate dataset size -> small rotations increase robustness",
                    score=0.6,
                )
            )

        # Class imbalance heuristic
        if isinstance(class_balance, dict) and class_balance:
            try:
                counts = list(class_balance.values())
                mx, mn = max(counts), min(counts)
                if mn > 0 and (mx / mn) > 5:
                    aug_list.append(
                        Augmentation(
                            name="ClassWiseOversample",
                            params={"method": "augment-minority", "target_ratio": "balanced"},
                            reason="Severe class imbalance -> oversample/augment minority classes",
                            score=0.95,
                        )
                    )
            except Exception:
                logger.debug("Class balance heuristic failed", exc_info=True)

        # Order: geometric -> color/entropy -> resolution -> class balance
        order = [a.name for a in aug_list]

        plan = AugmentationPlan(augmentations=aug_list, recommended_order=order, seed=self.seed)
        logger.info("Augmentation plan generated with %d augmentations", len(aug_list))
        return plan
