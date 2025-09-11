# src/imgshape/recommender.py
import os
from typing import Any, Dict, Union, Optional
from PIL import UnidentifiedImageError

from imgshape.shape import get_shape
from imgshape.analyze import get_entropy
from imgshape.augmentations import AugmentationRecommender, AugmentationPlan


def recommend_preprocessing(image_or_stats: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Suggest preprocessing steps for ML models.

    Supports:
      - str: path to a single image (backward compatible with v2.0)
      - dict: dataset_stats dict from analyze_dataset()
    """
    if isinstance(image_or_stats, str):
        image_path = image_or_stats
        try:
            shape = get_shape(image_path)
        except (FileNotFoundError, UnidentifiedImageError):
            return {"error": f"❌ Invalid image: {image_path}"}

        # Ensure shape is valid
        if not shape or len(shape) < 2:
            return {"error": "Shape could not be determined"}

        entropy = round(get_entropy(image_path), 2)
        height, width = shape[0], shape[1]
        channels = shape[2] if len(shape) == 3 else 1

        rec: Dict[str, Any] = {}

        # Suggest resize + candidate model
        if min(height, width) >= 224:
            rec["resize"] = (224, 224)
            rec["suggested_model"] = "MobileNet/ResNet"
        elif min(height, width) >= 96:
            rec["resize"] = (96, 96)
            rec["suggested_model"] = "EfficientNet-B0 (small)"
        elif min(height, width) <= 32:
            rec["resize"] = (32, 32)
            rec["suggested_model"] = "TinyNet/MNIST/CIFAR"
        else:
            rec["resize"] = (128, 128)
            rec["suggested_model"] = "General Use"

        rec["mode"] = "RGB" if channels == 3 else "Grayscale"
        rec["normalize"] = [0.5] * channels
        rec["entropy"] = entropy
        return rec

    # --- dataset stats dict case ---
    stats = image_or_stats
    channels = stats.get("channels", 3)
    rec: Dict[str, Any] = {
        "resize": {"size": [224, 224], "method": "bilinear"},
        "normalize": {
            "mean": [0.485, 0.456, 0.406][:channels],
            "std": [0.229, 0.224, 0.225][:channels],
        },
    }
    if channels == 1:
        rec["to_grayscale"] = True
    return rec


def recommend_dataset(dataset_stats: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Unified API for dataset-level recommendation.

    Returns a dict:
    {
      "preprocessing": {...},
      "augmentation_plan": {...}
    }
    """
    preprocessing = recommend_preprocessing(dataset_stats)
    ar = AugmentationRecommender(seed=seed)
    plan: AugmentationPlan = ar.recommend_for_dataset(dataset_stats)

    return {
        "preprocessing": preprocessing,
        "augmentation_plan": {
            "order": plan.recommended_order,
            "augmentations": [a.__dict__ for a in plan.augmentations],
            "seed": plan.seed,
        },
    }


def check_model_compatibility(folder_path: str, model_name: str) -> Dict[str, Any]:
    """
    Check if all images in the folder are compatible with the given model.

    Returns dict:
    {
      "model": str,
      "total": int,
      "passed": int,
      "failed": int,
      "issues": [(filename, reason), ...]
    }
    """
    total = 0
    passed = 0
    failed_list = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
                total += 1
                path = os.path.join(root, file)
                try:
                    shape = get_shape(path)
                    height, width = shape[0], shape[1]

                    # Basic rule: model_name implies min size
                    if "mobilenet" in model_name.lower():
                        if min(height, width) >= 224:
                            passed += 1
                        else:
                            failed_list.append((file, f"Too small: {shape}"))
                    else:
                        # Default threshold: 96px min
                        if min(height, width) >= 96:
                            passed += 1
                        else:
                            failed_list.append((file, f"Too small: {shape}"))

                except Exception as e:
                    failed_list.append((file, str(e)))

    return {
        "model": model_name,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "issues": failed_list,
    }
