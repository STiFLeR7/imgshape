# compatibility.py
import os
from imgshape.shape import get_shape

# Model compatibility rules (can expand later)
MODEL_REQUIREMENTS = {
    "mobilenet_v2": {"min_size": 224, "channels": 3},
    "resnet18": {"min_size": 224, "channels": 3},
    "efficientnet_b0": {"min_size": 128, "channels": 3},
    "mnist": {"min_size": 28, "channels": 1},
    "vit_tiny": {"min_size": 224, "channels": 3}
}

def check_model_compatibility(img_dir, model="mobilenet_v2"):
    if model not in MODEL_REQUIREMENTS:
        raise ValueError(f"Model '{model}' not supported.")

    req = MODEL_REQUIREMENTS[model]
    issues = []
    passed = 0
    total = 0

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        total += 1
        fpath = os.path.join(img_dir, fname)
        try:
            h, w, c = get_shape(fpath)
            if min(h, w) < req["min_size"] or c != req["channels"]:
                issues.append((fname, (h, w, c)))
            else:
                passed += 1
        except Exception as e:
            issues.append((fname, f"Error: {e}"))

    return {
        "total": total,
        "passed": passed,
        "issues": issues,
        "model": model,
        "requirement": req
    }
