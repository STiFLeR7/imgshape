# src/imgshape/recommender.py
import os
from PIL import Image, UnidentifiedImageError
from imgshape.shape import get_shape
from imgshape.analyze import get_entropy


def recommend_preprocessing(image_path):
    """Suggest preprocessing steps for ML models based on image size, channels, and entropy."""
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

    rec = {}

    # Suggest resize and model type
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


def check_model_compatibility(folder_path, model_name):
    """
    Check if all images in the folder are compatible with the given model.
    Returns: (total_images, passed, failed)
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
        "issues": failed_list
    }
