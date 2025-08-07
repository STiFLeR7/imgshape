# recommender.py
from PIL import Image
from imgshape.shape import get_shape
from imgshape.analyze import get_entropy

def recommend_preprocessing(image_path):
    shape = get_shape(image_path)
    entropy = get_entropy(image_path)

    height, width, channels = shape
    rec = {}

    # Suggest resize based on shape + entropy
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
    rec["entropy"] = round(entropy, 2)

    return rec
