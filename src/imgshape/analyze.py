# analyze.py
from PIL import Image, ImageStat, ImageFilter
import numpy as np
from collections import Counter

def get_entropy(image_path):
    """Returns entropy of an image."""
    img = Image.open(image_path).convert("L")
    return img.entropy()

def get_edge_density(image_path):
    """Returns edge pixel ratio after edge detection."""
    img = Image.open(image_path).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    edge_pixels = np.sum(arr > 50)  # threshold
    total_pixels = arr.size
    return edge_pixels / total_pixels

def get_dominant_color(image_path):
    """Returns the dominant color (as hex) in the image."""
    img = Image.open(image_path).convert("RGB").resize((50, 50))
    pixels = np.array(img).reshape(-1, 3)
    counts = Counter(map(tuple, pixels))
    dominant = counts.most_common(1)[0][0]
    return '#%02x%02x%02x' % dominant

def analyze_type(image_path):
    """Performs a lightweight image type analysis."""
    entropy = get_entropy(image_path)
    edge_density = get_edge_density(image_path)
    dominant_color = get_dominant_color(image_path)

    # Heuristic type guesser (can improve later)
    if entropy < 3.0 and edge_density < 0.01:
        guess = "document/scan"
    elif edge_density > 0.07:
        guess = "object-rich"
    elif entropy > 5.0:
        guess = "natural image"
    else:
        guess = "uncertain"

    return {
        "entropy": round(entropy, 2),
        "edge_density": round(edge_density, 3),
        "dominant_color": dominant_color,
        "guess_type": guess
    }
