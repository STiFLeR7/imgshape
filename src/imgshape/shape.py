# shape.py
from PIL import Image
import requests
from io import BytesIO
import os

def get_shape(path_or_pil):
    """Return shape (H, W, C) of a local image or PIL image."""
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")

    w, h = img.size
    c = len(img.getbands())
    return (h, w, c)

def get_shape_from_url(url: str):
    """Fetch image from URL and return its shape."""
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return get_shape(img)

def get_shape_batch(file_paths):
    shapes = {}
    for path in file_paths:
        try:
            shape = get_shape(path)
            shapes[path] = shape
        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")
    return shapes

