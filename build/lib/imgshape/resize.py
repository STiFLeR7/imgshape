import os
from PIL import Image, ImageOps

def resize_image(img_path, size, fmt="jpg", keep_aspect=True, save_dir=None, keep_original=False):
    img = Image.open(img_path).convert("RGB")
    basename = os.path.basename(img_path)
    filename, _ = os.path.splitext(basename)

    # Parse size
    if isinstance(size, int):
        target_size = (size, size)
    elif isinstance(size, str) and "x" in size:
        w, h = size.lower().split("x")
        target_size = (int(w), int(h))
    else:
        raise ValueError("Invalid size format. Use int or 'WIDTHxHEIGHT'.")

    # Resize with aspect ratio preserved
    if keep_aspect:
        img = ImageOps.pad(img, target_size, color=(0, 0, 0))
    else:
        img = img.resize(target_size)

    # Set output path
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{filename}.{fmt}")
    else:
        out_path = os.path.join(os.path.dirname(img_path), f"{filename}_resized.{fmt}")

    img.save(out_path, fmt.upper())

    if keep_original:
        return out_path, img_path
    return out_path


def batch_resize(folder_path, size, fmt="jpg", keep_structure=True, save_dir=None, keep_original=False):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                img_path = os.path.join(root, file)

                rel_path = os.path.relpath(root, folder_path)
                target_dir = os.path.join(save_dir, rel_path) if (save_dir and keep_structure) else save_dir

                out_path = resize_image(
                    img_path, size, fmt=fmt, save_dir=target_dir,
                    keep_aspect=True, keep_original=keep_original
                )
                results.append(out_path)
    return results
