import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imgshape.shape import get_shape_batch

sns.set(style="whitegrid")

def _get_image_paths(folder_path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return files

def plot_shape_distribution(folder_path, save=False):
    image_paths = _get_image_paths(folder_path)
    shapes_dict = get_shape_batch(image_paths)

    widths = []
    heights = []

    for shape in shapes_dict.values():
        if len(shape) == 3:
            h, w, _ = shape
            widths.append(w)
            heights.append(h)

    plt.figure(figsize=(10, 5))
    plt.hist(widths, bins=10, alpha=0.6, label='Widths', color='#5DADE2')
    plt.hist(heights, bins=10, alpha=0.6, label='Heights', color='#F5B041')
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.title("🧮 Image Size Distribution")
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs("output", exist_ok=True)
        plt.savefig("output/shape_distribution.png")
    plt.show()


def plot_image_dimensions(folder_path, save=False):
    image_paths = _get_image_paths(folder_path)
    shapes_dict = get_shape_batch(image_paths)

    widths = []
    heights = []

    for shape in shapes_dict.values():
        if len(shape) == 3:
            h, w, _ = shape
            widths.append(w)
            heights.append(h)

    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, alpha=0.6, c="#48C9B0", edgecolors='black')
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("🖼️ Image Dimension Scatter Plot")
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs("output", exist_ok=True)
        plt.savefig("output/dimension_scatter.png")
    plt.show()
