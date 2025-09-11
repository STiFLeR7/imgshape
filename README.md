
# 🖼️ imgshape — Smart Image Analysis & Preprocessing Toolkit (v2.1.3)

`imgshape` is a Python toolkit for **image shape detection**, **dataset inspection**, **preprocessing & augmentation recommendations**, **visualization**, **report generation**, and **PyTorch DataLoader helpers** — making it a **smarter dataset assistant** for ML/DL workflows.

![imgshape demo](assets/sample_images/Image_created_with_a_mobile_phone.png)

---

## ⚡️ Why use `imgshape`?

* 📐 Detect **image shapes** (H × W × C) for single files or whole datasets.
* 🔍 Compute **entropy**, **edge density**, **dominant color**, and guess image type.
* 🧠 Get **preprocessing recommendations** (resize, normalization, suitable model family).
* 🔄 **Augmentation recommender**: suggest flips, crops, color jitter, etc., based on dataset stats.
* 📊 **Visualizations**: size histograms, dimension scatter plots, channel distribution.
* ✅ **Model compatibility checks**: verify dataset readiness for models like `mobilenet_v2`, `resnet18`, etc.
* 📝 **Dataset reports**: export Markdown/HTML/PDF with stats, plots, preprocessing, and augmentation plans.
* 🔗 **Torch integration**: generate ready-to-use `torchvision.transforms` or even a `DataLoader`.
* 🌐 **GUI mode**: run a Gradio app for point-and-click analysis.

---

## 🚀 Installation

```bash
pip install imgshape
````

> Requires Python 3.8+
> Core deps: `Pillow`, `numpy`, `matplotlib`, `scikit-image`, `gradio`
> Optional extras:
>
> * `imgshape[torch]` → PyTorch / torchvision support
> * `imgshape[pdf]` → PDF report generation (`weasyprint`)
> * `imgshape[viz]` → prettier plots (`seaborn`)

---

## 💻 CLI Usage

```bash
# Shape detection
imgshape --path ./sample.jpg --shape

# Single image analysis
imgshape --path ./sample.jpg --analyze

# Preprocessing + augmentations
imgshape --path ./sample.jpg --recommend --augment

# Dataset compatibility check
imgshape --dir ./images --check mobilenet_v2

# Dataset visualization
imgshape --viz ./images

# Dataset report (md + html)
imgshape --path ./images --report --augment --report-format md,html --out report

# Torch integration (transform/DataLoader)
imgshape --path ./images --torchloader --augment --out transform_snippet.py

# Launch Gradio GUI
imgshape --web
```

---

## 📦 Python API

```python
from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.augmentations import AugmentationRecommender

print(get_shape("sample.jpg"))
print(analyze_type("sample.jpg"))
print(recommend_preprocessing("sample.jpg"))

# Augmentation plan
ar = AugmentationRecommender(seed=42)
plan = ar.recommend_for_dataset({"entropy_mean": 6.2, "image_count": 100})
print(plan.recommended_order)
```

---

## 📝 New in v2.1.1

* 🔄 **Augmentation recommender** (`--augment`, `augmentations.py`)
* 📝 **Dataset report generator** (`--report`, Markdown/HTML/PDF export)
* 🔗 **TorchLoader integration** (`--torchloader`, `to_dataloader`, `to_torch_transform`)
* 📊 **Improved visualizations** (works even for 1-image datasets)
* 🌐 **Modernized GUI** with analysis + recommendations tabs

---

## 📎 Resources

* [Source Code](https://github.com/STiFLeR7/imgshape)
* [Issues](https://github.com/STiFLeR7/imgshape/issues)
* License: MIT


