# 🖼️ imgshape — Smart Image Analysis & Preprocessing Toolkit (v2.2.0)

`imgshape` is a Python toolkit for **image shape detection**, **dataset inspection**, **preprocessing & augmentation recommendations**, **visualization**, **report generation**, and **PyTorch DataLoader helpers** — making it a **smarter dataset assistant** for ML/DL workflows.

![imgshape demo](assets/sample_images/Image_created_with_a_mobile_phone.png)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/imgshape?period=total\&units=INTERNATIONAL_SYSTEM\&left_color=BLACK\&right_color=GREEN\&left_text=downloads)](https://pepy.tech/projects/imgshape)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ⚡️ Why use `imgshape`?

* 📐 Detect **image shapes** (H × W × C) for single files or whole datasets.
* 🔍 Compute **entropy**, **edge density**, **dominant color**, and guess image type.
* 🧠 Get **preprocessing recommendations** (resize, normalization, suitable model family).
* 🔄 **Augmentation recommender**: suggest flips, crops, color jitter, etc., based on dataset stats.
* 📊 **Visualizations**: size histograms, dimension scatter plots, channel distribution.
* ✅ **Model compatibility checks**: verify dataset readiness for models like `mobilenet_v2`, `resnet18`, etc.
* 📝 **Dataset reports**: export Markdown/HTML/PDF with stats, plots, preprocessing, and augmentation plans.
* 🔗 **Torch integration**: generate ready-to-use `torchvision.transforms` or even a `DataLoader`.
* 🌐 **Interactive GUI modes**:

  * **Streamlit app** (`app.py`) → modern multi-tab UI
  * **Gradio app** (`--web`) → quick prototyping

---

## 🚀 Installation

```bash
pip install imgshape
```

> Requires Python 3.8+
> Core deps: `Pillow`, `numpy`, `matplotlib`, `scikit-image`, `streamlit`
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

# Launch Streamlit app
streamlit run app.py

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

## 📝 New in v2.2.0

* 🌐 **Streamlit App** (`app.py`) with **5 interactive tabs**:

  * **Shape** → instant image shape detection
  * **Analyze** → entropy, channels, and dataset visualization
  * **Recommend** → preprocessing + heuristic augmentation plan
  * **Report** → export dataset reports in Markdown/HTML
  * **TorchLoader** → export `torchvision.transforms` pipelines or snippets
* 🔗 **TorchLoader**:

  * Safe wrapper for Compose/snippet/no-op callable depending on availability.
  * Backward compatibility with old `(plan, preprocessing)` test calls.
* 🧠 **AugmentationRecommender**:

  * Deterministic heuristic plans with `.as_dict()` export.
  * Handles entropy, resolution, and imbalance.
* ✅ **Compatibility Fixes**:

  * `check_compatibility()` outputs structured results.
  * Deprecated alias `check_model_compatibility()` preserved.
* 📝 **Report Generators**:

  * Markdown + HTML outputs improved.
* ⚡️ **Test Suite**:

  * Fixed pytest failures in `compatibility`, `report`, and `torchloader`.
* 🎨 **UI Polishing**:

  * Defensive wrappers for `analyze_type`, `recommend_preprocessing`, TorchLoader.
  * Footer links to **Instagram, GitHub, HuggingFace, Kaggle, Medium**.

---

## 📎 Resources

* [Source Code](https://github.com/STiFLeR7/imgshape)
* [Issues](https://github.com/STiFLeR7/imgshape/issues)
* License: MIT
