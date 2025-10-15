# 🖼️ imgshape — Smart Dataset Intelligence Toolkit (v3.0.0 • Aurora)

`imgshape` is a modular Python toolkit for **image analysis**, **dataset inspection**, **augmentation & preprocessing recommendations**, **visualization**, and **pipeline export** — now evolved into a **Streamlit-powered dataset assistant** for modern ML/DL workflows.

![imgshape demo](assets/sample_images/imgshape.png)  
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/imgshape?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/projects/imgshape)

---

## ✨ What's New in v3.0.0 — *Aurora Major Release*

> A complete redesign: from a static CLI toolkit → to an intelligent dataset analysis framework.

**🧭 Highlights**
- **Full Streamlit App (`app.py`)** with 6 powerful tabs:
  - 📐 **Shape** → instant image shape detection  
  - 🔍 **Analyze** → entropy, color channels, dataset insights  
  - 🧠 **Recommend** → preprocessing & augmentation planning  
  - 🎨 **Augment Visualizer** → real-time augmentation previews  
  - 📄 **Reports** → export Markdown / HTML dataset reports  
  - 🔗 **Pipeline Export** → generate ready-to-run code snippets

**🧩 Modular Architecture**
- New `RecommendationPipeline` system for building, saving, and exporting end-to-end pipelines.
- Plugin framework (`/src/imgshape/plugins`) with support for:
  - `AnalyzerPlugin`
  - `RecommenderPlugin`
  - `ExporterPlugin`
- Unified lazy import system for ultra-fast startup.

**💡 Smart Recommendations**
- `RecommendEngine` provides preprocessing & augmentation strategies based on:
  - Entropy, resolution, and dataset diversity
  - User preferences (e.g. `preserve_aspect`, `low_res`)
  - Optional YAML profiles (`/profiles/`)

**📊 Dataset Analyzer Improvements**
- Counts only *unique readable images* (no overcount)
- Aggregates shapes, channels, entropy, and unreadable stats
- Sample summaries for representative examples

**📁 Reports**
- Markdown, HTML, and PDF (optional via `weasyprint` + `reportlab`)
- Embedded metadata, augmentations, and preprocessing recommendations

**🧰 CLI Modernization**
- `imgshape --web` → directly launches Streamlit UI  
- Extended with new actions:
  - `--pipeline-export`, `--pipeline-apply`, `--snapshot-save`, `--snapshot-diff`
- Plugin controls: `--plugin-list`, `--plugin-add`, `--plugin-remove`

---

## ⚙️ Installation

```bash
pip install imgshape
````

> Requires **Python 3.8+**
> Core dependencies: `Pillow`, `numpy`, `matplotlib`, `scikit-image`, `streamlit`

**Optional extras:**

| Extra             | Description                           |
| :---------------- | :------------------------------------ |
| `imgshape[torch]` | PyTorch / torchvision support         |
| `imgshape[pdf]`   | PDF report generation via WeasyPrint  |
| `imgshape[viz]`   | Advanced plots with Seaborn & Plotly  |
| `imgshape[ui]`    | Streamlit UI + profile parsing        |
| `imgshape[full]`  | Full suite with all optional features |

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

# Launch the Streamlit web UI
imgshape --web
```

---

## 🖥️ Streamlit Interface (v3)

> Run the visual interface directly:

```bash
streamlit run app.py
```

### Tabs Overview

| Tab                   | Function                                           |
| --------------------- | -------------------------------------------------- |
| 📐 Shape              | Detects image dimensions & color channels          |
| 🔍 Analyze            | Dataset entropy, shapes, and channel distributions |
| 🧠 Recommend          | Suggests preprocessing & augmentations             |
| 🎨 Augment Visualizer | Interactive augmentation intensity slider          |
| 📄 Reports            | Generates Markdown & HTML dataset summaries        |
| 🔗 Pipeline Export    | Exports pipelines as code (PyTorch/YAML/JSON)      |

---

## 🧠 Python API Example

```python
from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.pipeline import RecommendationPipeline

print(get_shape("sample.jpg"))
print(analyze_type("sample.jpg"))
print(recommend_preprocessing("sample.jpg"))

# Build a pipeline from a recommendation
rec = recommend_preprocessing("sample.jpg")
pipeline = RecommendationPipeline.from_recommender_output(rec)
print(pipeline.as_dict())
```

---

## 🧩 Plugins

Extend `imgshape` with your own plugins:

```python
# src/imgshape/plugins/custom_brightness.py
from imgshape.plugins import RecommenderPlugin

class CustomBrightnessPlugin(RecommenderPlugin):
    NAME = "CustomBrightness"

    def recommend(self, analysis):
        return [{"name": "adjust_brightness", "spec": {"factor": 1.2}}]
```

Then register it via CLI:

```bash
imgshape --plugin-add ./src/imgshape/plugins/custom_brightness.py
```

---

## 📝 Reports (Markdown, HTML, PDF)

```bash
# Markdown & HTML reports
imgshape --report --path ./datasets/cats --report-format md,html

# Generate PDF (requires extras)
pip install imgshape[pdf]
imgshape --report --path ./datasets/dogs --report-format pdf
```

---

## 🧪 Testing

Run all tests locally:

```bash
pytest -q
```

Or install dev tools:

```bash
pip install imgshape[dev]
black --check src tests
flake8 src tests
```

---

## 🧱 Developer & Build Guide

```bash
# Clean build artifacts
rm -rf dist build *.egg-info

# Build
python -m build

# Check metadata
twine check dist/*

# Upload (TestPyPI)
twine upload --repository testpypi dist/*

# Install locally
pip install dist/imgshape-3.0.0-py3-none-any.whl
```

---

## 🔗 Resources

* **Documentation:** [https://stifler7.github.io/imgshape](https://stifler7.github.io/imgshape)
* **GitHub Repository:** [https://github.com/STiFLeR7/imgshape](https://github.com/STiFLeR7/imgshape)
* **Issues:** [https://github.com/STiFLeR7/imgshape/issues](https://github.com/STiFLeR7/imgshape/issues)
* **License:** MIT

---

## 💫 Credits

Developed with ❤️ by **[Stifler](https://github.com/STiFLeR7)**
Researched / Developer
*Empowering AI at the Edge.*

---

## 🧭 Roadmap (v3.1.x)

* ONNX / TensorRT export for edge inference
* Auto-EDA visualization (class imbalance, histograms)
* Enhanced Streamlit dashboard with live metrics
* HuggingFace Spaces demo & CI/CD workflow

```

---

### 🧩 Summary of Key Updates
- Updated version → `v3.0.0 (Aurora)`  
- Removed Gradio references (Streamlit is now primary)  
- Added new **Pipeline**, **Plugins**, and **Recommender Engine** details  
- Expanded CLI + Streamlit examples  
- Ready for **PyPI rendering** and **GitHub preview**

