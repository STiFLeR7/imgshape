# 🖼️ imgshape — Smart Image Analysis & Preprocessing Toolkit (v2.0.1)

`imgshape` is a lightweight Python toolkit designed for **image shape detection**, **dataset inspection**, **preprocessing recommendation**, and **AI model compatibility checks** — all optimized for **ML/DL workflows**, both in research and production.

![alt text](image.png)
---

## ⚡️ Why use `imgshape`?

* 🔍 Automatically **detect shape**, **dominant color**, **entropy**, and **type** of an image.
* 🧠 Recommend preprocessing steps like resize dims, normalization, and suitable model types.
* 🖬 Analyze entire datasets to get size/shape distribution and dimension scatter plots.
* ✅ Check model compatibility (e.g. with `mobilenet_v2`, `resnet18`, etc.).
* 🌐 Supports **CLI**, **Python API**, and even a **Gradio-based GUI** for visual workflows.

---


## 🚀 Installation

```bash
pip install imgshape
```

> Requires Python 3.8+ and packages: Pillow, matplotlib, seaborn, numpy, scikit-image, gradio

---

## 💻 CLI Usage

```bash
imgshape --path ./sample.jpg                  # Get image shape
imgshape --path ./sample.jpg --analyze        # Analyze image type and entropy
imgshape --path ./sample.jpg --recommend      # Recommend preprocessing steps
imgshape --dir ./images --check mobilenet_v2  # Check dataset compatibility with a model
imgshape --batch --path ./folder              # Batch mode shape detection
imgshape --viz ./images                       # Visualize size/shape distribution
imgshape --web                                # Launch Gradio GUI
```




