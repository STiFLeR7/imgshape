# 📦 imgshape — Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [3.0.0] - 2025-10-15  
### 🌌 Aurora Major Release  
The **v3.0.0 "Aurora"** release transforms `imgshape` from a simple CLI toolkit into a **modular dataset intelligence framework** with Streamlit UI, pipeline export, and plugin ecosystem.

### 🚀 Major Highlights
- **Unified Streamlit Interface (`app.py` at repo root):**
  - 6 intuitive tabs:
    - **📐 Shape:** Image shape and dimension detection  
    - **🔍 Analyze:** Per-image and dataset-wide statistics (entropy, channels, shape distribution)  
    - **🧠 Recommend:** AI-assisted preprocessing & augmentation recommendations  
    - **🎨 Augment Visualizer:** Real-time augmentation previews with adjustable intensity  
    - **📄 Reports:** Export Markdown / HTML dataset reports  
    - **🔗 Pipeline Export:** Export pipelines as Python, JSON, or YAML  
  - New **liquid-glass dark theme** and responsive layout  
  - Full **Streamlit-based UX**, replacing legacy Gradio GUI  

- **CLI (`imgshape`) Modernization**
  - New `--web` flag → directly launches the Streamlit UI  
  - Extended pipeline commands:
    - `--pipeline-export`, `--pipeline-apply`, `--pipeline-dryrun`
  - Plugin system integration with `--plugin-list`, `--plugin-add`, `--plugin-remove`
  - Dataset snapshots: `--snapshot-save` and `--snapshot-diff`
  - Maintains backward compatibility for all v2.x flags

- **Pipeline & Plugin Architecture**
  - New classes: `RecommendationPipeline`, `PipelineStep`, and plugin bases (`AnalyzerPlugin`, `RecommenderPlugin`, `ExporterPlugin`)
  - Extensible via `/src/imgshape/plugins`
  - Export pipelines in multiple formats: `torchvision`, `json`, or `yaml`

- **Dataset Analyzer Improvements**
  - More accurate counting of unique readable images  
  - Smart handling of nested directories and corrupted files  
  - Aggregates shape, channel, and entropy distributions with per-sample summaries  
  - Optional verbose logging for debugging

- **Recommender Engine v3**
  - Unified `RecommendEngine` abstraction  
  - Profile-driven recommendations (`profiles/` YAML presets)  
  - Supports user preferences (`preserve_aspect`, `low_res`, etc.)

- **Report System**
  - Markdown, HTML, and optional PDF via `weasyprint` and `reportlab`  
  - Integrated into both CLI and Streamlit UI  

---

### ✨ Enhancements
- Lazy import system in `__init__.py` → lightweight and startup-fast  
- Streamlit deprecation fixes (`use_container_width=True`)  
- Modular build system with `pyproject.toml` + modern `setup.py`  
- New optional extras: `torch`, `ui`, `viz`, `pdf`, `plugins`, `full`, and `dev`
- Better dataset visualization with interactive shape plots (`plotly` or `matplotlib` fallback)

---

### 🐛 Fixes
- Fixed duplicate image counting in dataset analyzer  
- Improved entropy computation on grayscale inputs  
- Defensive error handling in plugin loader and augmentation visualizer  
- `--web` command now resolves `app.py` at repo root (not `src/`)  
- Fully backward compatible with legacy test suite (`pytest` verified)

---

### 🧪 Testing & Validation
- Complete local test suite (`pytest -q`)  
- Verified CLI, Streamlit, and lazy import consistency  
- Manual regression pass across dataset analysis, augmentation, and pipeline export  

---

### 🧱 Developer Experience
- Added `pyproject.toml` (PEP 621)  
- Modernized `setup.py` for PyPI packaging  
- Added build/test extras (`pip install imgshape[dev]`)  
- Ready for CI/CD via GitHub Actions  
- Supports reproducible builds via `python -m build && twine check dist/*`

---

### 🔗 Footer & Metadata
- Updated Streamlit footer:
  - Links to **GitHub**, **Medium**, **HuggingFace**, **Kaggle**, and **Instagram**
- Updated documentation URL:  
  [https://stifler7.github.io/imgshape/](https://stifler7.github.io/imgshape/)

---

## [2.2.0] - 2025-09-22
### 🚀 Major Release
- Introduced full **Streamlit App** (`app.py`) with 5 interactive tabs:
  - Shape, Analyze, Recommend, Report, and TorchLoader  
- CLI, Streamlit, and test suite aligned for consistent workflows.

### ✨ Features
- **TorchLoader:** Defensive wrapper supporting Compose/snippet output  
- **AugmentationRecommender:** Deterministic, entropy-aware augmentation plan  
- **Compatibility Checker:** `check_model_compatibility()` for datasets  
- **Report Generators:** Markdown, HTML, PDF export options  
- **Streamlit UX:** Sidebar inputs, dynamic outputs, social footer  

### 🐛 Fixes
- Fixed pytest issues across modules  
- Defensive handling for missing datasets and bad paths  
- Removed deprecated `use_column_width`  
- Fixed augmentation plan serialization  

### 🧪 Testing
- Full pytest suite passing locally  
- Backward compatibility ensured  

---

## [2.1.x] - 2025-06 → 2025-09
- Incremental fixes for augmentation, compatibility API, and report stability  
- Early Streamlit prototype introduced  
- Partial pytest alignment  

---

## [2.0.0] - 2025-04
- Major refactor: modularized analyze, recommender, augmentations, and CLI  
- Introduced initial test suite and CI compatibility  

---

## [1.x.x] - 2025
- Initial releases: core CLI tools for image shape detection and dataset analysis  

---

## 🧭 Next (Planned for 3.1.x)
- ONNX / TensorRT export helpers for edge inference  
- Auto-EDA previews (histograms, class imbalance visualization)  
- Enhanced Streamlit metrics dashboard  
- Dockerfile + HuggingFace Spaces demo  
- Full CI/CD release pipeline (pytest, lint, build, publish)
