# 📦 imgshape — Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [4.2.0] - 2026-04-16

### v4.2.0 "Bento Intelligence"

The **v4.2.0** release is a major UX overhaul coupled with deep semantic intelligence. It transforms the Atlas engine into a professional, modular workspace.

**Highlights:**
- **UX Refactor (Bento Grid)**: A complete UI redesign using a 12-column grid system for high information density and professional engineering aesthetic.
- **Semantic Drift 2.0**: Integrated vision-transformer (DINOv2) embeddings to detect semantic shifts in datasets beyond simple statistics.
- **Hyper-optimized GPU**: Vectorized PyTorch kernels for entropy and blur, reducing VRAM transfer overhead and improving batch throughput.
- **Domain Intelligence**: Specialized fingerprinting profiles and UI views for Medical (DICOM), Satellite, and OCR datasets.
- **Provenance IDs**: Enhanced audit trails for all rule-based decisions.

### [4.1.0] - 2026-03-01

### v4.1.0

The **v4.1.0** release enhances the v4 core with GPU acceleration, statistical dataset comparison, and improved decision traceability.

### 🚀 New Features

#### 1. **GPU Acceleration (PyTorch)**

- **Hardware-Aware Engine**: Automatically detects CUDA and offloads heavy computations
- **Accelerated Entropy**: Up to 15x faster entropy calculation for large datasets
- **Accelerated Blur Detection**: Laplacian variance computation using Torch convolutions
- **Direct Stat Projections**: Real-time signal analysis on GPU memory

#### 2. **Dataset Comparison & Drift**

- **Similarity Indexing**: Calculate geometric and semantic similarity between datasets (0.0 to 1.0)
- **Drift Detection**: Quantitative drift scoring using histogram intersection and metric deltas
- **Delta Reporting**: Automated Markdown reports highlighting statistical shifts between dataset versions
- **Signal Tracking**: Monitor changes in entropy stability and color distribution over time

#### 3. **Decision Intelligence v4.1**

- **Provenance IDs**: Every decision now linked to a specific rule ID (e.g., `R-MODEL-SIZE-LT-20`) for auditability
- **Reproducibility Hashing**: Unique SHA256 hashes for decision collections ensuring stable pipeline outcomes
- **Alternative Scoring**: Trade-offs and alternative choices now include normalized impact scores
- **Extended Metrics**: Added Aspect Ratio Clustering, Channel Variance, and 256-bin Color Histograms

### 🛠️ Improvements & Fixes

- **CLI**: Added `--compare`, `--drift`, and `--benchmark` commands
- **API**: New endpoints for comparison, drift analysis, and performance benchmarking
- **UI**: Modernized branding for v4.1.0 and added GPU status indicators
- **Metadata**: Enhanced dataset fingerprints with acceleration type and imgshape version tracking

---

## [4.0.0] - 2025-12-29

### 🌟 Atlas Major Release

The **v4.0.0 "Atlas"** release represents a fundamental architectural redesign of imgshape, shifting from heuristic-based recommendations to deterministic dataset intelligence with complete explainability.

### 🚀 Major Features

#### 1. **Deterministic Fingerprinting Engine**

- **5-Profile Fingerprinting System**
  - 📐 **Spatial Profile**: Image dimensions, aspect ratios, scale distribution analysis
  - 🔌 **Signal Profile**: Channel count, bit depth, dynamic range characterization
  - 📊 **Distribution Profile**: Entropy, skewness, color uniformity metrics
  - ✓ **Quality Profile**: Corruption detection, blur estimation, noise analysis
  - 🧠 **Semantic Profile**: Inferred content type (photographic, medical, aerial, etc.)

- **Stable Dataset Identities**
  - Canonical dataset URIs: `imgshape://vision/photographic/high-entropy`
  - SHA256-based dataset IDs for reproducible comparisons
  - Deterministic across runs and deployments
  - CI/CD-safe fingerprint locking (`.fingerprint_lock` files)

#### 2. **Rule-Based Decision Engine**

- **8 Core Decision Domains**
  - Model family selection (ResNet, MobileNet, ViT, EfficientNet, etc.)
  - Input dimension optimization (224×224, 512×512, custom)
  - Preprocessing strategy determination
  - Batch size recommendations based on dataset size
  - Optimizer selection (Adam, SGD, AdamW)
  - Augmentation strategy and intensity levels
  - Deployment target optimization (CPU, GPU, Edge, Mobile)
  - Training duration and early stopping configuration

- **Full Explainability**
  - Every decision includes 3-8 explicit reasons
  - Confidence scores with reasoning
  - Alternative recommendations with trade-offs
  - Metrics-based decision justification

#### 3. **Deployable Artifacts**

- **Production-Ready Exports**
  - JSON, YAML, Protocol Buffer formats
  - Version-controlled artifact storage
  - Checksum verification for integrity
  - Metadata for reproducibility
  - Git-friendly (supports `.fingerprint_lock` patterns)

#### 4. **Modern Web Interface (React)**

- **Interactive Dashboard**
  - Real-time fingerprint generation and visualization
  - Decision explorer with full reasoning display
  - Dataset statistics dashboard
  - Export functionality (JSON, YAML, PDF)
  - Modern responsive design with Tailwind CSS
  - Fast, lightweight (224 KB JS bundle)
  
- **Service Integration**
  - FastAPI backend (Python 3.8+)
  - CORS-enabled REST API
  - WebSocket support for real-time updates
  - Health check endpoints
  - Graceful fallback to Jinja2 templates

#### 5. **Enhanced CLI**

- **New v4 Commands**
  - `imgshape --fingerprint <path>` - Generate dataset fingerprint
  - `imgshape --atlas <path>` - Full analysis with all profiles
  - `imgshape --decisions <path>` - View decision recommendations
  - `imgshape --web` - Launch interactive React UI on port 8080

- **Command-Line Options**
  - `--task` - classification, detection, segmentation, etc.
  - `--deployment` - edge, gpu, cpu, mobile, server
  - `--priority` - speed, accuracy, balance
  - `--format` - json, yaml, protobuf
  - `--output` - save results to file
  - `--verbose` - detailed reasoning output

#### 6. **Docker & Cloud Deployment**

- **Multi-Stage Dockerfile**
  - Stage 1: Node.js 18-Alpine (builds React UI)
  - Stage 2: Python 3.12-slim (runs FastAPI service)
  - Optimized for Google Cloud Run
  - Health check support
  - Memory-efficient (~500MB)

- **Cloud Run Ready**
  - Pre-configured for GCP deployment
  - cloudbuild.yaml for CI/CD automation
  - `.gcloudignore` for optimized uploads
  - Environment-agnostic configuration

#### 7. **Testing & Validation**

- **Comprehensive Test Suite**
  - 33 tests across all modules
  - 26/33 passing (79% - 7 optional artifact tests)
  - Fingerprint extraction tests (9/9 passing)
  - Decision engine tests (7/7 passing)
  - Atlas orchestrator tests (4/4 passing)
  - API integration tests (6/6 passing)

- **Test Coverage**
  - Unit tests for all core modules
  - Integration tests for API endpoints
  - End-to-end tests for CLI commands
  - Docker image validation tests

### ✨ Key Improvements

- **No More Magic**: Every recommendation includes explicit reasoning
- **Reproducibility**: Deterministic fingerprints enable dataset locking
- **Framework Agnostic**: Works with PyTorch, TensorFlow, JAX, NumPy
- **Production Deployment**: Docker, Cloud Run, and REST API ready
- **Modern UX**: React-based interactive web interface
- **Extensible**: Plugin system for custom fingerprinting
- **CI/CD Friendly**: Fingerprint locking prevents dataset drift
- **Comprehensive Docs**: Full API reference and usage guides

### 🏗️ Architecture

**Old Approach (v3)**: Dataset → Heuristic Analysis → Recommendation  
**New Approach (v4 Atlas)**: Dataset → 5-Profile Fingerprinting → Rule-Based Decisions → Explainable Artifacts

### 📊 Performance

- **Fingerprint Generation**: 50K images in ~2-5 minutes
- **Decision Engine**: <100ms per decision
- **Web UI Bundle**: 224.92 KB (gzipped)
- **Docker Image**: ~500MB base, 1.2GB with dependencies
- **Memory Usage**: 256MB baseline, ~1GB under load

### 🔌 Plugin System

```python
from imgshape.plugins import FingerprintPlugin

class CustomProfiler(FingerprintPlugin):
    NAME = "custom_profiler_v1"
    def extract(self, dataset_path):
        return {...}
```

### 🌐 REST API

- **v4 Endpoints**
  - `POST /v4/fingerprint` - Get fingerprint for a dataset
  - `POST /v4/decisions` - Get recommendations
  - `POST /v4/analyze` - Full analysis
  - `GET /health` - Service health

- **Legacy Support**
  - `POST /analyze` - v3 compatibility
  - `POST /recommend` - v3 compatibility

### 📦 Installation

```bash
# Core package (minimal dependencies)
pip install imgshape

# With all features
pip install "imgshape[full]"

# Development
pip install "imgshape[dev]"
```

### 🐛 Breaking Changes

- Removed Streamlit integration (replaced with modern React UI)
- Removed v2 legacy APIs
- Command-line syntax updated (now uses `--fingerprint`, `--atlas`, `--decisions`)
- Python 3.8+ required (was 3.7+)

### 🔄 Migration from v3

```python
# v3 style (no longer recommended)
from imgshape.recommender import recommend_preprocessing
rec = recommend_preprocessing("image.jpg")

# v4 style (new)
from imgshape import Atlas
atlas = Atlas()
result = atlas.analyze("dataset/", task="classification")
```

### 📝 Documentation

- Full API documentation: <https://stifler7.github.io/imgshape>
- v4 Design Document: [v4.md](v4.md)
- Contributing Guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Deployment Guide: [CLOUD_RUN_DEPLOYMENT.md](CLOUD_RUN_DEPLOYMENT.md)

### 🙏 Contributors

Thanks to all contributors who made v4.0.0 possible!

---

## [3.0.0] - 2025-10-15  

### 🌌 Aurora Major Release

The **v3.0.0 "Aurora"** release transforms `imgshape` from a simple CLI toolkit into a **modular dataset intelligence framework** with Streamlit UI, pipeline export, and plugin ecosystem.

### 🚀 Major Highlights

- **Unified Streamlit Interface (`app.py` at repo root):**
  - 6 intuitive tabs: Shape, Analyze, Recommend, Augment Visualizer, Reports, Pipeline Export
  - New liquid-glass dark theme and responsive layout
  - Full Streamlit-based UX, replacing legacy Gradio GUI

- **CLI (`imgshape`) Modernization**
  - New `--web` flag → directly launches the Streamlit UI
  - Extended pipeline commands with snapshot support
  - Plugin system integration
  - Maintains backward compatibility for all v2.x flags

- **Pipeline & Plugin Architecture**
  - New classes: `RecommendationPipeline`, `PipelineStep`, and plugin bases
  - Extensible via `/src/imgshape/plugins`
  - Export pipelines in multiple formats: `torchvision`, `json`, or `yaml`

- **Dataset Analyzer Improvements**
  - More accurate counting of unique readable images
  - Smart handling of nested directories and corrupted files
  - Aggregates shape, channel, and entropy distributions
  - Optional verbose logging for debugging

- **Recommender Engine v3**
  - Unified `RecommendEngine` abstraction
  - Profile-driven recommendations (`profiles/` YAML presets)
  - Supports user preferences

- **Report System**
  - Markdown, HTML, and optional PDF export
  - Integrated into both CLI and Streamlit UI

### ✨ Enhancements

- Lazy import system → lightweight and startup-fast
- Streamlit deprecation fixes
- Modular build system with `pyproject.toml`
- New optional extras: `torch`, `ui`, `viz`, `pdf`, `plugins`, `full`, `dev`

### 🐛 Fixes

- Fixed duplicate image counting in dataset analyzer
- Improved entropy computation on grayscale inputs
- Defensive error handling in plugin loader
- Fully backward compatible with legacy test suite

### 🧪 Testing & Validation

- Complete local test suite (`pytest -q`)
- Verified CLI, Streamlit, and lazy import consistency
- Manual regression pass across all features

---

## [2.2.0] - 2025-09-22

### 🚀 Major Release

- Introduced full **Streamlit App** (`app.py`) with 5 interactive tabs
- CLI, Streamlit, and test suite aligned for consistent workflows
- TorchLoader, AugmentationRecommender, CompatibilityChecker features
- Markdown, HTML, PDF report export options
- Fixed pytest issues and defensive error handling

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

## 🎯 Roadmap

### Planned for 4.1.x

- ONNX / TensorRT export helpers for edge inference
- Auto-EDA visual previews (histograms, class imbalance)
- Enhanced metrics dashboard
- PyPI automated release workflow
- GPU-accelerated fingerprinting

### Planned for 5.x.x (Future)

- Multi-modal support (video, audio, 3D)
- Distributed fingerprinting (Dask, Ray)
- Federated learning dataset analysis
- Interactive dataset comparison UI
- Real-time dataset drift detection
