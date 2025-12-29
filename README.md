<div align="center">

# 📊 imgshape
### Dataset Intelligence Layer for Computer Vision

**v4.0.0 Atlas Edition**

[![PyPI Version](https://img.shields.io/pypi/v/imgshape?color=blue&style=for-the-badge)](https://pypi.org/project/imgshape/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/STiFLeR7/imgshape?style=for-the-badge&color=orange)](https://github.com/STiFLeR7/imgshape/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)

<br/>

![imgshape Atlas](service/imgshape.png)

<br/>

**Deterministic Dataset Fingerprinting & Intelligent Decision Making**<br/>
*Fingerprinting • Rule-Based Decisions • Explainable AI • Deployable Artifacts • Production Ready*

[🌐 **Live Demo**](https://imgshape.vercel.app/) • [**Documentation**](https://stifler7.github.io/imgshape) • [**v4 Guide**](#-imgshape-v400-atlas) • [**Report Bug**](https://github.com/STiFLeR7/imgshape/issues) • [**Request Feature**](https://github.com/STiFLeR7/imgshape/issues)

</div>

---

## 🚀 imgshape v4.0.0 (Atlas)

**Atlas** is a complete architectural redesign of imgshape, shifting from heuristic recommendations to deterministic dataset intelligence.

### Core Capabilities

| Feature | Description |
| :--- | :--- |
| **🔬 Deterministic Fingerprinting** | Stable, canonical dataset identities across runs and deployments |
| **🎯 Rule-Based Decisions** | Explainable, traceable decisions with full reasoning |
| **📐 Five-Profile System** | Spatial, Signal, Distribution, Quality, Semantic analysis |
| **📦 Deployable Artifacts** | CI-safe, version-controlled outputs for production |
| **🔓 No Hidden Logic** | Every decision includes complete rationale and confidence |
| **⚙️ Framework Agnostic** | Works with PyTorch, TensorFlow, JAX, or plain NumPy |

### Why Atlas?

**Before (v3):** "This dataset looks good for ResNet50."  
**Now (v4 Atlas):** "This dataset's fingerprint is `imgshape://vision/photographic/high-entropy`. For task=classification with priority=speed, we recommend MobileNetV3 because: [8 explicit reasons with metrics]."

---

## ⚡ Quick Start

### Installation

```bash
# Core package
pip install imgshape

# With web UI and full features
pip install "imgshape[full]"
```

### Python API (v4)

```python
from imgshape import Atlas

# Initialize the analyzer
atlas = Atlas()

# Analyze a dataset
result = atlas.analyze(
    dataset_path="path/to/images",
    task="classification",
    deployment="edge",
    priority="speed"
)

# Inspect results
print(f"Fingerprint: {result.fingerprint.dataset_uri}")
# Fingerprint: imgshape://vision/photographic/high-entropy

print(f"Recommended Model: {result.decisions['model_family'].selected}")
# Recommended Model: MobileNetV3

print(f"Reasoning: {result.decisions['model_family'].why}")
# Reasoning: [8 evidence points with metrics]

# Export for CI/CD
artifact = result.to_artifact()
artifact.save("dataset_analysis.json")
```

### Command Line (v4)

```bash
# Generate fingerprint
imgshape --fingerprint path/to/images --format json

# Run full analysis
imgshape --atlas path/to/images --task classification --output analysis.json

# View decisions
imgshape --decisions path/to/images --priority speed --deployment edge

# Interactive web UI
imgshape --web
# Opens http://localhost:8080 with modern React interface
```

### Web Interface

The **imgshape web UI** provides an interactive, modern interface for dataset analysis:

**Live Demo:** 🌐 [imgshape.vercel.app](https://imgshape.vercel.app/)

```bash
imgshape --web
```

**Features:**
- 📊 Real-time fingerprint generation and visualization
- 🎯 Interactive decision explorer with full reasoning
- 📈 Dataset statistics dashboard
- 💾 Export analysis results (JSON, YAML, PDF)
- 🚀 Deploy artifacts directly from the UI

![Dashboard UI](assets/dashboard.png)

---

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────┐
│         Atlas Orchestrator                       │
│  (Main coordination & result aggregation)        │
└────────────┬────────────────────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌────────┐ ┌──────┐ ┌─────────┐
│Finger- │ │Rules │ │Artifact │
│print   │ │Based │ │Generator│
│Engine  │ │Decis-│ │         │
│        │ │ion   │ │         │
└────────┘ └──────┘ └─────────┘
    │        │         │
    └────────┼─────────┘
             │
             ▼
    ┌────────────────┐
    │Result Bundle   │
    │ - Fingerprint  │
    │ - Decisions    │
    │ - Artifacts    │
    │ - Confidence   │
    └────────────────┘
```

### Fingerprint Profiles

Every dataset receives a **5-dimensional fingerprint**:

1. **Spatial Profile** - Image dimensions, aspect ratios, scale distribution
2. **Signal Profile** - Channel count, bit depth, dynamic range
3. **Distribution Profile** - Entropy, skewness, color uniformity
4. **Quality Profile** - Corruption rate, blur detection, noise estimation
5. **Semantic Profile** - Inferred content type (faces, objects, aerial, medical, etc.)

---

## 🎯 Decision Domains

Atlas makes deterministic decisions across 8 domains:

| Domain | Examples |
| :--- | :--- |
| **Model Family** | ResNet, MobileNet, ViT, EfficientNet, etc. |
| **Input Dimensions** | 224x224, 512x512, or custom based on content |
| **Preprocessing** | Normalization parameters, augmentation strategy |
| **Batch Size** | Based on memory constraints and convergence |
| **Optimizer** | Adam, SGD, AdamW based on dataset characteristics |
| **Augmentation** | RandAugment, MixUp, Cutmix, intensity levels |
| **Deployment Target** | CPU, GPU, Edge (TensorRT, ONNX), Mobile |
| **Training Duration** | Early stopping patience, epoch count, callbacks |

---

## 📊 Example Analysis Output

```json
{
  "fingerprint": {
    "dataset_uri": "imgshape://vision/photographic/high-entropy",
    "dataset_id": "sha256:abc123...",
    "sample_count": 50000,
    "spatial": {
      "resolution_class": "high",
      "aspect_ratio_variance": 0.23,
      "mean_dimensions": [1920, 1080]
    },
    "signal": {
      "channel_count": 3,
      "bit_depth": 8
    },
    "distribution": {
      "entropy": 7.84,
      "color_uniformity": 0.42
    },
    "quality": {
      "corruption_rate": 0.0,
      "blur_percentage": 3.2,
      "noise_estimate": "gaussian"
    },
    "semantic": {
      "inferred_type": "photographic",
      "confidence": 0.92
    }
  },
  "decisions": {
    "model_family": {
      "selected": "MobileNetV3",
      "confidence": 0.87,
      "why": [
        "Dataset has 50k images (suitable for efficient models)",
        "Spatial resolution is high (1920x1080 average)",
        "Photographic content with 0.23 aspect ratio variance",
        "Edge deployment prioritizes inference speed over accuracy",
        "MobileNetV3 offers 2.8x faster inference than ResNet50",
        "Maintains 91% of ResNet50 accuracy on ImageNet",
        "Works on CPU and mobile devices",
        "Recent architecture (2019) with good operator support"
      ],
      "alternatives": ["EfficientNetB1", "ResNet34"]
    },
    "input_dimensions": {
      "selected": [224, 224],
      "confidence": 0.95,
      "why": ["MobileNetV3 default", "High entropy favors standard sizes"]
    }
  },
  "artifacts": {
    "fingerprint_stable": true,
    "fingerprint_format": "v4",
    "export_formats": ["json", "yaml", "protobuf"]
  }
}
```

---

## 💻 Usage Patterns

### 1. CI/CD Integration

```bash
#!/bin/bash
# ci_check.sh - Ensure dataset integrity in your pipeline

imgshape --fingerprint data/train \
  --output fingerprint.json \
  --format json

# Compare with expected fingerprint
CURRENT=$(cat fingerprint.json | jq -r .dataset_id)
EXPECTED=$(cat .fingerprint_lock)

if [ "$CURRENT" != "$EXPECTED" ]; then
  echo "❌ Dataset changed! Update .fingerprint_lock"
  exit 1
fi

echo "✅ Dataset verified"
```

### 2. Training Script Integration

```python
from imgshape import Atlas

# In your training pipeline
atlas = Atlas()
analysis = atlas.analyze("data/train", task="classification")

# Use recommendations
model = create_model(
    architecture=analysis.decisions['model_family'].selected,
    input_size=analysis.decisions['input_dimensions'].selected
)

augmentation = get_augmentation_pipeline(
    analysis.decisions['augmentation'].selected
)

print(f"Fingerprint: {analysis.fingerprint.dataset_uri}")
print(f"Model: {model.__class__.__name__}")
```

### 3. Manual Inspection

```bash
# Generate comprehensive report
imgshape --atlas data/train \
  --task detection \
  --deployment gpu \
  --priority accuracy \
  --report analysis_report.md

# View decisions
imgshape --decisions data/train \
  --output decisions.json \
  --verbose
```

---

## 🔌 Plugin System

Extend imgshape with custom fingerprint extractors and decision rules.

```python
# plugins/medical_profiler.py
from imgshape.plugins import FingerprintPlugin

class MedicalProfiler(FingerprintPlugin):
    """Extract DICOM-specific attributes"""
    
    NAME = "medical_profiler_v1"
    
    def extract(self, dataset_path):
        # Custom logic for medical imaging
        return {
            "modality": "CT",
            "bit_depth": 16,
            "is_3d": True
        }
```

Register and use:

```bash
imgshape --plugin-add plugins/medical_profiler.py
imgshape --fingerprint medical_data/ --plugin medical_profiler_v1
```

---

## 📦 Installation Options

```bash
# Core (minimal dependencies)
pip install imgshape

# With PyTorch support
pip install "imgshape[torch]"

# With web UI (FastAPI + React)
pip install "imgshape[web]"

# With all features
pip install "imgshape[full]"

# Development (with testing tools)
pip install "imgshape[dev]"
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install dev dependencies
pip install "imgshape[dev]"

# Run all tests
pytest tests/ -v

# Run v4 specific tests
pytest tests/test_fingerprint.py tests/test_decision_engine.py -v

# Coverage
pytest --cov=imgshape tests/
```

Expected output: **26/33 passing** (7 optional artifact tests)

---

## 🌐 Web Service

Deploy imgshape as a REST API:

```bash
# Start the service
imgshape --web

# API Endpoints (v4)
# POST /v4/fingerprint    - Get dataset fingerprint
# POST /v4/decisions      - Get decisions for a dataset
# POST /v4/analyze        - Full analysis
# GET  /health            - Service health

# Legacy Endpoints (v3)
# POST /analyze           - v3 analyze
# POST /recommend         - v3 recommendations
```

### Docker Deployment

```bash
# Build
docker build -t imgshape:4.0.0 .

# Run
docker run -p 8080:8080 imgshape:4.0.0

# Cloud Run
gcloud run deploy imgshape --image gcr.io/your-project/imgshape:4.0.0
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run `pytest tests/` to verify
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📚 Additional Resources

- [Full Documentation](https://stifler7.github.io/imgshape)
- [v4 Atlas Design Document](https://github.com/STiFLeR7/imgshape/blob/master/v4.md)
- [API Reference](https://stifler7.github.io/imgshape/api)
- [Contributing Guide](CONTRIBUTING.md)
- [Issues & Feature Requests](https://github.com/STiFLeR7/imgshape/issues)

---

## 📄 License

imgshape is released under the [MIT License](LICENSE). See LICENSE file for details.

---

<div align="center">

**Built with 💜 by [Stifler](https://github.com/STiFLeR7)**

*Making dataset intelligence accessible to everyone.*

If you find imgshape useful, please consider:
- ⭐ Starring this repository
- 📢 Sharing with your colleagues
- 🐛 Reporting issues and suggesting features
- 🤝 Contributing code or documentation

</div>
