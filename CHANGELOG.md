## [2.2.0] - 2025-09-22
### 🚀 Major Release
- Full **Streamlit App** (`app.py`) introduced with 5 interactive tabs:  
  - **Shape** → instant image shape detection  
  - **Analyze** → entropy, channels, and dataset visualization  
  - **Recommend** → preprocessing + heuristic augmentation plan  
  - **Report** → export dataset reports in Markdown/HTML  
  - **TorchLoader** → export ready-to-use `torchvision.transforms` pipelines or code snippets  
- CLI, Streamlit, and test suite aligned for consistent workflows.

### ✨ Features
- **TorchLoader:**  
  - Defensive wrapper: returns Compose, snippet, or safe fallback.  
  - Works with real torchvision, fake monkeypatches (tests), or no torchvision.  
  - Backward compatibility for `(plan, preprocessing)` and `(config, recommendation)` calling styles.
- **AugmentationRecommender:**  
  - Deterministic heuristic-based augmentation plan (`as_dict()` export).  
  - Handles entropy, resolution, and class imbalance heuristics.  
- **Compatibility Checker:**  
  - `check_compatibility()` with structured dataset summary + recommendations.  
  - Backwards-compatible alias `check_model_compatibility()` preserved.
- **Report Generators:**  
  - Markdown, HTML, and PDF outputs with dataset summary + preprocessing + augmentations.  
- **Streamlit UX improvements:**  
  - Defensive wrappers for `analyze_type`, `recommend_preprocessing`, and TorchLoader.  
  - Sidebar inputs (dataset path, user prefs).  
  - Footer with links to social profiles.

### 🐛 Fixes
- Fixed multiple `pytest` failures across `compatibility`, `report`, and `torchloader`.  
- Defensive handling for missing datasets / bad paths.  
- Removed deprecated `use_column_width` → replaced with `use_container_width`.  
- Fixed augmentation serialization (`plan.as_dict()` instead of raw dataclass).

### 🧪 Testing
- All tests passing locally (`pytest -q`).  
- Backwards compatibility with old CLI and test suites maintained.  

### 🔗 Footer Links
- Added **Instagram, GitHub, HuggingFace, Kaggle, Medium** in Streamlit footer.

---

## [2.1.x] - 2025-06 → 2025-09
- Incremental fixes for augmentation, compatibility API, and report stability.
- Early Streamlit prototype introduced.
- Partial pytest alignment.

## [2.0.0] - 2025-04
- Major refactor: modularization of analyze, recommender, augmentations, and CLI.
- Added initial CI test suite.

## [1.x.x] - 2025
- Initial releases: basic CLI shape detection + dataset analysis.

---

## Next (Planned for 2.3.x)
- ONNX/TensorRT export helpers for edge inference.
- Auto-EDA previews (histograms, per-class imbalance plots).
- Dockerfile + HuggingFace Spaces demo.
- CI/CD pipeline (pytest, lint, build checks).
