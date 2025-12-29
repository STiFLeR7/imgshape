# Contributing to imgshape

Thank you for your interest in contributing to **imgshape**! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## 📜 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful, inclusive, and constructive in all interactions.

**Key Principles:**
- Be welcoming and inclusive
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🤝 How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:
1. **Check existing issues** to avoid duplicates
2. **Use the latest version** to verify the bug still exists
3. **Gather information**: error messages, stack traces, OS, Python version

**Submit a bug report:**
- Use a clear, descriptive title
- Describe the exact steps to reproduce
- Include expected vs. actual behavior
- Provide code samples, logs, or screenshots
- Mention your environment (OS, Python version, imgshape version)

**Example:**
```markdown
**Bug:** v4.analyze() crashes with IndexError on empty folders

**Environment:**
- imgshape version: 4.0.0
- Python: 3.12.0
- OS: Ubuntu 22.04

**Steps to reproduce:**
1. Create empty folder: `mkdir empty_dataset`
2. Run: `imgshape --atlas empty_dataset`
3. Crash with IndexError

**Expected:** Error message indicating empty dataset
**Actual:** Stack trace with IndexError
```

### Suggesting Features

We love new ideas! When suggesting a feature:
1. **Search existing issues** to see if it's already proposed
2. **Explain the use case** and why it's valuable
3. **Provide examples** of how it would work
4. **Consider alternatives** and mention them

**Template:**
```markdown
**Feature Request:** Add support for video frame analysis

**Use Case:**
I work with video datasets and need to analyze frame distributions 
before training video classification models.

**Proposed Solution:**
- Add `--video` flag to CLI
- Extract frames at configurable FPS
- Generate fingerprint from sampled frames

**Alternatives:**
- Use ffmpeg to extract frames manually (current workaround)
- Add as a plugin instead of core feature
```

### Contributing Code

We welcome code contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** from `master`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

---

## 🛠️ Development Setup

### Prerequisites

- **Python 3.8+** (3.12 recommended)
- **Git** for version control
- **Node.js 24.x** (for UI development)
- **pip** or **poetry** for package management

### Clone and Install

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/imgshape.git
cd imgshape

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
imgshape --version
pytest tests/ -v
```

### UI Development Setup

```bash
# Navigate to UI directory
cd ui/

# Install dependencies
npm install

# Start development server
npm run dev
# Opens http://localhost:5173

# Build for production
npm run build
```

### Running the Backend Locally

```bash
# Start FastAPI server
python -m uvicorn service.app:app --reload --port 8000

# Or use the CLI
imgshape --web
```

---

## 📁 Project Structure

```
imgshape/
├── src/imgshape/           # Core Python package
│   ├── __init__.py         # Public API exports
│   ├── analyze.py          # v3 analysis (legacy)
│   ├── atlas.py            # v4 Atlas orchestrator
│   ├── fingerprint.py      # v4 fingerprinting engine
│   ├── decision.py         # v4 decision rules
│   ├── cli.py              # Command-line interface
│   ├── server.py           # Web server (legacy)
│   ├── plugins/            # Plugin system
│   ├── profiles/           # Domain profiles (YAML)
│   └── schemas/            # Pydantic models
│
├── service/                # FastAPI backend (production)
│   ├── app.py              # Main API endpoints
│   └── static/             # Static assets
│
├── ui/                     # React/Vite frontend
│   ├── src/                # React components
│   ├── services/           # API client
│   ├── public/             # Public assets
│   └── package.json        # Node dependencies
│
├── tests/                  # Test suite
│   ├── test_analyze.py     # v3 tests
│   ├── test_fingerprint.py # v4 fingerprint tests
│   └── test_decision.py    # v4 decision tests
│
├── assets/                 # Documentation assets
├── Dockerfile              # Container definition
├── cloudbuild.yaml         # Google Cloud Build
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Package metadata
└── setup.py                # Package setup
```

---

## 💻 Coding Standards

### Python Style

We follow **PEP 8** with some modifications:

- **Line length:** 100 characters (not 79)
- **Quotes:** Prefer double quotes `"` over single quotes
- **Imports:** Use absolute imports, group by standard/third-party/local
- **Type hints:** Required for all public functions (Python 3.8+ compatible)
- **Docstrings:** Google style for all public functions/classes

**Example:**

```python
from __future__ import annotations  # For Python 3.8-3.11 compatibility

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def analyze_image(
    image_path: Path,
    task: str = "classification",
    verbose: bool = False
) -> dict[str, any]:
    """
    Analyze a single image and return metrics.

    Args:
        image_path: Path to the image file.
        task: Analysis task type (classification, detection, etc.).
        verbose: Enable detailed logging.

    Returns:
        Dictionary containing analysis results with keys:
        - dimensions: Tuple of (width, height)
        - format: Image format (JPEG, PNG, etc.)
        - mode: Color mode (RGB, L, etc.)

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If task is not supported.

    Examples:
        >>> result = analyze_image(Path("test.jpg"))
        >>> print(result["dimensions"])
        (1920, 1080)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Implementation...
    pass
```

### TypeScript/React Style

- **ESLint:** Follow the provided ESLint configuration
- **Prettier:** Auto-format with Prettier
- **Components:** Functional components with TypeScript
- **Naming:** PascalCase for components, camelCase for functions/variables

**Example:**

```typescript
interface AnalysisResult {
  fingerprint: string;
  confidence: number;
  decisions: Decision[];
}

export const AnalysisCard: React.FC<{ result: AnalysisResult }> = ({ result }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="analysis-card">
      <h3>{result.fingerprint}</h3>
      <p>Confidence: {result.confidence.toFixed(2)}</p>
    </div>
  );
};
```

### Code Formatting

We use automated formatters:

```bash
# Python: Black
pip install black
black src/ tests/

# Python: isort (import sorting)
pip install isort
isort src/ tests/

# TypeScript: Prettier
npm run format
```

### Linting

```bash
# Python: flake8
flake8 src/ tests/ --max-line-length=100

# Python: mypy (type checking)
mypy src/ --strict

# TypeScript: ESLint
npm run lint
```

---

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fingerprint.py -v

# Run with coverage
pytest --cov=imgshape tests/

# Run specific test
pytest tests/test_analyze.py::test_analyze_folder -v
```

### Writing Tests

All new features **must** include tests:

- **Unit tests** for individual functions
- **Integration tests** for workflows
- **Fixtures** for reusable test data

**Example:**

```python
import pytest
from pathlib import Path
from imgshape import Atlas


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a temporary dataset for testing."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    
    # Create sample images
    for i in range(10):
        img_path = dataset_dir / f"image_{i}.jpg"
        img_path.write_bytes(b"fake image data")
    
    return dataset_dir


def test_atlas_analyze(sample_dataset):
    """Test Atlas.analyze() on a sample dataset."""
    atlas = Atlas()
    result = atlas.analyze(sample_dataset, task="classification")
    
    assert result.fingerprint is not None
    assert result.fingerprint.sample_count == 10
    assert "model_family" in result.decisions
    assert result.decisions["model_family"].confidence > 0.5


def test_atlas_fingerprint_stability(sample_dataset):
    """Test that fingerprints are deterministic."""
    atlas = Atlas()
    
    result1 = atlas.fingerprint(sample_dataset)
    result2 = atlas.fingerprint(sample_dataset)
    
    assert result1.dataset_id == result2.dataset_id
```

### Test Coverage Requirements

- **Minimum coverage:** 70% overall
- **Critical paths:** 90%+ coverage for core functions
- **New features:** 80%+ coverage required

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update from master:** `git pull origin master --rebase`
2. **Run tests:** `pytest tests/ -v`
3. **Check code style:** `black src/ && flake8 src/`
4. **Update documentation** if needed
5. **Add CHANGELOG entry** (if significant change)

### PR Guidelines

**Good PR title examples:**
- ✅ `feat: add video frame analysis support`
- ✅ `fix: handle empty datasets gracefully`
- ✅ `docs: update API reference for v4 endpoints`
- ✅ `refactor: simplify fingerprint calculation`

**PR description should include:**
```markdown
## Summary
Brief description of what this PR does.

## Changes
- Added video analysis module
- Updated CLI to support --video flag
- Added tests for video processing

## Testing
- Tested with 5 video files (MP4, AVI, MOV)
- All existing tests pass
- Added 10 new tests

## Related Issues
Closes #123
Related to #456
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one reviewer** approval required
3. **Address review comments** or discuss
4. **Maintainer merges** after approval

### After Merge

- Delete your feature branch
- Update your fork: `git pull origin master`
- Celebrate! 🎉

---

## 🚀 Release Process

Releases are managed by maintainers:

### Version Numbering

We follow **Semantic Versioning** (SemVer):
- **Major (X.0.0):** Breaking changes
- **Minor (4.X.0):** New features, backward compatible
- **Patch (4.0.X):** Bug fixes, backward compatible

### Creating a Release

```bash
# Update version in pyproject.toml and setup.py
# Update CHANGELOG.md

# Tag the release
git tag -a v4.1.0 -m "Release v4.1.0 - Added video analysis"
git push origin v4.1.0

# Build and publish
python -m build
twine upload dist/*
```

---

## 📝 Documentation

Documentation lives in:
- **README.md:** Quick start and overview
- **API_ENDPOINTS.md:** API reference
- **v4.md:** v4 Atlas design document
- **Docstrings:** Inline code documentation

When adding features:
1. **Update docstrings** in code
2. **Add examples** to README if user-facing
3. **Update API_ENDPOINTS.md** if adding endpoints
4. **Add migration guide** if breaking change

---

## 🙋 Getting Help

- **Questions:** Open a [GitHub Discussion](https://github.com/STiFLeR7/imgshape/discussions)
- **Bugs:** File an [Issue](https://github.com/STiFLeR7/imgshape/issues)
- **Chat:** Join our community (link TBD)

---

## 🎯 Good First Issues

Looking for a place to start? Check issues labeled:
- `good first issue` - Beginner-friendly
- `help wanted` - Community contributions welcome
- `documentation` - Improve docs

---

## 📜 License

By contributing to imgshape, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

<div align="center">

**Thank you for contributing to imgshape!** 💜

*Making dataset intelligence accessible to everyone.*

</div>
