# Contributing to imgshape

Thank you for your interest in contributing to **imgshape**! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more. As of version 5.0.0 "Covenant", imgshape is a pure Python library and command-line tool. All Node.js/TypeScript frontend UI code has been permanently removed.

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
1. **Check existing issues** to avoid duplicates.
2. **Use the latest version** to verify the bug still exists.
3. **Gather information**: error messages, stack traces, OS, Python version, imgshape version.

**Submit a bug report:**
- Use a clear, descriptive title.
- Describe the exact steps to reproduce.
- Include expected vs. actual behavior.
- Provide code samples, logs, or screenshots.
- Mention your environment (OS, Python version, imgshape version).

**Example:**
```markdown
**Bug:** imgshape validate crashes with JSONDecodeError on corrupted .fingerprint_lock

**Environment:**
- imgshape version: 5.0.0
- Python: 3.12.0
- OS: Windows 11

**Steps to reproduce:**
1. Create a corrupted/invalid `.fingerprint_lock` file next to a contract.
2. Run: `imgshape validate datasets/ contract.yaml`
3. Crash with json.decoder.JSONDecodeError

**Expected:** Prints a friendly validation warning and exits gracefully.
**Actual:** Stack trace with JSONDecodeError.
```

### Suggesting Features

We love new ideas! When suggesting a feature:
1. **Search existing issues** to see if it's already proposed.
2. **Explain the use case** and why it's valuable.
3. **Provide examples** of how it would work.
4. **Consider alternatives** and mention them.

---

## 🛠️ Development Setup

### Prerequisites

- **Python 3.8+** (Python 3.10-3.12 recommended)
- **Git** for version control
- **pip** for package management

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/STiFLeR7/imgshape.git
cd imgshape

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies and extras in editable mode
pip install -e ".[dev,full]"

# Verify the CLI installation
imgshape --help

# Run the test suite
python -m pytest
```

---

## 📁 Project Structure

```
imgshape/
├── src/imgshape/           # Core Python package
│   ├── __init__.py         # Public package exports
│   ├── __main__.py         # Entrypoint
│   ├── atlas/              # Atlas engine (orchestrator and profilers)
│   ├── audit/              # Provenance & dataset fingerprint locking
│   ├── cli/                # Click CLI subcommands (validate, test, diff, fingerprint)
│   ├── contract/           # YAML contract loading, JSON schema validation, clause checking
│   ├── diff/               # Drift statistics and similarity comparison engines
│   ├── output/             # Output formatting (text, JSON, markdown reports)
│   ├── plugins/            # Extensible custom profiler hooks
│   ├── profiles/           # Domain-specific validation presets
│   ├── schemas/            # Validation JSON schemas for configs and contracts
│   └── test/               # pytest-like dataset test assertions
│
├── tests/                  # Test suite (v5 specifications & legacy suites)
│   ├── test_atlas_v5.py    # Atlas profiler tests
│   ├── test_contract_v5.py # YAML contract loaders & validator tests
│   ├── test_v5_cli.py      # CLI subcommand tests
│   └── test_v5_more.py     # Contradictions, custom severities, and edge cases
│
├── assets/                 # Documentation assets
├── requirements.txt        # Core package requirements (lean)
├── pyproject.toml          # Modern setuptools packaging and dependencies
└── setup.py                # Legacy setuptools wrapper (synchronized with pyproject.toml)
```

---

## 💻 Coding Standards

### Python Style

We follow **PEP 8** with these specific configurations:

- **Line length:** 100 characters
- **Quotes:** Prefer double quotes `"` over single quotes `'`
- **Imports:** Use absolute imports
- **Type hints:** Required for all public functions/methods
- **Docstrings:** Google style docstrings for public API items

**Example:**
```python
def validate_contract(
    dataset_path: Path,
    contract_path: Path,
    verbose: bool = False
) -> ValidationReport:
    """
    Validate a dataset directory against a YAML contract.

    Args:
        dataset_path: Path to the vision dataset directory.
        contract_path: Path to the contract file.
        verbose: If True, prints verbose validation logs.

    Returns:
        ValidationReport containing verdict, metadata, and violations list.

    Raises:
        FileNotFoundError: If the dataset or contract path does not exist.
    """
    # Implementation...
    pass
```

### Formatting & Linting

We use Black and Flake8 to maintain code cleanliness:

```bash
# Format your code before submitting
black src/ tests/

# Run static analysis
flake8 src/ tests/ --max-line-length=100
```

---

## 🧪 Testing Guidelines

### Running Tests

We use `pytest` as our primary testing framework:

```bash
# Run all tests (automatically skips GPU tests if CUDA/Torch is missing)
python -m pytest

# Run only v5 specific tests with verbose logs
python -m pytest tests/test_atlas_v5.py tests/test_contract_v5.py -v

# Run with coverage report
python -m pytest --cov=imgshape tests/
```

### Writing Tests

All new features or bug fixes **must** be accompanied by tests:
- **Unit tests** for contract loaders, profilers, and diff engines.
- **CLI integration tests** invoking Click `CliRunner`.
- Use `pytest` fixtures and mock directory generation to keep tests hermetic and deterministic.

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update from main:** `git pull origin main --rebase`
2. **Ensure all tests pass:** `python -m pytest`
3. **Format your code:** `black src/ tests/`
4. **Lint your code:** `flake8 src/ tests/ --max-line-length=100`
5. **Update documentation** or YAML schemas if you introduced new features or contract clauses.

### PR Description Template

Include a brief summary of:
- **What is the goal of this change?**
- **How was it accomplished?**
- **How was it tested?**
- **Any breaking changes?**

---

## 🚀 Release Process

Releases are managed by the core maintainers:

1. Versioning follows [Semantic Versioning 2.0.0](https://semver.org/).
2. Synchronize version bumps across [pyproject.toml](file:///D:/imgshape/pyproject.toml) and [setup.py](file:///D:/imgshape/setup.py).
3. Document all fixes and improvements in [CHANGELOG.md](file:///D:/imgshape/CHANGELOG.md).
4. Tag release commits and push to GitHub.
5. Build and upload packages to PyPI via:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

---

## 🙋 Getting Help

- **Questions & Ideas:** Join the conversation on [GitHub Discussions](https://github.com/STiFLeR7/imgshape/discussions).
- **Bug Reports:** Open an issue on [GitHub Issues](https://github.com/STiFLeR7/imgshape/issues).

<div align="center">

**Thank you for contributing to imgshape!** 💜

*Making dataset governance accessible and reliable for every machine learning engineer.*

</div>
