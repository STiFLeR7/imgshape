# setup.py — imgshape v4.0.0 (Atlas)
from setuptools import setup, find_packages

# read long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="imgshape",
    version="4.0.0",
    description=(
        "imgshape v4.0.0 (Atlas) — Dataset intelligence layer: "
        "deterministic fingerprinting and decision-making for ML pipelines. "
        "FastAPI web service, plugin system, and Atlas orchestrator."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stifler",
    author_email="hillaniljppatel@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.19.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        # optional feature groups
        "torch": [
            "torch>=1.12.0; platform_system != 'Windows' or python_version >= '3.8'",
            "torchvision>=0.13.0",
        ],
        "pdf": ["weasyprint>=53.0", "reportlab>=3.6.0", "pyyaml>=6.0"],
        "viz": ["plotly>=5.20.0", "seaborn>=0.12.0"],
        "web": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0", "jinja2>=3.1.0"],
        "plugins": ["importlib-metadata>=6.0", "types-Pillow>=9.0"],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=3.9",
            "pre-commit>=2.20",
            "mypy>=1.0",
            "build>=1.2",
            "twine>=4.0",
        ],
        "full": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "weasyprint>=53.0",
            "reportlab>=3.6.0",
            "pyyaml>=6.0",
            "plotly>=5.20.0",
            "seaborn>=0.12.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "jinja2>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imgshape=imgshape.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Framework :: FastAPI",
        "Development Status :: 5 - Production/Stable",
    ],
    python_requires=">=3.8",
    keywords=(
        "image-analysis dataset-analytics computer-vision fastapi atlas "
        "augmentation preprocessing pytorch pipeline fingerprinting decision-engine"
    ),
    url="https://github.com/STiFLeR7/imgshape",
    project_urls={
        "Homepage": "https://github.com/STiFLeR7/imgshape",
        "Source": "https://github.com/STiFLeR7/imgshape",
        "Issues": "https://github.com/STiFLeR7/imgshape/issues",
        "Documentation": "https://stifler7.github.io/imgshape/",
    },
)
