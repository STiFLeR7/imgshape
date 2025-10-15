# setup.py — imgshape v3.0.0 (Aurora)
from setuptools import setup, find_packages

# read long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="imgshape",
    version="3.0.0",
    description=(
        "imgshape — dataset intelligence for vision pipelines. "
        "Analyze, recommend, visualize, and export augmentation & preprocessing pipelines "
        "with Streamlit UI, plugin system, and lazy import architecture."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stifler",
    author_email="stiflerxd.ai@cudabit.live",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.19.0",
        "streamlit>=1.33.0",  # required for --web UI
    ],
    extras_require={
        # optional feature groups
        "torch": [
            "torch>=1.12.0; platform_system != 'Windows' or python_version >= '3.8'",
            "torchvision>=0.13.0",
        ],
        "pdf": ["weasyprint>=53.0", "reportlab>=3.6.0", "pyyaml>=6.0"],
        "viz": ["plotly>=5.20.0", "seaborn>=0.12.0"],
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
        "ui": ["streamlit>=1.33.0", "pyyaml>=6.0"],
        "full": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "weasyprint>=53.0",
            "reportlab>=3.6.0",
            "pyyaml>=6.0",
            "plotly>=5.20.0",
            "seaborn>=0.12.0",
            "streamlit>=1.33.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imgshape=imgshape.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Framework :: Streamlit",
        "Development Status :: 5 - Production/Stable",
    ],
    python_requires=">=3.8",
    keywords=(
        "image-analysis dataset-analytics computer-vision streamlit "
        "augmentation preprocessing pytorch pipeline edge-ai visualization"
    ),
    url="https://github.com/STiFLeR7/imgshape",
    project_urls={
        "Homepage": "https://github.com/STiFLeR7/imgshape",
        "Source": "https://github.com/STiFLeR7/imgshape",
        "Issues": "https://github.com/STiFLeR7/imgshape/issues",
        "Documentation": "https://stifler7.github.io/imgshape/",
    },
)
