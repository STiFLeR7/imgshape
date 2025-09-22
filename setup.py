from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="imgshape",
    version="2.2.0",
    description=(
        "imgshape — robust image shape detection, dataset analysis, preprocessing & "
        "augmentation recommendations, visualization, compatibility reports, and PyTorch loader helpers"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stifler",
    author_email="stiflerxd.ai@cudabit.live",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.19.0",
        "gradio>=3.50.0",  # bumped for modern Blocks/Tabs APIs
    ],
    extras_require={
        "torch": [
            "torch>=1.12.0; platform_system != 'Windows' or python_version >= '3.8'",
            "torchvision>=0.13.0",
        ],
        "pdf": ["weasyprint>=53.0"],
        "viz": ["seaborn>=0.11.0"],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=3.9",
            "pre-commit>=2.20",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imgshape=imgshape.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8",
    keywords="images dataset visualization augmentation pytorch machine-learning edge-ai",
    url="https://github.com/STiFLeR7/imgshape",
    project_urls={
        "Homepage": "https://github.com/STiFLeR7/imgshape",
        "Source": "https://github.com/STiFLeR7/imgshape",
        "Issues": "https://github.com/STiFLeR7/imgshape/issues",
        "Documentation": "https://stifler7.github.io/imgshape/",
    },
)
