from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="imgshape",
    version="2.0.1",
    description="Smart image shape & analysis tool for ML workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stifler",
    author_email="stiflerxd.ai@cudabit.live",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Pillow",
        "matplotlib",
        "seaborn",
        "gradio",
        "opencv-python",
        "numpy",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "imgshape=imgshape.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
