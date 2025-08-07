from setuptools import setup, find_packages

setup(
    name="imgshape",
    version="2.0.0",
    description="🖼️ Get image shapes, analyze type, recommend preprocessing, and check model compatibility.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Stifler",
    author_email="hillaniljppatel@gmail.com",
    url="https://github.com/STiFLeR7/imgshape",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-image>=0.19.0",
        "gradio>=3.0.0"
    ],
    entry_points={
        "console_scripts": [
            "imgshape=imgshape.cli:main"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
)
