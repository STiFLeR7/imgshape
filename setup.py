from setuptools import setup, find_packages
import pathlib

# Get long description from README.md
this_dir = pathlib.Path(__file__).parent
long_description = (this_dir / "README.md").read_text()

setup(
    name="imgshape",
    version="0.1.1",
    description="Resize image folders fast for AI/ML workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stifler",
    author_email="hillaniljppatel@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["Pillow"],
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
