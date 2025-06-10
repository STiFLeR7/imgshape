from setuptools import setup, find_packages

setup(
    name="imgshape",
    version="0.1.2",
    description="Resize image folders fast for AI/ML workflows",
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
