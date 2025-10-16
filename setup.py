#!/usr/bin/env python3
"""
Setup script for SPEED (Scalable Preprocessing of EEG Data For Self-Supervised Learning)
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements

setup(
    name="speed",
    version="0.1.0",
    author="Anders Gjølbye, Lina Skerath, William Lehn-Schiøler, Nicolas Langer, Lars Kai Hansen",
    author_email="",  # Add email if available
    description="Scalable Preprocessing of EEG Data For Self-Supervised Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/AndersGMadsen/SPEED",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "speed-preprocess=speed.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "speed": ["configs/*.yaml"],
    },
    keywords="eeg, preprocessing, machine learning, self-supervised learning, neuroscience",
    project_urls={
        "Bug Reports": "https://github.com/AndersGMadsen/SPEED/issues",
        "Source": "https://github.com/AndersGMadsen/SPEED",
        "Paper": "https://arxiv.org/abs/2408.08065",
    },
)
