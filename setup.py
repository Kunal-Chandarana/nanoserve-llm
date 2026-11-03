#!/usr/bin/env python3
"""
Setup script for vLLM Learning Project

This script provides easy installation and setup for the vLLM learning project.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nanoserve-llm",
    version="1.0.0",
    author="Kunal Chandarana",
    description="LLM inference engine built from scratch, benchmarked against vLLM",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Kunal-Chandarana/nanoserve-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "ui": [
            "gradio>=4.0.0",
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vllm-test=scripts.test_installation:main",
            "vllm-download=scripts.download_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Kunal-Chandarana/nanoserve-llm/issues",
        "Source": "https://github.com/Kunal-Chandarana/nanoserve-llm",
        "Documentation": "https://github.com/Kunal-Chandarana/nanoserve-llm/blob/main/README.md",
    },
)

