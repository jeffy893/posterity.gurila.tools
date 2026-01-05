#!/usr/bin/env python3
"""
Setup script for posterity.gurila.tools

Copyright (C) 2026 Jefferson Richards <jefferson@richards.plus>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="posterity-gurila-tools",
    version="1.0.0",
    author="Jefferson Richards",
    author_email="jefferson@richards.plus",
    description="Tactical simulation engine using Lanchester Laws and Markov Chain dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jefferson-richards/posterity.gurila.tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest-cov", "black", "flake8", "mypy"],
        "ar": ["opencv-python", "tensorflow"],  # Future AR integration
    },
    entry_points={
        "console_scripts": [
            "posterity-sim=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)