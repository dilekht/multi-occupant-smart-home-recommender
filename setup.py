"""
Setup script for Multi-Occupant Smart Home Recommender System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="multi-occupant-smart-home",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Multi-occupant context-aware smart home recommender system with conflict resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-occupant-smart-home-recommender",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/multi-occupant-smart-home-recommender/issues",
        "Documentation": "https://github.com/yourusername/multi-occupant-smart-home-recommender/docs",
        "Paper": "https://doi.org/XXXX",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Home Automation",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "mlxtend>=0.22.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-home-preprocess=experiments.run_preprocessing:main",
            "smart-home-train=experiments.run_glm_experiment:main",
            "smart-home-demo=experiments.run_recommendation_system:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "smart home",
        "activity recognition",
        "multi-occupant",
        "recommendation system",
        "conflict resolution",
        "FP-Growth",
        "machine learning",
        "IoT",
    ],
)
