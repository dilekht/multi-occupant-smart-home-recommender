"""
Multi-Occupant Smart Home Recommender System
=============================================

A comprehensive framework for multi-occupant context-aware smart home
recommendations with conflict detection and resolution.

Modules:
    preprocessing: Data preprocessing and feature engineering
    pattern_mining: FP-Growth multi-resident pattern mining
    models: Extended GLM models for activity prediction
    conflict_resolution: Conflict detection and resolution strategies

Example:
    >>> from src.conflict_resolution import SmartHomeRecommendationEngine
    >>> engine = SmartHomeRecommendationEngine()
    >>> result = engine.recommend_from_activities(11, 12)  # Sleep vs TV
    >>> print(result['resolution']['strategy'])
    'device'

Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@university.edu"

from .preprocessing import ARASPreprocessor
from .pattern_mining import MultiResidentPatternMiner
from .models import MultiResidentGLM
from .conflict_resolution import (
    ConflictResolver,
    SmartHomeRecommendationEngine,
    ConflictType,
    ResolutionStrategy,
)

__all__ = [
    "ARASPreprocessor",
    "MultiResidentPatternMiner", 
    "MultiResidentGLM",
    "ConflictResolver",
    "SmartHomeRecommendationEngine",
    "ConflictType",
    "ResolutionStrategy",
]
