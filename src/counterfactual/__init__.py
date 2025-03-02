"""
Counterfactual reasoning module for analyzing AI/ML implementation failures.

This package provides tools for:
1. Causal analysis: Identifying critical decision points in implementation
2. Structured comparison: Systematically comparing actual vs. alternative decisions
3. Pattern recognition: Identifying recurring failure patterns across implementations
4. Recommendation generation: Creating actionable recommendations based on analysis

The counterfactual reasoning approach helps identify "what could have been done differently"
to improve AI/ML implementations.
"""

from .causal_analysis import CausalAnalyzer, FailureCase, DecisionPoint
from .comparison import CounterfactualComparator, StructuredComparison, ComparisonDimension, ComparisonResult
from .pattern_recognition import PatternRecognizer, FailurePattern
from .recommendation import RecommendationGenerator, Recommendation
from .counterfactual_generator import CounterfactualGenerator

__all__ = [
    'CausalAnalyzer',
    'FailureCase',
    'DecisionPoint',
    'CounterfactualComparator',
    'StructuredComparison',
    'ComparisonDimension',
    'ComparisonResult',
    'PatternRecognizer',
    'FailurePattern',
    'RecommendationGenerator',
    'Recommendation',
    'CounterfactualGenerator'
]
