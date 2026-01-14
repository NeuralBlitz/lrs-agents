"""
Core mathematical components for Active Inference.

This module implements:
- Bayesian precision tracking
- Expected Free Energy calculation
- Tool abstraction (ToolLens)
- Tool registry with fallback chains
"""

from lrs.core.precision import PrecisionParameters, HierarchicalPrecision
from lrs.core.free_energy import (
    calculate_expected_free_energy,
    calculate_epistemic_value,
    calculate_pragmatic_value,
    precision_weighted_selection,
    PolicyEvaluation,
)
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.registry import ToolRegistry

__all__ = [
    "PrecisionParameters",
    "HierarchicalPrecision",
    "calculate_expected_free_energy",
    "calculate_epistemic_value",
    "calculate_pragmatic_value",
    "precision_weighted_selection",
    "PolicyEvaluation",
    "ToolLens",
    "ExecutionResult",
    "ToolRegistry",
]
