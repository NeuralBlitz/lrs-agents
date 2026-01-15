"""Core LRS-Agents components."""

from lrs.core.precision import PrecisionParameters, HierarchicalPrecision
from lrs.core.lens import ToolLens, ExecutionResult, ComposedLens
from lrs.core.registry import ToolRegistry

# Free energy imports - only import what exists
from lrs.core.free_energy import (
    calculate_epistemic_value,
    calculate_pragmatic_value,
    calculate_expected_free_energy,
    # evaluate_policy,  # Comment out if doesn't exist
    precision_weighted_selection,
    PolicyEvaluation,
)

__all__ = [
    "PrecisionParameters",
    "HierarchicalPrecision",
    "ToolLens",
    "ExecutionResult",
    "ComposedLens",
    "ToolRegistry",
    "calculate_epistemic_value",
    "calculate_pragmatic_value",
    "calculate_expected_free_energy",
    # "evaluate_policy",  # Comment out
    "precision_weighted_selection",
    "PolicyEvaluation",
]
