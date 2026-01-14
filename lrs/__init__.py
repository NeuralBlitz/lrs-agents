"""
LRS-Agents: Active Inference for Adaptive AI

LRS (Lambda-Reflexive Synthesis) is a framework for building adaptive AI agents
using Active Inference from neuroscience.

Key components:
- Precision tracking (Bayesian confidence)
- Expected Free Energy calculation
- Automatic exploration-exploitation balance
- Tool composition via categorical morphisms

Examples:
    >>> from lrs import create_lrs_agent
    >>> from langchain_anthropic import ChatAnthropic
    >>> 
    >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    >>> agent = create_lrs_agent(llm, tools=[...])
    >>> 
    >>> result = agent.invoke({"messages": [{"role": "user", "content": "Task"}]})
"""

from lrs.integration.langgraph import create_lrs_agent, LRSGraphBuilder
from lrs.core.precision import PrecisionParameters, HierarchicalPrecision
from lrs.core.free_energy import (
    calculate_expected_free_energy,
    calculate_epistemic_value,
    calculate_pragmatic_value,
    precision_weighted_selection,
)
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.registry import ToolRegistry

__version__ = "0.2.0"
__author__ = "LRS Contributors"
__license__ = "MIT"

__all__ = [
    # Main entry point
    "create_lrs_agent",
    "LRSGraphBuilder",
    # Core components
    "PrecisionParameters",
    "HierarchicalPrecision",
    "calculate_expected_free_energy",
    "calculate_epistemic_value",
    "calculate_pragmatic_value",
    "precision_weighted_selection",
    "ToolLens",
    "ExecutionResult",
    "ToolRegistry",
]
