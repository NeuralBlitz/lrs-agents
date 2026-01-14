"""LRS-Agents: Resilient AI agents via Active Inference."""

__version__ = "0.2.0"

# Don't import anything at package level to avoid circular imports
# Users should import from submodules:
#   from lrs.integration.langgraph import create_lrs_agent
#   from lrs.core import PrecisionParameters

__all__ = [
    "__version__",
]
