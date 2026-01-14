"""
LRS-Agents: Resilient AI agents via Active Inference.
"""

__version__ = "0.2.0"

# Import only what's needed at package level to avoid circular imports
# Let users import specific modules as needed

# Core components available at package level
from lrs.integration.langgraph import create_lrs_agent

__all__ = [
    "create_lrs_agent",
    "__version__",
]
