"""LRS-Agents: Resilient AI agents via Active Inference."""

__version__ = "0.2.1"

__all__ = ["__version__"]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "create_lrs_agent":
        from lrs.integration.langgraph import create_lrs_agent
        return create_lrs_agent
    raise AttributeError(f"module 'lrs' has no attribute '{name}'")


# Optional: Show banner on first import (disabled by default)
# Users can enable with: export LRS_SHOW_BANNER=1
import os
if os.getenv("LRS_SHOW_BANNER") == "1":
    from lrs.cli import show_banner
    show_banner(compact=True)
