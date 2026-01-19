"""Monitoring and visualization for LRS agents."""

from lrs.monitoring.tracker import LRSStateTracker, StateSnapshot
from lrs.monitoring.structured_logging import LRSLogger, create_logger_for_agent

# Make dashboard imports optional (requires streamlit)
try:
    from lrs.monitoring.dashboard import create_dashboard, run_dashboard
    _has_dashboard = True
except ImportError:
    _has_dashboard = False
    create_dashboard = None
    run_dashboard = None

__all__ = [
    "LRSStateTracker",
    "StateSnapshot",
    "LRSLogger",
    "create_logger_for_agent",
]

if _has_dashboard:
    __all__.extend(["create_dashboard", "run_dashboard"])
