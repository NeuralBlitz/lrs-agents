"""
Monitoring and observability for LRS-Agents.

This module provides:
- State tracking for history and analysis
- Real-time Streamlit dashboard
- Structured JSON logging for production
"""

from lrs.monitoring.tracker import LRSStateTracker
from lrs.monitoring.dashboard import create_dashboard, run_dashboard
from lrs.monitoring.structured_logging import LRSLogger, create_logger_for_agent

__all__ = [
    "LRSStateTracker",
    "create_dashboard",
    "run_dashboard",
    "LRSLogger",
    "create_logger_for_agent",
]
