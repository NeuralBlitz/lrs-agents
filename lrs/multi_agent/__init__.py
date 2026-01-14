"""
Multi-agent coordination for LRS-Agents.

This module provides:
- Social precision tracking (trust in other agents)
- Communication as information-seeking actions
- Shared world state
- Recursive theory-of-mind
- Multi-agent coordinator
"""

from lrs.multi_agent.social_precision import (
    SocialPrecisionTracker,
    SocialPrecisionParameters
)
from lrs.multi_agent.shared_state import SharedWorldState
from lrs.multi_agent.communication import (
    CommunicationLens,
    Message,
    MessageType
)
from lrs.multi_agent.multi_agent_free_energy import (
    calculate_social_free_energy,
    calculate_total_free_energy
)
from lrs.multi_agent.coordinator import MultiAgentCoordinator

__all__ = [
    "SocialPrecisionTracker",
    "SocialPrecisionParameters",
    "SharedWorldState",
    "CommunicationLens",
    "Message",
    "MessageType",
    "calculate_social_free_energy",
    "calculate_total_free_energy",
    "MultiAgentCoordinator",
]
