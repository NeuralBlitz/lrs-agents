"""Multi-agent coordination for LRS systems."""

from lrs.multi_agent.shared_state import SharedWorldState
from lrs.multi_agent.social_precision import (
    SocialPrecisionParameters,
    SocialPrecisionTracker,
)
from lrs.multi_agent.communication import (
    Message,
    MessageType,
    CommunicationLens,
)
from lrs.multi_agent.coordinator import MultiAgentCoordinator
from lrs.multi_agent.multi_agent_free_energy import (
    calculate_total_free_energy,
    should_communicate,
)

__all__ = [
    "SharedWorldState",
    "SocialPrecisionParameters",
    "SocialPrecisionTracker",
    "Message",
    "MessageType",
    "CommunicationLens",
    "MultiAgentCoordinator",
    "calculate_total_free_energy",
    "should_communicate",
]
