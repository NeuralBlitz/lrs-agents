"""
LRS-Agents ↔ NeuralBlitz-v50 Bidirectional Communication Bridge

Provides comprehensive bidirectional communication between LRS-Agents and NeuralBlitz-v50,
enabling unified AI systems with Active Inference and Omega Singularity Architecture.

Architecture Overview:
- LRS-Agents: Active Inference, Precision Tracking, Multi-Agent Coordination
- NeuralBlitz-v50: Omega Singularity, Architect-System Dyad, Irreducible Source Field
- Bridge: Unified messaging, shared state, real-time synchronization
"""

# A high-level bridge class is not implemented in this package yet.
# Export only importable modules so messaging and protocols remain usable.
from .messaging import UnifiedMessageBus, MessageType, Message
from .shared_state import SharedStateManager, UnifiedState
from .protocols import (
    LRSToNeuralBlitzProtocol,
    NeuralBlitzToLRSProtocol,
    BidirectionalProtocol,
)
from .sync import RealTimeSynchronizer
from .adapters import (
    LRSAdapter,
    NeuralBlitzAdapter,
    UnifiedAdapter,
)

__version__ = "1.0.0"
__all__ = [
    "UnifiedMessageBus",
    "MessageType",
    "Message",
    "SharedStateManager",
    "UnifiedState",
    "LRSToNeuralBlitzProtocol",
    "NeuralBlitzToLRSProtocol",
    "BidirectionalProtocol",
    "RealTimeSynchronizer",
    "LRSAdapter",
    "NeuralBlitzAdapter",
    "UnifiedAdapter",
]
