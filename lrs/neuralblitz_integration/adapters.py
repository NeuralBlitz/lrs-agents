"""
System Adapters for LRS-Agents ↔ NeuralBlitz-v50 Integration

Provides adapter classes to bridge LRS-Agents and NeuralBlitz-v50 systems,
handling data transformation, protocol translation, and interface compatibility.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

from .messaging import UnifiedMessageBus, Message, MessageType
from .shared_state import SharedStateManager, StateType
from .protocols import BidirectionalProtocol, ProtocolConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """LRS agent state representation."""

    agent_id: str
    beliefs: Dict[str, Any]
    policies: List[str]
    precision: Dict[str, float]
    free_energy: float
    active_policy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "beliefs": self.beliefs,
            "policies": self.policies,
            "precision": self.precision,
            "free_energy": self.free_energy,
            "active_policy": self.active_policy,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class NeuralBlitzState:
    """NeuralBlitz system state representation."""

    source_state: Dict[str, Any]
    intent_vector: Dict[str, Any]
    architect_dyad: Dict[str, Any]
    attestation: Optional[Dict[str, Any]] = None
    verification_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_state": self.source_state,
            "intent_vector": self.intent_vector,
            "architect_dyad": self.architect_dyad,
            "attestation": self.attestation,
            "verification_status": self.verification_status,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }


class LRSAdapter:
    """Adapter for LRS-Agents system integration."""

    def __init__(
        self, agent_id: str, message_bus: UnifiedMessageBus, state_manager: SharedStateManager
    ):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.state_manager = state_manager
        self.agent_states: Dict[str, AgentState] = {}
        self.precision_history: List[Dict[str, Any]] = []
        self.free_energy_history: List[float] = []

    async def register_agent(self, agent_state: AgentState) -> bool:
        """Register an LRS agent with the adapter."""
        try:
            self.agent_states[agent_state.agent_id] = agent_state

            # Store in shared state
            await self.state_manager.set_state(
                key=f"lrs_agent_{agent_state.agent_id}",
                value=agent_state.to_dict(),
                state_type=StateType.LRS_AGENT_STATE,
                source="lrs_adapter",
            )

            # Broadcast agent state
            message = Message(
                type=MessageType.LRS_AGENT_STATE,
                source=f"lrs_adapter_{self.agent_id}",
                payload=agent_state.to_dict(),
            )
            await self.message_bus.publish(message)

            logger.info(f"LRS agent {agent_state.agent_id} registered")
            return True
        except Exception as e:
            logger.error(f"Error registering LRS agent: {e}")
            return False

    async def update_precision(self, agent_id: str, precision_data: Dict[str, float]) -> bool:
        """Update agent precision and broadcast."""
        try:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].precision.update(precision_data)

            # Store precision update
            precision_record = {
                "agent_id": agent_id,
                "precision": precision_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.precision_history.append(precision_record)

            # Store in shared state
            await self.state_manager.set_state(
                key=f"lrs_precision_{agent_id}",
                value=precision_record,
                state_type=StateType.LRS_PRECISION,
                source="lrs_adapter",
            )

            # Broadcast precision update
            message = Message(
                type=MessageType.LRS_PRECISION_UPDATE,
                source=f"lrs_adapter_{self.agent_id}",
                payload=precision_record,
            )
            await self.message_bus.publish(message)

            logger.debug(f"Precision updated for agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating precision: {e}")
            return False

    async def update_free_energy(self, agent_id: str, free_energy: float) -> bool:
        """Update agent free energy and broadcast."""
        try:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].free_energy = free_energy

            # Store free energy
            self.free_energy_history.append(free_energy)

            # Store in shared state
            free_energy_data = {
                "agent_id": agent_id,
                "free_energy": free_energy,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.state_manager.set_state(
                key=f"lrs_free_energy_{agent_id}",
                value=free_energy_data,
                state_type=StateType.LRS_FREE_ENERGY,
                source="lrs_adapter",
            )

            # Broadcast free energy update
            message = Message(
                type=MessageType.LRS_FREE_ENERGY,
                source=f"lrs_adapter_{self.agent_id}",
                payload=free_energy_data,
            )
            await self.message_bus.publish(message)

            logger.debug(f"Free energy updated for agent {agent_id}: {free_energy}")
            return True
        except Exception as e:
            logger.error(f"Error updating free energy: {e}")
            return False

    async def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state."""
        return self.agent_states.get(agent_id)

    async def get_all_agents(self) -> List[AgentState]:
        """Get all registered agents."""
        return list(self.agent_states.values())

    def get_precision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get precision history."""
        return self.precision_history[-limit:]

    def get_free_energy_history(self, limit: int = 100) -> List[float]:
        """Get free energy history."""
        return self.free_energy_history[-limit:]


class NeuralBlitzAdapter:
    """Adapter for NeuralBlitz-v50 system integration."""

    def __init__(
        self, system_id: str, message_bus: UnifiedMessageBus, state_manager: SharedStateManager
    ):
        self.system_id = system_id
        self.message_bus = message_bus
        self.state_manager = state_manager
        self.current_state: Optional[NeuralBlitzState] = None
        self.attestation_history: List[Dict[str, Any]] = []

    async def initialize_source_state(self, source_state: Dict[str, Any]) -> bool:
        """Initialize NeuralBlitz source state."""
        try:
            if not self.current_state:
                self.current_state = NeuralBlitzState(
                    source_state=source_state, intent_vector={}, architect_dyad={}
                )
            else:
                self.current_state.source_state.update(source_state)

            # Store in shared state
            await self.state_manager.set_state(
                key=f"neuralblitz_source_{self.system_id}",
                value=source_state,
                state_type=StateType.NEURALBLITZ_SOURCE,
                source="neuralblitz_adapter",
            )

            # Broadcast source state
            message = Message(
                type=MessageType.NEURALBLITZ_SOURCE_STATE,
                source=f"neuralblitz_adapter_{self.system_id}",
                payload={"system_id": self.system_id, "source_state": source_state},
            )
            await self.message_bus.publish(message)

            logger.info(f"NeuralBlitz source state initialized for {self.system_id}")
            return True
        except Exception as e:
            logger.error(f"Error initializing source state: {e}")
            return False

    async def update_intent_vector(self, intent_vector: Dict[str, Any]) -> bool:
        """Update NeuralBlitz intent vector."""
        try:
            if self.current_state:
                self.current_state.intent_vector.update(intent_vector)
            else:
                self.current_state = NeuralBlitzState(
                    source_state={}, intent_vector=intent_vector, architect_dyad={}
                )

            # Store in shared state
            await self.state_manager.set_state(
                key=f"neuralblitz_intent_{self.system_id}",
                value=intent_vector,
                state_type=StateType.NEURALBLITZ_INTENT,
                source="neuralblitz_adapter",
            )

            # Broadcast intent vector
            message = Message(
                type=MessageType.NEURALBLITZ_INTENT_VECTOR,
                source=f"neuralblitz_adapter_{self.system_id}",
                payload={"system_id": self.system_id, "intent_vector": intent_vector},
            )
            await self.message_bus.publish(message)

            logger.debug(f"Intent vector updated for {self.system_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating intent vector: {e}")
            return False

    async def update_architect_dyad(self, architect_dyad: Dict[str, Any]) -> bool:
        """Update architect dyad information."""
        try:
            if self.current_state:
                self.current_state.architect_dyad.update(architect_dyad)
            else:
                self.current_state = NeuralBlitzState(
                    source_state={}, intent_vector={}, architect_dyad=architect_dyad
                )

            # Broadcast architect dyad
            message = Message(
                type=MessageType.NEURALBLITZ_ARCHITECT_DYAD,
                source=f"neuralblitz_adapter_{self.system_id}",
                payload={"system_id": self.system_id, "architect_dyad": architect_dyad},
            )
            await self.message_bus.publish(message)

            logger.debug(f"Architect dyad updated for {self.system_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating architect dyad: {e}")
            return False

    async def add_attestation(self, attestation: Dict[str, Any]) -> bool:
        """Add attestation information."""
        try:
            if self.current_state:
                self.current_state.attestation = attestation
                self.current_state.verification_status = "verified"

            # Store in shared state
            await self.state_manager.set_state(
                key=f"neuralblitz_attestation_{self.system_id}",
                value=attestation,
                state_type=StateType.NEURALBLITZ_ATTESTATION,
                source="neuralblitz_adapter",
            )

            # Add to history
            attestation_record = {
                "system_id": self.system_id,
                "attestation": attestation,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.attestation_history.append(attestation_record)

            # Broadcast attestation
            message = Message(
                type=MessageType.NEURALBLITZ_ATTESTATION,
                source=f"neuralblitz_adapter_{self.system_id}",
                payload=attestation_record,
            )
            await self.message_bus.publish(message)

            logger.info(f"Attestation added for {self.system_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding attestation: {e}")
            return False

    async def send_verification(self, verification_data: Dict[str, Any]) -> bool:
        """Send verification response."""
        try:
            message = Message(
                type=MessageType.NEURALBLITZ_VERIFICATION,
                source=f"neuralblitz_adapter_{self.system_id}",
                payload=verification_data,
            )
            await self.message_bus.publish(message)

            logger.debug(f"Verification sent from {self.system_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending verification: {e}")
            return False

    def get_current_state(self) -> Optional[NeuralBlitzState]:
        """Get current NeuralBlitz state."""
        return self.current_state

    def get_attestation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get attestation history."""
        return self.attestation_history[-limit:]


class UnifiedAdapter:
    """Unified adapter combining LRS and NeuralBlitz capabilities."""

    def __init__(self, agent_id: str, system_id: str):
        self.agent_id = agent_id
        self.system_id = system_id
        self.message_bus: Optional[UnifiedMessageBus] = None
        self.state_manager: Optional[SharedStateManager] = None
        self.bidirectional_protocol: Optional[BidirectionalProtocol] = None
        self.lrs_adapter: Optional[LRSAdapter] = None
        self.neuralblitz_adapter: Optional[NeuralBlitzAdapter] = None

    async def initialize(
        self, message_bus: UnifiedMessageBus, state_manager: SharedStateManager
    ) -> bool:
        """Initialize the unified adapter."""
        try:
            self.message_bus = message_bus
            self.state_manager = state_manager

            # Create protocol configurations
            lrs_config = ProtocolConfig(
                source_system=f"lrs_agent_{self.agent_id}", target_system="neuralblitz"
            )
            neuralblitz_config = ProtocolConfig(
                source_system=f"neuralblitz_system_{self.system_id}", target_system="lrs"
            )

            # Create bidirectional protocol
            self.bidirectional_protocol = BidirectionalProtocol(
                lrs_config, neuralblitz_config, message_bus, state_manager
            )

            # Create individual adapters
            self.lrs_adapter = LRSAdapter(self.agent_id, message_bus, state_manager)
            self.neuralblitz_adapter = NeuralBlitzAdapter(
                self.system_id, message_bus, state_manager
            )

            # Start protocol
            await self.bidirectional_protocol.start()

            logger.info(f"Unified adapter initialized for {self.agent_id}/{self.system_id}")
            return True
        except Exception as e:
            logger.error(f"Error initializing unified adapter: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the unified adapter."""
        if self.bidirectional_protocol:
            await self.bidirectional_protocol.stop()
        logger.info(f"Unified adapter shutdown for {self.agent_id}/{self.system_id}")

    async def synchronize_systems(self) -> bool:
        """Synchronize LRS and NeuralBlitz systems."""
        if self.bidirectional_protocol:
            return await self.bidirectional_protocol.sync_states()
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "agent_id": self.agent_id,
            "system_id": self.system_id,
            "initialized": self.message_bus is not None,
            "lrs_agents": len(self.lrs_adapter.agent_states) if self.lrs_adapter else 0,
            "neuralblitz_active": self.neuralblitz_adapter.current_state is not None
            if self.neuralblitz_adapter
            else False,
            "protocol_stats": self.bidirectional_protocol.get_stats()
            if self.bidirectional_protocol
            else {},
        }
