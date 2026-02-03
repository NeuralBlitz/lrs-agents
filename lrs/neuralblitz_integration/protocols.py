"""
Communication Protocols for LRS-Agents ↔ NeuralBlitz-v50 Integration

Defines standardized protocols for bidirectional communication between
LRS-Agents Active Inference systems and NeuralBlitz-v50 Omega Architecture.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
import logging

from .messaging import Message, MessageType, UnifiedMessageBus
from .shared_state import StateType, SharedStateManager

logger = logging.getLogger(__name__)


@dataclass
class ProtocolConfig:
    """Configuration for communication protocols."""

    source_system: str
    target_system: str
    heartbeat_interval: float = 30.0  # seconds
    timeout: float = 10.0  # seconds
    max_retries: int = 3
    enable_compression: bool = False
    enable_encryption: bool = False
    custom_headers: Dict[str, str] = field(default_factory=dict)


class BaseProtocol(ABC):
    """Base class for communication protocols."""

    def __init__(
        self,
        config: ProtocolConfig,
        message_bus: UnifiedMessageBus,
        state_manager: SharedStateManager,
    ):
        self.config = config
        self.message_bus = message_bus
        self.state_manager = state_manager
        self.running = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_handlers: Dict[MessageType, Callable] = {}

    @abstractmethod
    async def start(self) -> None:
        """Start the protocol."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the protocol."""
        pass

    async def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        await self.message_bus.subscribe(message_type, self._handle_message)

    async def unregister_handler(self, message_type: MessageType) -> None:
        """Unregister a message handler."""
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            await self.message_bus.unsubscribe(message_type, self._handle_message)

    async def _handle_message(self, message: Message) -> None:
        """Handle incoming messages."""
        handler = self.message_handlers.get(message.type)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.type}: {e}")
        else:
            logger.warning(f"No handler for message type {message.type}")

    async def send_message(self, message: Message) -> bool:
        """Send a message."""
        message.source = self.config.source_system
        message.destination = self.config.target_system
        return await self.message_bus.publish(message)

    async def _start_heartbeat(self) -> None:
        """Start heartbeat task."""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _stop_heartbeat(self) -> None:
        """Stop heartbeat task."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop."""
        while self.running:
            try:
                heartbeat = await self.message_bus.create_heartbeat(
                    self.config.source_system, {"timestamp": datetime.utcnow().isoformat()}
                )
                await self.send_message(heartbeat)
                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying


class LRSToNeuralBlitzProtocol(BaseProtocol):
    """Protocol for LRS-Agents to NeuralBlitz-v50 communication."""

    def __init__(
        self,
        config: ProtocolConfig,
        message_bus: UnifiedMessageBus,
        state_manager: SharedStateManager,
    ):
        super().__init__(config, message_bus, state_manager)

    async def start(self) -> None:
        """Start LRS to NeuralBlitz protocol."""
        self.running = True
        await self._register_default_handlers()
        await self._start_heartbeat()
        logger.info("LRSToNeuralBlitzProtocol started")

    async def stop(self) -> None:
        """Stop LRS to NeuralBlitz protocol."""
        self.running = False
        await self._stop_heartbeat()
        await self._unregister_default_handlers()
        logger.info("LRSToNeuralBlitzProtocol stopped")

    async def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        # Register handlers for messages from NeuralBlitz
        await self.register_handler(
            MessageType.NEURALBLITZ_SOURCE_STATE, self._handle_neuralblitz_source_state
        )
        await self.register_handler(
            MessageType.NEURALBLITZ_INTENT_VECTOR, self._handle_neuralblitz_intent_vector
        )
        await self.register_handler(
            MessageType.NEURALBLITZ_VERIFICATION, self._handle_neuralblitz_verification
        )

    async def _unregister_default_handlers(self) -> None:
        """Unregister default message handlers."""
        await self.unregister_handler(MessageType.NEURALBLITZ_SOURCE_STATE)
        await self.unregister_handler(MessageType.NEURALBLITZ_INTENT_VECTOR)
        await self.unregister_handler(MessageType.NEURALBLITZ_VERIFICATION)

    async def send_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Send LRS agent state to NeuralBlitz."""
        message = Message(
            type=MessageType.LRS_AGENT_STATE,
            payload={
                "agent_id": agent_id,
                "state": state,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        return await self.send_message(message)

    async def send_precision_update(self, precision_data: Dict[str, Any]) -> bool:
        """Send precision update to NeuralBlitz."""
        message = Message(type=MessageType.LRS_PRECISION_UPDATE, payload=precision_data)
        return await self.send_message(message)

    async def send_free_energy(self, free_energy: Dict[str, Any]) -> bool:
        """Send free energy data to NeuralBlitz."""
        message = Message(type=MessageType.LRS_FREE_ENERGY, payload=free_energy)
        return await self.send_message(message)

    async def send_coordination_update(self, coordination_data: Dict[str, Any]) -> bool:
        """Send multi-agent coordination data to NeuralBlitz."""
        message = Message(type=MessageType.LRS_COORDINATION, payload=coordination_data)
        return await self.send_message(message)

    async def _handle_neuralblitz_source_state(self, message: Message) -> None:
        """Handle NeuralBlitz source state updates."""
        await self.state_manager.set_state(
            key=f"neuralblitz_source_{message.payload.get('id', 'default')}",
            value=message.payload,
            state_type=StateType.NEURALBLITZ_SOURCE,
            source="neuralblitz",
        )

    async def _handle_neuralblitz_intent_vector(self, message: Message) -> None:
        """Handle NeuralBlitz intent vector updates."""
        await self.state_manager.set_state(
            key=f"neuralblitz_intent_{message.payload.get('id', 'default')}",
            value=message.payload,
            state_type=StateType.NEURALBLITZ_INTENT,
            source="neuralblitz",
        )

    async def _handle_neuralblitz_verification(self, message: Message) -> None:
        """Handle NeuralBlitz verification responses."""
        await self.state_manager.set_state(
            key="neuralblitz_verification",
            value=message.payload,
            state_type=StateType.NEURALBLITZ_ATTESTATION,
            source="neuralblitz",
        )


class NeuralBlitzToLRSProtocol(BaseProtocol):
    """Protocol for NeuralBlitz-v50 to LRS-Agents communication."""

    def __init__(
        self,
        config: ProtocolConfig,
        message_bus: UnifiedMessageBus,
        state_manager: SharedStateManager,
    ):
        super().__init__(config, message_bus, state_manager)

    async def start(self) -> None:
        """Start NeuralBlitz to LRS protocol."""
        self.running = True
        await self._register_default_handlers()
        await self._start_heartbeat()
        logger.info("NeuralBlitzToLRSProtocol started")

    async def stop(self) -> None:
        """Stop NeuralBlitz to LRS protocol."""
        self.running = False
        await self._stop_heartbeat()
        await self._unregister_default_handlers()
        logger.info("NeuralBlitzToLRSProtocol stopped")

    async def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        # Register handlers for messages from LRS
        await self.register_handler(MessageType.LRS_AGENT_STATE, self._handle_lrs_agent_state)
        await self.register_handler(
            MessageType.LRS_PRECISION_UPDATE, self._handle_lrs_precision_update
        )
        await self.register_handler(MessageType.LRS_FREE_ENERGY, self._handle_lrs_free_energy)

    async def _unregister_default_handlers(self) -> None:
        """Unregister default message handlers."""
        await self.unregister_handler(MessageType.LRS_AGENT_STATE)
        await self.unregister_handler(MessageType.LRS_PRECISION_UPDATE)
        await self.unregister_handler(MessageType.LRS_FREE_ENERGY)

    async def send_source_state(self, source_state: Dict[str, Any]) -> bool:
        """Send NeuralBlitz source state to LRS."""
        message = Message(type=MessageType.NEURALBLITZ_SOURCE_STATE, payload=source_state)
        return await self.send_message(message)

    async def send_intent_vector(self, intent_vector: Dict[str, Any]) -> bool:
        """Send NeuralBlitz intent vector to LRS."""
        message = Message(type=MessageType.NEURALBLITZ_INTENT_VECTOR, payload=intent_vector)
        return await self.send_message(message)

    async def send_architect_dyad(self, dyad: Dict[str, Any]) -> bool:
        """Send architect dyad information to LRS."""
        message = Message(type=MessageType.NEURALBLITZ_ARCHITECT_DYAD, payload=dyad)
        return await self.send_message(message)

    async def send_attestation(self, attestation: Dict[str, Any]) -> bool:
        """Send attestation information to LRS."""
        message = Message(type=MessageType.NEURALBLITZ_ATTESTATION, payload=attestation)
        return await self.send_message(message)

    async def _handle_lrs_agent_state(self, message: Message) -> None:
        """Handle LRS agent state updates."""
        await self.state_manager.set_state(
            key=f"lrs_agent_{message.payload.get('agent_id', 'default')}",
            value=message.payload,
            state_type=StateType.LRS_AGENT_STATE,
            source="lrs",
        )

    async def _handle_lrs_precision_update(self, message: Message) -> None:
        """Handle LRS precision updates."""
        await self.state_manager.set_state(
            key="lrs_precision",
            value=message.payload,
            state_type=StateType.LRS_PRECISION,
            source="lrs",
        )

    async def _handle_lrs_free_energy(self, message: Message) -> None:
        """Handle LRS free energy updates."""
        await self.state_manager.set_state(
            key="lrs_free_energy",
            value=message.payload,
            state_type=StateType.LRS_FREE_ENERGY,
            source="lrs",
        )


class BidirectionalProtocol:
    """Combines both protocols for full bidirectional communication."""

    def __init__(
        self,
        lrs_config: ProtocolConfig,
        neuralblitz_config: ProtocolConfig,
        message_bus: UnifiedMessageBus,
        state_manager: SharedStateManager,
    ):
        self.lrs_to_neuralblitz = LRSToNeuralBlitzProtocol(lrs_config, message_bus, state_manager)
        self.neuralblitz_to_lrs = NeuralBlitzToLRSProtocol(
            neuralblitz_config, message_bus, state_manager
        )
        self.message_bus = message_bus
        self.state_manager = state_manager

    async def start(self) -> None:
        """Start both protocols."""
        await self.lrs_to_neuralblitz.start()
        await self.neuralblitz_to_lrs.start()
        logger.info("BidirectionalProtocol started")

    async def stop(self) -> None:
        """Stop both protocols."""
        await self.lrs_to_neuralblitz.stop()
        await self.neuralblitz_to_lrs.stop()
        logger.info("BidirectionalProtocol stopped")

    async def sync_states(self) -> bool:
        """Synchronize states between both systems."""
        try:
            # Create sync request
            sync_request = Message(
                type=MessageType.SYNC_REQUEST,
                source="bidirectional_protocol",
                destination="broadcast",
                payload={
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": datetime.utcnow().timestamp(),
                },
            )

            await self.message_bus.publish(sync_request)
            logger.info("State synchronization initiated")
            return True
        except Exception as e:
            logger.error(f"Error during state synchronization: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "message_bus": self.message_bus.get_stats(),
            "state_manager": self.state_manager.get_stats(),
            "lrs_protocol": {
                "running": self.lrs_to_neuralblitz.running,
                "handlers": len(self.lrs_to_neuralblitz.message_handlers),
            },
            "neuralblitz_protocol": {
                "running": self.neuralblitz_to_lrs.running,
                "handlers": len(self.neuralblitz_to_lrs.message_handlers),
            },
        }
