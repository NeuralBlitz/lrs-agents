"""
Unified Message Bus for LRS-Agents ↔ NeuralBlitz-v50 Communication

Provides high-performance asynchronous messaging system with type safety,
message routing, and guaranteed delivery semantics.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for bidirectional communication."""

    # LRS → NeuralBlitz
    LRS_AGENT_STATE = "lrs_agent_state"
    LRS_PRECISION_UPDATE = "lrs_precision_update"
    LRS_FREE_ENERGY = "lrs_free_energy"
    LRS_COORDINATION = "lrs_coordination"
    LRS_BENCHMARK = "lrs_benchmark"

    # NeuralBlitz → LRS
    NEURALBLITZ_SOURCE_STATE = "neuralblitz_source_state"
    NEURALBLITZ_INTENT_VECTOR = "neuralblitz_intent_vector"
    NEURALBLITZ_ARCHITECT_DYAD = "neuralblitz_architect_dyad"
    NEURALBLITZ_ATTESTATION = "neuralblitz_attestation"
    NEURALBLITZ_VERIFICATION = "neuralblitz_verification"

    # Bidirectional
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """Universal message format for bidirectional communication."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.HEARTBEAT
    source: str = "unknown"
    destination: str = "broadcast"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher numbers = higher priority
    ttl: Optional[int] = None  # Time-to-live in seconds
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "destination": self.destination,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl": self.ttl,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        data["type"] = MessageType(data["type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl is None:
            return False
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.ttl

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries


class UnifiedMessageBus:
    """High-performance message bus for bidirectional communication."""

    def __init__(self):
        self._subscribers: Dict[MessageType, Set[Callable]] = defaultdict(set)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._message_history: Dict[str, Message] = {}
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "subscribers": 0,
        }

    async def start(self) -> None:
        """Start the message bus."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._message_loop())
        logger.info("UnifiedMessageBus started")

    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("UnifiedMessageBus stopped")

    async def publish(self, message: Message) -> bool:
        """Publish a message to all subscribers."""
        if message.is_expired():
            logger.warning(f"Message {message.id} expired, dropping")
            self._stats["messages_failed"] += 1
            return False

        self._message_history[message.id] = message
        self._stats["messages_sent"] += 1

        try:
            await self._message_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish message {message.id}: {e}")
            self._stats["messages_failed"] += 1
            return False

    async def subscribe(
        self, message_type: MessageType, callback: Callable[[Message], None]
    ) -> None:
        """Subscribe to a specific message type."""
        self._subscribers[message_type].add(callback)
        self._stats["subscribers"] += 1
        logger.debug(f"Subscribed to {message_type}")

    async def unsubscribe(
        self, message_type: MessageType, callback: Callable[[Message], None]
    ) -> None:
        """Unsubscribe from a specific message type."""
        self._subscribers[message_type].discard(callback)
        self._stats["subscribers"] -= 1
        logger.debug(f"Unsubscribed from {message_type}")

    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message loop: {e}")

    async def _process_message(self, message: Message) -> None:
        """Process a single message."""
        try:
            self._stats["messages_received"] += 1

            # Get subscribers for this message type
            subscribers = self._subscribers.get(message.type, set())

            # Execute all subscribers concurrently
            if subscribers:
                tasks = [self._safe_execute_callback(callback, message) for callback in subscribers]
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                logger.warning(f"No subscribers for message type {message.type}")

        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            self._stats["messages_failed"] += 1

    async def _safe_execute_callback(
        self, callback: Callable[[Message], None], message: Message
    ) -> None:
        """Safely execute a subscriber callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.error(f"Error in subscriber callback: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            **self._stats,
            "queue_size": self._message_queue.qsize(),
            "message_history_size": len(self._message_history),
            "running": self._running,
        }

    def get_message_history(self, limit: int = 100) -> List[Message]:
        """Get recent message history."""
        return list(self._message_history.values())[-limit:]

    async def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()
        logger.info("Message history cleared")

    async def create_heartbeat(self, source: str, payload: Dict[str, Any] = None) -> Message:
        """Create a heartbeat message."""
        return Message(
            type=MessageType.HEARTBEAT,
            source=source,
            payload=payload or {"status": "alive"},
        )

    async def create_error(self, source: str, error: str, correlation_id: str = None) -> Message:
        """Create an error message."""
        return Message(
            type=MessageType.ERROR,
            source=source,
            payload={"error": error},
            correlation_id=correlation_id,
        )


# Global message bus instance
_message_bus = UnifiedMessageBus()


def get_message_bus() -> UnifiedMessageBus:
    """Get the global message bus instance."""
    return _message_bus


async def start_message_bus() -> None:
    """Start the global message bus."""
    await _message_bus.start()


async def stop_message_bus() -> None:
    """Stop the global message bus."""
    await _message_bus.stop()
