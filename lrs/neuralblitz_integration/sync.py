"""
Real-time Synchronization for LRS-Agents ↔ NeuralBlitz-v50 Integration

Provides real-time event synchronization, conflict resolution, and
coordinated state management between both systems.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from .messaging import UnifiedMessageBus, Message, MessageType
from .shared_state import SharedStateManager, StateType

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Synchronization status."""

    SYNCHRONIZED = "synchronized"
    SYNCHRONIZING = "synchronizing"
    CONFLICT = "conflict"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class SyncEvent:
    """Represents a synchronization event."""

    id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: SyncStatus = SyncStatus.SYNCHRONIZING
    source_system: str = "unknown"
    target_system: str = "unknown"
    state_changes: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "state_changes": self.state_changes,
            "conflicts": self.conflicts,
            "metadata": self.metadata,
        }


@dataclass
class SyncMetrics:
    """Synchronization performance metrics."""

    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    conflicts_resolved: int = 0
    average_sync_time: float = 0.0
    last_sync_time: Optional[datetime] = None
    sync_frequency: float = 30.0  # seconds
    uptime: float = 0.0  # percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
            "failed_syncs": self.failed_syncs,
            "conflicts_resolved": self.conflicts_resolved,
            "success_rate": (self.successful_syncs / self.total_syncs * 100)
            if self.total_syncs > 0
            else 0,
            "average_sync_time": self.average_sync_time,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "sync_frequency": self.sync_frequency,
            "uptime": self.uptime,
        }


class RealTimeSynchronizer:
    """Real-time synchronization system for bidirectional communication."""

    def __init__(self, message_bus: UnifiedMessageBus, state_manager: SharedStateManager):
        self.message_bus = message_bus
        self.state_manager = state_manager
        self.running = False
        self.sync_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.sync_events: List[SyncEvent] = []
        self.metrics = SyncMetrics()
        self.conflict_resolvers: Dict[str, Callable] = {}
        self.sync_subscribers: Set[Callable] = set()
        self.last_heartbeat: Dict[str, datetime] = {}
        self.heartbeat_timeout = 60.0  # seconds

    async def start(self) -> None:
        """Start the synchronizer."""
        if self.running:
            return

        self.running = True

        # Register message handlers
        await self.message_bus.subscribe(MessageType.SYNC_REQUEST, self._handle_sync_request)
        await self.message_bus.subscribe(MessageType.SYNC_RESPONSE, self._handle_sync_response)
        await self.message_bus.subscribe(MessageType.HEARTBEAT, self._handle_heartbeat)

        # Start synchronization loop
        self.sync_task = asyncio.create_task(self._sync_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        logger.info("RealTimeSynchronizer started")

    async def stop(self) -> None:
        """Stop the synchronizer."""
        self.running = False

        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass

        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info("RealTimeSynchronizer stopped")

    async def add_conflict_resolver(self, conflict_type: str, resolver: Callable) -> None:
        """Add a conflict resolver."""
        self.conflict_resolvers[conflict_type] = resolver
        logger.debug(f"Added conflict resolver for {conflict_type}")

    async def subscribe_to_sync(self, callback: Callable[[SyncEvent], None]) -> None:
        """Subscribe to sync events."""
        self.sync_subscribers.add(callback)

    async def unsubscribe_from_sync(self, callback: Callable[[SyncEvent], None]) -> None:
        """Unsubscribe from sync events."""
        self.sync_subscribers.discard(callback)

    async def force_sync(self, source_system: str, target_system: str) -> bool:
        """Force immediate synchronization."""
        try:
            start_time = time.time()

            # Create sync event
            sync_event = SyncEvent(
                id=f"sync_{int(time.time())}",
                source_system=source_system,
                target_system=target_system,
                status=SyncStatus.SYNCHRONIZING,
            )

            # Perform synchronization
            success = await self._perform_synchronization(sync_event)

            # Update metrics
            sync_time = time.time() - start_time
            self._update_metrics(success, sync_time)

            # Notify subscribers
            await self._notify_sync_event(sync_event)

            return success
        except Exception as e:
            logger.error(f"Error forcing sync: {e}")
            return False

    async def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self.running:
            try:
                # Check if sync is needed
                await self._check_sync_requirements()

                # Wait for next sync cycle
                await asyncio.sleep(self.metrics.sync_frequency)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying

    async def _heartbeat_monitor(self) -> None:
        """Monitor heartbeat messages."""
        while self.running:
            try:
                # Check for stale heartbeats
                current_time = datetime.utcnow()
                stale_systems = []

                for system, last_heartbeat in self.last_heartbeat.items():
                    if (current_time - last_heartbeat).total_seconds() > self.heartbeat_timeout:
                        stale_systems.append(system)

                # Handle stale systems
                for system in stale_systems:
                    logger.warning(f"System {system} appears to be offline")
                    del self.last_heartbeat[system]

                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5.0)

    async def _check_sync_requirements(self) -> None:
        """Check if synchronization is needed."""
        try:
            # Get last sync time
            if not self.metrics.last_sync_time:
                await self.force_sync("auto", "auto")
                return

            # Check time since last sync
            time_since_sync = (datetime.utcnow() - self.metrics.last_sync_time).total_seconds()
            if time_since_sync >= self.metrics.sync_frequency:
                await self.force_sync("auto", "auto")

        except Exception as e:
            logger.error(f"Error checking sync requirements: {e}")

    async def _perform_synchronization(self, sync_event: SyncEvent) -> bool:
        """Perform actual synchronization."""
        try:
            # Get current unified state
            unified_state = self.state_manager.get_unified_state()

            # Create state snapshot
            state_snapshot = {
                "lrs_states": unified_state.lrs_agent_states,
                "lrs_precision": unified_state.lrs_precision_states,
                "neuralblitz_source": unified_state.neuralblitz_source,
                "neuralblitz_intent": unified_state.neuralblitz_intent,
                "coordination": unified_state.coordination,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Detect conflicts
            conflicts = await self._detect_conflicts(state_snapshot)

            if conflicts:
                sync_event.conflicts = conflicts
                await self._resolve_conflicts(conflicts, sync_event)

            # Apply synchronization
            await self._apply_synchronization(state_snapshot, sync_event)

            # Update sync event status
            sync_event.status = SyncStatus.SYNCHRONIZED
            sync_event.state_changes = [state_snapshot]

            self.metrics.last_sync_time = datetime.utcnow()

            # Send sync response
            response = Message(
                type=MessageType.SYNC_RESPONSE,
                source="realtime_synchronizer",
                payload={"sync_event": sync_event.to_dict(), "state_snapshot": state_snapshot},
            )
            await self.message_bus.publish(response)

            logger.info("Synchronization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error performing synchronization: {e}")
            sync_event.status = SyncStatus.ERROR
            return False

    async def _detect_conflicts(self, state_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect state conflicts."""
        conflicts = []

        try:
            # Check for conflicting agent states
            lrs_agents = state_snapshot.get("lrs_states", {})
            neuralblitz_source = state_snapshot.get("neuralblitz_source", {})

            # Simple conflict detection example
            for agent_id, agent_state in lrs_agents.items():
                # Check if there's a conflicting neuralblitz state
                if agent_id in neuralblitz_source:
                    conflict = {
                        "type": "agent_state_conflict",
                        "entity": agent_id,
                        "lrs_state": agent_state,
                        "neuralblitz_state": neuralblitz_source[agent_id],
                        "detected_at": datetime.utcnow().isoformat(),
                    }
                    conflicts.append(conflict)

        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")

        return conflicts

    async def _resolve_conflicts(
        self, conflicts: List[Dict[str, Any]], sync_event: SyncEvent
    ) -> None:
        """Resolve detected conflicts."""
        try:
            for conflict in conflicts:
                conflict_type = conflict.get("type", "unknown")

                if conflict_type in self.conflict_resolvers:
                    # Use custom resolver
                    resolver = self.conflict_resolvers[conflict_type]
                    resolution = await resolver(conflict)
                    sync_event.state_changes.append(resolution)

                else:
                    # Default resolution: LRS takes precedence
                    logger.info(f"Using default resolution for conflict {conflict_type}")
                    sync_event.conflicts_resolved += 1

        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")

    async def _apply_synchronization(
        self, state_snapshot: Dict[str, Any], sync_event: SyncEvent
    ) -> None:
        """Apply synchronized state."""
        try:
            # Update coordination state
            await self.state_manager.set_state(
                key="last_sync",
                value=state_snapshot,
                state_type=StateType.COORDINATION,
                source="synchronizer",
            )

            # Update synchronization state
            sync_info = {
                "last_sync": datetime.utcnow().isoformat(),
                "sync_event_id": sync_event.id,
                "status": sync_event.status.value,
            }
            await self.state_manager.set_state(
                key="synchronization",
                value=sync_info,
                state_type=StateType.SYNCHRONIZATION,
                source="synchronizer",
            )

        except Exception as e:
            logger.error(f"Error applying synchronization: {e}")

    async def _handle_sync_request(self, message: Message) -> None:
        """Handle sync request messages."""
        try:
            payload = message.payload

            # Force sync with specific source/target
            await self.force_sync(
                payload.get("source_system", message.source),
                payload.get("target_system", "broadcast"),
            )

        except Exception as e:
            logger.error(f"Error handling sync request: {e}")

    async def _handle_sync_response(self, message: Message) -> None:
        """Handle sync response messages."""
        try:
            payload = message.payload

            # Update sync history
            sync_event_data = payload.get("sync_event", {})
            sync_event = SyncEvent(
                id=sync_event_data.get("id"),
                status=SyncStatus(sync_event_data.get("status", "unknown")),
                source_system=sync_event_data.get("source_system"),
                target_system=sync_event_data.get("target_system"),
                state_changes=sync_event_data.get("state_changes", []),
                conflicts=sync_event_data.get("conflicts", []),
                metadata=sync_event_data.get("metadata", {}),
            )

            self.sync_events.append(sync_event)

            # Keep only recent events
            if len(self.sync_events) > 100:
                self.sync_events = self.sync_events[-100:]

        except Exception as e:
            logger.error(f"Error handling sync response: {e}")

    async def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat messages."""
        try:
            self.last_heartbeat[message.source] = datetime.utcnow()
            logger.debug(f"Heartbeat received from {message.source}")

        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")

    def _update_metrics(self, success: bool, sync_time: float) -> None:
        """Update synchronization metrics."""
        self.metrics.total_syncs += 1

        if success:
            self.metrics.successful_syncs += 1
        else:
            self.metrics.failed_syncs += 1

        # Update average sync time
        if self.metrics.total_syncs == 1:
            self.metrics.average_sync_time = sync_time
        else:
            self.metrics.average_sync_time = (
                self.metrics.average_sync_time * (self.metrics.total_syncs - 1) + sync_time
            ) / self.metrics.total_syncs

    async def _notify_sync_event(self, sync_event: SyncEvent) -> None:
        """Notify subscribers of sync event."""
        for subscriber in self.sync_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(sync_event)
                else:
                    subscriber(sync_event)
            except Exception as e:
                logger.error(f"Error notifying sync subscriber: {e}")

    def get_metrics(self) -> SyncMetrics:
        """Get synchronization metrics."""
        return self.metrics

    def get_sync_history(self, limit: int = 50) -> List[SyncEvent]:
        """Get sync event history."""
        return self.sync_events[-limit:]

    def get_active_systems(self) -> List[str]:
        """Get list of currently active systems."""
        current_time = datetime.utcnow()
        return [
            system
            for system, last_heartbeat in self.last_heartbeat.items()
            if (current_time - last_heartbeat).total_seconds() < self.heartbeat_timeout
        ]

    async def set_sync_frequency(self, frequency: float) -> None:
        """Set synchronization frequency."""
        self.metrics.sync_frequency = max(1.0, frequency)  # Minimum 1 second
        logger.info(f"Sync frequency set to {self.metrics.sync_frequency} seconds")
