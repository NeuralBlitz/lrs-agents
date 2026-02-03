"""
Shared State Management for LRS-Agents ↔ NeuralBlitz-v50 Integration

Provides unified state management with synchronization, versioning, and conflict resolution
between LRS-Agents Active Inference systems and NeuralBlitz-v50 Omega Architecture.
"""

import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
import weakref
from collections import defaultdict

logger = logging.getLogger(__name__)


class StateType(Enum):
    """Types of shared state."""

    # LRS States
    LRS_AGENT_STATE = "lrs_agent_state"
    LRS_PRECISION = "lrs_precision"
    LRS_FREE_ENERGY = "lrs_free_energy"
    LRS_MULTI_AGENT = "lrs_multi_agent"

    # NeuralBlitz States
    NEURALBLITZ_SOURCE = "neuralblitz_source"
    NEURALBLITZ_INTENT = "neuralblitz_intent"
    NEURALBLITZ_DYAD = "neuralblitz_dyad"
    NEURALBLITZ_ATTESTATION = "neuralblitz_attestation"

    # Shared States
    COORDINATION = "coordination"
    SYNCHRONIZATION = "synchronization"
    CONFIGURATION = "configuration"


class ConflictResolution(Enum):
    """Strategies for resolving state conflicts."""

    LAST_WRITE_WINS = "last_write_wins"
    MERGE = "merge"
    CUSTOM = "custom"
    IGNORE = "ignore"


@dataclass
class StateEntry:
    """Single state entry with metadata."""

    key: str
    value: Any
    type: StateType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    source: str = "unknown"
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate checksum after initialization."""
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification."""
        import hashlib

        content = f"{self.key}{self.type.value}{self.value}{self.version}{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "source": self.source,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateEntry":
        """Create from dictionary."""
        data["type"] = StateType(data["type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def is_valid(self) -> bool:
        """Verify state entry integrity."""
        return self.checksum == self._calculate_checksum()


@dataclass
class UnifiedState:
    """Unified state container for both systems."""

    # LRS States
    lrs_agent_states: Dict[str, Any] = field(default_factory=dict)
    lrs_precision_states: Dict[str, Any] = field(default_factory=dict)
    lrs_free_energy: Dict[str, Any] = field(default_factory=dict)
    lrs_multi_agent: Dict[str, Any] = field(default_factory=dict)

    # NeuralBlitz States
    neuralblitz_source: Dict[str, Any] = field(default_factory=dict)
    neuralblitz_intent: Dict[str, Any] = field(default_factory=dict)
    neuralblitz_dyad: Dict[str, Any] = field(default_factory=dict)
    neuralblitz_attestation: Dict[str, Any] = field(default_factory=dict)

    # Shared States
    coordination: Dict[str, Any] = field(default_factory=dict)
    synchronization: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lrs_agent_states": self.lrs_agent_states,
            "lrs_precision_states": self.lrs_precision_states,
            "lrs_free_energy": self.lrs_free_energy,
            "lrs_multi_agent": self.lrs_multi_agent,
            "neuralblitz_source": self.neuralblitz_source,
            "neuralblitz_intent": self.neuralblitz_intent,
            "neuralblitz_dyad": self.neuralblitz_dyad,
            "neuralblitz_attestation": self.neuralblitz_attestation,
            "coordination": self.coordination,
            "synchronization": self.synchronization,
            "configuration": self.configuration,
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedState":
        """Create from dictionary."""
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class SharedStateManager:
    """Manages shared state between LRS-Agents and NeuralBlitz-v50."""

    def __init__(
        self, conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
    ):
        self.state = UnifiedState()
        self.state_entries: Dict[str, StateEntry] = {}
        self.subscribers: Dict[StateType, Set[Callable]] = defaultdict(set)
        self.lock = asyncio.Lock()
        self.conflict_resolution = conflict_resolution
        self.version_counter = 0
        self.stats = {
            "state_updates": 0,
            "state_reads": 0,
            "conflicts": 0,
            "subscribers": 0,
        }

    async def set_state(
        self,
        key: str,
        value: Any,
        state_type: StateType,
        source: str = "unknown",
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Set a state entry."""
        async with self.lock:
            try:
                # Check for conflicts
                existing_entry = self.state_entries.get(key)
                if existing_entry and existing_entry.value != value:
                    await self._handle_conflict(key, existing_entry, value, source)

                # Create new state entry
                entry = StateEntry(
                    key=key,
                    value=value,
                    type=state_type,
                    source=source,
                    version=existing_entry.version + 1 if existing_entry else 1,
                    metadata=metadata or {},
                )

                # Update storage
                self.state_entries[key] = entry
                await self._update_unified_state(entry)

                self.stats["state_updates"] += 1
                await self._notify_subscribers(state_type, key, value, source)

                logger.debug(f"State set: {key} = {value} (from {source})")
                return True

            except Exception as e:
                logger.error(f"Error setting state {key}: {e}")
                return False

    async def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state entry value."""
        async with self.lock:
            self.stats["state_reads"] += 1
            entry = self.state_entries.get(key)
            return entry.value if entry else default

    async def get_state_entry(self, key: str) -> Optional[StateEntry]:
        """Get a complete state entry."""
        async with self.lock:
            return self.state_entries.get(key)

    async def delete_state(self, key: str) -> bool:
        """Delete a state entry."""
        async with self.lock:
            if key in self.state_entries:
                entry = self.state_entries[key]
                del self.state_entries[key]
                await self._remove_from_unified_state(entry)
                await self._notify_subscribers(entry.type, key, None, "deleted")
                logger.debug(f"State deleted: {key}")
                return True
            return False

    async def subscribe(
        self, state_type: StateType, callback: Callable[[str, Any, str], None]
    ) -> None:
        """Subscribe to state changes."""
        self.subscribers[state_type].add(callback)
        self.stats["subscribers"] += 1
        logger.debug(f"Subscribed to {state_type} state changes")

    async def unsubscribe(
        self, state_type: StateType, callback: Callable[[str, Any, str], None]
    ) -> None:
        """Unsubscribe from state changes."""
        self.subscribers[state_type].discard(callback)
        self.stats["subscribers"] -= 1
        logger.debug(f"Unsubscribed from {state_type} state changes")

    async def _update_unified_state(self, entry: StateEntry) -> None:
        """Update the unified state container."""
        state_map = self._get_state_map(entry.type)
        if state_map is not None:
            state_map[entry.key] = entry.value
            self.state.last_updated = datetime.utcnow()
            self.state.version += 1

    async def _remove_from_unified_state(self, entry: StateEntry) -> None:
        """Remove from unified state container."""
        state_map = self._get_state_map(entry.type)
        if state_map is not None and entry.key in state_map:
            del state_map[entry.key]
            self.state.last_updated = datetime.utcnow()
            self.state.version += 1

    def _get_state_map(self, state_type: StateType) -> Optional[Dict[str, Any]]:
        """Get the appropriate state map for a type."""
        mapping = {
            StateType.LRS_AGENT_STATE: self.state.lrs_agent_states,
            StateType.LRS_PRECISION: self.state.lrs_precision_states,
            StateType.LRS_FREE_ENERGY: self.state.lrs_free_energy,
            StateType.LRS_MULTI_AGENT: self.state.lrs_multi_agent,
            StateType.NEURALBLITZ_SOURCE: self.state.neuralblitz_source,
            StateType.NEURALBLITZ_INTENT: self.state.neuralblitz_intent,
            StateType.NEURALBLITZ_DYAD: self.state.neuralblitz_dyad,
            StateType.NEURALBLITZ_ATTESTATION: self.state.neuralblitz_attestation,
            StateType.COORDINATION: self.state.coordination,
            StateType.SYNCHRONIZATION: self.state.synchronization,
            StateType.CONFIGURATION: self.state.configuration,
        }
        return mapping.get(state_type)

    async def _handle_conflict(
        self, key: str, existing: StateEntry, new_value: Any, source: str
    ) -> None:
        """Handle state conflicts based on resolution strategy."""
        self.stats["conflicts"] += 1
        logger.warning(
            f"State conflict for {key}: existing={existing.value}, new={new_value}, source={source}"
        )

        if self.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
            # Default behavior - just overwrite
            pass
        elif self.conflict_resolution == ConflictResolution.MERGE:
            # Attempt to merge values
            try:
                if isinstance(existing.value, dict) and isinstance(new_value, dict):
                    merged = {**existing.value, **new_value}
                    existing.value = merged
                    logger.info(f"Merged state for {key}: {merged}")
                else:
                    logger.warning(f"Cannot merge non-dict values for {key}")
            except Exception as e:
                logger.error(f"Error merging state for {key}: {e}")

    async def _notify_subscribers(
        self, state_type: StateType, key: str, value: Any, source: str
    ) -> None:
        """Notify subscribers of state changes."""
        subscribers = self.subscribers.get(state_type, set())
        if subscribers:
            tasks = [self._safe_notify(callback, key, value, source) for callback in subscribers]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_notify(
        self, callback: Callable[[str, Any, str], None], key: str, value: Any, source: str
    ) -> None:
        """Safely notify a subscriber."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(key, value, source)
            else:
                callback(key, value, source)
        except Exception as e:
            logger.error(f"Error notifying subscriber: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        return {
            **self.stats,
            "total_states": len(self.state_entries),
            "state_types": len(set(entry.type for entry in self.state_entries.values())),
            "last_updated": self.state.last_updated.isoformat(),
            "version": self.state.version,
        }

    def get_unified_state(self) -> UnifiedState:
        """Get the complete unified state."""
        return self.state

    async def clear_all_states(self) -> None:
        """Clear all states."""
        async with self.lock:
            self.state_entries.clear()
            self.state = UnifiedState()
            logger.info("All states cleared")

    async def export_state(self) -> str:
        """Export state to JSON string."""
        return json.dumps(
            {
                "state": self.state.to_dict(),
                "entries": [entry.to_dict() for entry in self.state_entries.values()],
                "stats": self.get_stats(),
            },
            indent=2,
        )

    async def import_state(self, json_data: str) -> bool:
        """Import state from JSON string."""
        try:
            data = json.loads(json_data)
            self.state = UnifiedState.from_dict(data["state"])
            self.state_entries = {
                entry["key"]: StateEntry.from_dict(entry) for entry in data["entries"]
            }
            logger.info("State imported successfully")
            return True
        except Exception as e:
            logger.error(f"Error importing state: {e}")
            return False


# Global state manager instance
_state_manager = SharedStateManager()


def get_state_manager() -> SharedStateManager:
    """Get the global state manager instance."""
    return _state_manager


async def start_state_manager() -> None:
    """Initialize the state manager."""
    logger.info("SharedStateManager initialized")
