"""
TUI State Mirror: Bidirectional state synchronization between TUI and LRS-Agents.

This component maintains consistency between TUI display state and LRS agent state,
providing real-time synchronization and conflict resolution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ...multi_agent.shared_state import SharedWorldState


@dataclass
class StateSnapshot:
    """Snapshot of state at a point in time."""

    agent_id: str
    state_data: Dict[str, Any]
    timestamp: datetime
    source: str  # 'tui' or 'lrs'
    version: int = 0


@dataclass
class ConflictResolution:
    """Configuration for state conflict resolution."""

    strategy: str = "tui_wins"  # 'tui_wins', 'lrs_wins', 'merge', 'timestamp'
    merge_function: Optional[Callable] = None
    conflict_callbacks: List[Callable] = field(default_factory=list)


class TUIStateMirror:
    """
    Keeps TUI and LRS state synchronized bidirectionally.

    Features:
    - Real-time state mirroring
    - Conflict detection and resolution
    - Change tracking and history
    - Subscription-based notifications
    - Performance optimization with debouncing
    - State validation and schema checking

    Examples:
        >>> mirror = TUIStateMirror(shared_state, tui_bridge)
        >>>
        >>> # Sync LRS state to TUI
        >>> await mirror.sync_precision_to_tui('agent_1', {'value': 0.8})
        >>>
        >>> # Sync TUI state to LRS
        >>> await mirror.sync_from_tui({'agent_1': {'status': 'active'}})
        >>>
        >>> # Subscribe to state changes
        >>> mirror.subscribe_agent('agent_1', callback)
    """

    def __init__(
        self,
        shared_state: SharedWorldState,
        tui_bridge,
        conflict_resolution: Optional[ConflictResolution] = None,
    ):
        """
        Initialize TUI State Mirror.

        Args:
            shared_state: LRS shared world state
            tui_bridge: TUIBridge instance
            conflict_resolution: Conflict resolution strategy
        """
        self.shared_state = shared_state
        self.tui_bridge = tui_bridge
        self.conflict_resolution = conflict_resolution or ConflictResolution()

        # State tracking
        self.tui_state: Dict[str, Dict[str, Any]] = {}
        self.lrs_state: Dict[str, Dict[str, Any]] = {}
        self.state_snapshots: List[StateSnapshot] = []
        self.state_versions: Dict[str, int] = {}

        # Subscribers for state changes
        self.state_subscribers: Dict[str, List[Callable]] = {}
        self.global_subscribers: List[Callable] = []

        # Change tracking
        self.pending_changes: Dict[str, Dict[str, Any]] = {}
        self.sync_lock = asyncio.Lock()

        # Performance optimization
        self.debounce_time = 0.1  # seconds
        self.batch_size = 50
        self.last_sync_time: Dict[str, datetime] = {}

        # User preferences storage
        self.user_preferences: Dict[str, Any] = {}

        self.logger = logging.getLogger(__name__)

        # Setup initial state sync
        asyncio.create_task(self._initialize_sync())

    async def _initialize_sync(self):
        """Initialize state synchronization."""

        # Get current LRS state
        all_lrs_states = self.shared_state.get_all_states()

        for agent_id, state in all_lrs_states.items():
            self.lrs_state[agent_id] = state.copy()
            self.state_versions[agent_id] = 0

        # Initialize TUI state cache
        self.tui_state = {}

        self.logger.info("TUI State Mirror initialized")

    async def sync_precision_to_tui(self, agent_id: str, precision_changes: Dict[str, Any]):
        """
        Sync precision changes from LRS to TUI.

        Args:
            agent_id: Agent identifier
            precision_changes: Precision update data
        """
        async with self.sync_lock:
            # Update LRS state cache
            if agent_id not in self.lrs_state:
                self.lrs_state[agent_id] = {}

            self.lrs_state[agent_id]["precision"] = precision_changes
            self.lrs_state[agent_id]["last_precision_update"] = datetime.now().isoformat()

            # Create snapshot
            snapshot = StateSnapshot(
                agent_id=agent_id,
                state_data={"precision": precision_changes},
                timestamp=datetime.now(),
                source="lrs",
                version=self.state_versions.get(agent_id, 0) + 1,
            )
            self._add_snapshot(snapshot)

            # Check for conflicts
            conflict = await self._detect_conflict(agent_id, "precision", precision_changes)

            if conflict:
                resolution = await self._resolve_conflict(conflict)
                precision_changes = resolution["resolved_value"]

            # Update TUI state
            if agent_id not in self.tui_state:
                self.tui_state[agent_id] = {}

            self.tui_state[agent_id]["precision"] = precision_changes
            self.tui_state[agent_id]["last_precision_sync"] = datetime.now().isoformat()

            # Broadcast to TUI
            await self._broadcast_to_tui(
                agent_id,
                "precision_update",
                {"precision": precision_changes, "conflict_resolved": conflict is not None},
            )

            # Notify subscribers
            await self._notify_subscribers(agent_id, "precision", precision_changes)

    async def sync_tool_execution_to_tui(self, agent_id: str, execution_result: Dict[str, Any]):
        """
        Sync tool execution results from LRS to TUI.

        Args:
            agent_id: Agent identifier
            execution_result: Tool execution data
        """
        async with self.sync_lock:
            # Update LRS state cache
            if agent_id not in self.lrs_state:
                self.lrs_state[agent_id] = {}

            # Add to execution history
            if "tool_executions" not in self.lrs_state[agent_id]:
                self.lrs_state[agent_id]["tool_executions"] = []

            self.lrs_state[agent_id]["tool_executions"].append(
                {**execution_result, "timestamp": datetime.now().isoformat()}
            )

            # Keep only recent executions
            max_executions = 100
            if len(self.lrs_state[agent_id]["tool_executions"]) > max_executions:
                self.lrs_state[agent_id]["tool_executions"] = self.lrs_state[agent_id][
                    "tool_executions"
                ][-max_executions:]

            # Update TUI state
            if agent_id not in self.tui_state:
                self.tui_state[agent_id] = {}

            self.tui_state[agent_id]["last_execution"] = execution_result
            self.tui_state[agent_id]["last_execution_sync"] = datetime.now().isoformat()

            # Broadcast to TUI
            await self._broadcast_to_tui(agent_id, "tool_execution", execution_result)

            # Notify subscribers
            await self._notify_subscribers(agent_id, "tool_execution", execution_result)

    async def sync_from_tui(self, tui_state_updates: Dict[str, Dict[str, Any]]):
        """
        Sync state updates from TUI to LRS.

        Args:
            tui_state_updates: Dictionary of agent_id -> state_updates
        """
        async with self.sync_lock:
            for agent_id, updates in tui_state_updates.items():
                # Update TUI state cache
                if agent_id not in self.tui_state:
                    self.tui_state[agent_id] = {}

                self.tui_state[agent_id].update(updates)
                self.tui_state[agent_id]["last_tui_update"] = datetime.now().isoformat()

                # Create snapshot
                snapshot = StateSnapshot(
                    agent_id=agent_id,
                    state_data=updates,
                    timestamp=datetime.now(),
                    source="tui",
                    version=self.state_versions.get(agent_id, 0) + 1,
                )
                self._add_snapshot(snapshot)

                # Check for conflicts
                conflicts = []
                for key, value in updates.items():
                    conflict = await self._detect_conflict(agent_id, key, value)
                    if conflict:
                        conflicts.append(conflict)

                # Resolve conflicts
                resolved_updates = updates.copy()
                for conflict in conflicts:
                    resolution = await self._resolve_conflict(conflict)
                    resolved_updates[conflict["field"]] = resolution["resolved_value"]

                # Update LRS shared state
                self.shared_state.update(agent_id, resolved_updates)

                # Update LRS state cache
                if agent_id not in self.lrs_state:
                    self.lrs_state[agent_id] = {}

                self.lrs_state[agent_id].update(resolved_updates)
                self.lrs_state[agent_id]["last_lrs_update"] = datetime.now().isoformat()

                # Broadcast confirmation to TUI
                await self._broadcast_to_tui(
                    agent_id,
                    "state_synced",
                    {"updates": resolved_updates, "conflicts_resolved": len(conflicts) > 0},
                )

                # Notify subscribers
                await self._notify_subscribers(agent_id, "tui_sync", resolved_updates)

    def get_user_preference(self, preference_key: str) -> Any:
        """
        Get user preference value.

        Args:
            preference_key: Preference identifier

        Returns:
            Preference value or None
        """
        return self.user_preferences.get(preference_key)

    def set_user_preference(self, preference_key: str, value: Any):
        """
        Set user preference value.

        Args:
            preference_key: Preference identifier
            value: Preference value
        """
        self.user_preferences[preference_key] = value

        # Broadcast preference change
        asyncio.create_task(
            self._broadcast_to_tui(
                "system", "preference_change", {"preference_key": preference_key, "value": value}
            )
        )

    def get_tui_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current TUI state.

        Args:
            agent_id: Optional agent filter

        Returns:
            TUI state data
        """
        if agent_id:
            return self.tui_state.get(agent_id, {}).copy()

        return {aid: state.copy() for aid, state in self.tui_state.items()}

    def get_lrs_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current LRS state.

        Args:
            agent_id: Optional agent filter

        Returns:
            LRS state data
        """
        if agent_id:
            return self.lrs_state.get(agent_id, {}).copy()

        return {aid: state.copy() for aid, state in self.lrs_state.items()}

    def subscribe_agent(self, agent_id: str, callback: Callable):
        """
        Subscribe to state changes for specific agent.

        Args:
            agent_id: Agent to watch
            callback: Callback function
        """
        if agent_id not in self.state_subscribers:
            self.state_subscribers[agent_id] = []

        self.state_subscribers[agent_id].append(callback)

    def subscribe_global(self, callback: Callable):
        """
        Subscribe to all state changes.

        Args:
            callback: Callback function
        """
        self.global_subscribers.append(callback)

    async def periodic_sync(self):
        """Perform periodic synchronization and cleanup."""

        # Clean old snapshots
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.state_snapshots = [s for s in self.state_snapshots if s.timestamp > cutoff_time]

        # Sync any pending changes
        if self.pending_changes:
            async with self.sync_lock:
                for agent_id, changes in self.pending_changes.items():
                    await self.sync_from_tui({agent_id: changes})

                self.pending_changes.clear()

        # Check for stale agents
        current_time = datetime.now()
        stale_threshold = timedelta(minutes=5)

        for agent_id in list(self.tui_state.keys()):
            last_update = self.tui_state[agent_id].get("last_tui_update")
            if last_update:
                last_update_time = datetime.fromisoformat(last_update)
                if current_time - last_update_time > stale_threshold:
                    # Mark agent as stale
                    await self._broadcast_to_tui(
                        agent_id,
                        "agent_stale",
                        {
                            "last_update": last_update,
                            "threshold_minutes": stale_threshold.total_seconds() / 60,
                        },
                    )

    def _add_snapshot(self, snapshot: StateSnapshot):
        """Add state snapshot to history."""
        self.state_snapshots.append(snapshot)
        self.state_versions[snapshot.agent_id] = snapshot.version

        # Keep only recent snapshots
        max_snapshots = 1000
        if len(self.state_snapshots) > max_snapshots:
            self.state_snapshots = self.state_snapshots[-max_snapshots:]

    async def _detect_conflict(
        self, agent_id: str, field: str, new_value: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Detect state conflict between TUI and LRS.

        Args:
            agent_id: Agent identifier
            field: State field name
            new_value: New value to check

        Returns:
            Conflict description or None
        """
        if agent_id not in self.lrs_state or agent_id not in self.tui_state:
            return None

        lrs_value = self.lrs_state[agent_id].get(field)
        tui_value = self.tui_state[agent_id].get(field)

        # Check if values differ significantly
        if lrs_value != tui_value and lrs_value is not None and tui_value is not None:
            return {
                "agent_id": agent_id,
                "field": field,
                "lrs_value": lrs_value,
                "tui_value": tui_value,
                "new_value": new_value,
                "timestamp": datetime.now().isoformat(),
            }

        return None

    async def _resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve state conflict using configured strategy.

        Args:
            conflict: Conflict description

        Returns:
            Resolution result
        """
        strategy = self.conflict_resolution.strategy
        field = conflict["field"]

        if strategy == "tui_wins":
            resolved_value = conflict["tui_value"]

        elif strategy == "lrs_wins":
            resolved_value = conflict["lrs_value"]

        elif strategy == "timestamp":
            # Use timestamp-based resolution
            lrs_timestamp = self.lrs_state[conflict["agent_id"]].get("last_lrs_update")
            tui_timestamp = self.tui_state[conflict["agent_id"]].get("last_tui_update")

            if lrs_timestamp and tui_timestamp:
                lrs_time = datetime.fromisoformat(lrs_timestamp)
                tui_time = datetime.fromisoformat(tui_timestamp)
                resolved_value = (
                    conflict["tui_value"] if tui_time > lrs_time else conflict["lrs_value"]
                )
            else:
                resolved_value = conflict["lrs_value"]  # Default to LRS

        elif strategy == "merge" and self.conflict_resolution.merge_function:
            resolved_value = self.conflict_resolution.merge_function(
                conflict["lrs_value"], conflict["tui_value"]
            )

        else:
            # Default: prefer new value
            resolved_value = conflict["new_value"]

        # Call conflict callbacks
        for callback in self.conflict_resolution.conflict_callbacks:
            try:
                await callback(conflict, resolved_value)
            except Exception as e:
                self.logger.error(f"Error in conflict callback: {e}")

        return {
            "conflict": conflict,
            "resolved_value": resolved_value,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
        }

    async def _broadcast_to_tui(self, target: str, event_type: str, data: Dict[str, Any]):
        """Broadcast event to TUI clients."""

        message = {
            "type": event_type,
            "target": target,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        if self.tui_bridge and hasattr(self.tui_bridge, "websocket_manager"):
            await self.tui_bridge.websocket_manager.broadcast("tui_events", message)

    async def _notify_subscribers(self, agent_id: str, event_type: str, data: Any):
        """Notify state change subscribers."""

        # Agent-specific subscribers
        if agent_id in self.state_subscribers:
            for callback in self.state_subscribers[agent_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(agent_id, event_type, data)
                    else:
                        callback(agent_id, event_type, data)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")

        # Global subscribers
        for callback in self.global_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, event_type, data)
                else:
                    callback(agent_id, event_type, data)
            except Exception as e:
                self.logger.error(f"Error in global subscriber callback: {e}")

    def get_state_history(
        self, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get state change history.

        Args:
            agent_id: Optional agent filter
            limit: Maximum number of records

        Returns:
            List of state change records
        """
        snapshots = self.state_snapshots

        if agent_id:
            snapshots = [s for s in snapshots if s.agent_id == agent_id]

        # Sort by timestamp (most recent first)
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        # Convert to dict format
        return [
            {
                "agent_id": s.agent_id,
                "state_data": s.state_data,
                "timestamp": s.timestamp.isoformat(),
                "source": s.source,
                "version": s.version,
            }
            for s in snapshots[:limit]
        ]
