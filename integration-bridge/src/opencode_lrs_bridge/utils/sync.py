"""
State synchronization and conflict resolution between opencode and LRS-Agents.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import structlog

from ..config.settings import IntegrationBridgeConfig
from ..models.schemas import (
    AgentState,
    StateSyncRequest,
    StateSyncResponse,
    ConflictResolutionStrategy,
    PrecisionData,
    AgentStatus,
)

logger = structlog.get_logger(__name__)


class StateConflictType(str, Enum):
    """Types of state conflicts."""

    VALUE_CONFLICT = "value_conflict"
    STRUCTURE_CONFLICT = "structure_conflict"
    TIMESTAMP_CONFLICT = "timestamp_conflict"
    PRECISION_CONFLICT = "precision_conflict"
    STATUS_CONFLICT = "status_conflict"


class StateConflict:
    """Represents a state conflict between two systems."""

    def __init__(
        self,
        conflict_type: StateConflictType,
        field_path: str,
        lrs_value: Any,
        opencode_value: Any,
        timestamp: datetime,
    ):
        self.conflict_type = conflict_type
        self.field_path = field_path
        self.lrs_value = lrs_value
        self.opencode_value = opencode_value
        self.timestamp = timestamp
        self.resolved_value: Optional[Any] = None
        self.resolution_strategy: Optional[ConflictResolutionStrategy] = None


class StateMerger:
    """Merges state from different systems."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config

    async def merge_states(
        self,
        lrs_state: Dict[str, Any],
        opencode_state: Dict[str, Any],
        strategy: ConflictResolutionStrategy,
        field_path: str = "",
    ) -> Tuple[Dict[str, Any], List[StateConflict]]:
        """Merge two state dictionaries."""
        conflicts = []
        merged_state = {}

        # Get all unique keys from both states
        all_keys = set(lrs_state.keys()) | set(opencode_state.keys())

        for key in all_keys:
            current_path = f"{field_path}.{key}" if field_path else key

            if key in lrs_state and key in opencode_state:
                lrs_value = lrs_state[key]
                opencode_value = opencode_state[key]

                # Check for conflicts
                if self._has_conflict(lrs_value, opencode_value):
                    conflict = StateConflict(
                        conflict_type=self._determine_conflict_type(
                            lrs_value, opencode_value
                        ),
                        field_path=current_path,
                        lrs_value=lrs_value,
                        opencode_value=opencode_value,
                        timestamp=datetime.utcnow(),
                    )
                    conflicts.append(conflict)

                    # Resolve conflict based on strategy
                    resolved_value = await self._resolve_conflict(conflict, strategy)
                    merged_state[key] = resolved_value
                else:
                    # No conflict, use lrs value (or merge if both are dicts)
                    if isinstance(lrs_value, dict) and isinstance(opencode_value, dict):
                        merged_child, child_conflicts = await self.merge_states(
                            lrs_value, opencode_value, strategy, current_path
                        )
                        merged_state[key] = merged_child
                        conflicts.extend(child_conflicts)
                    else:
                        merged_state[key] = lrs_value

            elif key in lrs_state:
                merged_state[key] = lrs_state
            else:
                merged_state[key] = opencode_state

        return merged_state, conflicts

    def _has_conflict(self, lrs_value: Any, opencode_value: Any) -> bool:
        """Check if two values conflict."""
        # Simple equality check for most types
        if lrs_value != opencode_value:
            return True

        # For timestamps, check if they're significantly different
        if isinstance(lrs_value, str) and isinstance(opencode_value, str):
            try:
                lrs_time = datetime.fromisoformat(lrs_value.replace("Z", "+00:00"))
                opencode_time = datetime.fromisoformat(
                    opencode_value.replace("Z", "+00:00")
                )

                # Consider it a conflict if times differ by more than 1 second
                return abs((lrs_time - opencode_time).total_seconds()) > 1.0
            except (ValueError, AttributeError):
                pass

        return False

    def _determine_conflict_type(
        self, lrs_value: Any, opencode_value: Any
    ) -> StateConflictType:
        """Determine the type of conflict."""
        if isinstance(lrs_value, dict) and isinstance(opencode_value, dict):
            return StateConflictType.STRUCTURE_CONFLICT

        if isinstance(lrs_value, (int, float)) and isinstance(
            opencode_value, (int, float)
        ):
            return StateConflictType.VALUE_CONFLICT

        if isinstance(lrs_value, str) and isinstance(opencode_value, str):
            # Check if they look like timestamps
            try:
                datetime.fromisoformat(lrs_value.replace("Z", "+00:00"))
                datetime.fromisoformat(opencode_value.replace("Z", "+00:00"))
                return StateConflictType.TIMESTAMP_CONFLICT
            except ValueError:
                pass

            # Check if they look like status values
            lrs_lower = lrs_value.lower()
            opencode_lower = opencode_value.lower()
            if any(
                status in lrs_lower
                for status in ["idle", "active", "error", "completed"]
            ):
                return StateConflictType.STATUS_CONFLICT

            return StateConflictType.VALUE_CONFLICT

        # Check for precision data
        if (
            "precision" in str(type(lrs_value)).lower()
            or "precision" in str(type(opencode_value)).lower()
        ):
            return StateConflictType.PRECISION_CONFLICT

        return StateConflictType.VALUE_CONFLICT

    async def _resolve_conflict(
        self, conflict: StateConflict, strategy: ConflictResolutionStrategy
    ) -> Any:
        """Resolve a conflict using the specified strategy."""
        conflict.resolution_strategy = strategy

        if strategy == ConflictResolutionStrategy.TUI_WINS:
            conflict.resolved_value = conflict.opencode_value
            return conflict.opencode_value

        elif strategy == ConflictResolutionStrategy.LRS_WINS:
            conflict.resolved_value = conflict.lrs_value
            return conflict.lrs_value

        elif strategy == ConflictResolutionStrategy.TIMESTAMP:
            return self._resolve_by_timestamp(conflict)

        elif strategy == ConflictResolutionStrategy.MERGE:
            return await self._merge_values(conflict)

        # Default to LRS wins
        return conflict.lrs_value

    def _resolve_by_timestamp(self, conflict: StateConflict) -> Any:
        """Resolve conflict by choosing the most recent timestamp."""
        # For timestamp conflicts, choose the more recent time
        if conflict.conflict_type == StateConflictType.TIMESTAMP_CONFLICT:
            try:
                lrs_time = datetime.fromisoformat(
                    conflict.lrs_value.replace("Z", "+00:00")
                )
                opencode_time = datetime.fromisoformat(
                    conflict.opencode_value.replace("Z", "+00:00")
                )

                resolved_value = (
                    conflict.lrs_value
                    if lrs_time > opencode_time
                    else conflict.opencode_value
                )
                conflict.resolved_value = resolved_value
                return resolved_value
            except ValueError:
                pass

        # For other conflicts, prefer LRS (more likely to have accurate state)
        conflict.resolved_value = conflict.lrs_value
        return conflict.lrs_value

    async def _merge_values(self, conflict: StateConflict) -> Any:
        """Merge conflicting values when possible."""
        if isinstance(conflict.lrs_value, dict) and isinstance(
            conflict.opencode_value, dict
        ):
            # Recursively merge dictionaries
            merged, _ = await self.merge_states(
                conflict.lrs_value,
                conflict.opencode_value,
                ConflictResolutionStrategy.MERGE,
                conflict.field_path,
            )
            conflict.resolved_value = merged
            return merged

        elif isinstance(conflict.lrs_value, list) and isinstance(
            conflict.opencode_value, list
        ):
            # Merge lists by combining and deduplicating
            merged_list = list(set(conflict.lrs_value + conflict.opencode_value))
            conflict.resolved_value = merged_list
            return merged_list

        elif conflict.conflict_type == StateConflictType.PRECISION_CONFLICT:
            # For precision conflicts, use average or more conservative value
            if isinstance(conflict.lrs_value, (int, float)) and isinstance(
                conflict.opencode_value, (int, float)
            ):
                # Use the lower (more conservative) precision value
                merged_value = min(conflict.lrs_value, conflict.opencode_value)
                conflict.resolved_value = merged_value
                return merged_value

        # Default to LRS value
        conflict.resolved_value = conflict.lrs_value
        return conflict.lrs_value


class StateSynchronizer:
    """Manages state synchronization between opencode and LRS-Agents."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.state_merger = StateMerger(config)
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.sync_history: Dict[str, List[StateSyncResponse]] = {}
        self.last_sync_times: Dict[str, datetime] = {}
        self.sync_locks: Dict[str, asyncio.Lock] = {}

    async def sync_agent_state(
        self,
        agent_id: str,
        lrs_state: Optional[Dict[str, Any]] = None,
        opencode_state: Optional[Dict[str, Any]] = None,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.TIMESTAMP,
    ) -> StateSyncResponse:
        """Synchronize agent state between systems."""
        # Acquire lock for this agent
        if agent_id not in self.sync_locks:
            self.sync_locks[agent_id] = asyncio.Lock()

        async with self.sync_locks[agent_id]:
            return await self._perform_sync(
                agent_id, lrs_state, opencode_state, strategy
            )

    async def _perform_sync(
        self,
        agent_id: str,
        lrs_state: Optional[Dict[str, Any]],
        opencode_state: Optional[Dict[str, Any]],
        strategy: ConflictResolutionStrategy,
    ) -> StateSyncResponse:
        """Perform the actual synchronization."""
        try:
            # Get current cached state if new state not provided
            if lrs_state is None or opencode_state is None:
                current_state = self.agent_states.get(
                    agent_id, {"lrs": {}, "opencode": {}}
                )

                if lrs_state is None:
                    lrs_state = current_state.get("lrs", {})
                if opencode_state is None:
                    opencode_state = current_state.get("opencode", {})

            # Merge states
            merged_state, conflicts = await self.state_merger.merge_states(
                lrs_state, opencode_state, strategy
            )

            # Create response
            response = StateSyncResponse(
                success=True,
                resolved_state=merged_state,
                conflicts=[c.field_path for c in conflicts],
                strategy_used=strategy,
                timestamp=datetime.utcnow(),
            )

            # Update cached state
            self.agent_states[agent_id] = {
                "lrs": lrs_state,
                "opencode": opencode_state,
                "merged": merged_state,
                "last_sync": datetime.utcnow(),
            }

            # Store sync history
            if agent_id not in self.sync_history:
                self.sync_history[agent_id] = []

            self.sync_history[agent_id].append(response)

            # Keep only last 50 syncs per agent
            if len(self.sync_history[agent_id]) > 50:
                self.sync_history[agent_id] = self.sync_history[agent_id][-50:]

            self.last_sync_times[agent_id] = datetime.utcnow()

            logger.info(
                "Agent state synchronized",
                agent_id=agent_id,
                conflicts_found=len(conflicts),
                strategy=strategy,
            )

            return response

        except Exception as e:
            logger.error(
                "State synchronization failed", agent_id=agent_id, error=str(e)
            )

            return StateSyncResponse(
                success=False,
                resolved_state={},
                conflicts=[f"Sync error: {str(e)}"],
                strategy_used=strategy,
                timestamp=datetime.utcnow(),
            )

    async def resolve_conflict_manually(
        self, agent_id: str, field_path: str, resolved_value: Any
    ) -> StateSyncResponse:
        """Manually resolve a specific conflict."""
        if agent_id not in self.agent_states:
            raise ValueError(f"No cached state found for agent {agent_id}")

        current_state = self.agent_states[agent_id]["merged"]

        # Update the specific field
        self._update_nested_field(current_state, field_path, resolved_value)

        # Create sync response
        response = StateSyncResponse(
            success=True,
            resolved_state=current_state,
            conflicts=[],
            strategy_used=ConflictResolutionStrategy.MERGE,
            timestamp=datetime.utcnow(),
        )

        # Update cache
        self.agent_states[agent_id]["merged"] = current_state

        return response

    def _update_nested_field(self, state: Dict[str, Any], field_path: str, value: Any):
        """Update a nested field in the state dictionary."""
        keys = field_path.split(".")
        current = state

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get_sync_history(
        self, agent_id: str, limit: int = 20
    ) -> List[StateSyncResponse]:
        """Get synchronization history for an agent."""
        if agent_id not in self.sync_history:
            return []

        return self.sync_history[agent_id][-limit:]

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current synchronized state for an agent."""
        if agent_id not in self.agent_states:
            return None

        return self.agent_states[agent_id].get("merged", {})

    async def force_sync_all_agents(self, strategy: ConflictResolutionStrategy):
        """Force synchronization for all agents."""
        sync_tasks = []

        for agent_id in list(self.agent_states.keys()):
            cached_state = self.agent_states[agent_id]
            task = asyncio.create_task(
                self._perform_sync(
                    agent_id,
                    cached_state.get("lrs", {}),
                    cached_state.get("opencode", {}),
                    strategy,
                )
            )
            sync_tasks.append(task)

        if sync_tasks:
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)

            successful_syncs = sum(
                1 for r in results if isinstance(r, StateSyncResponse) and r.success
            )
            logger.info(
                "Force sync completed",
                total_agents=len(sync_tasks),
                successful=successful_syncs,
            )

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        total_agents = len(self.agent_states)
        total_syncs = sum(len(history) for history in self.sync_history.values())

        conflict_strategies = {}
        for history in self.sync_history.values():
            for sync in history:
                strategy = sync.strategy_used
                conflict_strategies[strategy] = conflict_strategies.get(strategy, 0) + 1

        recent_syncs = 0
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        for last_sync in self.last_sync_times.values():
            if last_sync > one_hour_ago:
                recent_syncs += 1

        return {
            "total_agents": total_agents,
            "total_synchronizations": total_syncs,
            "agents_synced_recently": recent_syncs,
            "conflict_resolution_strategies": conflict_strategies,
            "active_sync_locks": len(
                [lock for lock in self.sync_locks.values() if lock.locked()]
            ),
        }


class StateChangeDetector:
    """Detects and notifies about state changes."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.state_hashes: Dict[str, str] = {}
        self.change_handlers: List[Callable] = []

    def register_change_handler(self, handler: Callable):
        """Register a handler for state change notifications."""
        self.change_handlers.append(handler)

    async def detect_changes(self, agent_id: str, new_state: Dict[str, Any]) -> bool:
        """Detect if state has changed since last check."""
        state_json = json.dumps(new_state, sort_keys=True, default=str)
        new_hash = hashlib.sha256(state_json.encode()).hexdigest()

        old_hash = self.state_hashes.get(agent_id)

        if old_hash != new_hash:
            self.state_hashes[agent_id] = new_hash

            # Notify change handlers
            for handler in self.change_handlers:
                try:
                    await handler(agent_id, new_state, old_hash is not None)
                except Exception as e:
                    logger.error("Change handler error", error=str(e))

            return True

        return False

    def has_state(self, agent_id: str) -> bool:
        """Check if we have cached state for an agent."""
        return agent_id in self.state_hashes
