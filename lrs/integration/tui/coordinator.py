"""
TUI Multi-Agent Coordinator: Enhanced coordination with TUI awareness.

This component extends the base MultiAgentCoordinator to provide TUI-specific
features like shared dashboard state, TUI-driven coordination, and user
interaction handling.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

from ...multi_agent.coordinator import MultiAgentCoordinator
from ...multi_agent.shared_state import SharedWorldState
from ...core.registry import ToolRegistry
from .state_mirror import TUIStateMirror
from .tool import TUIInteractionTool


@dataclass
class TUIAgentConfig:
    """Configuration for TUI-aware agents."""

    agent_id: str
    agent_type: str
    tui_visible: bool = True
    tui_panel: Optional[str] = None
    user_interactions: List[str] = field(default_factory=list)
    dashboard_metrics: List[str] = field(default_factory=list)
    coordination_group: Optional[str] = None


@dataclass
class TUICoordinationEvent:
    """Event for TUI-driven coordination."""

    event_type: str
    agent_ids: List[str]
    data: Dict[str, Any]
    timestamp: datetime
    user_initiated: bool = False
    tui_context: Dict[str, Any] = field(default_factory=dict)


class TUIMultiAgentCoordinator(MultiAgentCoordinator):
    """
    Enhanced multi-agent coordinator with TUI integration.

    Extends the base coordinator with:
    - TUI-specific agent management
    - Shared dashboard state coordination
    - User interaction handling
    - TUI-driven coordination events
    - Real-time visualization of coordination
    - Agent grouping and team management

    Examples:
        >>> coordinator = TUIMultiAgentCoordinator(tui_bridge, shared_state)
        >>>
        >>> # Register TUI-aware agent
        >>> config = TUIAgentConfig(
        ...     agent_id="agent_1",
        ...     agent_type="lrs",
        ...     tui_panel="main",
        ...     dashboard_metrics=["precision", "task_progress"]
        ... )
        >>> coordinator.register_tui_agent(config)
        >>>
        >>> # Coordinate agents via TUI
        >>> await coordinator.coordinate_via_tui(
        ...     agent_ids=["agent_1", "agent_2"],
        ...     coordination_type="collaborative_task",
        ...     task_data={"goal": "analyze_data"}
        ... )
    """

    def __init__(self, tui_bridge, shared_state: SharedWorldState, tool_registry: ToolRegistry):
        """
        Initialize TUI Multi-Agent Coordinator.

        Args:
            tui_bridge: TUIBridge instance
            shared_state: Shared world state
            tool_registry: Tool registry
        """
        super().__init__(shared_state)

        self.tui_bridge = tui_bridge
        self.tool_registry = tool_registry

        # TUI-specific agent management
        self.tui_agents: Dict[str, TUIAgentConfig] = {}
        self.coordination_groups: Dict[str, List[str]] = {}

        # TUI state management
        self.dashboard_state: Dict[str, Any] = {}
        self.coordination_history: List[TUICoordinationEvent] = []

        # Event handlers
        self.coordination_handlers: Dict[str, Callable] = {}
        self.user_interaction_handlers: Dict[str, Callable] = {}

        # Background tasks
        self.coordination_task: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(__name__)

        # Setup TUI-specific coordination
        self._setup_tui_coordination()

        # Start background tasks
        self._start_background_tasks()

    def register_tui_agent(self, config: TUIAgentConfig):
        """
        Register agent with TUI-specific configuration.

        Args:
            config: TUI agent configuration
        """
        self.tui_agents[config.agent_id] = config

        # Add to coordination group if specified
        if config.coordination_group:
            if config.coordination_group not in self.coordination_groups:
                self.coordination_groups[config.coordination_group] = []
            self.coordination_groups[config.coordination_group].append(config.agent_id)

        # Initialize agent TUI state
        self._initialize_agent_tui_state(config)

        # Setup user interaction handlers
        for interaction_type in config.user_interactions:
            self._setup_user_interaction_handler(config.agent_id, interaction_type)

        self.logger.info(f"Registered TUI agent: {config.agent_id}")

        # Broadcast agent registration
        asyncio.create_task(self._broadcast_agent_registration(config))

    def unregister_tui_agent(self, agent_id: str):
        """
        Unregister TUI agent.

        Args:
            agent_id: Agent ID to unregister
        """
        if agent_id in self.tui_agents:
            config = self.tui_agents[agent_id]

            # Remove from coordination groups
            if config.coordination_group and config.coordination_group in self.coordination_groups:
                self.coordination_groups[config.coordination_group].remove(agent_id)

                # Clean up empty groups
                if not self.coordination_groups[config.coordination_group]:
                    del self.coordination_groups[config.coordination_group]

            # Remove agent config
            del self.tui_agents[agent_id]

            self.logger.info(f"Unregistered TUI agent: {agent_id}")

            # Broadcast agent unregistration
            asyncio.create_task(self._broadcast_agent_unregistration(agent_id))

    async def coordinate_via_tui(
        self,
        agent_ids: List[str],
        coordination_type: str,
        data: Dict[str, Any],
        user_initiated: bool = False,
    ) -> Dict[str, Any]:
        """
        Coordinate agents through TUI interface.

        Args:
            agent_ids: Agents to coordinate
            coordination_type: Type of coordination
            data: Coordination data
            user_initiated: Whether initiated by user

        Returns:
            Coordination result
        """
        try:
            # Validate agents
            for agent_id in agent_ids:
                if agent_id not in self.tui_agents:
                    raise ValueError(f"Agent {agent_id} not registered as TUI agent")

            # Create coordination event
            event = TUICoordinationEvent(
                event_type=coordination_type,
                agent_ids=agent_ids,
                data=data,
                timestamp=datetime.now(),
                user_initiated=user_initiated,
            )

            # Add to history
            self.coordination_history.append(event)

            # Keep history manageable
            if len(self.coordination_history) > 1000:
                self.coordination_history = self.coordination_history[-1000:]

            # Handle coordination based on type
            handler = self.coordination_handlers.get(coordination_type)
            if handler:
                result = await handler(event)
            else:
                result = await self._handle_default_coordination(event)

            # Update dashboard state
            await self._update_dashboard_coordination(event, result)

            # Broadcast coordination event
            await self._broadcast_coordination_event(event, result)

            return {
                "coordination_id": str(id(event)),
                "agent_ids": agent_ids,
                "coordination_type": coordination_type,
                "result": result,
                "timestamp": event.timestamp.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error in TUI coordination: {e}")
            raise

    async def get_tui_dashboard_state(self) -> Dict[str, Any]:
        """
        Get comprehensive TUI dashboard state.

        Returns:
            Dashboard state data
        """
        # Collect agent states
        agent_states = {}
        for agent_id, config in self.tui_agents.items():
            agent_state = self.shared_state.get_agent_state(agent_id)
            if agent_state:
                # Filter to dashboard metrics
                dashboard_data = {}
                for metric in config.dashboard_metrics:
                    if metric in agent_state:
                        dashboard_data[metric] = agent_state[metric]

                agent_states[agent_id] = {
                    "config": {
                        "agent_type": config.agent_type,
                        "tui_panel": config.tui_panel,
                        "coordination_group": config.coordination_group,
                    },
                    "state": dashboard_data,
                    "last_update": agent_state.get("last_update"),
                }

        # Get coordination group states
        group_states = {}
        for group_name, agent_ids in self.coordination_groups.items():
            group_states[group_name] = {
                "agent_ids": agent_ids,
                "active_coordinations": len(
                    [
                        event
                        for event in self.coordination_history
                        if event.agent_ids == agent_ids
                        and datetime.now() - event.timestamp < timedelta(minutes=5)
                    ]
                ),
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "agents": agent_states,
            "coordination_groups": group_states,
            "recent_coordinations": [
                {
                    "event_type": event.event_type,
                    "agent_ids": event.agent_ids,
                    "timestamp": event.timestamp.isoformat(),
                    "user_initiated": event.user_initiated,
                }
                for event in self.coordination_history[-20:]
            ],
            "system_metrics": {
                "total_agents": len(self.tui_agents),
                "active_groups": len(self.coordination_groups),
                "recent_coordinations": len(
                    [
                        event
                        for event in self.coordination_history
                        if datetime.now() - event.timestamp < timedelta(hours=1)
                    ]
                ),
            },
        }

    async def handle_user_interaction(self, interaction_data: Dict[str, Any]):
        """
        Handle user interaction from TUI.

        Args:
            interaction_data: User interaction details
        """
        try:
            interaction_type = interaction_data.get("type")
            agent_id = interaction_data.get("agent_id")

            if not interaction_type:
                return

            # Route to specific handler
            handler_key = f"{agent_id}:{interaction_type}" if agent_id else interaction_type
            handler = self.user_interaction_handlers.get(handler_key)

            if handler:
                await handler(interaction_data)
            else:
                await self._handle_default_user_interaction(interaction_data)

        except Exception as e:
            self.logger.error(f"Error handling user interaction: {e}")

    def _setup_tui_coordination(self):
        """Setup TUI-specific coordination handlers."""

        # Register default coordination handlers
        self.coordination_handlers.update(
            {
                "collaborative_task": self._handle_collaborative_task,
                "resource_sharing": self._handle_resource_sharing,
                "state_synchronization": self._handle_state_synchronization,
                "load_balancing": self._handle_load_balancing,
                "error_recovery": self._handle_error_recovery,
            }
        )

    def _setup_user_interaction_handler(self, agent_id: str, interaction_type: str):
        """Setup handler for specific user interaction."""
        handler_key = f"{agent_id}:{interaction_type}"

        # Default handlers
        if interaction_type == "tool_override":
            self.user_interaction_handlers[handler_key] = self._handle_tool_override
        elif interaction_type == "precision_adjustment":
            self.user_interaction_handlers[handler_key] = self._handle_precision_adjustment
        elif interaction_type == "task_reassignment":
            self.user_interaction_handlers[handler_key] = self._handle_task_reassignment

    def _initialize_agent_tui_state(self, config: TUIAgentConfig):
        """Initialize TUI state for agent."""

        initial_state = {
            "tui_agent_config": {
                "agent_type": config.agent_type,
                "tui_visible": config.tui_visible,
                "tui_panel": config.tui_panel,
                "coordination_group": config.coordination_group,
            },
            "tui_registered_at": datetime.now().isoformat(),
            "tui_last_coordination": None,
        }

        self.shared_state.update(config.agent_id, initial_state)

    async def _handle_collaborative_task(self, event: TUICoordinationEvent) -> Dict[str, Any]:
        """Handle collaborative task coordination."""

        task_data = event.data

        # Distribute task among agents
        task_distribution = {}
        for i, agent_id in enumerate(event.agent_ids):
            task_distribution[agent_id] = {
                "task_id": task_data.get("task_id"),
                "role": f"participant_{i + 1}",
                "subtask": f"{task_data.get('goal', 'task')}_part_{i + 1}",
                "collaborators": [aid for aid in event.agent_ids if aid != agent_id],
            }

        # Update agent states with task assignments
        for agent_id, assignment in task_distribution.items():
            self.shared_state.update(
                agent_id,
                {
                    "current_task": assignment,
                    "collaboration_active": True,
                    "collaboration_id": str(id(event)),
                },
            )

        return {
            "status": "initiated",
            "task_distribution": task_distribution,
            "collaboration_id": str(id(event)),
        }

    async def _handle_resource_sharing(self, event: TUICoordinationEvent) -> Dict[str, Any]:
        """Handle resource sharing coordination."""

        resource_data = event.data
        resource_type = resource_data.get("resource_type")
        sharing_agent = resource_data.get("sharing_agent")
        receiving_agents = [aid for aid in event.agent_ids if aid != sharing_agent]

        # Update agent states with resource sharing info
        for agent_id in receiving_agents:
            self.shared_state.update(
                agent_id,
                {
                    "shared_resources": {
                        resource_type: {
                            "source": sharing_agent,
                            "timestamp": event.timestamp.isoformat(),
                            "status": "available",
                        }
                    }
                },
            )

        self.shared_state.update(
            sharing_agent,
            {
                "shared_resources_outbound": {
                    resource_type: {
                        "recipients": receiving_agents,
                        "timestamp": event.timestamp.isoformat(),
                    }
                }
            },
        )

        return {
            "status": "shared",
            "resource_type": resource_type,
            "sharing_agent": sharing_agent,
            "receiving_agents": receiving_agents,
        }

    async def _handle_state_synchronization(self, event: TUICoordinationEvent) -> Dict[str, Any]:
        """Handle state synchronization coordination."""

        sync_data = event.data
        state_keys = sync_data.get("state_keys", [])
        source_agent = sync_data.get("source_agent")

        if source_agent and source_agent in event.agent_ids:
            # Get source agent state
            source_state = self.shared_state.get_agent_state(source_agent)

            # Filter to requested state keys
            filtered_state = {key: source_state[key] for key in state_keys if key in source_state}

            # Distribute to other agents
            for agent_id in event.agent_ids:
                if agent_id != source_agent:
                    self.shared_state.update(
                        agent_id,
                        {
                            "synchronized_state": filtered_state,
                            "sync_source": source_agent,
                            "sync_timestamp": event.timestamp.isoformat(),
                        },
                    )

            return {
                "status": "synchronized",
                "source_agent": source_agent,
                "state_keys": state_keys,
                "target_agents": [aid for aid in event.agent_ids if aid != source_agent],
            }

        return {"status": "no_source_specified"}

    async def _handle_load_balancing(self, event: TUICoordinationEvent) -> Dict[str, Any]:
        """Handle load balancing coordination."""

        load_data = event.data

        # Get current load metrics for all agents
        agent_loads = {}
        for agent_id in event.agent_ids:
            state = self.shared_state.get_agent_state(agent_id)
            load = state.get("current_load", 0.0)
            agent_loads[agent_id] = load

        # Find most and least loaded agents
        most_loaded = max(agent_loads.items(), key=lambda x: x[1])
        least_loaded = min(agent_loads.items(), key=lambda x: x[1])

        # Redistribute if load difference is significant
        if most_loaded[1] - least_loaded[1] > 0.3:
            # Suggest task redistribution
            return {
                "status": "redistribution_suggested",
                "redistribute_from": most_loaded[0],
                "redistribute_to": least_loaded[0],
                "load_difference": most_loaded[1] - least_loaded[1],
            }

        return {"status": "balanced"}

    async def _handle_error_recovery(self, event: TUICoordinationEvent) -> Dict[str, Any]:
        """Handle error recovery coordination."""

        error_data = event.data
        failed_agent = error_data.get("failed_agent")
        error_type = error_data.get("error_type")

        if failed_agent and failed_agent in event.agent_ids:
            # Find healthy agents for recovery
            healthy_agents = [aid for aid in event.agent_ids if aid != failed_agent]

            # Update failed agent state
            self.shared_state.update(
                failed_agent,
                {
                    "error_status": {
                        "error_type": error_type,
                        "timestamp": event.timestamp.isoformat(),
                        "recovery_coordinated": True,
                    }
                },
            )

            # Prepare recovery for healthy agents
            for agent_id in healthy_agents:
                self.shared_state.update(
                    agent_id,
                    {
                        "recovery_assignment": {
                            "failed_agent": failed_agent,
                            "error_type": error_type,
                            "timestamp": event.timestamp.isoformat(),
                        }
                    },
                )

            return {
                "status": "recovery_coordinated",
                "failed_agent": failed_agent,
                "recovery_agents": healthy_agents,
                "error_type": error_type,
            }

        return {"status": "no_failed_agent_specified"}

    async def _handle_default_coordination(self, event: TUICoordinationEvent) -> Dict[str, Any]:
        """Handle unknown coordination types."""

        # Default coordination logic
        for agent_id in event.agent_ids:
            self.shared_state.update(
                agent_id,
                {
                    "coordination_event": {
                        "type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data,
                    }
                },
            )

        return {
            "status": "default_handling",
            "agent_ids": event.agent_ids,
            "event_type": event.event_type,
        }

    async def _handle_tool_override(self, interaction_data: Dict[str, Any]):
        """Handle tool override user interaction."""

        agent_id = interaction_data.get("agent_id")
        tool_name = interaction_data.get("tool_name")
        override_action = interaction_data.get("action")

        # Update agent state with tool override
        self.shared_state.update(
            agent_id,
            {
                "tool_override": {
                    "tool_name": tool_name,
                    "action": override_action,
                    "timestamp": datetime.now().isoformat(),
                    "user_initiated": True,
                }
            },
        )

        self.logger.info(f"User tool override for {agent_id}: {tool_name} -> {override_action}")

    async def _handle_precision_adjustment(self, interaction_data: Dict[str, Any]):
        """Handle precision adjustment user interaction."""

        agent_id = interaction_data.get("agent_id")
        precision_value = interaction_data.get("precision_value")

        # Update agent precision
        self.shared_state.update(
            agent_id,
            {
                "precision_adjustment": {
                    "new_value": precision_value,
                    "timestamp": datetime.now().isoformat(),
                    "user_initiated": True,
                }
            },
        )

        self.logger.info(f"User precision adjustment for {agent_id}: {precision_value}")

    async def _handle_task_reassignment(self, interaction_data: Dict[str, Any]):
        """Handle task reassignment user interaction."""

        from_agent = interaction_data.get("from_agent")
        to_agent = interaction_data.get("to_agent")
        task_data = interaction_data.get("task_data")

        # Update task assignment
        self.shared_state.update(
            from_agent,
            {
                "task_reassigned_outbound": {
                    "to_agent": to_agent,
                    "task_data": task_data,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

        self.shared_state.update(
            to_agent,
            {
                "task_reassigned_inbound": {
                    "from_agent": from_agent,
                    "task_data": task_data,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

        self.logger.info(f"User task reassignment: {from_agent} -> {to_agent}")

    async def _handle_default_user_interaction(self, interaction_data: Dict[str, Any]):
        """Handle unknown user interaction types."""

        # Log interaction for future analysis
        self.shared_state.update(
            "system",
            {
                "user_interaction_log": {
                    "interaction": interaction_data,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

    async def _update_dashboard_coordination(
        self, event: TUICoordinationEvent, result: Dict[str, Any]
    ):
        """Update dashboard state with coordination info."""

        coordination_key = f"{event.event_type}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}"

        self.dashboard_state[coordination_key] = {
            "event": {
                "type": event.event_type,
                "agent_ids": event.agent_ids,
                "user_initiated": event.user_initiated,
            },
            "result": result,
            "timestamp": event.timestamp.isoformat(),
        }

        # Keep dashboard state manageable
        if len(self.dashboard_state) > 100:
            oldest_keys = sorted(self.dashboard_state.keys())[:50]
            for key in oldest_keys:
                del self.dashboard_state[key]

    async def _broadcast_agent_registration(self, config: TUIAgentConfig):
        """Broadcast agent registration to TUI."""

        await self.tui_bridge.websocket_manager.broadcast(
            "tui_events",
            {
                "type": "agent_registered",
                "agent_id": config.agent_id,
                "config": {
                    "agent_type": config.agent_type,
                    "tui_panel": config.tui_panel,
                    "coordination_group": config.coordination_group,
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _broadcast_agent_unregistration(self, agent_id: str):
        """Broadcast agent unregistration to TUI."""

        await self.tui_bridge.websocket_manager.broadcast(
            "tui_events",
            {
                "type": "agent_unregistered",
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _broadcast_coordination_event(
        self, event: TUICoordinationEvent, result: Dict[str, Any]
    ):
        """Broadcast coordination event to TUI."""

        await self.tui_bridge.websocket_manager.broadcast(
            "tui_events",
            {
                "type": "coordination_event",
                "coordination": {
                    "event_type": event.event_type,
                    "agent_ids": event.agent_ids,
                    "result": result,
                    "user_initiated": event.user_initiated,
                    "timestamp": event.timestamp.isoformat(),
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _start_background_tasks(self):
        """Start background coordination tasks."""

        self.coordination_task = asyncio.create_task(self._coordination_monitoring_loop())

    async def _coordination_monitoring_loop(self):
        """Background loop for coordination monitoring."""
        while True:
            try:
                # Monitor for coordination opportunities
                await self._monitor_coordination_opportunities()

                # Clean old coordination history
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.coordination_history = [
                    event for event in self.coordination_history if event.timestamp > cutoff_time
                ]

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in coordination monitoring: {e}")
                await asyncio.sleep(10)

    async def _monitor_coordination_opportunities(self):
        """Monitor for potential coordination opportunities."""

        # Check for agents with similar tasks
        agent_tasks = {}
        for agent_id in self.tui_agents:
            state = self.shared_state.get_agent_state(agent_id)
            current_task = state.get("current_task")
            if current_task:
                task_key = current_task.get("task_id", current_task.get("goal", "unknown"))
                if task_key not in agent_tasks:
                    agent_tasks[task_key] = []
                agent_tasks[task_key].append(agent_id)

        # Suggest coordination for agents with similar tasks
        for task_key, agents in agent_tasks.items():
            if len(agents) > 1:
                # Check if coordination already active
                existing_coordination = any(
                    event.event_type == "collaborative_task"
                    and set(event.agent_ids) == set(agents)
                    and datetime.now() - event.timestamp < timedelta(minutes=10)
                    for event in self.coordination_history
                )

                if not existing_coordination:
                    # Suggest coordination opportunity
                    await self.tui_bridge.websocket_manager.broadcast(
                        "tui_events",
                        {
                            "type": "coordination_opportunity",
                            "task_key": task_key,
                            "suggested_agents": agents,
                            "coordination_type": "collaborative_task",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )


# Import timedelta for coordination monitoring
from datetime import timedelta
