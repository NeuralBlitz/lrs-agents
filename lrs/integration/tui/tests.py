"""
Comprehensive tests for TUI integration.

This module provides extensive test coverage for all TUI integration components,
including unit tests, integration tests, and performance benchmarks.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Import TUI integration components
from ..tool import TUIInteractionTool
from ..state_mirror import TUIStateMirror
from ..precision_mapper import TUIPrecisionMapper, ConfidenceLevel
from ..coordinator import TUIMultiAgentCoordinator, TUIAgentConfig
from ..config import TUIConfigManager, TUIIntegrationConfig


class TestTUIConfig:
    """Test TUI configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TUIIntegrationConfig()

        assert config.websocket.port == 8000
        assert config.rest.port == 8001
        assert config.enable_websockets is True
        assert config.enable_rest_api is True
        assert config.debug is False
        assert config.environment == "development"

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "websocket": {"port": 8080},
            "rest": {"port": 8081},
            "debug": True,
            "environment": "production",
        }

        config = TUIConfigManager._dict_to_config(config_dict)

        assert config.websocket.port == 8080
        assert config.rest.port == 8081
        assert config.debug is True
        assert config.environment == "production"

    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = TUIConfigManager()

        # Valid config
        assert config_manager.validate_config() is True

        # Invalid ports
        config_manager.config.websocket.port = 70000
        assert config_manager.validate_config() is False

        # Same ports
        config_manager.config.websocket.port = 8000
        config_manager.config.rest.port = 8000
        assert config_manager.validate_config() is False

    def test_config_update(self):
        """Test configuration updates."""
        config_manager = TUIConfigManager()

        updates = {"websocket": {"port": 8080}, "debug": True}

        success = config_manager.update_config(updates)
        assert success is True
        assert config_manager.config.websocket.port == 8080
        assert config_manager.config.debug is True


class TestTUIInteractionTool:
    """Test TUI Interaction Tool."""

    @pytest.fixture
    def mock_tui_bridge(self):
        """Mock TUI bridge for testing."""
        bridge = Mock()
        bridge.websocket_manager = Mock()
        bridge.websocket_manager.broadcast = AsyncMock()
        bridge.state_mirror = Mock()
        bridge.state_mirror.get_user_preference = Mock(return_value="test_value")
        bridge.shared_state = Mock()
        bridge.shared_state.get_all_states = Mock(return_value=["agent_1", "agent_2"])

        return bridge

    @pytest.fixture
    def tui_tool(self, mock_tui_bridge):
        """Create TUI interaction tool."""
        return TUIInteractionTool(mock_tui_bridge)

    def test_tool_initialization(self, tui_tool, mock_tui_bridge):
        """Test tool initialization."""
        assert tui_tool.name == "tui_interaction"
        assert tui_tool.tui_bridge == mock_tui_bridge
        assert "query" in tui_tool.input_schema["properties"]["action"]["enum"]

    @pytest.mark.asyncio
    async def test_query_user_preference(self, tui_tool, mock_tui_bridge):
        """Test querying user preferences."""
        state = {"action": "query", "query_type": "user_preference", "preference_key": "theme"}

        result = tui_tool.get(state)

        assert result.success is True
        assert "response" in result.value
        assert result.value["response"]["user_preferences"]["theme"] == "test_value"
        assert result.prediction_error == 0.1

    @pytest.mark.asyncio
    async def test_command_notification(self, tui_tool, mock_tui_bridge):
        """Test sending notification command."""
        state = {
            "action": "command",
            "command": "show_notification",
            "message": "Test notification",
            "notification_type": "info",
        }

        result = tui_tool.get(state)

        assert result.success is True
        assert result.value["response"]["notification_sent"] is True

        # Verify broadcast was called
        mock_tui_bridge.websocket_manager.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_update(self, tui_tool, mock_tui_bridge):
        """Test state update action."""
        state = {"action": "state_update", "state_data": {"status": "active", "progress": 0.75}}

        result = tui_tool.get(state)

        assert result.success is True
        assert result.value["response"]["state_updated"] is True

        # Verify shared state was updated
        mock_tui_bridge.shared_state.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_input_request(self, tui_tool):
        """Test user input request."""
        state = {"action": "user_input", "prompt": "Please enter a value:", "timeout": 5.0}

        result = tui_tool.get(state)

        assert result.success is False  # Timeout expected
        assert "timed out" in result.error

    def test_belief_state_update(self, tui_tool):
        """Test belief state update (set operation)."""
        state = {"existing_key": "existing_value"}
        observation = {
            "response": {"user_preferences": {"theme": "dark"}, "tui_state": {"panel": "main"}}
        }

        updated_state = tui_tool.set(state, observation)

        assert "tui_context" in updated_state
        assert updated_state["tui_context"]["user_preferences"]["theme"] == "dark"
        assert updated_state["tui_context"]["tui_state"]["panel"] == "main"
        assert "last_interaction" in updated_state["tui_context"]

    def test_success_rate_calculation(self, tui_tool):
        """Test success rate calculation."""
        # Initially no calls
        assert tui_tool.success_rate == 0.5

        # Simulate some calls
        tui_tool.call_count = 10
        tui_tool.failure_count = 2

        assert tui_tool.success_rate == 0.8


class TestTUIStateMirror:
    """Test TUI State Mirror."""

    @pytest.fixture
    def mock_shared_state(self):
        """Mock shared state."""
        state = Mock()
        state.get_agent_state = Mock(return_value={"precision": {"value": 0.8}})
        state.update = Mock()
        state.get_all_states = Mock(return_value={"agent_1": {"status": "active"}})
        return state

    @pytest.fixture
    def mock_tui_bridge(self):
        """Mock TUI bridge."""
        bridge = Mock()
        bridge.websocket_manager = Mock()
        bridge.websocket_manager.broadcast = AsyncMock()
        return bridge

    @pytest.fixture
    def state_mirror(self, mock_shared_state, mock_tui_bridge):
        """Create state mirror."""
        return TUIStateMirror(mock_shared_state, mock_tui_bridge)

    @pytest.mark.asyncio
    async def test_precision_sync_to_tui(self, state_mirror):
        """Test precision synchronization to TUI."""
        precision_changes = {"value": 0.85, "confidence": "high"}

        await state_mirror.sync_precision_to_tui("agent_1", precision_changes)

        # Verify LRS state updated
        assert "precision" in state_mirror.lrs_state["agent_1"]
        assert state_mirror.lrs_state["agent_1"]["precision"]["value"] == 0.85

        # Verify TUI state updated
        assert "precision" in state_mirror.tui_state["agent_1"]
        assert state_mirror.tui_state["agent_1"]["precision"]["value"] == 0.85

    @pytest.mark.asyncio
    async def test_tool_execution_sync(self, state_mirror):
        """Test tool execution synchronization."""
        execution_result = {"tool": "search_tool", "success": True, "prediction_error": 0.1}

        await state_mirror.sync_tool_execution_to_tui("agent_1", execution_result)

        # Verify execution history
        executions = state_mirror.lrs_state["agent_1"]["tool_executions"]
        assert len(executions) == 1
        assert executions[0]["tool"] == "search_tool"

    @pytest.mark.asyncio
    async def test_sync_from_tui(self, state_mirror):
        """Test synchronization from TUI."""
        tui_updates = {"agent_1": {"status": "inactive", "user_preference": "dark_theme"}}

        await state_mirror.sync_from_tui(tui_updates)

        # Verify TUI state updated
        assert state_mirror.tui_state["agent_1"]["status"] == "inactive"

        # Verify shared state updated
        state_mirror.shared_state.update.assert_called_once()

    def test_user_preferences(self, state_mirror):
        """Test user preference management."""
        # Test setting and getting preferences
        state_mirror.set_user_preference("theme", "dark")
        assert state_mirror.get_user_preference("theme") == "dark"

        # Test non-existent preference
        assert state_mirror.get_user_preference("nonexistent") is None

    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self, state_mirror):
        """Test conflict detection and resolution."""
        # Set initial states
        state_mirror.lrs_state["agent_1"] = {"status": "active"}
        state_mirror.tui_state["agent_1"] = {"status": "inactive"}

        # Test conflict detection
        conflict = await state_mirror._detect_conflict("agent_1", "status", "pending")
        assert conflict is not None
        assert conflict["lrs_value"] == "active"
        assert conflict["tui_value"] == "inactive"

        # Test conflict resolution
        resolution = await state_mirror._resolve_conflict(conflict)
        assert "resolved_value" in resolution
        assert resolution["strategy"] == "tui_wins"

    def test_state_history(self, state_mirror):
        """Test state change history."""
        # Add some history
        from ..state_mirror import StateSnapshot

        snapshot = StateSnapshot(
            agent_id="agent_1",
            state_data={"status": "active"},
            timestamp=datetime.now(),
            source="lrs",
        )
        state_mirror._add_snapshot(snapshot)

        # Get history
        history = state_mirror.get_state_history("agent_1")
        assert len(history) == 1
        assert history[0]["agent_id"] == "agent_1"
        assert history[0]["source"] == "lrs"


class TestTUIPrecisionMapper:
    """Test TUI Precision Mapper."""

    @pytest.fixture
    def precision_mapper(self):
        """Create precision mapper."""
        return TUIPrecisionMapper()

    def test_precision_to_confidence(self, precision_mapper):
        """Test precision to confidence mapping."""
        # Test high precision
        precision_data = {"value": 0.9}
        confidence = precision_mapper.precision_to_confidence(precision_data)

        assert confidence.level == ConfidenceLevel.VERY_HIGH
        assert confidence.score == 90.0
        assert confidence.color == "#10b981"
        assert confidence.icon == "confidence-very-high"

        # Test low precision
        precision_data = {"value": 0.2}
        confidence = precision_mapper.precision_to_confidence(precision_data)

        assert confidence.level == ConfidenceLevel.VERY_LOW
        assert confidence.score == 20.0
        assert confidence.color == "#ef4444"

    def test_adaptation_alert_generation(self, precision_mapper):
        """Test adaptation alert generation."""
        adaptation_event = {
            "agent_id": "agent_1",
            "precision_before": 0.8,
            "precision_after": 0.3,
            "trigger_tool": "search_tool",
        }

        alert = precision_mapper.adaptation_to_tui_alert(adaptation_event)

        assert alert.agent_id == "agent_1"
        assert alert.precision_before == 0.8
        assert alert.precision_after == 0.3
        assert "drop" in alert.alert_type
        assert alert.severity.value == "error"  # Should be error due to low final precision
        assert len(alert.suggested_actions) > 0

    def test_multiple_confidence_indicators(self, precision_mapper):
        """Test multiple confidence indicators."""
        precision_values = [0.9, 0.6, 0.3, 0.1]
        indicators = precision_mapper.get_confidence_indicators(precision_values)

        assert len(indicators) == 4
        assert indicators[0].level == ConfidenceLevel.VERY_HIGH
        assert indicators[1].level == ConfidenceLevel.MEDIUM
        assert indicators[2].level == ConfidenceLevel.LOW
        assert indicators[3].level == ConfidenceLevel.VERY_LOW

    def test_precision_summary(self, precision_mapper):
        """Test precision summary generation."""
        precision_data = {"value": 0.75, "agent_id": "agent_1"}
        summary = precision_mapper.get_precision_summary("agent_1", precision_data)

        assert summary["agent_id"] == "agent_1"
        assert summary["current_precision"] == 0.75
        assert "confidence" in summary
        assert "historical" in summary
        assert "prediction" in summary
        assert summary["confidence"]["level"] == ConfidenceLevel.HIGH.value

    def test_dashboard_metrics(self, precision_mapper):
        """Test dashboard metrics calculation."""
        all_precisions = {
            "agent_1": {"value": 0.9},
            "agent_2": {"value": 0.6},
            "agent_3": {"value": 0.2},
            "agent_4": {"value": 0.8},
        }

        metrics = precision_mapper.get_dashboard_metrics(all_precisions)

        assert metrics["total_agents"] == 4
        assert metrics["average_precision"] == pytest.approx(0.625, rel=1e-2)
        assert metrics["system_health"] == "fair"
        assert len(metrics["confidence_distribution"]) == 5  # All confidence levels
        assert len(metrics["critical_agents"]) == 1  # Only agent_3 is critical


class TestTUIMultiAgentCoordinator:
    """Test TUI Multi-Agent Coordinator."""

    @pytest.fixture
    def mock_tui_bridge(self):
        """Mock TUI bridge."""
        bridge = Mock()
        bridge.websocket_manager = Mock()
        bridge.websocket_manager.broadcast = AsyncMock()
        bridge.state_mirror = Mock()
        bridge.shared_state = Mock()
        bridge.shared_state.get_agent_state = Mock(return_value={"status": "active"})
        bridge.shared_state.update = Mock()
        return bridge

    @pytest.fixture
    def mock_shared_state(self):
        """Mock shared state."""
        state = Mock()
        state.get_agent_state = Mock(return_value={"status": "active"})
        state.update = Mock()
        state.get_all_states = Mock(return_value={})
        return state

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["tool1", "tool2"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tui_bridge, mock_shared_state, mock_tool_registry):
        """Create TUI coordinator."""
        return TUIMultiAgentCoordinator(mock_tui_bridge, mock_shared_state, mock_tool_registry)

    def test_register_tui_agent(self, coordinator):
        """Test TUI agent registration."""
        config = TUIAgentConfig(
            agent_id="agent_1",
            agent_type="lrs",
            tui_panel="main",
            dashboard_metrics=["precision", "status"],
            coordination_group="group_1",
        )

        coordinator.register_tui_agent(config)

        assert "agent_1" in coordinator.tui_agents
        assert coordinator.tui_agents["agent_1"].agent_type == "lrs"
        assert "group_1" in coordinator.coordination_groups
        assert "agent_1" in coordinator.coordination_groups["group_1"]

    def test_unregister_tui_agent(self, coordinator):
        """Test TUI agent unregistration."""
        # First register an agent
        config = TUIAgentConfig(agent_id="agent_1", agent_type="lrs", coordination_group="group_1")
        coordinator.register_tui_agent(config)

        # Then unregister
        coordinator.unregister_tui_agent("agent_1")

        assert "agent_1" not in coordinator.tui_agents
        assert "group_1" not in coordinator.coordination_groups

    @pytest.mark.asyncio
    async def test_coordinate_via_tui(self, coordinator):
        """Test TUI-based coordination."""
        # Register agents first
        config1 = TUIAgentConfig(agent_id="agent_1", agent_type="lrs")
        config2 = TUIAgentConfig(agent_id="agent_2", agent_type="lrs")
        coordinator.register_tui_agent(config1)
        coordinator.register_tui_agent(config2)

        # Coordinate agents
        result = await coordinator.coordinate_via_tui(
            agent_ids=["agent_1", "agent_2"],
            coordination_type="collaborative_task",
            data={"task_id": "task_1", "goal": "analyze_data"},
            user_initiated=True,
        )

        assert result["agent_ids"] == ["agent_1", "agent_2"]
        assert result["coordination_type"] == "collaborative_task"
        assert "result" in result
        assert result["result"]["status"] == "initiated"

    @pytest.mark.asyncio
    async def test_collaborative_task_coordination(self, coordinator):
        """Test collaborative task coordination."""
        from ..coordinator import TUICoordinationEvent

        event = TUICoordinationEvent(
            event_type="collaborative_task",
            agent_ids=["agent_1", "agent_2"],
            data={"task_id": "task_1", "goal": "analyze_data"},
            timestamp=datetime.now(),
        )

        result = await coordinator._handle_collaborative_task(event)

        assert result["status"] == "initiated"
        assert "task_distribution" in result
        assert len(result["task_distribution"]) == 2
        assert "agent_1" in result["task_distribution"]
        assert "agent_2" in result["task_distribution"]

    @pytest.mark.asyncio
    async def test_load_balancing_coordination(self, coordinator):
        """Test load balancing coordination."""
        from ..coordinator import TUICoordinationEvent

        event = TUICoordinationEvent(
            event_type="load_balancing",
            agent_ids=["agent_1", "agent_2"],
            data={},
            timestamp=datetime.now(),
        )

        # Mock different load levels
        coordinator.shared_state.get_agent_state.side_effect = [
            {"current_load": 0.8},  # agent_1 - high load
            {"current_load": 0.2},  # agent_2 - low load
        ]

        result = await coordinator._handle_load_balancing(event)

        assert result["status"] == "redistribution_suggested"
        assert "redistribute_from" in result
        assert "redistribute_to" in result


class TestIntegration:
    """Integration tests for TUI components."""

    @pytest.mark.asyncio
    async def test_end_to_end_precision_flow(self):
        """Test end-to-end precision data flow."""
        # Create mock components
        mock_shared_state = Mock()
        mock_shared_state.get_agent_state = Mock(return_value={})
        mock_shared_state.update = Mock()
        mock_shared_state.get_all_states = Mock(return_value={})

        mock_tui_bridge = Mock()
        mock_tui_bridge.websocket_manager = Mock()
        mock_tui_bridge.websocket_manager.broadcast = AsyncMock()
        mock_tui_bridge.shared_state = mock_shared_state

        # Create components
        state_mirror = TUIStateMirror(mock_shared_state, mock_tui_bridge)
        precision_mapper = TUIPrecisionMapper()

        # Simulate precision update from LRS
        precision_changes = {"value": 0.75, "alpha": 7.0, "beta": 3.0}
        await state_mirror.sync_precision_to_tui("agent_1", precision_changes)

        # Map to TUI confidence
        confidence = precision_mapper.precision_to_confidence(precision_changes)

        # Verify results
        assert confidence.level == ConfidenceLevel.HIGH
        assert confidence.score == 75.0
        assert "precision" in state_mirror.lrs_state["agent_1"]
        assert "precision" in state_mirror.tui_state["agent_1"]

        # Verify WebSocket broadcast was called
        mock_tui_bridge.websocket_manager.broadcast.assert_called()

    @pytest.mark.asyncio
    async def test_tool_execution_flow(self):
        """Test tool execution flow through TUI."""
        # Create mock components
        mock_shared_state = Mock()
        mock_shared_state.get_agent_state = Mock(return_value={"status": "active"})
        mock_shared_state.update = Mock()

        mock_tui_bridge = Mock()
        mock_tui_bridge.websocket_manager = Mock()
        mock_tui_bridge.websocket_manager.broadcast = AsyncMock()
        mock_tui_bridge.shared_state = mock_shared_state
        mock_tui_bridge.state_mirror = Mock()
        mock_tui_bridge.state_mirror.get_user_preference = Mock(return_value="test")

        # Create TUI interaction tool
        tui_tool = TUIInteractionTool(mock_tui_bridge)

        # Simulate tool execution request
        state = {
            "action": "command",
            "command": "show_notification",
            "message": "Tool executed successfully",
            "notification_type": "success",
        }

        result = tui_tool.get(state)

        # Verify execution
        assert result.success is True
        assert "notification_sent" in result.value

        # Verify WebSocket broadcast
        mock_tui_bridge.websocket_manager.broadcast.assert_called_once()

        # Verify broadcast message content
        call_args = mock_tui_bridge.websocket_manager.broadcast.call_args
        message = call_args[0][1]  # Second argument is the message
        assert message["type"] == "notification"
        assert message["message"] == "Tool executed successfully"
        assert message["notification_type"] == "success"


class TestPerformance:
    """Performance tests for TUI components."""

    @pytest.mark.asyncio
    async def test_precision_mapping_performance(self):
        """Test precision mapping performance."""
        precision_mapper = TUIPrecisionMapper()

        # Test with large number of precision values
        precision_values = [i / 1000.0 for i in range(1000)]

        import time

        start_time = time.time()

        indicators = precision_mapper.get_confidence_indicators(precision_values)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        assert duration < 0.1  # 100ms
        assert len(indicators) == 1000

    @pytest.mark.asyncio
    async def test_state_sync_performance(self):
        """Test state synchronization performance."""
        mock_shared_state = Mock()
        mock_shared_state.get_agent_state = Mock(return_value={})
        mock_shared_state.update = Mock()

        mock_tui_bridge = Mock()
        mock_tui_bridge.websocket_manager = Mock()
        mock_tui_bridge.websocket_manager.broadcast = AsyncMock()

        state_mirror = TUIStateMirror(mock_shared_state, mock_tui_bridge)

        # Test with many simultaneous updates
        updates = {f"agent_{i}": {"precision": {"value": i / 100.0}} for i in range(100)}

        import time

        start_time = time.time()

        await state_mirror.sync_from_tui(updates)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        assert duration < 0.5  # 500ms
        assert len(state_mirror.tui_state) == 100


# Test configuration for pytest
def pytest_configure(config):
    """Configure pytest for TUI tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
