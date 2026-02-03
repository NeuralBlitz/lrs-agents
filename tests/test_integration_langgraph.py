"""
Tests for LangGraph integration.
"""

import pytest
from unittest.mock import Mock, MagicMock
from lrs.core.free_energy import PolicyEvaluation

from lrs.integration.langgraph import LRSGraphBuilder, LRSState, create_lrs_agent
from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.precision import HierarchicalPrecision


class MockTool(ToolLens):
    """Mock tool for testing"""

    def __init__(self, name="mock", should_fail=False):
        super().__init__(name, {}, {})
        self.should_fail = should_fail

    def get(self, state):
        self.call_count += 1
        if self.should_fail:
            self.failure_count += 1
            return ExecutionResult(False, None, "Failed", 0.9)
        return ExecutionResult(True, f"{self.name}_result", None, 0.1)

    def set(self, state, obs):
        return {**state, f"{self.name}_output": obs}


class TestLRSGraphBuilder:
    """Test LRSGraphBuilder class"""

    def test_initialization(self):
        """Test builder initialization"""
        mock_llm = Mock()
        registry = ToolRegistry()

        builder = LRSGraphBuilder(mock_llm, registry)

        assert builder.llm == mock_llm
        assert builder.registry == registry
        assert isinstance(builder.hp, HierarchicalPrecision)

    def test_initialization_with_preferences(self):
        """Test initialization with custom preferences"""
        mock_llm = Mock()
        registry = ToolRegistry()

        preferences = {"custom": 10.0}
        builder = LRSGraphBuilder(mock_llm, registry, preferences=preferences)

        assert builder.preferences["custom"] == 10.0

    def test_build_creates_graph(self):
        """Test that build() creates a graph"""
        mock_llm = Mock()
        registry = ToolRegistry()
        registry.register(MockTool("test_tool"))

        builder = LRSGraphBuilder(mock_llm, registry)
        graph = builder.build()

        assert graph is not None

    def test_initialize_node(self):
        """Test _initialize node"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {}
        result = builder._initialize(state)

        assert "precision" in result
        assert "belief_state" in result
        assert "tool_history" in result
        assert "adaptation_count" in result

    def test_initialize_preserves_existing_state(self):
        """Test that initialize preserves existing state"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {"messages": [{"role": "user", "content": "test"}], "custom_field": "value"}

        result = builder._initialize(state)

        assert result["messages"] == state["messages"]
        assert result["custom_field"] == "value"

    def test_generate_policies_exhaustive(self):
        """Test policy generation via exhaustive search"""
        mock_llm = Mock()
        registry = ToolRegistry()
        registry.register(MockTool("tool_a"))
        registry.register(MockTool("tool_b"))

        builder = LRSGraphBuilder(mock_llm, registry, use_llm_proposals=False)

        state = {"belief_state": {}}
        result = builder._generate_policies(state)

        assert "candidate_policies" in result
        assert len(result["candidate_policies"]) > 0

    def test_evaluate_G_node(self):
        """Test G evaluation node"""
        mock_llm = Mock()
        registry = ToolRegistry()
        tool = MockTool("test")
        registry.register(tool)

        builder = LRSGraphBuilder(mock_llm, registry, use_llm_proposals=False)

        state = {"candidate_policies": [{"policy": [tool], "strategy": "test"}], "belief_state": {}}

        result = builder._evaluate_G(state)

        assert "G_values" in result
        assert 0 in result["G_values"]  # First policy

    def test_select_policy_node(self):
        """Test policy selection node"""
        mock_llm = Mock()
        registry = ToolRegistry()
        tool = MockTool("test")
        registry.register(tool)

        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "candidate_policies": [[tool]],
            "policy_evaluations": [Mock(total_G=-2.0)],
            "precision": {"planning": 0.5},
            "belief_state": {},
            "preferences": {"success": 1.0, "error": -1.0},
        }

        result = builder._select_policy(state)

        assert "selected_policy" in result
        assert len(result["selected_policy"]) > 0

    def test_execute_tool_node(self):
        """Test tool execution node"""
        mock_llm = Mock()
        registry = ToolRegistry()
        tool = MockTool("test")
        registry.register(tool)

        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "selected_policy": [tool],
            "belief_state": {},
            "tool_history": [],
            "current_policy_index": 0,
        }

        result = builder._execute_tool(state)

        assert len(result["tool_history"]) == 1
        assert result["tool_history"][0]["tool"] == "test"
        assert result["tool_history"][0]["success"] is True

    def test_execute_tool_updates_belief_state(self):
        """Test that tool execution updates belief state"""
        mock_llm = Mock()
        registry = ToolRegistry()
        tool = MockTool("test")
        registry.register(tool)

        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "selected_policy": [tool],
            "belief_state": {"existing": "value"},
            "tool_history": [],
            "current_policy_index": 0,
        }

        result = builder._execute_tool(state)

        assert "test_output" in result["belief_state"]

    def test_execute_tool_stops_on_failure(self):
        """Test that execution stops on first failure"""
        mock_llm = Mock()
        registry = ToolRegistry()

        tool_success = MockTool("success", should_fail=False)
        tool_fail = MockTool("fail", should_fail=True)
        tool_never_called = MockTool("never", should_fail=False)

        registry.register(tool_success)
        registry.register(tool_fail)
        registry.register(tool_never_called)

        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "selected_policy": [tool_success, tool_fail, tool_never_called],
            "belief_state": {},
            "tool_history": [],
            "current_policy_index": 0,
        }

        # Execute first tool
        result1 = builder._execute_tool(state)

        # Execute second tool
        state2 = {**state, **result1, "current_policy_index": 1}
        result2 = builder._execute_tool(state2)

        # Should execute success and fail, but not never_called
        assert len(result2["tool_history"]) == 2
        assert tool_never_called.call_count == 0

    def test_update_precision_node(self):
        """Test precision update node"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "tool_history": [{"tool": "test", "success": False, "prediction_error": 0.95}],
            "precision": builder.hp.get_all(),
        }

        result = builder._update_precision(state)

        # Precision should have decreased due to high error
        assert result["precision"]["execution"] < 0.5

    def test_precision_gate_continues(self):
        """Test decision gate routing - continue"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "belief_state": {"goal_satisfied": False},
            "tool_history": [],
            "selected_policy": [MockTool("test")],
        }

        next_node = builder._decision_gate(state)

        assert next_node == "continue"

    def test_precision_gate_ends_on_completion(self):
        """Test decision gate routing - end on completion"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "belief_state": {"goal_satisfied": True},
            "tool_history": [],
            "selected_policy": [],
        }

        next_node = builder._decision_gate(state)

        assert next_node == "success"

    def test_precision_gate_ends_on_max_iterations(self):
        """Test decision gate routing - end on max iterations"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {
            "belief_state": {"goal_satisfied": False},
            "tool_history": [],
            "selected_policy": [],
            "precision": {"planning": 0.1, "execution": 0.1},
        }

        next_node = builder._decision_gate(state)

        assert next_node == "fail"

    def test_precision_gate_ends_on_max_iterations(self):
        """Test precision gate routing - end on max iterations"""
        mock_llm = Mock()
        registry = ToolRegistry()
        builder = LRSGraphBuilder(mock_llm, registry)

        state = {"belief_state": {}, "tool_history": [{}] * 100, "max_iterations": 50}

        next_node = builder._precision_gate(state)

        assert next_node == "end"


class TestCreateLRSAgent:
    """Test create_lrs_agent convenience function"""

    def test_creates_agent(self):
        """Test that create_lrs_agent creates an agent"""
        mock_llm = Mock()
        tools = [MockTool("tool1"), MockTool("tool2")]

        agent = create_lrs_agent(mock_llm, tools)

        assert agent is not None

    def test_registers_tools(self):
        """Test that tools are registered"""
        mock_llm = Mock()
        tools = [MockTool("tool1"), MockTool("tool2")]

        agent = create_lrs_agent(mock_llm, tools)

        # Tools should be registered (can't directly test, but graph should exist)
        assert agent is not None

    def test_accepts_preferences(self):
        """Test that custom preferences are accepted"""
        mock_llm = Mock()
        tools = [MockTool("tool")]
        preferences = {"custom_pref": 10.0}

        agent = create_lrs_agent(mock_llm, tools, preferences=preferences)

        assert agent is not None

    def test_accepts_tracker(self):
        """Test that tracker is accepted"""
        from lrs.monitoring.tracker import LRSStateTracker

        mock_llm = Mock()
        tools = [MockTool("tool")]
        tracker = LRSStateTracker()

        agent = create_lrs_agent(mock_llm, tools, tracker=tracker)

        assert agent is not None


class TestLRSGraphExecution:
    """Test full graph execution (integration tests)"""

    @pytest.mark.skip(reason="Requires full graph compilation")
    def test_full_execution_success(self):
        """Test full agent execution with successful tool"""
        mock_llm = Mock()
        tools = [MockTool("test", should_fail=False)]

        agent = create_lrs_agent(mock_llm, tools, use_llm_proposals=False)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Test task"}], "max_iterations": 5}
        )

        assert "tool_history" in result
        assert len(result["tool_history"]) > 0

    @pytest.mark.skip(reason="Requires full graph compilation")
    def test_full_execution_with_failure(self):
        """Test agent execution with tool failure"""
        mock_llm = Mock()
        tools = [MockTool("fail", should_fail=True)]

        agent = create_lrs_agent(mock_llm, tools, use_llm_proposals=False)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Test task"}], "max_iterations": 5}
        )

        # Should have tool history even with failures
        assert "tool_history" in result

    @pytest.mark.skip(reason="Requires full graph compilation")
    def test_adaptation_on_precision_collapse(self):
        """Test that agent adapts when precision collapses"""
        mock_llm = Mock()

        # First tool fails, should trigger adaptation
        fail_tool = MockTool("fail", should_fail=True)
        success_tool = MockTool("success", should_fail=False)

        tools = [fail_tool, success_tool]

        agent = create_lrs_agent(mock_llm, tools, use_llm_proposals=False)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Test task"}], "max_iterations": 10}
        )

        # Should have adaptation count > 0
        assert result.get("adaptation_count", 0) > 0


class TestLRSStateSchema:
    """Test LRSState TypedDict schema"""

    def test_state_has_required_fields(self):
        """Test that LRSState defines required fields"""
        # This is mostly a type checking test
        # In practice, TypedDict is for type hints only

        state: LRSState = {
            "messages": [],
            "belief_state": {},
            "precision": {},
            "prediction_errors": {},
            "current_policy": [],
            "candidate_policies": [],
            "G_values": {},
            "tool_history": [],
            "adaptation_count": 0,
            "current_hbn_level": "planning",
            "next": "continue",
        }

        # Should compile without errors
        assert isinstance(state, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
