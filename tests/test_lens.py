"""
Tests for ToolLens and composition.
"""

import pytest

from lrs.core.lens import ToolLens, ExecutionResult, ComposedLens


class SimpleTool(ToolLens):
    """Simple test tool"""
    def __init__(self, name="simple", should_fail=False):
        super().__init__(name, {}, {})
        self.should_fail = should_fail
    
    def get(self, state):
        self.call_count += 1
        if self.should_fail:
            self.failure_count += 1
            return ExecutionResult(False, None, "Failed", 0.9)
        return ExecutionResult(True, f"{self.name}_output", None, 0.1)
    
    def set(self, state, observation):
        return {**state, self.name: observation}


class TestExecutionResult:
    """Test ExecutionResult dataclass"""
    
    def test_successful_result(self):
        """Test creating successful result"""
        result = ExecutionResult(
            success=True,
            value="data",
            error=None,
            prediction_error=0.1
        )
        
        assert result.success is True
        assert result.value == "data"
        assert result.error is None
        assert result.prediction_error == 0.1
    
    def test_failed_result(self):
        """Test creating failed result"""
        result = ExecutionResult(
            success=False,
            value=None,
            error="Something broke",
            prediction_error=0.95
        )
        
        assert result.success is False
        assert result.value is None
        assert result.error == "Something broke"
    
    def test_prediction_error_validation(self):
        """Prediction error must be in [0, 1]"""
        with pytest.raises(ValueError):
            ExecutionResult(True, "data", None, prediction_error=-0.1)
        
        with pytest.raises(ValueError):
            ExecutionResult(True, "data", None, prediction_error=1.5)


class TestToolLens:
    """Test ToolLens base class"""
    
    def test_initialization(self):
        """Test tool initialization"""
        tool = SimpleTool("test_tool")
        
        assert tool.name == "test_tool"
        assert tool.call_count == 0
        assert tool.failure_count == 0
    
    def test_successful_execution(self):
        """Test successful tool execution"""
        tool = SimpleTool("test", should_fail=False)
        
        result = tool.get({})
        
        assert result.success is True
        assert result.value == "test_output"
        assert tool.call_count == 1
        assert tool.failure_count == 0
    
    def test_failed_execution(self):
        """Test failed tool execution"""
        tool = SimpleTool("test", should_fail=True)
        
        result = tool.get({})
        
        assert result.success is False
        assert result.error == "Failed"
        assert tool.call_count == 1
        assert tool.failure_count == 1
    
    def test_state_update(self):
        """Test state update via set()"""
        tool = SimpleTool("test")
        
        state = {'existing': 'data'}
        new_state = tool.set(state, "observation")
        
        assert 'existing' in new_state
        assert new_state['test'] == "observation"
    
    def test_success_rate(self):
        """Test success rate calculation"""
        tool = SimpleTool("test", should_fail=False)
        
        # Execute multiple times
        for _ in range(10):
            tool.get({})
        
        assert tool.success_rate == 1.0
        
        # Now fail once
        tool.should_fail = True
        tool.get({})
        
        assert abs(tool.success_rate - (10/11)) < 0.01


class TestLensComposition:
    """Test lens composition via >> operator"""
    
    def test_simple_composition(self):
        """Test composing two lenses"""
        tool_a = SimpleTool("a")
        tool_b = SimpleTool("b")
        
        composed = tool_a >> tool_b
        
        assert isinstance(composed, ComposedLens)
        assert composed.left == tool_a
        assert composed.right == tool_b
    
    def test_composed_execution_success(self):
        """Test executing composed lens (both succeed)"""
        tool_a = SimpleTool("a", should_fail=False)
        tool_b = SimpleTool("b", should_fail=False)
        
        composed = tool_a >> tool_b
        result = composed.get({})
        
        assert result.success is True
        assert result.value == "b_output"  # Right tool's output
        assert tool_a.call_count == 1
        assert tool_b.call_count == 1
    
    def test_composed_short_circuit_on_failure(self):
        """Test that composition short-circuits on first failure"""
        tool_a = SimpleTool("a", should_fail=True)  # Fails
        tool_b = SimpleTool("b", should_fail=False)
        
        composed = tool_a >> tool_b
        result = composed.get({})
        
        assert result.success is False
        assert result.error == "Failed"
        assert tool_a.call_count == 1
        assert tool_b.call_count == 0  # Should not be called
    
    def test_multi_level_composition(self):
        """Test composing multiple lenses"""
        tool_a = SimpleTool("a")
        tool_b = SimpleTool("b")
        tool_c = SimpleTool("c")
        
        composed = tool_a >> tool_b >> tool_c
        
        result = composed.get({})
        
        assert result.success is True
        assert tool_a.call_count == 1
        assert tool_b.call_count == 1
        assert tool_c.call_count == 1
    
    def test_composed_state_threading(self):
        """Test that state threads through composition"""
        tool_a = SimpleTool("a")
        tool_b = SimpleTool("b")
        
        composed = tool_a >> tool_b
        
        initial_state = {'initial': 'value'}
        result = composed.get(initial_state)
        
        # State should be updated by both tools
        final_state = composed.set(initial_state, result.value)
        
        # Both tools should have updated state
        # (exact behavior depends on set() implementation)
    
    def test_composition_name(self):
        """Test composed lens name"""
        tool_a = SimpleTool("fetch")
        tool_b = SimpleTool("parse")
        
        composed = tool_a >> tool_b
        
        assert "fetch" in composed.name
        assert "parse" in composed.name
        assert ">>" in composed.name


class TestLensStatistics:
    """Test lens statistics tracking"""
    
    def test_call_count_increments(self):
        """Call count should increment on each execution"""
        tool = SimpleTool("test")
        
        for i in range(5):
            tool.get({})
            assert tool.call_count == i + 1
    
    def test_failure_count_increments(self):
        """Failure count should increment on failures"""
        tool = SimpleTool("test", should_fail=True)
        
        for i in range(3):
            tool.get({})
            assert tool.failure_count == i + 1
    
    def test_success_rate_calculation(self):
        """Success rate should be accurate"""
        tool = SimpleTool("test")
        
        # No calls yet
        assert tool.success_rate == 0.5  # Neutral prior
        
        # 7 successes, 3 failures
        tool.should_fail = False
        for _ in range(7):
            tool.get({})
        
        tool.should_fail = True
        for _ in range(3):
            tool.get({})
        
        assert abs(tool.success_rate - 0.7) < 0.01


class TestLensEdgeCases:
    """Test edge cases"""
    
    def test_empty_state(self):
        """Should handle empty state dict"""
        tool = SimpleTool("test")
        
        result = tool.get({})
        assert result.success is True
    
    def test_none_observation(self):
        """Should handle None observation in set()"""
        tool = SimpleTool("test")
        
        state = tool.set({'existing': 'data'}, None)
        assert 'existing' in state
    
    def test_multiple_compositions(self):
        """Should handle arbitrary composition depth"""
        tools = [SimpleTool(f"tool_{i}") for i in range(10)]
        
        composed = tools[0]
        for tool in tools[1:]:
            composed = composed >> tool
        
        result = composed.get({})
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
