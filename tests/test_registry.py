"""
Tests for tool registry.
"""

import pytest

from lrs.core.registry import ToolRegistry
from lrs.core.lens import ToolLens, ExecutionResult


class DummyTool(ToolLens):
    """Dummy tool for testing"""
    def __init__(self, name, input_type="string", output_type="string"):
        super().__init__(
            name,
            input_schema={'type': input_type},
            output_schema={'type': output_type}
        )
    
    def get(self, state):
        return ExecutionResult(True, "output", None, 0.1)
    
    def set(self, state, obs):
        return state


class TestToolRegistry:
    """Test ToolRegistry class"""
    
    def test_initialization(self):
        """Test empty registry initialization"""
        registry = ToolRegistry()
        
        assert len(registry.tools) == 0
        assert len(registry.alternatives) == 0
        assert len(registry.statistics) == 0
    
    def test_register_tool(self):
        """Test registering a tool"""
        registry = ToolRegistry()
        tool = DummyTool("test_tool")
        
        registry.register(tool)
        
        assert "test_tool" in registry.tools
        assert registry.tools["test_tool"] == tool
    
    def test_register_with_alternatives(self):
        """Test registering tool with alternatives"""
        registry = ToolRegistry()
        tool = DummyTool("primary")
        alt1 = DummyTool("alternative_1")
        alt2 = DummyTool("alternative_2")
        
        registry.register(tool, alternatives=["alternative_1", "alternative_2"])
        registry.register(alt1)
        registry.register(alt2)
        
        alts = registry.find_alternatives("primary")
        assert "alternative_1" in alts
        assert "alternative_2" in alts
    
    def test_get_tool(self):
        """Test retrieving tool by name"""
        registry = ToolRegistry()
        tool = DummyTool("my_tool")
        
        registry.register(tool)
        
        retrieved = registry.get_tool("my_tool")
        assert retrieved == tool
    
    def test_get_nonexistent_tool(self):
        """Test retrieving non-existent tool"""
        registry = ToolRegistry()
        
        retrieved = registry.get_tool("nonexistent")
        assert retrieved is None
    
    def test_find_alternatives_no_alternatives(self):
        """Test finding alternatives when none exist"""
        registry = ToolRegistry()
        tool = DummyTool("tool")
        
        registry.register(tool)
        
        alts = registry.find_alternatives("tool")
        assert alts == []
    
    def test_list_tools(self):
        """Test listing all tool names"""
        registry = ToolRegistry()
        
        tools = [DummyTool(f"tool_{i}") for i in range(5)]
        for tool in tools:
            registry.register(tool)
        
        tool_names = registry.list_tools()
        assert len(tool_names) == 5
        assert "tool_0" in tool_names
        assert "tool_4" in tool_names


class TestToolStatistics:
    """Test statistics tracking"""
    
    def test_statistics_initialization(self):
        """Statistics should be initialized on registration"""
        registry = ToolRegistry()
        tool = DummyTool("test")
        
        registry.register(tool)
        
        stats = registry.get_statistics("test")
        assert stats is not None
        assert stats['success_rate'] == 0.5  # Neutral prior
        assert stats['call_count'] == 0
    
    def test_update_statistics_success(self):
        """Test updating statistics with successful execution"""
        registry = ToolRegistry()
        tool = DummyTool("test")
        registry.register(tool)
        
        registry.update_statistics("test", success=True, prediction_error=0.1)
        
        stats = registry.get_statistics("test")
        assert stats['call_count'] == 1
        assert stats['failure_count'] == 0
        assert stats['success_rate'] == 1.0
    
    def test_update_statistics_failure(self):
        """Test updating statistics with failed execution"""
        registry = ToolRegistry()
        tool = DummyTool("test")
        registry.register(tool)
        
        registry.update_statistics("test", success=False, prediction_error=0.9)
        
        stats = registry.get_statistics("test")
        assert stats['call_count'] == 1
        assert stats['failure_count'] == 1
        assert stats['success_rate'] == 0.0
    
    def test_running_average_prediction_error(self):
        """Test running average of prediction errors"""
        registry = ToolRegistry()
        tool = DummyTool("test")
        registry.register(tool)
        
        # Update with different errors
        registry.update_statistics("test", True, 0.1)
        registry.update_statistics("test", True, 0.3)
        registry.update_statistics("test", True, 0.2)
        
        stats = registry.get_statistics("test")
        expected_avg = (0.1 + 0.3 + 0.2) / 3
        assert abs(stats['avg_prediction_error'] - expected_avg) < 0.01
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        registry = ToolRegistry()
        tool = DummyTool("test")
        registry.register(tool)
        
        # 7 successes, 3 failures
        for _ in range(7):
            registry.update_statistics("test", success=True, prediction_error=0.1)
        for _ in range(3):
            registry.update_statistics("test", success=False, prediction_error=0.9)
        
        stats = registry.get_statistics("test")
        assert abs(stats['success_rate'] - 0.7) < 0.01


class TestSchemaCompatibility:
    """Test schema compatibility checking"""
    
    def test_discover_compatible_tools_same_type(self):
        """Test discovering tools with compatible types"""
        registry = ToolRegistry()
        
        tool_a = DummyTool("a", input_type="string", output_type="string")
        tool_b = DummyTool("b", input_type="string", output_type="string")
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        compatible = registry.discover_compatible_tools(
            input_schema={'type': 'string'},
            output_schema={'type': 'string'}
        )
        
        assert "a" in compatible
        assert "b" in compatible
    
    def test_discover_compatible_tools_different_type(self):
        """Test that incompatible types are not matched"""
        registry = ToolRegistry()
        
        tool_a = DummyTool("a", input_type="string", output_type="string")
        tool_b = DummyTool("b", input_type="number", output_type="number")
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        compatible = registry.discover_compatible_tools(
            input_schema={'type': 'string'},
            output_schema={'type': 'string'}
        )
        
        assert "a" in compatible
        assert "b" not in compatible
    
    def test_object_schema_required_fields(self):
        """Test object schema with required fields"""
        registry = ToolRegistry()
        
        tool = DummyTool("test")
        tool.input_schema = {
            'type': 'object',
            'required': ['field_a', 'field_b']
        }
        registry.register(tool)
        
        # Should match if all required fields present
        compatible = registry.discover_compatible_tools(
            input_schema={'type': 'object', 'required': ['field_a', 'field_b']},
            output_schema={'type': 'string'}
        )
        
        assert "test" in compatible
        
        # Should not match if missing required field
        compatible = registry.discover_compatible_tools(
            input_schema={'type': 'object', 'required': ['field_a']},
            output_schema={'type': 'string'}
        )
        
        assert "test" not in compatible  # Tool requires more fields


class TestRegistryEdgeCases:
    """Test edge cases"""
    
    def test_register_duplicate_tool(self):
        """Test registering tool with duplicate name"""
        registry = ToolRegistry()
        
        tool1 = DummyTool("same_name")
        tool2 = DummyTool("same_name")
        
        registry.register(tool1)
        registry.register(tool2)
        
        # Should overwrite
        assert registry.get_tool("same_name") == tool2
    
    def test_update_statistics_before_registration(self):
        """Test updating statistics for unregistered tool"""
        registry = ToolRegistry()
        
        # Should create statistics entry
        registry.update_statistics("new_tool", success=True, prediction_error=0.1)
        
        stats = registry.get_statistics("new_tool")
        assert stats is not None
    
    def test_get_statistics_nonexistent(self):
        """Test getting statistics for non-existent tool"""
        registry = ToolRegistry()
        
        stats = registry.get_statistics("nonexistent")
        assert stats is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
