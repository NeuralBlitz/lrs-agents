"""Tests for LangChain adapter."""

import pytest

from lrs.integration.langchain_adapter import (
    LangChainToolLens,
    wrap_langchain_tool,
)

# ... rest of tests


class MockLangChainTool:
    """Mock LangChain BaseTool"""

    def __init__(self, name="mock_tool", should_fail=False):
        self.name = name
        self.description = "Mock tool for testing"
        self.should_fail = should_fail
        self.args_schema = None

    def run(self, **kwargs):
        if self.should_fail:
            raise Exception("Tool failed")
        return f"Result for {kwargs.get('input', kwargs)}"


class TestLangChainToolLens:
    """Test LangChainToolLens wrapper"""

    def test_initialization(self):
        """Test wrapper initialization"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        assert lens.name == "mock_tool"
        assert lens.tool == tool

    def test_successful_execution(self):
        """Test successful tool execution"""
        tool = MockLangChainTool(should_fail=False)
        lens = LangChainToolLens(tool)

        result = lens.get({"input": "test"})

        assert result.success is True
        assert "Result for" in result.value
        assert result.prediction_error < 0.5

    def test_failed_execution(self):
        """Test failed tool execution"""
        tool = MockLangChainTool(should_fail=True)
        lens = LangChainToolLens(tool)

        result = lens.get({"input": "test"})

        assert result.success is False
        assert result.error is not None
        assert result.prediction_error > 0.7

    def test_timeout_handling(self):
        """Test timeout handling"""
        tool = MockLangChainTool()

        # Mock a slow tool
        original_run = tool.run

        def slow_run(input_data):
            import time

            time.sleep(2)
            return original_run(input_data)

        tool.run = slow_run

        lens = LangChainToolLens(tool, timeout=1)

        result = lens.get({"input": "test"})

        # Should timeout
        assert result.success is False
        assert "Timeout" in result.error or "timeout" in result.error.lower()
        assert result.prediction_error > 0.7

    def test_state_update(self):
        """Test state update via set()"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        state = {"existing": "value"}
        new_state = lens.set(state, "observation")

        assert "existing" in new_state
        assert f"{tool.name}_output" in new_state

    def test_default_error_function(self):
        """Test default error calculation"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        # Empty result
        error_empty = lens._default_error_fn("", {})
        assert error_empty > 0.5

        # None result
        error_none = lens._default_error_fn(None, {})
        assert error_none > 0.5

        # Valid string result
        error_valid = lens._default_error_fn("result", {"type": "string"})
        assert error_valid < 0.3

    def test_custom_error_function(self):
        """Test custom error function"""
        tool = MockLangChainTool()

        def custom_error_fn(result, schema):
            return 0.5  # Always return 0.5

        lens = LangChainToolLens(tool, error_fn=custom_error_fn)

        result = lens.get({"input": "test"})

        assert result.prediction_error == 0.5

    def test_call_count_increments(self):
        """Test that call count increments"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        for i in range(5):
            lens.get({"input": "test"})
            assert lens.call_count == i + 1

    def test_failure_count_increments(self):
        """Test that failure count increments on errors"""
        tool = MockLangChainTool(should_fail=True)
        lens = LangChainToolLens(tool)

        for i in range(3):
            lens.get({"input": "test"})
            assert lens.failure_count == i + 1


class TestWrapLangChainTool:
    """Test wrap_langchain_tool convenience function"""

    def test_wrap_creates_lens(self):
        """Test that wrap creates LangChainToolLens"""
        tool = MockLangChainTool()

        lens = wrap_langchain_tool(tool)

        assert isinstance(lens, LangChainToolLens)

    def test_wrap_accepts_kwargs(self):
        """Test that wrap accepts additional kwargs"""
        tool = MockLangChainTool()

        lens = wrap_langchain_tool(tool, timeout=5.0)

        assert lens.timeout == 5.0

    def test_wrapped_tool_executes(self):
        """Test that wrapped tool executes correctly"""
        tool = MockLangChainTool()
        lens = wrap_langchain_tool(tool)

        result = lens.get({"input": "test"})

        assert result.success is True


class TestSchemaExtraction:
    """Test schema extraction from LangChain tools"""

    def test_extract_input_schema_with_pydantic(self):
        """Test extracting input schema from Pydantic model"""
        from pydantic import BaseModel, Field

        class TestSchema(BaseModel):
            input_text: str = Field(description="Input text")
            count: int = Field(default=1)

        tool = MockLangChainTool()
        tool.args_schema = TestSchema

        lens = LangChainToolLens(tool)

        # Should have extracted schema
        assert "type" in lens.input_schema
        assert lens.input_schema["type"] == "object"

    def test_extract_input_schema_fallback(self):
        """Test fallback schema when no Pydantic model"""
        tool = MockLangChainTool()
        tool.args_schema = None

        lens = LangChainToolLens(tool)

        # Should use fallback
        assert lens.input_schema["type"] == "object"
        assert "input" in lens.input_schema["properties"]

    def test_type_mismatch_error(self):
        """Test error calculation for type mismatches"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        # Expected string, got number
        error = lens._default_error_fn(123, {"type": "string"})

        assert error == 0.1  # Default heuristic only checks None/empty

    def test_correct_type_low_error(self):
        """Test low error for correct types"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        # Expected string, got string
        error = lens._default_error_fn("result", {"type": "string"})

        assert error < 0.3

    def test_empty_result_moderate_error(self):
        """Test moderate error for empty results"""
        tool = MockLangChainTool()
        lens = LangChainToolLens(tool)

        error = lens._default_error_fn("", {"type": "string"})

        assert error == 0.7  # Empty returns 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
