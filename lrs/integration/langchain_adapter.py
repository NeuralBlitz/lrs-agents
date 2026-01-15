"""LangChain integration for LRS-Agents."""

from typing import Dict, Any, Optional, Callable
import time
import platform
import threading

from langchain.tools import BaseTool

from lrs.core.lens import ToolLens, ExecutionResult


def _extract_input_schema(tool: BaseTool) -> Dict[str, Any]:
    """Extract input schema from LangChain tool."""
    if hasattr(tool, 'args_schema') and tool.args_schema:
        try:
            # Try Pydantic V2 method first
            return tool.args_schema.model_json_schema()
        except AttributeError:
            # Fall back to Pydantic V1
            return tool.args_schema.schema()
    return {}


def _extract_output_schema(tool: BaseTool) -> Dict[str, Any]:
    """Extract output schema from LangChain tool."""
    # Most LangChain tools return strings or dicts
    return {
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        }
    }


class LangChainToolLens(ToolLens):
    """
    Wraps a LangChain tool as a ToolLens.
    
    Provides timeout handling, prediction error calculation,
    and statistics tracking for any LangChain tool.
    
    Args:
        tool: LangChain BaseTool to wrap
        timeout: Maximum execution time in seconds (default: 30.0)
        error_fn: Custom function to calculate prediction error
    
    Example:
        >>> from langchain.tools import Tool
        >>> lc_tool = Tool(name="search", func=lambda q: f"Results for {q}")
        >>> lrs_tool = LangChainToolLens(lc_tool, timeout=10.0)
        >>> result = lrs_tool.get({"query": "test"})
    """
    
    def __init__(
        self,
        tool: BaseTool,
        timeout: float = 30.0,
        error_fn: Optional[Callable] = None
    ):
        """Initialize LangChain tool wrapper."""
        input_schema = _extract_input_schema(tool)
        output_schema = _extract_output_schema(tool)
        
        super().__init__(
            name=tool.name,
            input_schema=input_schema,
            output_schema=output_schema
        )
        
        self.tool = tool
        self.timeout = timeout
        self.error_fn = error_fn or self._default_error_fn
    
    def _default_error_fn(self, result: Any, output_schema: Dict) -> float:
        """Default prediction error calculation."""
        if result is None:
            return 0.9  # High surprise for null
        elif isinstance(result, str) and len(result) == 0:
            return 0.7  # Medium surprise for empty
        else:
            return 0.1  # Low surprise for success
    
    def get(self, state: Dict[str, Any]) -> ExecutionResult:
        """Execute LangChain tool with timeout."""
        self.call_count += 1
        start_time = time.time()
        
        try:
            # Platform-specific timeout handling
            if platform.system() == 'Windows':
                # Windows doesn't support SIGALRM - use threading
                result_container = {'result': None, 'error': None}
                
                def target():
                    try:
                        result_container['result'] = self.tool.run(**state)
                    except Exception as e:
                        result_container['error'] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=self.timeout)
                
                if thread.is_alive():
                    # Timeout occurred
                    self.failure_count += 1
                    execution_time = time.time() - start_time
                    return ExecutionResult(
                        success=False,
                        value=None,
                        error=f"Timeout after {self.timeout}s",
                        prediction_error=0.7
                    )
                
                if result_container['error']:
                    raise result_container['error']
                
                tool_result = result_container['result']
            
            else:
                # Unix-like systems can use signal
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Timeout after {self.timeout}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
                
                try:
                    tool_result = self.tool.run(**state)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            # Success
            execution_time = time.time() - start_time
            prediction_error = self.error_fn(tool_result, self.output_schema)
            
            return ExecutionResult(
                success=True,
                value=tool_result,
                error=None,
                prediction_error=prediction_error
            )
        
        except TimeoutError as e:
            self.failure_count += 1
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.7
            )
        
        except Exception as e:
            self.failure_count += 1
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.9
            )
    
    def set(self, state: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        """Update state with tool result."""
        return {
            **state,
            f'{self.name}_result': obs
        }


def wrap_langchain_tool(
    tool: BaseTool,
    timeout: float = 30.0,
    error_fn: Optional[Callable] = None
) -> LangChainToolLens:
    """
    Convenience function to wrap a LangChain tool.
    
    Args:
        tool: LangChain BaseTool to wrap
        timeout: Maximum execution time in seconds
        error_fn: Optional custom error calculation function
    
    Returns:
        LangChainToolLens: Wrapped tool ready for LRS use
    
    Example:
        >>> from langchain_community.tools import DuckDuckGoSearchRun
        >>> search = wrap_langchain_tool(DuckDuckGoSearchRun(), timeout=10.0)
    """
    return LangChainToolLens(tool, timeout=timeout, error_fn=error_fn)
