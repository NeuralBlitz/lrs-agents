"""
LangChain tool integration for LRS-Agents.

Wraps LangChain tools as ToolLens objects with automatic prediction error calculation.
"""

from typing import Any, Dict, Optional, Callable
import signal
from langchain_core.tools import BaseTool

from lrs.core.lens import ToolLens, ExecutionResult


class LangChainToolLens(ToolLens):
    """
    Wrapper that converts LangChain tools to ToolLens.
    
    Automatically calculates prediction errors based on:
    - Tool execution success/failure
    - Output schema validation
    - Execution time (timeouts)
    
    Examples:
        >>> from langchain_community.tools import ShellTool
        >>> 
        >>> shell = ShellTool()
        >>> lens = LangChainToolLens(shell)
        >>> 
        >>> result = lens.get({"commands": ["ls -la"]})
        >>> print(result.prediction_error)  # 0.1 if success, 0.9 if failure
    """
    
    def __init__(
        self,
        tool: BaseTool,
        error_fn: Optional[Callable[[Any, Dict], float]] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize LangChain tool wrapper.
        
        Args:
            tool: LangChain BaseTool instance
            error_fn: Optional custom prediction error function
                Signature: (result, expected_schema) -> float in [0, 1]
            timeout: Optional timeout in seconds
        """
        # Extract schema from LangChain tool
        input_schema = self._extract_input_schema(tool)
        output_schema = self._extract_output_schema(tool)
        
        super().__init__(
            name=tool.name,
            input_schema=input_schema,
            output_schema=output_schema
        )
        
        self.tool = tool
        self.error_fn = error_fn or self._default_error_fn
        self.timeout = timeout
    
    def _extract_input_schema(self, tool: BaseTool) -> Dict:
        """Extract input schema from LangChain tool"""
        if hasattr(tool, 'args_schema') and tool.args_schema:
            # Pydantic model to JSON schema
            return tool.args_schema.schema()
        else:
            # Fallback to simple schema
            return {
                'type': 'object',
                'properties': {
                    'input': {'type': 'string'}
                }
            }
    
    def _extract_output_schema(self, tool: BaseTool) -> Dict:
        """Extract expected output schema"""
        # Most LangChain tools return strings
        return {
            'type': 'string',
            'description': tool.description if hasattr(tool, 'description') else ''
        }
    
    def get(self, state: dict) -> ExecutionResult:
        """
        Execute LangChain tool and calculate prediction error.
        
        Args:
            state: Input state matching tool's args_schema
        
        Returns:
            ExecutionResult with prediction_error based on outcome
        """
        self.call_count += 1
        
        try:
            # Execute tool with timeout
            if self.timeout:
                # Set timeout signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Tool execution timed out")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
            
            # Call LangChain tool
            result = self.tool.run(state)
            
            if self.timeout:
                signal.alarm(0)  # Cancel timeout
                signal.signal(signal.SIGALRM, old_handler)
            
            # Calculate prediction error
            error = self.error_fn(result, self.output_schema)
            
            return ExecutionResult(
                success=True,
                value=result,
                error=None,
                prediction_error=error
            )
        
        except TimeoutError as e:
            self.failure_count += 1
            if self.timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return ExecutionResult(
                success=False,
                value=None,
                error=f"Timeout after {self.timeout}s",
                prediction_error=0.8  # Timeouts are surprising
            )
        
        except Exception as e:
            self.failure_count += 1
            if self.timeout:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except:
                    pass
            
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.9  # Exceptions are very surprising
            )
    
    def set(self, state: dict, observation: Any) -> dict:
        """
        Update belief state with tool output.
        
        Args:
            state: Current belief state
            observation: Tool output
        
        Returns:
            Updated belief state
        """
        # Store output with tool name as key
        return {
            **state,
            f'{self.name}_output': observation,
            'last_tool': self.name
        }
    
    def _default_error_fn(self, result: Any, expected_schema: Dict) -> float:
        """
        Default prediction error calculation.
        
        Heuristics:
        - Empty/None result → 0.6 (moderate surprise)
        - String result matches expected → 0.1 (low surprise)
        - Unexpected type → 0.5 (medium surprise)
        
        Args:
            result: Tool output
            expected_schema: Expected output schema
        
        Returns:
            Prediction error in [0, 1]
        """
        if result is None or result == "":
            return 0.6
        
        expected_type = expected_schema.get('type', 'string')
        
        if expected_type == 'string' and isinstance(result, str):
            return 0.1  # As expected
        elif expected_type == 'number' and isinstance(result, (int, float)):
            return 0.1
        elif expected_type == 'boolean' and isinstance(result, bool):
            return 0.1
        elif expected_type == 'object' and isinstance(result, dict):
            return 0.1
        elif expected_type == 'array' and isinstance(result, list):
            return 0.1
        else:
            return 0.5  # Type mismatch


def wrap_langchain_tool(
    tool: BaseTool,
    **kwargs
) -> LangChainToolLens:
    """
    Convenience function to wrap LangChain tools.
    
    Args:
        tool: LangChain BaseTool
        **kwargs: Passed to LangChainToolLens constructor
    
    Returns:
        ToolLens wrapper
    
    Examples:
        >>> from langchain_community.tools import ShellTool
        >>> 
        >>> lens = wrap_langchain_tool(ShellTool(), timeout=5.0)
        >>> 
        >>> # Use in LRS agent
        >>> from lrs import create_lrs_agent
        >>> agent = create_lrs_agent(llm, tools=[lens])
    """
    return LangChainToolLens(tool, **kwargs)
