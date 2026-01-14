"""
ToolLens: Categorical abstraction for tools.

A lens is a bidirectional morphism:
- get: Execute the tool (forward)
- set: Update belief state (backward)

Lenses compose via the >> operator, creating pipelines with automatic
error propagation.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ExecutionResult:
    """
    Result of executing a tool.
    
    Attributes:
        success: Whether execution succeeded
        value: Return value (None if failed)
        error: Error message (None if succeeded)
        prediction_error: How surprising this outcome was [0, 1]
    
    Examples:
        >>> # Successful execution
        >>> result = ExecutionResult(
        ...     success=True,
        ...     value="Data fetched",
        ...     error=None,
        ...     prediction_error=0.1  # Expected success
        ... )
        >>> 
        >>> # Failed execution
        >>> result = ExecutionResult(
        ...     success=False,
        ...     value=None,
        ...     error="API timeout",
        ...     prediction_error=0.9  # Unexpected failure
        ... )
    """
    success: bool
    value: Optional[Any]
    error: Optional[str]
    prediction_error: float
    
    def __post_init__(self):
        """Validate prediction error is in [0, 1]"""
        if not 0.0 <= self.prediction_error <= 1.0:
            raise ValueError(f"prediction_error must be in [0, 1], got {self.prediction_error}")


class ToolLens(ABC):
    """
    Abstract base class for tools as lenses.
    
    A lens has two operations:
    1. get(state) → ExecutionResult: Execute the tool
    2. set(state, observation) → state: Update belief state
    
    Lenses compose via >> operator:
        lens_a >> lens_b >> lens_c
    
    This creates a pipeline where:
    - Data flows forward through get operations
    - Belief updates flow backward through set operations
    - Errors propagate automatically
    
    Attributes:
        name: Tool identifier
        input_schema: JSON schema for inputs
        output_schema: JSON schema for outputs
        call_count: Number of times get() has been called
        failure_count: Number of times get() has failed
    
    Examples:
        >>> class FetchTool(ToolLens):
        ...     def get(self, state):
        ...         data = fetch(state['url'])
        ...         return ExecutionResult(True, data, None, 0.1)
        ...     
        ...     def set(self, state, observation):
        ...         return {**state, 'data': observation}
        >>> 
        >>> class ParseTool(ToolLens):
        ...     def get(self, state):
        ...         parsed = json.loads(state['data'])
        ...         return ExecutionResult(True, parsed, None, 0.05)
        ...     
        ...     def set(self, state, observation):
        ...         return {**state, 'parsed': observation}
        >>> 
        >>> # Compose
        >>> pipeline = FetchTool() >> ParseTool()
        >>> result = pipeline.get({'url': 'api.com/data'})
    """
    
    def __init__(
        self,
        name: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any]
    ):
        """
        Initialize tool lens.
        
        Args:
            name: Unique tool identifier
            input_schema: JSON schema for expected inputs
            output_schema: JSON schema for expected outputs
        """
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.call_count = 0
        self.failure_count = 0
    
    @abstractmethod
    def get(self, state: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the tool (forward operation).
        
        Args:
            state: Current agent state
        
        Returns:
            ExecutionResult with value and prediction error
        
        Note:
            Implementations should update call_count and failure_count
        """
        pass
    
    @abstractmethod
    def set(self, state: Dict[str, Any], observation: Any) -> Dict[str, Any]:
        """
        Update belief state with observation (backward operation).
        
        Args:
            state: Current state
            observation: Tool output
        
        Returns:
            Updated state
        """
        pass
    
    def __rshift__(self, other: 'ToolLens') -> 'ComposedLens':
        """
        Compose this lens with another: self >> other
        
        Args:
            other: Lens to compose with
        
        Returns:
            ComposedLens representing the pipeline
        
        Examples:
            >>> pipeline = fetch_tool >> parse_tool >> validate_tool
        """
        return ComposedLens(self, other)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate from history"""
        if self.call_count == 0:
            return 0.5  # Neutral prior
        return 1.0 - (self.failure_count / self.call_count)


class ComposedLens(ToolLens):
    """
    Composition of two lenses.
    
    Created via >> operator. Handles:
    - Forward data flow (left.get then right.get)
    - Backward belief update (right.set then left.set)
    - Error short-circuiting (stop on first failure)
    
    Attributes:
        left: First lens in composition
        right: Second lens in composition
    """
    
    def __init__(self, left: ToolLens, right: ToolLens):
        """
        Create composed lens.
        
        Args:
            left: First lens
            right: Second lens
        """
        super().__init__(
            name=f"{left.name}>>{right.name}",
            input_schema=left.input_schema,
            output_schema=right.output_schema
        )
        self.left = left
        self.right = right
    
    def get(self, state: Dict[str, Any]) -> ExecutionResult:
        """
        Execute composed lens (left then right).
        
        If left fails, short-circuit and return left's error.
        Otherwise, execute right with left's output.
        
        Args:
            state: Input state
        
        Returns:
            ExecutionResult from final lens (or first failure)
        """
        self.call_count += 1
        
        # Execute left lens
        left_result = self.left.get(state)
        
        if not left_result.success:
            # Short-circuit on failure
            self.failure_count += 1
            return left_result
        
        # Update state with left's output
        intermediate_state = self.left.set(state, left_result.value)
        
        # Execute right lens
        right_result = self.right.get(intermediate_state)
        
        if not right_result.success:
            self.failure_count += 1
        
        return right_result
    
    def set(self, state: Dict[str, Any], observation: Any) -> Dict[str, Any]:
        """
        Update state (right then left, backward flow).
        
        Args:
            state: Current state
            observation: Final observation
        
        Returns:
            Fully updated state
        """
        # Update from right (final observation)
        state = self.right.set(state, observation)
        
        # Update from left (intermediate state preserved)
        state = self.left.set(state, state)
        
        return state
