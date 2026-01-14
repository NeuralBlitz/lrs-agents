"""
Tool registry with natural transformation discovery.

Manages tools and their fallback chains. Automatically discovers
alternative tools based on schema compatibility.
"""

from typing import Dict, List, Optional, Any
from lrs.core.lens import ToolLens


class ToolRegistry:
    """
    Registry for managing tools and their alternatives.
    
    Features:
    - Register tools with explicit fallback chains
    - Discover compatible alternatives via schema matching
    - Track tool statistics for Free Energy calculation
    
    Attributes:
        tools: Dict mapping tool names to ToolLens objects
        alternatives: Dict mapping tool names to lists of alternative names
        statistics: Dict tracking execution history per tool
    
    Examples:
        >>> registry = ToolRegistry()
        >>> 
        >>> # Register primary tool with alternatives
        >>> registry.register(
        ...     api_tool,
        ...     alternatives=["cache_tool", "fallback_tool"]
        ... )
        >>> 
        >>> # Register alternatives
        >>> registry.register(cache_tool)
        >>> registry.register(fallback_tool)
        >>> 
        >>> # Find alternatives when primary fails
        >>> alts = registry.find_alternatives("api_tool")
        >>> print(alts)
        ['cache_tool', 'fallback_tool']
    """
    
    def __init__(self):
        """Initialize empty registry"""
        self.tools: Dict[str, ToolLens] = {}
        self.alternatives: Dict[str, List[str]] = {}
        self.statistics: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        tool: ToolLens,
        alternatives: Optional[List[str]] = None
    ):
        """
        Register a tool with optional alternatives.
        
        Args:
            tool: ToolLens to register
            alternatives: List of alternative tool names (fallback chain)
        
        Examples:
            >>> registry.register(
            ...     APITool(),
            ...     alternatives=["CacheTool", "LocalTool"]
            ... )
        """
        self.tools[tool.name] = tool
        
        if alternatives:
            self.alternatives[tool.name] = alternatives
        
        # Initialize statistics
        if tool.name not in self.statistics:
            self.statistics[tool.name] = {
                'success_rate': 0.5,  # Neutral prior
                'avg_prediction_error': 0.5,
                'error_variance': 0.0,
                'call_count': 0,
                'failure_count': 0
            }
    
    def get_tool(self, name: str) -> Optional[ToolLens]:
        """
        Retrieve tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            ToolLens or None if not found
        """
        return self.tools.get(name)
    
    def find_alternatives(self, tool_name: str) -> List[str]:
        """
        Find registered alternatives for a tool.
        
        Args:
            tool_name: Name of primary tool
        
        Returns:
            List of alternative tool names (may be empty)
        
        Examples:
            >>> alts = registry.find_alternatives("api_tool")
            >>> for alt_name in alts:
            ...     alt_tool = registry.get_tool(alt_name)
            ...     result = alt_tool.get(state)
        """
        return self.alternatives.get(tool_name, [])
    
    def discover_compatible_tools(
        self,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any]
    ) -> List[str]:
        """
        Discover tools compatible with given schemas.
        
        Uses structural matching to find tools that could serve as
        natural transformations (alternatives).
        
        Args:
            input_schema: Required input schema
            output_schema: Required output schema
        
        Returns:
            List of compatible tool names
        
        Examples:
            >>> compatible = registry.discover_compatible_tools(
            ...     input_schema={'type': 'object', 'required': ['url']},
            ...     output_schema={'type': 'string'}
            ... )
        """
        compatible = []
        
        for name, tool in self.tools.items():
            if self._schemas_compatible(tool.input_schema, input_schema):
                if self._schemas_compatible(tool.output_schema, output_schema):
                    compatible.append(name)
        
        return compatible
    
    def _schemas_compatible(
        self,
        schema_a: Dict[str, Any],
        schema_b: Dict[str, Any]
    ) -> bool:
        """
        Check if two JSON schemas are compatible.
        
        Simplified check: types must match.
        Full implementation would use jsonschema library.
        
        Args:
            schema_a: First schema
            schema_b: Second schema
        
        Returns:
            True if compatible
        """
        # Simple type check
        type_a = schema_a.get('type')
        type_b = schema_b.get('type')
        
        if type_a != type_b:
            return False
        
        # Check required fields for objects
        if type_a == 'object':
            req_a = set(schema_a.get('required', []))
            req_b = set(schema_b.get('required', []))
            
            # schema_a must provide all fields required by schema_b
            if not req_b.issubset(req_a):
                return False
        
        return True
    
    def update_statistics(
        self,
        tool_name: str,
        success: bool,
        prediction_error: float
    ):
        """
        Update execution statistics for a tool.
        
        Used by Free Energy calculation to estimate success probabilities
        and epistemic values.
        
        Args:
            tool_name: Name of executed tool
            success: Whether execution succeeded
            prediction_error: Observed prediction error
        
        Examples:
            >>> registry.update_statistics("api_tool", success=True, prediction_error=0.1)
        """
        if tool_name not in self.statistics:
            self.statistics[tool_name] = {
                'success_rate': 0.5,
                'avg_prediction_error': 0.5,
                'error_variance': 0.0,
                'call_count': 0,
                'failure_count': 0
            }
        
        stats = self.statistics[tool_name]
        
        # Update counts
        stats['call_count'] += 1
        if not success:
            stats['failure_count'] += 1
        
        # Update success rate (running average)
        stats['success_rate'] = 1.0 - (stats['failure_count'] / stats['call_count'])
        
        # Update prediction error average
        n = stats['call_count']
        old_avg = stats['avg_prediction_error']
        new_avg = old_avg + (prediction_error - old_avg) / n
        stats['avg_prediction_error'] = new_avg
        
        # Update variance (Welford's online algorithm)
        if n > 1:
            old_var = stats['error_variance']
            stats['error_variance'] = old_var + (prediction_error - old_avg) * (prediction_error - new_avg)
    
    def get_statistics(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve statistics for a tool.
        
        Args:
            tool_name: Tool name
        
        Returns:
            Statistics dict or None
        """
        return self.statistics.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
