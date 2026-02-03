"""
AutoGPT integration for LRS-Agents.

Replaces AutoGPT's command execution loop with LRS Active Inference dynamics.
"""

from typing import Dict, List, Any, Optional, Callable

from lrs.core.lens import ToolLens, ExecutionResult
from lrs.core.registry import ToolRegistry
from lrs.integration.langgraph import create_lrs_agent  # Direct import



class AutoGPTCommand(ToolLens):
    """
    Wraps AutoGPT command as ToolLens.
    
    AutoGPT commands are functions that agents can execute.
    This wrapper adds prediction error tracking.
    """
    
    def __init__(self, command_name: str, command_func: Callable, description: str):
        """
        Initialize AutoGPT command wrapper.
        
        Args:
            command_name: Name of the command
            command_func: Function to execute
            description: Human-readable description
        """
        super().__init__(
            name=command_name,
            input_schema={
                'type': 'object',
                'properties': {
                    'args': {'type': 'object'}
                }
            },
            output_schema={'type': 'string'}
        )
        
        self.command_func = command_func
        self.description = description
    
    def get(self, state: dict) -> ExecutionResult:
        """Execute AutoGPT command"""
        self.call_count += 1
        
        try:
            args = state.get('args', {})
            result = self.command_func(**args)
            
            # Determine prediction error based on result
            if isinstance(result, dict) and result.get('error'):
                self.failure_count += 1
                return ExecutionResult(
                    success=False,
                    value=None,
                    error=result.get('error'),
                    prediction_error=0.9
                )
            else:
                return ExecutionResult(
                    success=True,
                    value=result,
                    error=None,
                    prediction_error=0.1
                )
        
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.95
            )
    
    def set(self, state: dict, observation: Any) -> dict:
        """Update state with command result"""
        return {
            **state,
            f'{self.name}_result': observation,
            'last_command': self.name
        }


class LRSAutoGPTAgent:
    """
    AutoGPT agent powered by LRS Active Inference.
    
    Replaces AutoGPT's standard execution loop with:
    - Precision tracking
    - Expected Free Energy calculation
    - Automatic adaptation on failures
    
    Examples:
        >>> def browse_website(url: str) -> str:
        ...     return requests.get(url).text
        >>> 
        >>> def write_file(filename: str, content: str) -> dict:
        ...     with open(filename, 'w') as f:
        ...         f.write(content)
        ...     return {'status': 'success'}
        >>> 
        >>> agent = LRSAutoGPTAgent(
        ...     name="ResearchAgent",
        ...     role="Research assistant",
        ...     commands={
        ...         'browse': browse_website,
        ...         'write': write_file
        ...     }
        ... )
        >>> 
        >>> result = agent.run("Research AI safety and write report")
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        commands: Dict[str, Callable],
        llm: Any,
        goals: Optional[List[str]] = None
    ):
        """
        Initialize LRS AutoGPT agent.
        
        Args:
            name: Agent name
            role: Agent role description
            commands: Dictionary of {name: function} commands
            llm: Language model for policy generation
            goals: Optional list of goals
        """
        self.name = name
        self.role = role
        self.goals = goals or []
        
        # Convert commands to ToolLens
        self.registry = ToolRegistry()
        for cmd_name, cmd_func in commands.items():
            lens = AutoGPTCommand(
                command_name=cmd_name,
                command_func=cmd_func,
                description=cmd_func.__doc__ or f"Execute {cmd_name}"
            )
            self.registry.register(lens)
        
        # Create LRS agent
        self.agent = create_lrs_agent(
            llm=llm,
            tools=list(self.registry.tools.values()),
            preferences={
                'goal_achieved': 10.0,
                'error': -5.0,
                'cost': -0.1
            }
        )
    
    def run(self, task: str, max_iterations: int = 25) -> Dict:
        """
        Execute task using LRS dynamics.
        
        Args:
            task: Task description
            max_iterations: Maximum execution steps
        
        Returns:
            Execution results with precision trajectory
        """
        state = {
            'messages': [{
                'role': 'user',
                'content': f"""You are {self.name}, a {self.role}.

Goals:
{chr(10).join(f'- {goal}' for goal in self.goals)}

Task: {task}

Available commands: {', '.join(self.registry.tools.keys())}

Generate a plan to achieve this task."""
            }],
            'belief_state': {
                'task': task,
                'goals': self.goals,
                'completed': False
            },
            'max_iterations': max_iterations
        }
        
        result = self.agent.invoke(state)
        
        return {
            'success': result['belief_state'].get('completed', False),
            'precision_trajectory': result.get('precision_history', []),
            'adaptations': result.get('adaptation_count', 0),
            'tool_usage': result.get('tool_history', []),
            'final_state': result['belief_state']
        }


def convert_autogpt_to_lrs(
    autogpt_config: Dict,
    llm: Any
) -> LRSAutoGPTAgent:
    """
    Convert AutoGPT configuration to LRS agent.
    
    Args:
        autogpt_config: AutoGPT agent configuration
            Must contain: 'name', 'role', 'commands'
        llm: Language model
    
    Returns:
        LRS-powered AutoGPT agent
    
    Examples:
        >>> config = {
        ...     'name': 'FileOrganizer',
        ...     'role': 'File organization assistant',
        ...     'commands': {
        ...         'list_files': lambda path: os.listdir(path),
        ...         'move_file': lambda src, dst: shutil.move(src, dst)
        ...     },
        ...     'goals': ['Organize files by type']
        ... }
        >>> 
        >>> agent = convert_autogpt_to_lrs(config, llm)
    """
    return LRSAutoGPTAgent(
        name=autogpt_config['name'],
        role=autogpt_config['role'],
        commands=autogpt_config['commands'],
        llm=llm,
        goals=autogpt_config.get('goals', [])
    )
