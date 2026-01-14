"""
Structured logging for LRS-Agents.

Provides JSON-formatted logs for production monitoring and analysis.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class LRSLogger:
    """
    Structured logger for LRS agents.
    
    Logs events in JSON format for easy parsing and analysis.
    Captures:
    - Precision changes
    - Policy selections
    - Tool executions
    - Adaptation events
    - Performance metrics
    
    Examples:
        >>> logger = LRSLogger(agent_id="agent_1", log_file="agent.jsonl")
        >>> 
        >>> logger.log_precision_update(
        ...     level='execution',
        ...     old_value=0.8,
        ...     new_value=0.4,
        ...     prediction_error=0.95
        ... )
        >>> 
        >>> logger.log_tool_execution(
        ...     tool_name="api_fetch",
        ...     success=False,
        ...     execution_time=0.5,
        ...     prediction_error=0.9,
        ...     error_message="Timeout"
        ... )
    """
    
    def __init__(
        self,
        agent_id: str,
        log_file: Optional[str] = None,
        console: bool = True,
        level: int = logging.INFO
    ):
        """
        Initialize structured logger.
        
        Args:
            agent_id: Unique identifier for this agent
            log_file: Optional file path for JSON logs
            console: Whether to also log to console
            level: Logging level
        """
        self.agent_id = agent_id
        self.session_id = f"{agent_id}_{int(time.time())}"
        
        # Create logger
        self.logger = logging.getLogger(f"lrs.{agent_id}")
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # JSON file handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            )
            self.logger.addHandler(console_handler)
    
    def _log(self, event_type: str, data: Dict[str, Any], level: int = logging.INFO):
        """Internal logging method"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': self.agent_id,
            'session_id': self.session_id,
            'event_type': event_type,
            'data': data
        }
        
        self.logger.log(level, json.dumps(log_entry))
    
    # Event-specific logging methods
    
    def log_precision_update(
        self,
        level: str,
        old_value: float,
        new_value: float,
        prediction_error: float,
        propagated: bool = False
    ):
        """
        Log precision update event.
        
        Args:
            level: Precision level (abstract/planning/execution)
            old_value: Previous precision value
            new_value: New precision value
            prediction_error: Triggering prediction error
            propagated: Whether error propagated from lower level
        """
        self._log('precision_update', {
            'level': level,
            'old_value': round(old_value, 4),
            'new_value': round(new_value, 4),
            'delta': round(new_value - old_value, 4),
            'prediction_error': round(prediction_error, 4),
            'propagated': propagated
        })
    
    def log_policy_selection(
        self,
        policies: list,
        selected_index: int,
        G_values: list,
        precision: float
    ):
        """
        Log policy selection via G.
        
        Args:
            policies: List of candidate policies
            selected_index: Index of selected policy
            G_values: Expected Free Energy values
            precision: Current precision value
        """
        self._log('policy_selection', {
            'num_policies': len(policies),
            'selected_index': selected_index,
            'G_values': [round(g, 4) for g in G_values],
            'selected_G': round(G_values[selected_index], 4),
            'precision': round(precision, 4)
        })
    
    def log_tool_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        prediction_error: float,
        error_message: Optional[str] = None
    ):
        """
        Log tool execution.
        
        Args:
            tool_name: Name of executed tool
            success: Whether execution succeeded
            execution_time: Execution time in seconds
            prediction_error: Observed prediction error
            error_message: Error message if failed
        """
        self._log('tool_execution', {
            'tool': tool_name,
            'success': success,
            'execution_time_ms': round(execution_time * 1000, 2),
            'prediction_error': round(prediction_error, 4),
            'error': error_message
        }, level=logging.WARNING if not success else logging.INFO)
    
    def log_adaptation_event(
        self,
        trigger: str,
        old_precision: Dict[str, float],
        new_precision: Dict[str, float],
        action_taken: str
    ):
        """
        Log adaptation event.
        
        Args:
            trigger: What triggered the adaptation
            old_precision: Precision before adaptation
            new_precision: Precision after adaptation
            action_taken: Action taken by agent
        """
        self._log('adaptation', {
            'trigger': trigger,
            'old_precision': {k: round(v, 4) for k, v in old_precision.items()},
            'new_precision': {k: round(v, 4) for k, v in new_precision.items()},
            'action': action_taken
        }, level=logging.WARNING)
    
    def log_performance_metrics(
        self,
        total_steps: int,
        success_rate: float,
        avg_precision: float,
        adaptation_count: int,
        execution_time: float
    ):
        """
        Log aggregate performance metrics.
        
        Args:
            total_steps: Total execution steps
            success_rate: Overall success rate
            avg_precision: Average precision value
            adaptation_count: Number of adaptations
            execution_time: Total execution time
        """
        self._log('performance_metrics', {
            'total_steps': total_steps,
            'success_rate': round(success_rate, 4),
            'avg_precision': round(avg_precision, 4),
            'adaptation_count': adaptation_count,
            'total_time_s': round(execution_time, 2),
            'steps_per_second': round(total_steps / execution_time, 2) if execution_time > 0 else 0
        })
    
    def log_error(
        self,
        error_type: str,
        message: str,
        stack_trace: Optional[str] = None
    ):
        """
        Log error event.
        
        Args:
            error_type: Type of error
            message: Error message
            stack_trace: Optional stack trace
        """
        self._log('error', {
            'error_type': error_type,
            'message': message,
            'stack_trace': stack_trace
        }, level=logging.ERROR)


def create_logger_for_agent(agent_id: str, **kwargs) -> LRSLogger:
    """
    Create logger for LRS agent.
    
    Args:
        agent_id: Agent identifier
        **kwargs: Passed to LRSLogger
    
    Returns:
        Configured logger instance
    
    Examples:
        >>> logger = create_logger_for_agent(
        ...     "production_agent_1",
        ...     log_file="logs/agent.jsonl",
        ...     console=True
        ... )
    """
    return LRSLogger(agent_id=agent_id, **kwargs)
