"""
State tracking for LRS agents.

Maintains a rolling history of agent states for analysis and visualization.
"""

from typing import List, Dict, Any, Optional
from collections import deque
from dataclimport dataclass
from datetime import datetime
import json


@dataclass
class StateSnapshot:
    """
    Snapshot of agent state at a specific point in time.
    
    Attributes:
        timestamp: When this snapshot was taken
        precision: Precision values at all levels
        prediction_errors: Recent prediction errors
        tool_history: Tool execution history
        adaptation_count: Number of adaptations so far
        belief_state: Current beliefs
    """
    timestamp: datetime
    precision: Dict[str, float]
    prediction_errors: List[float]
    tool_history: List[Dict[str, Any]]
    adaptation_count: int
    belief_state: Dict[str, Any]


class LRSStateTracker:
    """
    Tracks agent state history for monitoring and analysis.
    
    Maintains a rolling window of state snapshots with configurable size.
    Used by the dashboard and for post-execution analysis.
    
    Examples:
        >>> tracker = LRSStateTracker(max_history=100)
        >>> 
        >>> # Track state during execution
        >>> for step in agent_execution:
        ...     tracker.track_state(step)
        >>> 
        >>> # Analyze precision trajectory
        >>> precision_history = tracker.get_precision_trajectory('execution')
        >>> print(f"Average precision: {sum(precision_history) / len(precision_history)}")
        >>> 
        >>> # Get adaptation events
        >>> adaptations = tracker.get_adaptation_events()
        >>> print(f"Total adaptations: {len(adaptations)}")
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize state tracker.
        
        Args:
            max_history: Maximum number of states to keep in history
        """
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.adaptation_events: List[Dict[str, Any]] = []
    
    def track_state(self, state: Dict[str, Any]):
        """
        Track a new state snapshot.
        
        Args:
            state: Current agent state (LRSState dict)
        
        Examples:
            >>> tracker.track_state({
            ...     'precision': {'execution': 0.7, 'planning': 0.6},
            ...     'tool_history': [...],
            ...     'belief_state': {...}
            ... })
        """
        # Extract relevant information
        precision = state.get('precision', {})
        tool_history = state.get('tool_history', [])
        adaptation_count = state.get('adaptation_count', 0)
        belief_state = state.get('belief_state', {})
        
        # Extract recent prediction errors
        recent_errors = []
        if tool_history:
            recent_errors = [
                entry.get('prediction_error', 0.0)
                for entry in tool_history[-10:]  # Last 10
            ]
        
        # Create snapshot
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            precision=precision.copy(),
            prediction_errors=recent_errors,
            tool_history=tool_history.copy(),
            adaptation_count=adaptation_count,
            belief_state=belief_state.copy()
        )
        
        # Add to history
        self.history.append(snapshot)
        
        # Check for adaptation events
        if len(self.history) > 1:
            prev_adaptations = self.history[-2].adaptation_count
            curr_adaptations = adaptation_count
            
            if curr_adaptations > prev_adaptations:
                # New adaptation occurred
                self._record_adaptation_event(state)
    
    def _record_adaptation_event(self, state: Dict[str, Any]):
        """Record an adaptation event with context"""
        tool_history = state.get('tool_history', [])
        precision = state.get('precision', {})
        
        # Find the tool that triggered adaptation
        trigger_tool = None
        trigger_error = None
        
        if tool_history:
            latest = tool_history[-1]
            if latest.get('prediction_error', 0) > 0.7:
                trigger_tool = latest.get('tool')
                trigger_error = latest.get('prediction_error')
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'trigger_tool': trigger_tool,
            'trigger_error': trigger_error,
            'precision_before': self.history[-2].precision if len(self.history) > 1 else {},
            'precision_after': precision,
            'adaptation_number': state.get('adaptation_count', 0)
        }
        
        self.adaptation_events.append(event)
    
    def get_precision_trajectory(self, level: str = 'execution') -> List[float]:
        """
        Get precision trajectory for a specific level.
        
        Args:
            level: Precision level ('abstract', 'planning', or 'execution')
        
        Returns:
            List of precision values over time
        
        Examples:
            >>> trajectory = tracker.get_precision_trajectory('execution')
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(trajectory)
            >>> plt.show()
        """
        return [
            snapshot.precision.get(level, 0.5)
            for snapshot in self.history
        ]
    
    def get_all_precision_trajectories(self) -> Dict[str, List[float]]:
        """
        Get precision trajectories for all levels.
        
        Returns:
            Dict mapping level names to precision trajectories
        """
        return {
            'abstract': self.get_precision_trajectory('abstract'),
            'planning': self.get_precision_trajectory('planning'),
            'execution': self.get_precision_trajectory('execution')
        }
    
    def get_prediction_errors(self) -> List[float]:
        """
        Get all prediction errors from history.
        
        Returns:
            Flat list of all prediction errors
        """
        errors = []
        for snapshot in self.history:
            errors.extend(snapshot.prediction_errors)
        return errors
    
    def get_adaptation_events(self) -> List[Dict[str, Any]]:
        """
        Get all recorded adaptation events.
        
        Returns:
            List of adaptation event dicts
        """
        return self.adaptation_events.copy()
    
    def get_tool_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate tool usage statistics.
        
        Returns:
            Dict mapping tool names to stats (calls, successes, avg_error)
        
        Examples:
            >>> stats = tracker.get_tool_usage_stats()
            >>> for tool, data in stats.items():
            ...     print(f"{tool}: {data['success_rate']:.1%} success rate")
        """
        tool_stats = {}
        
        for snapshot in self.history:
            for entry in snapshot.tool_history:
                tool_name = entry.get('tool')
                if not tool_name:
                    continue
                
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        'calls': 0,
                        'successes': 0,
                        'failures': 0,
                        'total_error': 0.0,
                        'errors': []
                    }
                
                stats = tool_stats[tool_name]
                stats['calls'] += 1
                
                if entry.get('success'):
                    stats['successes'] += 1
                else:
                    stats['failures'] += 1
                
                error = entry.get('prediction_error', 0.0)
                stats['total_error'] += error
                stats['errors'].append(error)
        
        # Calculate derived stats
        for tool_name, stats in tool_stats.items():
            if stats['calls'] > 0:
                stats['success_rate'] = stats['successes'] / stats['calls']
                stats['avg_error'] = stats['total_error'] / stats['calls']
            else:
                stats['success_rate'] = 0.0
                stats['avg_error'] = 0.0
        
        return tool_stats
    
    def get_current_state(self) -> Optional[StateSnapshot]:
        """
        Get most recent state snapshot.
        
        Returns:
            Latest StateSnapshot or None if no history
        """
        if self.history:
            return self.history[-1]
        return None
    
    def export_history(self, filepath: str):
        """
        Export history to JSON file.
        
        Args:
            filepath: Output file path
        
        Examples:
            >>> tracker.export_history('agent_history.json')
        """
        data = {
            'snapshots': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'precision': snapshot.precision,
                    'prediction_errors': snapshot.prediction_errors,
                    'tool_history': snapshot.tool_history,
                    'adaptation_count': snapshot.adaptation_count,
                    'belief_state': snapshot.belief_state
                }
                for snapshot in self.history
            ],
            'adaptation_events': self.adaptation_events
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all tracked history"""
        self.history.clear()
        self.adaptation_events.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of tracked execution.
        
        Returns:
            Dict with summary metrics
        
        Examples:
            >>> summary = tracker.get_summary()
            >>> print(f"Total steps: {summary['total_steps']}")
            >>> print(f"Adaptations: {summary['total_adaptations']}")
        """
        if not self.history:
            return {
                'total_steps': 0,
                'total_adaptations': 0,
                'avg_precision': 0.0,
                'final_precision': {}
            }
        
        precision_trajectories = self.get_all_precision_trajectories()
        
        # Calculate average precision across all levels
        all_precisions = []
        for trajectory in precision_trajectories.values():
            all_precisions.extend(trajectory)
        
        avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
        
        return {
            'total_steps': len(self.history),
            'total_adaptations': len(self.adaptation_events),
            'avg_precision': avg_precision,
            'final_precision': self.history[-1].precision,
            'tool_usage': self.get_tool_usage_stats()
        }
