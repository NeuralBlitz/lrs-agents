"""
Shared world state for multi-agent systems.

Provides a common observable state that all agents can read and update.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from threading import Lock
import json


class SharedWorldState:
    """
    Thread-safe shared state for multi-agent coordination.
    
    All agents can:
    - Read the shared state
    - Write updates to the shared state
    - Subscribe to state changes
    
    Examples:
        >>> state = SharedWorldState()
        >>> 
        >>> # Agent A writes
        >>> state.update("agent_a", {"status": "working", "task": "fetch_data"})
        >>> 
        >>> # Agent B reads
        >>> a_state = state.get_agent_state("agent_a")
        >>> print(a_state["status"])
        "working"
        >>> 
        >>> # Agent B updates
        >>> state.update("agent_b", {"status": "idle"})
        >>> 
        >>> # View all agents
        >>> all_states = state.get_all_states()
    """
    
    def __init__(self):
        """Initialize shared state"""
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._history: List[Dict[str, Any]] = []
        self._subscribers: Dict[str, List[callable]] = {}
    
    def update(self, agent_id: str, updates: Dict[str, Any]):
        """
        Update state for an agent.
        
        Args:
            agent_id: Agent making the update
            updates: State updates (merged with existing state)
        
        Examples:
            >>> state.update("agent_a", {"position": (10, 20), "task": "move"})
        """
        with self._lock:
            if agent_id not in self._state:
                self._state[agent_id] = {}
            
            # Merge updates
            self._state[agent_id].update(updates)
            self._state[agent_id]['last_update'] = datetime.now().isoformat()
            
            # Record in history
            self._history.append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'updates': updates.copy()
            })
            
            # Notify subscribers
            self._notify_subscribers(agent_id, updates)
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state for an agent.
        
        Args:
            agent_id: Agent to query
        
        Returns:
            Agent's state dict or None
        """
        with self._lock:
            return self._state.get(agent_id, {}).copy()
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get state for all agents.
        
        Returns:
            Dict mapping agent IDs to their states
        """
        with self._lock:
            return {
                agent_id: state.copy()
                for agent_id, state in self._state.items()
            }
    
    def get_other_agents(self, agent_id: str) -> List[str]:
        """
        Get list of other agent IDs.
        
        Args:
            agent_id: Requesting agent
        
        Returns:
            List of other agent IDs
        """
        with self._lock:
            return [aid for aid in self._state.keys() if aid != agent_id]
    
    def subscribe(self, agent_id: str, callback: callable):
        """
        Subscribe to state changes for an agent.
        
        Args:
            agent_id: Agent to watch
            callback: Function called on updates, signature: (agent_id, updates)
        
        Examples:
            >>> def on_update(agent_id, updates):
            ...     print(f"{agent_id} updated: {updates}")
            >>> 
            >>> state.subscribe("agent_a", on_update)
        """
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        
        self._subscribers[agent_id].append(callback)
    
    def _notify_subscribers(self, agent_id: str, updates: Dict[str, Any]):
        """Notify subscribers of state change"""
        if agent_id in self._subscribers:
            for callback in self._subscribers[agent_id]:
                try:
                    callback(agent_id, updates)
                except Exception as e:
                    print(f"Error in subscriber callback: {e}")
    
    def get_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get state update history.
        
        Args:
            agent_id: Optional filter by agent
            limit: Maximum number of records
        
        Returns:
            List of history records
        """
        with self._lock:
            history = self._history
            
            if agent_id:
                history = [h for h in history if h['agent_id'] == agent_id]
            
            return history[-limit:]
    
    def export_state(self, filepath: str):
        """
        Export current state to JSON file.
        
        Args:
            filepath: Output file path
        """
        with self._lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'states': self._state,
                'history': self._history[-1000:]  # Last 1000 updates
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all state (for testing)"""
        with self._lock:
            self._state.clear()
            self._history.clear()
