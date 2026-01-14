"""
Multi-agent coordinator for LRS systems.

Manages turn-taking, communication, and shared state for multiple agents.
"""

from typing import List, Dict, Any, Optional
import time

from lrs.multi_agent.shared_state import SharedWorldState
from lrs.multi_agent.social_precision import SocialPrecisionTracker
from lrs.multi_agent.communication import CommunicationLens


class MultiAgentCoordinator:
    """
    Coordinate multiple LRS agents.
    
    Provides:
    - Shared world state
    - Turn-based execution
    - Communication infrastructure
    - Social precision tracking
    
    Examples:
        >>> coordinator = MultiAgentCoordinator()
        >>> 
        >>> # Register agents
        >>> coordinator.register_agent("agent_a", agent_a)
        >>> coordinator.register_agent("agent_b", agent_b)
        >>> 
        >>> # Run coordination
        >>> results = coordinator.run(
        ...     task="Coordinate warehouse operations",
        ...     max_rounds=10
        ... )
    """
    
    def __init__(self):
        """Initialize coordinator"""
        self.shared_state = SharedWorldState()
        self.agents: Dict[str, Any] = {}
        self.social_trackers: Dict[str, SocialPrecisionTracker] = {}
        self.communication_tools: Dict[str, CommunicationLens] = {}
    
    def register_agent(self, agent_id: str, agent: Any):
        """
        Register an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent: LRS agent instance
        """
        self.agents[agent_id] = agent
        
        # Create social precision tracker
        self.social_trackers[agent_id] = SocialPrecisionTracker(agent_id)
        
        # Create communication tool
        self.communication_tools[agent_id] = CommunicationLens(
            agent_id,
            self.shared_state
        )
        
        # Register other agents for social tracking
        for other_id in self.agents.keys():
            if other_id != agent_id:
                self.social_trackers[agent_id].register_agent(other_id)
                self.social_trackers[other_id].register_agent(agent_id)
    
    def run(
        self,
        task: str,
        max_rounds: int = 20,
        turn_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run multi-agent coordination.
        
        Args:
            task: Task description
            max_rounds: Maximum coordination rounds
            turn_order: Optional fixed turn order (default: round-robin)
        
        Returns:
            Coordination results
        
        Examples:
            >>> results = coordinator.run(
            ...     task="Package items for shipping",
            ...     max_rounds=15
            ... )
            >>> 
            >>> print(f"Rounds: {results['total_rounds']}")
            >>> print(f"Messages: {results['total_messages']}")
        """
        if turn_order is None:
            turn_order = list(self.agents.keys())
        
        start_time = time.time()
        total_messages = 0
        
        # Initialize task in shared state
        self.shared_state.update("coordinator", {
            'task': task,
            'status': 'running',
            'round': 0
        })
        
        # Run rounds
        for round_num in range(max_rounds):
            self.shared_state.update("coordinator", {'round': round_num})
            
            # Each agent takes a turn
            for agent_id in turn_order:
                agent = self.agents[agent_id]
                
                # Get agent's view of world
                world_state = self.shared_state.get_all_states()
                
                # Check for messages
                comm_tool = self.communication_tools[agent_id]
                messages = comm_tool.receive_messages()
                
                # Agent decides and acts
                state = {
                    'messages': [{
                        'role': 'user',
                        'content': f"Task: {task}\nRound: {round_num}\nWorld state: {world_state}"
                    }],
                    'belief_state': {
                        'task': task,
                        'round': round_num,
                        'world_state': world_state,
                        'incoming_messages': messages
                    },
                    'max_iterations': 5  # Limited steps per turn
                }
                
                result = agent.invoke(state)
                
                # Update shared state with agent's actions
                self.shared_state.update(agent_id, {
                    'last_action': result.get('tool_history', [])[-1] if result.get('tool_history') else None,
                    'precision': result.get('precision', {}),
                    'completed': result.get('belief_state', {}).get('completed', False)
                })
                
                # Count messages
                if result.get('tool_history'):
                    for entry in result['tool_history']:
                        if 'send_message' in entry.get('tool', ''):
                            total_messages += 1
                
                # Update social precision based on predictions
                self._update_social_precision(agent_id, world_state, result)
            
            # Check termination
            all_states = self.shared_state.get_all_states()
            if all(s.get('completed', False) for s in all_states.values() if s):
                break
        
        execution_time = time.time() - start_time
        
        # Aggregate results
        return {
            'total_rounds': round_num + 1,
            'total_messages': total_messages,
            'execution_time': execution_time,
            'final_state': self.shared_state.get_all_states(),
            'social_precisions': {
                agent_id: tracker.get_all_social_precisions()
                for agent_id, tracker in self.social_trackers.items()
            }
        }
    
    def _update_social_precision(
        self,
        agent_id: str,
        world_state: Dict,
        result: Dict
    ):
        """Update social precision based on action predictions"""
        tracker = self.social_trackers[agent_id]
        
        # For each other agent, compare predicted vs observed action
        for other_id in self.agents.keys():
            if other_id == agent_id:
                continue
            
            # Predict what other agent did
            predicted = tracker.predict_action(other_id, world_state)
            
            # Observe what they actually did
            other_state = world_state.get(other_id, {})
            observed = other_state.get('last_action')
            
            # Update social precision
            if predicted and observed:
                tracker.update_social_precision(
                    other_id,
                    predicted_action=predicted,
                    observed_action=observed
                )
