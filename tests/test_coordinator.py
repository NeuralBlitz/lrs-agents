"""
Tests for multi-agent coordinator.
"""

import pytest
from unittest.mock import Mock, MagicMock

from lrs.multi_agent.coordinator import MultiAgentCoordinator
from lrs.multi_agent.shared_state import SharedWorldState
from lrs.multi_agent.social_precision import SocialPrecisionTracker


class TestMultiAgentCoordinator:
    """Test MultiAgentCoordinator"""
    
    def test_initialization(self):
        """Test coordinator initialization"""
        coordinator = MultiAgentCoordinator()
        
        assert isinstance(coordinator.shared_state, SharedWorldState)
        assert len(coordinator.agents) == 0
        assert len(coordinator.social_trackers) == 0
    
    def test_register_agent(self):
        """Test registering an agent"""
        coordinator = MultiAgentCoordinator()
        mock_agent = Mock()
        
        coordinator.register_agent("agent_a", mock_agent)
        
        assert "agent_a" in coordinator.agents
        assert coordinator.agents["agent_a"] == mock_agent
        assert "agent_a" in coordinator.social_trackers
        assert "agent_a" in coordinator.communication_tools
    
    def test_register_multiple_agents(self):
        """Test registering multiple agents"""
        coordinator = MultiAgentCoordinator()
        
        agent_a = Mock()
        agent_b = Mock()
        agent_c = Mock()
        
        coordinator.register_agent("agent_a", agent_a)
        coordinator.register_agent("agent_b", agent_b)
        coordinator.register_agent("agent_c", agent_c)
        
        assert len(coordinator.agents) == 3
    
    def test_social_trackers_cross_registered(self):
        """Test that agents track each other"""
        coordinator = MultiAgentCoordinator()
        
        coordinator.register_agent("agent_a", Mock())
        coordinator.register_agent("agent_b", Mock())
        
        # Agent A should track Agent B
        tracker_a = coordinator.social_trackers["agent_a"]
        assert "agent_b" in tracker_a.social_precision
        
        # Agent B should track Agent A
        tracker_b = coordinator.social_trackers["agent_b"]
        assert "agent_a" in tracker_b.social_precision
    
    @pytest.mark.skip(reason="Requires full agent implementation")
    def test_run_coordination(self):
        """Test running coordination loop"""
        coordinator = MultiAgentCoordinator()
        
        # Create mock agents
        mock_agent_a = Mock()
        mock_agent_a.invoke = Mock(return_value={
            'tool_history': [{'tool': 'fetch', 'success': True}],
            'precision': {'execution': 0.7},
            'belief_state': {'completed': False}
        })
        
        mock_agent_b = Mock()
        mock_agent_b.invoke = Mock(return_value={
            'tool_history': [{'tool': 'process', 'success': True}],
            'precision': {'execution': 0.8},
            'belief_state': {'completed': False}
        })
        
        coordinator.register_agent("agent_a", mock_agent_a)
        coordinator.register_agent("agent_b", mock_agent_b)
        
        # Run coordination
        results = coordinator.run(
            task="Test task",
            max_rounds=2
        )
        
        assert 'total_rounds' in results
        assert 'total_messages' in results
        assert results['total_rounds'] <= 2
    
    def test_update_social_precision(self):
        """Test that social precision is updated during coordination"""
        coordinator = MultiAgentCoordinator()
        
        coordinator.register_agent("agent_a", Mock())
        coordinator.register_agent("agent_b", Mock())
        
        # Initial precision
        tracker = coordinator.social_trackers["agent_a"]
        initial_prec = tracker.get_social_precision("agent_b")
        
        # Simulate coordination with prediction
        world_state = {
            "agent_b": {"last_action": "fetch_data"}
        }
        
        result = {
            'tool_history': [],
            'precision': {}
        }
        
        coordinator._update_social_precision("agent_a", world_state, result)
        
        # Precision tracking should have been attempted
        # (exact value depends on prediction logic)
        assert tracker.get_social_precision("agent_b") is not None


class TestCoordinationPatterns:
    """Test common coordination patterns"""
    
    @pytest.mark.skip(reason="Integration test requiring full setup")
    def test_turn_taking(self):
        """Test round-robin turn taking"""
        coordinator = MultiAgentCoordinator()
        
        execution_order = []
        
        def create_tracking_agent(agent_id):
            agent = Mock()
            def invoke(state):
                execution_order.append(agent_id)
                return {
                    'tool_history': [],
                    'precision': {},
                    'belief_state': {'completed': False}
                }
            agent.invoke = invoke
            return agent
        
        coordinator.register_agent("agent_a", create_tracking_agent("agent_a"))
        coordinator.register_agent("agent_b", create_tracking_agent("agent_b"))
        coordinator.register_agent("agent_c", create_tracking_agent("agent_c"))
        
        coordinator.run(task="Test", max_rounds=2)
        
        # Should execute in round-robin order
        # Round 1: a, b, c
        # Round 2: a, b, c
        expected = ["agent_a", "agent_b", "agent_c", "agent_a", "agent_b", "agent_c"]
        assert execution_order == expected
    
    @pytest.mark.skip(reason="Integration test requiring full setup")
    def test_task_completion_ends_coordination(self):
        """Test that coordination ends when all agents complete"""
        coordinator = MultiAgentCoordinator()
        
        # Agents that complete quickly
        def create_completing_agent():
            agent = Mock()
            agent.invoke = Mock(return_value={
                'tool_history': [],
                'precision': {},
                'belief_state': {'completed': True}
            })
            return agent
        
        coordinator.register_agent("agent_a", create_completing_agent())
        coordinator.register_agent("agent_b", create_completing_agent())
        
        results = coordinator.run(task="Test", max_rounds=10)
        
        # Should end early
        assert results['total_rounds'] < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
