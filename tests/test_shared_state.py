"""
Tests for shared world state.
"""

import pytest
from threading import Thread

from lrs.multi_agent.shared_state import SharedWorldState


class TestSharedWorldState:
    """Test SharedWorldState"""
    
    def test_initialization(self):
        """Test initialization"""
        state = SharedWorldState()
        
        assert len(state._state) == 0
        assert len(state._history) == 0
    
    def test_update_state(self):
        """Test updating agent state"""
        state = SharedWorldState()
        
        state.update("agent_a", {"status": "working", "task": "fetch_data"})
        
        agent_state = state.get_agent_state("agent_a")
        
        assert agent_state['status'] == "working"
        assert agent_state['task'] == "fetch_data"
        assert 'last_update' in agent_state
    
    def test_update_merges_with_existing(self):
        """Test that updates merge with existing state"""
        state = SharedWorldState()
        
        state.update("agent_a", {"field1": "value1"})
        state.update("agent_a", {"field2": "value2"})
        
        agent_state = state.get_agent_state("agent_a")
        
        assert agent_state['field1'] == "value1"
        assert agent_state['field2'] == "value2"
    
    def test_get_agent_state_nonexistent(self):
        """Test getting state for non-existent agent"""
        state = SharedWorldState()
        
        agent_state = state.get_agent_state("nonexistent")
        
        assert agent_state == {}
    
    def test_get_all_states(self):
        """Test getting all agent states"""
        state = SharedWorldState()
        
        state.update("agent_a", {"status": "working"})
        state.update("agent_b", {"status": "idle"})
        state.update("agent_c", {"status": "waiting"})
        
        all_states = state.get_all_states()
        
        assert len(all_states) == 3
        assert "agent_a" in all_states
        assert "agent_b" in all_states
        assert "agent_c" in all_states
    
    def test_get_other_agents(self):
        """Test getting list of other agents"""
        state = SharedWorldState()
        
        state.update("agent_a", {"status": "working"})
        state.update("agent_b", {"status": "idle"})
        state.update("agent_c", {"status": "waiting"})
        
        others = state.get_other_agents("agent_a")
        
        assert len(others) == 2
        assert "agent_b" in others
        assert "agent_c" in others
        assert "agent_a" not in others
    
    def test_history_recording(self):
        """Test that history is recorded"""
        state = SharedWorldState()
        
        state.update("agent_a", {"action": "fetch"})
        state.update("agent_b", {"action": "process"})
        
        history = state.get_history()
        
        assert len(history) == 2
        assert history[0]['agent_id'] == "agent_a"
        assert history[1]['agent_id'] == "agent_b"
    
    def test_history_filtering_by_agent(self):
        """Test filtering history by agent"""
        state = SharedWorldState()
        
        state.update("agent_a", {"action": "fetch"})
        state.update("agent_b", {"action": "process"})
        state.update("agent_a", {"action": "cache"})
        
        history = state.get_history(agent_id="agent_a")
        
        assert len(history) == 2
        assert all(h['agent_id'] == "agent_a" for h in history)
    
    def test_history_limit(self):
        """Test history limit"""
        state = SharedWorldState()
        
        # Create 150 updates
        for i in range(150):
            state.update("agent_a", {"count": i})
        
        history = state.get_history(limit=50)
        
        assert len(history) == 50
        # Should be most recent 50
        assert history[-1]['updates']['count'] == 149
    
    def test_subscribe_to_updates(self):
        """Test subscribing to state changes"""
        state = SharedWorldState()
        
        updates_received = []
        
        def callback(agent_id, updates):
            updates_received.append((agent_id, updates))
        
        state.subscribe("agent_a", callback)
        
        state.update("agent_a", {"status": "working"})
        
        assert len(updates_received) == 1
        assert updates_received[0][0] == "agent_a"
        assert updates_received[0][1]['status'] == "working"
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers for same agent"""
        state = SharedWorldState()
        
        received_1 = []
        received_2 = []
        
        state.subscribe("agent_a", lambda aid, u: received_1.append(u))
        state.subscribe("agent_a", lambda aid, u: received_2.append(u))
        
        state.update("agent_a", {"test": "value"})
        
        assert len(received_1) == 1
        assert len(received_2) == 1
    
    def test_subscriber_error_handling(self):
        """Test that subscriber errors don't break updates"""
        state = SharedWorldState()
        
        def bad_callback(agent_id, updates):
            raise Exception("Subscriber error")
        
        state.subscribe("agent_a", bad_callback)
        
        # Should not raise exception
        state.update("agent_a", {"test": "value"})
        
        # State should still be updated
        agent_state = state.get_agent_state("agent_a")
        assert agent_state['test'] == "value"
    
    def test_export_state(self):
        """Test exporting state to file"""
        import tempfile
        
        state = SharedWorldState()
        state.update("agent_a", {"status": "working"})
        state.update("agent_b", {"status": "idle"})
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            state.export_state(filepath)
            
            # Read back
            import json
            with open(filepath) as f:
                data = json.load(f)
            
            assert 'states' in data
            assert 'history' in data
            assert len(data['states']) == 2
        finally:
            import os
            os.unlink(filepath)
    
    def test_clear_state(self):
        """Test clearing all state"""
        state = SharedWorldState()
        
        state.update("agent_a", {"status": "working"})
        state.update("agent_b", {"status": "idle"})
        
        state.clear()
        
        assert len(state._state) == 0
        assert len(state._history) == 0
    
    def test_thread_safety(self):
        """Test thread-safe updates"""
        state = SharedWorldState()
        
        def update_worker(agent_id, count):
            for i in range(count):
                state.update(agent_id, {"count": i})
        
        threads = [
            Thread(target=update_worker, args=("agent_a", 50)),
            Thread(target=update_worker, args=("agent_b", 50)),
            Thread(target=update_worker, args=("agent_c", 50))
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All updates should be recorded
        assert len(state.get_all_states()) == 3
        
        # History should have all updates (150 total)
        assert len(state._history) == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
