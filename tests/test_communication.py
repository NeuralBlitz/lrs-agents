"""
Tests for multi-agent communication.
"""

import pytest

from lrs.multi_agent.communication import (
    Message,
    MessageType,
    CommunicationLens
)
from lrs.multi_agent.shared_state import SharedWorldState


class TestMessage:
    """Test Message dataclass"""
    
    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(
            from_agent="agent_a",
            to_agent="agent_b",
            message_type=MessageType.QUERY,
            content="What is your status?"
        )
        
        assert msg.from_agent == "agent_a"
        assert msg.to_agent == "agent_b"
        assert msg.message_type == MessageType.QUERY
        assert msg.content == "What is your status?"
    
    def test_message_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated"""
        msg = Message(
            from_agent="agent_a",
            to_agent="agent_b",
            message_type=MessageType.INFORM,
            content="Status: idle"
        )
        
        assert msg.timestamp is not None
    
    def test_message_types(self):
        """Test different message types"""
        types = [
            MessageType.QUERY,
            MessageType.INFORM,
            MessageType.REQUEST,
            MessageType.ACKNOWLEDGE,
            MessageType.ERROR
        ]
        
        for msg_type in types:
            msg = Message(
                from_agent="a",
                to_agent="b",
                message_type=msg_type,
                content="test"
            )
            assert msg.message_type == msg_type
    
    def test_message_in_reply_to(self):
        """Test message replies"""
        original = Message(
            from_agent="agent_a",
            to_agent="agent_b",
            message_type=MessageType.QUERY,
            content="Question?"
        )
        
        reply = Message(
            from_agent="agent_b",
            to_agent="agent_a",
            message_type=MessageType.INFORM,
            content="Answer",
            in_reply_to="msg_123"
        )
        
        assert reply.in_reply_to == "msg_123"


class TestCommunicationLens:
    """Test CommunicationLens"""
    
    def test_initialization(self):
        """Test lens initialization"""
        shared_state = SharedWorldState()
        
        comm_lens = CommunicationLens(
            agent_id="agent_a",
            shared_state=shared_state
        )
        
        assert comm_lens.agent_id == "agent_a"
        assert comm_lens.shared_state == shared_state
    
    def test_send_message_success(self):
        """Test sending a message successfully"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        result = comm_lens.get({
            'to_agent': 'agent_b',
            'message_type': 'query',
            'content': 'What is your task?'
        })
        
        assert result.success is True
        assert result.value['sent'] is True
        assert 'message_id' in result.value
    
    def test_send_message_updates_shared_state(self):
        """Test that sending updates shared state"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        comm_lens.get({
            'to_agent': 'agent_b',
            'message_type': 'inform',
            'content': 'Status update'
        })
        
        # Check shared state for agent_b
        agent_b_state = shared_state.get_agent_state("agent_b")
        
        assert 'incoming_message' in agent_b_state
        assert agent_b_state['incoming_message']['from'] == 'agent_a'
    
    def test_send_message_missing_fields(self):
        """Test sending message with missing fields"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        result = comm_lens.get({
            'to_agent': 'agent_b'
            # Missing message_type and content
        })
        
        assert result.success is False
        assert "Missing required fields" in result.error
    
    def test_receive_messages(self):
        """Test receiving messages"""
        shared_state = SharedWorldState()
        
        comm_a = CommunicationLens("agent_a", shared_state)
        comm_b = CommunicationLens("agent_b", shared_state)
        
        # Agent A sends to Agent B
        comm_a.get({
            'to_agent': 'agent_b',
            'message_type': 'query',
            'content': 'Hello'
        })
        
        # Agent B checks for messages
        messages = comm_b.receive_messages()
        
        assert len(messages) == 1
        assert messages[0].from_agent == 'agent_a'
        assert messages[0].content == 'Hello'
    
    def test_receive_messages_empty(self):
        """Test receiving when no messages"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        messages = comm_lens.receive_messages()
        
        assert messages == []
    
    def test_message_storage(self):
        """Test that sent messages are stored"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        comm_lens.get({
            'to_agent': 'agent_b',
            'message_type': 'inform',
            'content': 'Message 1'
        })
        
        comm_lens.get({
            'to_agent': 'agent_c',
            'message_type': 'query',
            'content': 'Message 2'
        })
        
        # Should have stored both messages
        assert len(comm_lens.sent_messages) == 2
    
    def test_prediction_error_for_communication(self):
        """Test that communication has low prediction error (high info gain)"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        result = comm_lens.get({
            'to_agent': 'agent_b',
            'message_type': 'query',
            'content': 'Test'
        })
        
        # Communication reduces uncertainty → low prediction error
        assert result.prediction_error < 0.5
    
    def test_state_update_increments_counter(self):
        """Test that set() increments communication counter"""
        shared_state = SharedWorldState()
        comm_lens = CommunicationLens("agent_a", shared_state)
        
        state = {}
        
        observation = {'sent': True, 'message_id': 'msg_1'}
        new_state = comm_lens.set(state, observation)
        
        assert new_state['communication_count'] == 1
        
        # Send another
        newer_state = comm_lens.set(new_state, observation)
        assert newer_state['communication_count'] == 2
    
    def test_message_cost(self):
        """Test that message cost can be configured"""
        shared_state = SharedWorldState()
        
        comm_lens = CommunicationLens(
            "agent_a",
            shared_state,
            message_cost=0.5
        )
        
        assert comm_lens.message_cost == 0.5


class TestCommunicationPatterns:
    """Test common communication patterns"""
    
    def test_query_response_pattern(self):
        """Test query-response communication pattern"""
        shared_state = SharedWorldState()
        
        comm_a = CommunicationLens("agent_a", shared_state)
        comm_b = CommunicationLens("agent_b", shared_state)
        
        # Agent A queries Agent B
        query_result = comm_a.get({
            'to_agent': 'agent_b',
            'message_type': 'query',
            'content': 'What is your status?'
        })
        
        query_id = query_result.value['message_id']
        
        # Agent B receives query
        messages = comm_b.receive_messages()
        assert len(messages) == 1
        
        # Agent B responds
        comm_b.get({
            'to_agent': 'agent_a',
            'message_type': 'inform',
            'content': 'Status: working',
            'in_reply_to': query_id
        })
        
        # Agent A receives response
        responses = comm_a.receive_messages()
        assert len(responses) == 1
        assert responses[0].message_type == MessageType.INFORM
    
    def test_broadcast_to_multiple_agents(self):
        """Test broadcasting to multiple agents"""
        shared_state = SharedWorldState()
        
        comm_a = CommunicationLens("agent_a", shared_state)
        
        # Send to multiple agents
        for agent_id in ['agent_b', 'agent_c', 'agent_d']:
            comm_a.get({
                'to_agent': agent_id,
                'message_type': 'inform',
                'content': 'Broadcast message'
            })
        
        # All agents should have received
        for agent_id in ['agent_b', 'agent_c', 'agent_d']:
            state = shared_state.get_agent_state(agent_id)
            assert 'incoming_message' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
