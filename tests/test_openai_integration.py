"""
Tests for OpenAI Assistants integration.
"""

import pytest
from unittest.mock import Mock
import json

from lrs.integration.openai_assistants import (
    OpenAIAssistantLens,
    OpenAIAssistantPolicyGenerator
)


class MockOpenAIClient:
    """Mock OpenAI client"""
    def __init__(self):
        self.beta = Mock()
        self.beta.threads = Mock()
        self.beta.assistants = Mock()


class TestOpenAIAssistantLens:
    """Test OpenAIAssistantLens"""
    
    def test_initialization(self):
        """Test lens initialization"""
        client = MockOpenAIClient()
        
        lens = OpenAIAssistantLens(
            client=client,
            assistant_id="asst_123"
        )
        
        assert lens.client == client
        assert lens.assistant_id == "asst_123"
    
    def test_temperature_adaptation(self):
        """Test temperature adaptation based on precision"""
        client = MockOpenAIClient()
        lens = OpenAIAssistantLens(
            client=client,
            assistant_id="asst_123",
            temperature=0.7
        )
        
        # Low precision → high temperature
        temp_low = lens._adapt_temperature(0.2)
        
        # High precision → low temperature
        temp_high = lens._adapt_temperature(0.9)
        
        assert temp_low > temp_high
    
    def test_successful_query(self):
        """Test successful assistant query"""
        client = MockOpenAIClient()
        
        # Mock thread creation
        mock_thread = Mock()
        mock_thread.id = "thread_123"
        client.beta.threads.create = Mock(return_value=mock_thread)
        
        # Mock message creation
        client.beta.threads.messages.create = Mock()
        
        # Mock run creation
        mock_run = Mock()
        mock_run.id = "run_123"
        client.beta.threads.runs.create = Mock(return_value=mock_run)
        
        # Mock run completion
        mock_completed_run = Mock()
        mock_completed_run.status = "completed"
        client.beta.threads.runs.retrieve = Mock(return_value=mock_completed_run)
        
        # Mock messages retrieval
        mock_message = Mock()
        mock_message.content = [Mock()]
        mock_message.content[0].text = Mock()
        mock_message.content[0].text.value = json.dumps({
            "proposals": [
                {
                    "policy_id": 1,
                    "tools": ["tool_a"],
                    "estimated_success_prob": 0.8,
                    "expected_information_gain": 0.3,
                    "strategy": "exploit",
                    "rationale": "Test"
                }
            ]
        })
        
        mock_messages = Mock()
        mock_messages.data = [mock_message]
        client.beta.threads.messages.list = Mock(return_value=mock_messages)
        
        lens = OpenAIAssistantLens(
            client=client,
            assistant_id="asst_123"
        )
        
        result = lens.get({
            'query': 'Generate proposals',
            'precision': 0.5
        })
        
        assert result.success is True
        assert 'proposals' in result.value
    
    def test_timeout_handling(self):
        """Test timeout when assistant doesn't respond"""
        client = MockOpenAIClient()
        
        # Mock thread
        mock_thread = Mock()
        mock_thread.id = "thread_123"
        client.beta.threads.create = Mock(return_value=mock_thread)
        client.beta.threads.messages.create = Mock()
        
        # Mock run
        mock_run = Mock()
        mock_run.id = "run_123"
        client.beta.threads.runs.create = Mock(return_value=mock_run)
        
        # Mock run that never completes
        mock_pending_run = Mock()
        mock_pending_run.status = "in_progress"
        client.beta.threads.runs.retrieve = Mock(return_value=mock_pending_run)
        
        lens = OpenAIAssistantLens(
            client=client,
            assistant_id="asst_123",
            max_wait=1  # Short timeout for testing
        )
        
        result = lens.get({
            'query': 'Generate proposals',
            'precision': 0.5
        })
        
        assert result.success is False
        assert "didn't respond" in result.error or "Timeout" in result.error
    
    def test_failed_run(self):
        """Test handling of failed assistant run"""
        client = MockOpenAIClient()
        
        mock_thread = Mock()
        mock_thread.id = "thread_123"
        client.beta.threads.create = Mock(return_value=mock_thread)
        client.beta.threads.messages.create = Mock()
        
        mock_run = Mock()
        mock_run.id = "run_123"
        client.beta.threads.runs.create = Mock(return_value=mock_run)
        
        # Mock failed run
        mock_failed_run = Mock()
        mock_failed_run.status = "failed"
        client.beta.threads.runs.retrieve = Mock(return_value=mock_failed_run)
        
        lens = OpenAIAssistantLens(
            client=client,
            assistant_id="asst_123"
        )
        
        result = lens.get({
            'query': 'Generate proposals',
            'precision': 0.5
        })
        
        assert result.success is False
        assert "failed" in result.error.lower()


class TestOpenAIAssistantPolicyGenerator:
    """Test OpenAIAssistantPolicyGenerator"""
    
    def test_initialization_creates_assistant(self):
        """Test that initialization creates assistant"""
        client = MockOpenAIClient()
        
        mock_assistant = Mock()
        mock_assistant.id = "asst_123"
        client.beta.assistants.create = Mock(return_value=mock_assistant)
        
        generator = OpenAIAssistantPolicyGenerator(
            client=client,
            model="gpt-4-turbo-preview"
        )
        
        assert generator.assistant_id == "asst_123"
        assert client.beta.assistants.create.called
    
    def test_initialization_uses_existing_assistant(self):
        """Test using existing assistant ID"""
        client = MockOpenAIClient()
        
        generator = OpenAIAssistantPolicyGenerator(
            client=client,
            assistant_id="asst_existing"
        )
        
        assert generator.assistant_id == "asst_existing"
    
    def test_assistant_instructions_include_lrs_concepts(self):
        """Test that created assistant has LRS-specific instructions"""
        client = MockOpenAIClient()
        
        mock_assistant = Mock()
        mock_assistant.id = "asst_123"
        client.beta.assistants.create = Mock(return_value=mock_assistant)
        
        generator = OpenAIAssistantPolicyGenerator(client=client)
        
        # Check that instructions contain key concepts
        call_args = client.beta.assistants.create.call_args
        instructions = call_args[1]['instructions']
        
        assert "Active Inference" in instructions
        assert "policy" in instructions.lower()
        assert "precision" in instructions.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
