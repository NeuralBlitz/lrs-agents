"""Shared test fixtures and configuration."""

import pytest
from unittest.mock import Mock
from typing import Dict, Any, List


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_state() -> Dict[str, Any]:
    """Sample agent state for testing."""
    return {
        'messages': [
            {'role': 'user', 'content': 'Test task'}
        ],
        'belief_state': {'goal': 'test'},
        'precision': {'execution': 0.5, 'planning': 0.5, 'abstract': 0.5},
        'tool_history': [],
        'adaptation_count': 0
    }


@pytest.fixture
def sample_preferences() -> Dict[str, float]:
    """Sample preferences for testing."""
    return {
        'success': 5.0,
        'error': -3.0,
        'step_cost': -0.1
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(
        content='{"proposals": []}'
    ))
    return llm
