"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_state():
    """Sample agent state for testing"""
    return {
        'messages': [{'role': 'user', 'content': 'Test task'}],
        'belief_state': {'goal': 'test'},
        'precision': {
            'execution': 0.5,
            'planning': 0.5,
            'abstract': 0.5
        },
        'tool_history': [],
        'adaptation_count': 0
    }


@pytest.fixture
def sample_preferences():
    """Sample preferences for testing"""
    return {
        'success': 5.0,
        'error': -3.0,
        'step_cost': -0.1
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    from unittest.mock import Mock
    
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="Mock response"))
    
    return llm
