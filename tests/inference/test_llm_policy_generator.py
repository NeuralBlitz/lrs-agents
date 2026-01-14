import json
from unittest.mock import Mock, MagicMock

import pytest

from lrs.core.precision import PrecisionParameters
from lrs.inference.llm_policy_generator import create_mock_generator


def test_create_mock_generator():
    """Verify that the mock generator produces valid proposals."""

    # 1. Setup
    generator = create_mock_generator(num_proposals=3)

    # 2. Execution
    proposals = generator.generate_proposals(
        context={'goal': 'Test goal'},
        precision=PrecisionParameters(alpha=1.0, beta=1.0)  # Corresponds to precision=0.5
    )

    # 3. Assertion
    assert len(proposals) == 3, "Should generate the requested number of proposals"

    # Deeper validation
    first_proposal = proposals[0]
    assert 'tools' in first_proposal
    assert 'reasoning' in first_proposal
    assert first_proposal['strategy'] in ['exploitation', 'exploration', 'balanced']
