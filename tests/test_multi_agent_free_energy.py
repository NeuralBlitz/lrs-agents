"""
Tests for multi-agent Free Energy calculations.
"""

import pytest

from lrs.multi_agent.multi_agent_free_energy import (
    calculate_social_free_energy,
    calculate_total_free_energy,
    should_communicate_based_on_G
)
from lrs.core.lens import ToolLens, ExecutionResult


class DummyTool(ToolLens):
    """Dummy tool for testing"""
    def __init__(self, name):
        super().__init__(name, {}, {})
    
    def get(self, state):
        return ExecutionResult(True, "result", None, 0.1)
    
    def set(self, state, obs):
        return state


class TestSocialFreeEnergy:
    """Test social Free Energy calculation"""
    
    def test_empty_social_precisions(self):
        """Test with no other agents"""
        G_social = calculate_social_free_energy({})
        
        assert G_social == 0.0
    
    def test_high_social_precision_low_G(self):
        """Test that high trust → low social Free Energy"""
        social_precisions = {
            'agent_b': 0.9,  # High trust
            'agent_c': 0.8
        }
        
        G_social = calculate_social_free_energy(social_precisions)
        
        # Low uncertainty → low G
        assert G_social < 0.5
    
    def test_low_social_precision_high_G(self):
        """Test that low trust → high social Free Energy"""
        social_precisions = {
            'agent_b': 0.2,  # Low trust
            'agent_c': 0.3
        }
        
        G_social = calculate_social_free_energy(social_precisions)
        
        # High uncertainty → high G
        assert G_social > 1.0
    
    def test_mixed_social_precisions(self):
        """Test mixed trust levels"""
        social_precisions = {
            'agent_b': 0.8,  # High trust
            'agent_c': 0.3   # Low trust
        }
        
        G_social = calculate_social_free_energy(social_precisions)
        
        # Should be moderate
        assert 0.5 < G_social < 1.5
    
    def test_weight_parameter(self):
        """Test social weight parameter"""
        social_precisions = {'agent_b': 0.5}
        
        G_default = calculate_social_free_energy(social_precisions, weight=1.0)
        G_weighted = calculate_social_free_energy(social_precisions, weight=2.0)
        
        assert G_weighted == 2.0 * G_default


class TestTotalFreeEnergy:
    """Test total Free Energy (environmental + social)"""
    
    def test_combines_environmental_and_social(self):
        """Test that total G combines both components"""
        policy = [DummyTool("tool_a")]
        state = {}
        preferences = {'success': 5.0}
        social_precisions = {'agent_b': 0.5}
        
        G_total = calculate_total_free_energy(
            policy, state, preferences, social_precisions
        )
        
        # Should be a finite number
        assert isinstance(G_total, float)
    
    def test_high_social_uncertainty_increases_G(self):
        """Test that social uncertainty increases total G"""
        policy = [DummyTool("tool_a")]
        state = {}
        preferences = {'success': 5.0}
        
        # High trust
        G_high_trust = calculate_total_free_energy(
            policy, state, preferences,
            social_precisions={'agent_b': 0.9}
        )
        
        # Low trust
        G_low_trust = calculate_total_free_energy(
            policy, state, preferences,
            social_precisions={'agent_b': 0.2}
        )
        
        # Low trust should have higher G
        assert G_low_trust > G_high_trust
    
    def test_social_weight_parameter(self):
        """Test social weight affects total G"""
        policy = [DummyTool("tool_a")]
        state = {}
        preferences = {'success': 5.0}
        social_precisions = {'agent_b': 0.3}
        
        G_low_weight = calculate_total_free_energy(
            policy, state, preferences, social_precisions,
            social_weight=0.5
        )
        
        G_high_weight = calculate_total_free_energy(
            policy, state, preferences, social_precisions,
            social_weight=2.0
        )
        
        # Higher weight → more influence of social uncertainty
        assert G_high_weight != G_low_weight
    
    def test_empty_social_precisions(self):
        """Test with no social component"""
        policy = [DummyTool("tool_a")]
        state = {}
        preferences = {'success': 5.0}
        
        G_total = calculate_total_free_energy(
            policy, state, preferences,
            social_precisions={}
        )
        
        # Should still compute (just environmental G)
        assert isinstance(G_total, float)


class TestCommunicationDecision:
    """Test communication decision based on G"""
    
    def test_communicate_when_G_lower(self):
        """Test communicate when G(communicate) < G(no communicate)"""
        G_communicate = -1.5
        G_no_communicate = 0.5
        
        should_comm = should_communicate_based_on_G(
            G_communicate,
            G_no_communicate,
            precision=0.9  # High precision → deterministic
        )
        
        assert should_comm is True
    
    def test_dont_communicate_when_G_higher(self):
        """Test don't communicate when G(communicate) > G(no communicate)"""
        G_communicate = 1.5
        G_no_communicate = -0.5
        
        should_comm = should_communicate_based_on_G(
            G_communicate,
            G_no_communicate,
            precision=0.9
        )
        
        assert should_comm is False
    
    def test_stochastic_selection_low_precision(self):
        """Test stochastic selection with low precision"""
        import numpy as np
        
        G_communicate = -1.0
        G_no_communicate = 0.0
        
        np.random.seed(42)
        
        # Run multiple times with low precision
        decisions = [
            should_communicate_based_on_G(
                G_communicate,
                G_no_communicate,
                precision=0.3
            )
            for _ in range(100)
        ]
        
        # Should have some diversity (not all True)
        # But should favor communication (lower G)
        comm_count = sum(decisions)
        
        assert 50 < comm_count < 100  # Mostly communicate, some exploration
    
    def test_deterministic_selection_high_precision(self):
        """Test deterministic selection with high precision"""
        G_communicate = -1.0
        G_no_communicate = 0.0
        
        # Run multiple times with high precision
        decisions = [
            should_communicate_based_on_G(
                G_communicate,
                G_no_communicate,
                precision=0.95
            )
            for _ in range(50)
        ]
        
        # Should always communicate (lower G)
        assert all(decisions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
