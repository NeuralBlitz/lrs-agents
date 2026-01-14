"""
Tests for social precision tracking (multi-agent).
"""

import pytest

from lrs.multi_agent.social_precision import (
    SocialPrecisionTracker,
    SocialPrecisionParameters,
    RecursiveBeliefState
)


class TestSocialPrecisionParameters:
    """Test SocialPrecisionParameters"""
    
    def test_initialization(self):
        """Test default initialization"""
        params = SocialPrecisionParameters()
        
        assert params.alpha == 5.0
        assert params.beta == 5.0
        # Social precision has different learning rates
        assert params.learning_rate_gain < 0.1
        assert params.learning_rate_loss > 0.2
    
    def test_slower_gain_than_environmental(self):
        """Test that social precision gains slower than environmental"""
        from lrs.core.precision import PrecisionParameters
        
        social = SocialPrecisionParameters()
        environmental = PrecisionParameters()
        
        assert social.learning_rate_gain < environmental.learning_rate_gain
    
    def test_faster_loss_than_environmental(self):
        """Test that social precision loses faster"""
        from lrs.core.precision import PrecisionParameters
        
        social = SocialPrecisionParameters()
        environmental = PrecisionParameters()
        
        assert social.learning_rate_loss > environmental.learning_rate_loss


class TestSocialPrecisionTracker:
    """Test SocialPrecisionTracker"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = SocialPrecisionTracker("agent_a")
        
        assert tracker.agent_id == "agent_a"
        assert len(tracker.social_precision) == 0
    
    def test_register_agent(self):
        """Test registering another agent"""
        tracker = SocialPrecisionTracker("agent_a")
        
        tracker.register_agent("agent_b")
        
        assert "agent_b" in tracker.social_precision
        assert tracker.get_social_precision("agent_b") == 0.5
    
    def test_update_correct_prediction_increases_precision(self):
        """Test that correct predictions increase social precision"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        initial = tracker.get_social_precision("agent_b")
        
        # Correct prediction
        tracker.update_social_precision(
            "agent_b",
            predicted_action="fetch_data",
            observed_action="fetch_data"
        )
        
        final = tracker.get_social_precision("agent_b")
        
        assert final > initial
    
    def test_update_incorrect_prediction_decreases_precision(self):
        """Test that incorrect predictions decrease social precision"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        initial = tracker.get_social_precision("agent_b")
        
        # Incorrect prediction
        tracker.update_social_precision(
            "agent_b",
            predicted_action="fetch_data",
            observed_action="use_cache"  # Different!
        )
        
        final = tracker.get_social_precision("agent_b")
        
        assert final < initial
    
    def test_get_all_social_precisions(self):
        """Test getting all social precisions"""
        tracker = SocialPrecisionTracker("agent_a")
        
        tracker.register_agent("agent_b")
        tracker.register_agent("agent_c")
        
        all_prec = tracker.get_all_social_precisions()
        
        assert "agent_b" in all_prec
        assert "agent_c" in all_prec
        assert all_prec["agent_b"] == 0.5
    
    def test_should_communicate_low_social_precision(self):
        """Test communication decision with low social precision"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        # Lower social precision
        for _ in range(5):
            tracker.update_social_precision("agent_b", "fetch", "cache")
        
        # High environmental precision
        should_comm = tracker.should_communicate(
            "agent_b",
            threshold=0.5,
            env_precision=0.8
        )
        
        assert should_comm is True
    
    def test_should_not_communicate_high_social_precision(self):
        """Test no communication with high social precision"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        # Raise social precision
        for _ in range(10):
            tracker.update_social_precision("agent_b", "fetch", "fetch")
        
        should_comm = tracker.should_communicate(
            "agent_b",
            threshold=0.5,
            env_precision=0.8
        )
        
        assert should_comm is False
    
    def test_should_not_communicate_low_env_precision(self):
        """Test no communication when env precision also low"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        # Low social precision but also low env precision
        for _ in range(5):
            tracker.update_social_precision("agent_b", "fetch", "cache")
        
        should_comm = tracker.should_communicate(
            "agent_b",
            threshold=0.5,
            env_precision=0.3  # Low env precision
        )
        
        # Problem might not be social
        assert should_comm is False
    
    def test_action_history_recording(self):
        """Test that action history is recorded"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        tracker.update_social_precision("agent_b", "action_1", "action_1")
        tracker.update_social_precision("agent_b", "action_2", "action_3")
        
        history = tracker.get_action_history("agent_b")
        
        assert len(history) == 2
        assert history[0]['predicted'] == "action_1"
        assert history[1]['observed'] == "action_3"
    
    def test_predict_action_from_history(self):
        """Test action prediction from history"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        # Record some actions
        tracker.update_social_precision("agent_b", "fetch", "fetch")
        tracker.update_social_precision("agent_b", "cache", "cache")
        
        # Predict next action (simple: returns most recent)
        predicted = tracker.predict_action("agent_b", {})
        
        assert predicted == "cache"
    
    def test_predict_action_no_history(self):
        """Test prediction with no history"""
        tracker = SocialPrecisionTracker("agent_a")
        tracker.register_agent("agent_b")
        
        predicted = tracker.predict_action("agent_b", {})
        
        assert predicted is None


class TestRecursiveBeliefState:
    """Test RecursiveBeliefState (theory-of-mind)"""
    
    def test_initialization(self):
        """Test initialization"""
        beliefs = RecursiveBeliefState("agent_a")
        
        assert beliefs.agent_id == "agent_a"
        assert beliefs.my_precision == 0.5
    
    def test_set_my_precision(self):
        """Test setting own precision"""
        beliefs = RecursiveBeliefState("agent_a")
        
        beliefs.set_my_precision(0.8)
        
        assert beliefs.my_precision == 0.8
    
    def test_set_belief_about_other(self):
        """Test setting belief about other agent's precision"""
        beliefs = RecursiveBeliefState("agent_a")
        
        beliefs.set_belief_about_other("agent_b", 0.7)
        
        assert beliefs.belief_about_other["agent_b"] == 0.7
    
    def test_set_belief_about_other_belief(self):
        """Test setting belief about other's belief about me"""
        beliefs = RecursiveBeliefState("agent_a")
        
        # I think Agent B thinks my precision is 0.8
        beliefs.set_belief_about_other_belief("agent_b", 0.8)
        
        assert beliefs.belief_about_other_belief["agent_b"] == 0.8
    
    def test_should_share_uncertainty_when_mismatch(self):
        """Test sharing uncertainty when there's a mismatch"""
        beliefs = RecursiveBeliefState("agent_a")
        
        # I'm uncertain
        beliefs.set_my_precision(0.3)
        
        # But Agent B thinks I'm confident
        beliefs.set_belief_about_other_belief("agent_b", 0.8)
        
        should_share = beliefs.should_share_uncertainty("agent_b")
        
        assert should_share is True
    
    def test_should_not_share_uncertainty_when_aligned(self):
        """Test not sharing when beliefs are aligned"""
        beliefs = RecursiveBeliefState("agent_a")
        
        # I'm confident
        beliefs.set_my_precision(0.8)
        
        # Agent B also thinks I'm confident
        beliefs.set_belief_about_other_belief("agent_b", 0.8)
        
        should_share = beliefs.should_share_uncertainty("agent_b")
        
        assert should_share is False
    
    def test_should_seek_help_when_appropriate(self):
        """Test seeking help when I'm uncertain and other is confident"""
        beliefs = RecursiveBeliefState("agent_a")
        
        # I'm uncertain
        beliefs.set_my_precision(0.3)
        
        # Agent B is confident
        beliefs.set_belief_about_other("agent_b", 0.8)
        
        should_seek = beliefs.should_seek_help("agent_b")
        
        assert should_seek is True
    
    def test_should_not_seek_help_when_both_uncertain(self):
        """Test not seeking help when other agent also uncertain"""
        beliefs = RecursiveBeliefState("agent_a")
        
        # I'm uncertain
        beliefs.set_my_precision(0.3)
        
        # Agent B is also uncertain
        beliefs.set_belief_about_other("agent_b", 0.3)
        
        should_seek = beliefs.should_seek_help("agent_b")
        
        assert should_seek is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
