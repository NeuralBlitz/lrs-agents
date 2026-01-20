"""Tests for precision tracking."""

import pytest
from lrs.core.precision import (
    PrecisionParameters,
    HierarchicalPrecision,
    beta_mean,
    beta_variance,
)


class TestPrecisionParameters:
    """Test PrecisionParameters class."""
    
    def test_initialization(self):
        """Test default initialization."""
        precision = PrecisionParameters()
        
        assert precision.value == 0.5
        assert precision.alpha == 1.0
        assert precision.beta == 1.0
        assert 0.0 < precision.variance < 1.0
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        precision = PrecisionParameters(
            alpha=5.0,
            beta=2.0,
            gain_learning_rate=0.05,
            loss_learning_rate=0.15
        )
        
        assert precision.alpha == 5.0
        assert precision.beta == 2.0
        assert precision.gain_learning_rate == 0.05
        assert precision.loss_learning_rate == 0.15
    
    def test_update_low_error_increases_precision(self):
        """Test precision increases with low prediction error."""
        precision = PrecisionParameters()
        initial = precision.value
        
        precision.update(prediction_error=0.1)
        
        assert precision.value > initial
        assert precision.alpha > 1.0
    
    def test_update_high_error_decreases_precision(self):
        """Test precision decreases with high prediction error."""
        precision = PrecisionParameters()
        initial = precision.value
        
        precision.update(prediction_error=0.9)
        
        assert precision.value < initial
        assert precision.beta > 1.0
    
    def test_asymmetric_learning(self):
        """Test asymmetric learning rates."""
        precision = PrecisionParameters(
            gain_learning_rate=0.1,
            loss_learning_rate=0.2
        )
        
        initial = precision.value
        
        # Small success
        precision.update(0.1)
        after_success = precision.value
        
        # Reset
        precision.reset()
        
        # Small failure
        precision.update(0.9)
        after_failure = precision.value
        
        # Failure should change precision more than success
        success_change = abs(after_success - initial)
        failure_change = abs(after_failure - initial)
        
        assert failure_change > success_change
    
    def test_should_adapt_true(self):
        """Test should_adapt returns True when below threshold."""
        precision = PrecisionParameters(adaptation_threshold=0.4)
        
        # Drive precision down
        for _ in range(10):
            precision.update(0.95)
        
        assert precision.should_adapt()
    
    def test_should_adapt_false(self):
        """Test should_adapt returns False when above threshold."""
        precision = PrecisionParameters(adaptation_threshold=0.3)
        
        assert not precision.should_adapt()
    
    def test_reset(self):
        """Test reset functionality."""
        precision = PrecisionParameters()
        
        precision.update(0.9)
        precision.update(0.9)
        
        precision.reset()
        
        assert precision.alpha == 1.0
        assert precision.beta == 1.0
        assert precision.value == 0.5
    
    def test_get_all(self):
        """Test get_all returns all statistics."""
        precision = PrecisionParameters(alpha=5.0, beta=3.0)
        
        stats = precision.get_all()
        
        assert 'value' in stats
        assert 'alpha' in stats
        assert 'beta' in stats
        assert 'variance' in stats
        assert stats['alpha'] == 5.0
        assert stats['beta'] == 3.0
    
    def test_backward_compatibility_properties(self):
        """Test backward compatibility aliases."""
        precision = PrecisionParameters(
            gain_learning_rate=0.15,
            loss_learning_rate=0.25,
            adaptation_threshold=0.35
        )
        
        assert precision.learning_rate_gain == 0.15
        assert precision.learning_rate_loss == 0.25
        assert precision.threshold == 0.35


class TestHierarchicalPrecision:
    """Test hierarchical precision tracking."""
    
    def test_initialization(self):
        """Test hierarchical precision initialization."""
        hp = HierarchicalPrecision()
        
        # Float properties
        assert hp.abstract == 0.5
        assert hp.planning == 0.5
        assert hp.execution == 0.5
        
        # PrecisionParameters objects
        assert isinstance(hp.get_level('abstract'), PrecisionParameters)
        assert isinstance(hp.get_level('planning'), PrecisionParameters)
        assert isinstance(hp.get_level('execution'), PrecisionParameters)
    
    def test_get_level(self):
        """Test get_level returns PrecisionParameters."""
        hp = HierarchicalPrecision()
        
        abstract = hp.get_level('abstract')
        
        assert isinstance(abstract, PrecisionParameters)
        assert abstract.value == 0.5
    
    def test_get_level_invalid(self):
        """Test get_level raises error for invalid level."""
        hp = HierarchicalPrecision()
        
        with pytest.raises(ValueError):
            hp.get_level('invalid')
    
    def test_update_execution_no_propagation(self):
        """Test execution update without propagation."""
        hp = HierarchicalPrecision()
        
        initial_planning = hp.planning
        initial_abstract = hp.abstract
        
        # Low error - shouldn't propagate (below 0.7 threshold)
        hp.update('execution', 0.3)
        
        assert hp.execution != 0.5  # Changed
        assert hp.planning == initial_planning  # Unchanged
        assert hp.abstract == initial_abstract  # Unchanged
    
    def test_update_execution_with_propagation(self):
        """Test execution error propagates upward."""
        hp = HierarchicalPrecision()
        
        initial_planning = hp.planning
        
        # High error - should propagate (above 0.7 threshold)
        hp.update('execution', 0.95)
        
        assert hp.execution < 0.5
        assert hp.planning < initial_planning  # Should change due to propagation
    
    def test_update_planning_propagates_to_abstract(self):
        """Test planning error propagates to abstract."""
        hp = HierarchicalPrecision()
        
        initial_abstract = hp.abstract
        
        # High error - should propagate
        hp.update('planning', 0.95)
        
        assert hp.planning < 0.5
        assert hp.abstract < initial_abstract  # Should change due to propagation
    
    def test_reset(self):
        """Test reset functionality."""
        hp = HierarchicalPrecision()
        
        # Make some updates
        hp.update('execution', 0.95)
        hp.update('planning', 0.95)
        
        # Verify changed
        assert hp.execution < 0.5
        assert hp.planning < 0.5
        
        # Reset
        hp.reset()
        
        # Verify reset
        assert hp.execution == 0.5
        assert hp.planning == 0.5
        assert hp.abstract == 0.5
    
    def test_get_all_values(self):
        """Test getting all values as dict."""
        hp = HierarchicalPrecision()
        
        values = hp.get_all_values()
        
        assert values == {
            'abstract': 0.5,
            'planning': 0.5,
            'execution': 0.5
        }
    
    def test_should_adapt(self):
        """Test should_adapt checks correct level."""
        hp = HierarchicalPrecision()
        
        # Drive execution precision down
        for _ in range(10):
            hp.update('execution', 0.95)
        
        assert hp.should_adapt('execution')


class TestPrecisionEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_error(self):
        """Test update with zero error."""
        precision = PrecisionParameters()
        initial = precision.value
        
        precision.update(0.0)
        
        assert precision.value > initial
    
    def test_max_error(self):
        """Test update with maximum error."""
        precision = PrecisionParameters()
        initial = precision.value
        
        precision.update(1.0)
        
        assert precision.value < initial
    
    def test_threshold_boundary(self):
        """Test adaptation threshold boundary."""
        precision = PrecisionParameters(adaptation_threshold=0.3)
        
        # Drive precision to just above threshold
        precision.alpha = 1.5
        precision.beta = 3.5  # value = 0.3
        
        assert not precision.should_adapt()
        
        # One more failure
        precision.update(0.9)
        
        assert precision.should_adapt()
    
    def test_variance_decreases_with_updates(self):
        """Test variance decreases as we get more data."""
        precision = PrecisionParameters()
        initial_variance = precision.variance
        
        for _ in range(20):
            precision.update(0.5)
        
        assert precision.variance < initial_variance


class TestPrecisionStatistics:
    """Test precision statistical properties."""
    
    def test_convergence_with_consistent_low_error(self):
        """Test precision converges with consistent success."""
        precision = PrecisionParameters()
        
        for _ in range(50):
            precision.update(0.1)
        
        # Should stabilize at high precision
        # Note: With default learning rates, this converges to ~0.73
        assert precision.value > 0.7
    
    def test_convergence_with_consistent_high_error(self):
        """Test precision converges with consistent failure."""
        precision = PrecisionParameters()
        
        for _ in range(50):
            precision.update(0.9)
        
        # Should stabilize at low precision
        assert precision.value < 0.3
    
    def test_recovery_from_collapse(self):
        """Test precision can recover after collapse."""
        precision = PrecisionParameters()
        
        # Collapse precision
        for _ in range(20):
            precision.update(0.9)
        
        low_precision = precision.value
        
        # Recovery
        for _ in range(20):
            precision.update(0.1)
        
        # Should recover (though not fully due to asymmetric learning)
        assert precision.value > low_precision
    
    def test_oscillation_with_alternating_errors(self):
        """Test behavior with alternating successes and failures."""
        precision = PrecisionParameters()
        
        for i in range(20):
            if i % 2 == 0:
                precision.update(0.1)  # Success
            else:
                precision.update(0.9)  # Failure
        
        # Should stabilize somewhere in middle
        # Due to asymmetric learning, will be slightly below 0.5
        assert 0.3 < precision.value < 0.6


class TestBetaHelpers:
    """Test Beta distribution helper functions."""
    
    def test_beta_mean(self):
        """Test beta_mean calculation."""
        assert beta_mean(1.0, 1.0) == 0.5
        assert beta_mean(2.0, 2.0) == 0.5
        assert beta_mean(3.0, 1.0) == 0.75
        assert beta_mean(1.0, 3.0) == 0.25
    
    def test_beta_variance(self):
        """Test beta_variance calculation."""
        var = beta_variance(1.0, 1.0)
        assert 0.0 < var < 1.0
        
        # Higher alpha+beta should give lower variance
        var1 = beta_variance(1.0, 1.0)
        var2 = beta_variance(10.0, 10.0)
        assert var2 < var1
