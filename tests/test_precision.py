"""
Tests for precision tracking components.
"""

import pytest
import numpy as np

from lrs.core.precision import PrecisionParameters, HierarchicalPrecision


class TestPrecisionParameters:
    """Test PrecisionParameters class"""
    
    def test_initialization(self):
        """Test default initialization"""
        precision = PrecisionParameters()
        
        assert precision.alpha == 5.0
        assert precision.beta == 5.0
        assert precision.value == 0.5  # E[γ] = 5/(5+5)
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        precision = PrecisionParameters(alpha=10.0, beta=5.0)
        
        assert precision.alpha == 10.0
        assert precision.beta == 5.0
        assert abs(precision.value - 0.667) < 0.01
    
    def test_update_low_error_increases_precision(self):
        """Low prediction error should increase precision"""
        precision = PrecisionParameters()
        initial_value = precision.value
        
        new_value = precision.update(prediction_error=0.1)
        
        assert new_value > initial_value
        assert precision.alpha > 5.0  # Alpha increased
        assert precision.beta == 5.0  # Beta unchanged
    
    def test_update_high_error_decreases_precision(self):
        """High prediction error should decrease precision"""
        precision = PrecisionParameters()
        initial_value = precision.value
        
        new_value = precision.update(prediction_error=0.9)
        
        assert new_value < initial_value
        assert precision.alpha == 5.0  # Alpha unchanged
        assert precision.beta > 5.0  # Beta increased
    
    def test_asymmetric_learning(self):
        """Loss should be faster than gain (asymmetric learning)"""
        precision = PrecisionParameters(
            learning_rate_gain=0.1,
            learning_rate_loss=0.2
        )
        
        # Gain
        precision.update(0.1)
        alpha_gain = precision.alpha - 5.0
        
        # Reset
        precision.alpha = 5.0
        precision.beta = 5.0
        
        # Loss
        precision.update(0.9)
        beta_loss = precision.beta - 5.0
        
        assert beta_loss > alpha_gain
    
    def test_variance_calculation(self):
        """Test variance calculation"""
        precision = PrecisionParameters(alpha=10.0, beta=10.0)
        
        variance = precision.variance
        
        # Variance should be positive
        assert variance > 0
        
        # Higher α and β → lower variance (more certain)
        precision2 = PrecisionParameters(alpha=100.0, beta=100.0)
        assert precision2.variance < variance
    
    def test_reset(self):
        """Test reset to initial prior"""
        precision = PrecisionParameters()
        
        # Update several times
        for _ in range(10):
            precision.update(np.random.random())
        
        # Reset
        precision.reset()
        
        assert precision.alpha == 5.0
        assert precision.beta == 5.0
        assert precision.value == 0.5


class TestHierarchicalPrecision:
    """Test HierarchicalPrecision class"""
    
    def test_initialization(self):
        """Test default initialization"""
        hp = HierarchicalPrecision()
        
        assert hp.abstract.value == 0.5
        assert hp.planning.value == 0.5
        assert hp.execution.value == 0.5
    
    def test_get_level(self):
        """Test getting precision for specific level"""
        hp = HierarchicalPrecision()
        
        assert hp.get_level('abstract') == 0.5
        assert hp.get_level('planning') == 0.5
        assert hp.get_level('execution') == 0.5
    
    def test_get_level_invalid(self):
        """Test error on invalid level"""
        hp = HierarchicalPrecision()
        
        with pytest.raises(ValueError):
            hp.get_level('invalid_level')
    
    def test_get_all(self):
        """Test getting all precision values"""
        hp = HierarchicalPrecision()
        
        all_prec = hp.get_all()
        
        assert 'abstract' in all_prec
        assert 'planning' in all_prec
        assert 'execution' in all_prec
    
    def test_update_execution_no_propagation(self):
        """Small error at execution should not propagate"""
        hp = HierarchicalPrecision(propagation_threshold=0.7)
        
        initial_planning = hp.planning.value
        
        # Small error (below threshold)
        updated = hp.update('execution', prediction_error=0.3)
        
        # Only execution should change
        assert 'execution' in updated
        assert 'planning' not in updated
        assert hp.planning.value == initial_planning
    
    def test_update_execution_with_propagation(self):
        """Large error at execution should propagate to planning"""
        hp = HierarchicalPrecision(propagation_threshold=0.7)
        
        initial_planning = hp.planning.value
        initial_abstract = hp.abstract.value
        
        # Large error (above threshold)
        updated = hp.update('execution', prediction_error=0.95)
        
        # Should update execution and planning
        assert 'execution' in updated
        assert 'planning' in updated
        assert hp.planning.value < initial_planning
        
        # Abstract might or might not update depending on attenuation
        # (attenuated error might fall below threshold)
    
    def test_update_planning_propagates_to_abstract(self):
        """High error at planning should propagate to abstract"""
        hp = HierarchicalPrecision(propagation_threshold=0.7)
        
        initial_abstract = hp.abstract.value
        
        # High error at planning
        updated = hp.update('planning', prediction_error=0.9)
        
        assert 'planning' in updated
        assert 'abstract' in updated
        assert hp.abstract.value < initial_abstract
    
    def test_attenuation_factor(self):
        """Test that errors are attenuated when propagating"""
        hp = HierarchicalPrecision(
            propagation_threshold=0.7,
            attenuation_factor=0.5
        )
        
        # Error of 0.9 at execution
        # Attenuated to 0.45 for planning (0.9 * 0.5)
        # Below threshold → no further propagation
        updated = hp.update('execution', prediction_error=0.9)
        
        # Should update planning but not abstract
        assert 'execution' in updated
        assert 'planning' in updated
        # Abstract may or may not be in updated depending on second attenuation
    
    def test_reset(self):
        """Test reset all levels"""
        hp = HierarchicalPrecision()
        
        # Update several times
        for _ in range(5):
            hp.update('execution', np.random.random())
            hp.update('planning', np.random.random())
        
        # Reset
        hp.reset()
        
        assert hp.abstract.value == 0.5
        assert hp.planning.value == 0.5
        assert hp.execution.value == 0.5


class TestPrecisionEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_precision_bounds(self):
        """Precision should stay in [0, 1]"""
        precision = PrecisionParameters()
        
        # Many successful updates
        for _ in range(100):
            precision.update(0.0)
        
        assert 0 <= precision.value <= 1
        
        # Many failed updates
        precision.reset()
        for _ in range(100):
            precision.update(1.0)
        
        assert 0 <= precision.value <= 1
    
    def test_zero_prediction_error(self):
        """Test with zero prediction error"""
        precision = PrecisionParameters()
        
        new_value = precision.update(0.0)
        
        assert new_value > 0.5  # Should increase
    
    def test_one_prediction_error(self):
        """Test with maximum prediction error"""
        precision = PrecisionParameters()
        
        new_value = precision.update(1.0)
        
        assert new_value < 0.5  # Should decrease
    
    def test_threshold_boundary(self):
        """Test behavior at threshold boundary"""
        precision = PrecisionParameters(threshold=0.5)
        
        # Exactly at threshold
        precision.update(0.5)
        
        # Should trigger gain (error < threshold is false at boundary)
        # Depending on implementation, may need adjustment
    
    def test_very_high_alpha_beta(self):
        """Test with very confident prior"""
        precision = PrecisionParameters(alpha=1000.0, beta=1000.0)
        
        # Should be very resistant to change
        initial = precision.value
        precision.update(0.0)
        
        assert abs(precision.value - initial) < 0.01


class TestPrecisionStatistics:
    """Test statistical properties of precision tracking"""
    
    def test_convergence_with_consistent_low_error(self):
        """Precision should converge high with consistent success"""
        precision = PrecisionParameters()
        
        # Simulate 50 successful executions
        for _ in range(50):
            precision.update(0.1)
        
        assert precision.value > 0.8
    
    def test_convergence_with_consistent_high_error(self):
        """Precision should converge low with consistent failure"""
        precision = PrecisionParameters()
        
        # Simulate 50 failed executions
        for _ in range(50):
            precision.update(0.9)
        
        assert precision.value < 0.2
    
    def test_recovery_from_collapse(self):
        """Precision should recover after collapse if errors improve"""
        precision = PrecisionParameters()
        
        # Collapse precision
        for _ in range(10):
            precision.update(0.95)
        
        collapsed_value = precision.value
        assert collapsed_value < 0.4
        
        # Recover with consistent success
        for _ in range(20):
            precision.update(0.1)
        
        assert precision.value > collapsed_value
        assert precision.value > 0.5
    
    def test_noise_resistance(self):
        """Precision should handle noisy signals"""
        precision = PrecisionParameters()
        
        # Simulate noisy but generally good performance
        np.random.seed(42)
        errors = np.random.beta(2, 8, size=100)  # Skewed toward low errors
        
        for error in errors:
            precision.update(error)
        
        # Should settle somewhere reasonable
        assert 0.3 < precision.value < 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
