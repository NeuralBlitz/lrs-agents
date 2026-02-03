"""
Tests for LRSStateTracker.
"""

import pytest
from datetime import datetime
import tempfile
import os

from lrs.monitoring.tracker import LRSStateTracker, StateSnapshot


class TestStateSnapshot:
    """Test StateSnapshot dataclass"""

    def test_snapshot_creation(self):
        """Test creating a snapshot"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            precision={"execution": 0.7, "planning": 0.6},
            prediction_errors=[0.1, 0.3, 0.2],
            tool_history=[{"tool": "fetch", "success": True}],
            adaptation_count=0,
            belief_state={"goal": "test"},
        )

        assert snapshot.precision["execution"] == 0.7
        assert len(snapshot.prediction_errors) == 3
        assert snapshot.adaptation_count == 0


class TestLRSStateTracker:
    """Test LRSStateTracker"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = LRSStateTracker(max_history=100)

        assert len(tracker.history) == 0
        assert len(tracker.adaptation_events) == 0

    def test_track_state(self):
        """Test tracking a state"""
        tracker = LRSStateTracker()

        state = {
            "precision": {"execution": 0.7},
            "tool_history": [{"tool": "fetch", "success": True, "prediction_error": 0.1}],
            "adaptation_count": 0,
            "belief_state": {},
        }

        tracker.track_state(state)

        assert len(tracker.history) == 1

    def test_max_history_limit(self):
        """Test that history is limited"""
        tracker = LRSStateTracker(max_history=5)

        for i in range(10):
            tracker.track_state(
                {"precision": {}, "tool_history": [], "adaptation_count": 0, "belief_state": {}}
            )

        assert len(tracker.history) == 5

    def test_get_precision_trajectory(self):
        """Test getting precision trajectory"""
        tracker = LRSStateTracker()

        for i in range(5):
            tracker.track_state(
                {
                    "precision": {"execution": 0.5 + i * 0.1},
                    "tool_history": [],
                    "adaptation_count": 0,
                    "belief_state": {},
                }
            )

        trajectory = tracker.get_precision_trajectory("execution")

        assert len(trajectory) == 5
        assert trajectory[0] == 0.5
        assert trajectory[4] == 0.9

    def test_get_all_precision_trajectories(self):
        """Test getting all precision trajectories"""
        tracker = LRSStateTracker()

        tracker.track_state(
            {
                "precision": {"execution": 0.7, "planning": 0.6, "abstract": 0.5},
                "tool_history": [],
                "adaptation_count": 0,
                "belief_state": {},
            }
        )

        trajectories = tracker.get_all_precision_trajectories()

        assert "execution" in trajectories
        assert "planning" in trajectories
        assert "abstract" in trajectories

    def test_get_prediction_errors(self):
        """Test getting prediction errors"""
        tracker = LRSStateTracker()

        tracker.track_state(
            {
                "precision": {},
                "tool_history": [{"prediction_error": 0.1}, {"prediction_error": 0.3}],
                "adaptation_count": 0,
                "belief_state": {},
            }
        )

        tracker.track_state(
            {
                "precision": {},
                "tool_history": [{"prediction_error": 0.5}],
                "adaptation_count": 0,
                "belief_state": {},
            }
        )

        errors = tracker.get_prediction_errors()

        # Should flatten all errors
        assert 0.1 in errors
        assert 0.3 in errors
        assert 0.5 in errors

    def test_adaptation_event_detection(self):
        """Test that adaptation events are detected"""
        tracker = LRSStateTracker()

        # First state - no adaptation
        tracker.track_state(
            {"precision": {}, "tool_history": [], "adaptation_count": 0, "belief_state": {}}
        )

        # Second state - adaptation occurred
        tracker.track_state(
            {
                "precision": {"execution": 0.3},
                "tool_history": [{"tool": "fetch", "prediction_error": 0.95}],
                "adaptation_count": 1,
                "belief_state": {},
            }
        )

        events = tracker.get_adaptation_events()

        assert len(events) == 1
        assert events[0]["adaptation_number"] == 1

    def test_get_tool_usage_stats(self):
        """Test getting tool usage statistics"""
        tracker = LRSStateTracker()

        # Track multiple executions
        for _ in range(3):
            tracker.track_state(
                {
                    "precision": {},
                    "tool_history": [{"tool": "fetch", "success": True, "prediction_error": 0.1}],
                    "adaptation_count": 0,
                    "belief_state": {},
                }
            )

        for _ in range(2):
            tracker.track_state(
                {
                    "precision": {},
                    "tool_history": [{"tool": "fetch", "success": False, "prediction_error": 0.9}],
                    "adaptation_count": 0,
                    "belief_state": {},
                }
            )

        stats = tracker.get_tool_usage_stats()

        assert "fetch" in stats
        assert stats["fetch"]["calls"] == 5
        assert stats["fetch"]["successes"] == 3
        assert stats["fetch"]["failures"] == 2
        assert abs(stats["fetch"]["success_rate"] - 0.6) < 0.01

    def test_get_current_state(self):
        """Test getting current state"""
        tracker = LRSStateTracker()

        # No state yet
        assert tracker.get_current_state() is None

        # Add state
        tracker.track_state(
            {
                "precision": {"execution": 0.7},
                "tool_history": [],
                "adaptation_count": 0,
                "belief_state": {},
            }
        )

        current = tracker.get_current_state()

        assert current is not None
        assert current.precision["execution"] == 0.7

    def test_export_history(self):
        """Test exporting history to file"""
        tracker = LRSStateTracker()

        tracker.track_state(
            {
                "precision": {"execution": 0.7},
                "tool_history": [{"tool": "fetch", "success": True, "prediction_error": 0.1}],
                "adaptation_count": 0,
                "belief_state": {},
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            tracker.export_history(filepath)

            # Read back
            import json

            with open(filepath) as f:
                data = json.load(f)

            assert "snapshots" in data
            assert "adaptation_events" in data
            assert len(data["snapshots"]) == 1
        finally:
            os.unlink(filepath)

    def test_clear_history(self):
        """Test clearing history"""
        tracker = LRSStateTracker()

        for _ in range(5):
            tracker.track_state(
                {"precision": {}, "tool_history": [], "adaptation_count": 0, "belief_state": {}}
            )

        tracker.clear()

        assert len(tracker.history) == 0
        assert len(tracker.adaptation_events) == 0

    def test_get_summary(self):
        """Test getting summary statistics"""
        tracker = LRSStateTracker()

        for i in range(5):
            tracker.track_state(
                {
                    "precision": {"execution": 0.5 + i * 0.1},
                    "tool_history": [{"tool": "fetch", "success": True, "prediction_error": 0.1}],
                    "adaptation_count": i,
                    "belief_state": {},
                }
            )

        summary = tracker.get_summary()

        assert summary["total_steps"] == 5
        # Each of the 5 states has adaptation_count = i, so adaptation events should match
        assert summary["total_adaptations"] == 4  # Events detected when count increases
        assert "avg_precision" in summary
        assert "final_precision" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
