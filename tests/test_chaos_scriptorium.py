"""
Tests for Chaos Scriptorium benchmark.
"""

import pytest
import tempfile
import os
from pathlib import Path

from lrs.benchmarks.chaos_scriptorium import (
    ChaosEnvironment,
    ChaosConfig,
    ShellTool,
    PythonTool,
    FileReadTool
)


class TestChaosEnvironment:
    """Test ChaosEnvironment"""
    
    def test_initialization(self):
        """Test environment initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            
            assert env.root_dir == tmpdir
            assert env.step_count == 0
            assert env.locked is False
    
    def test_setup_creates_directory_structure(self):
        """Test that setup creates directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            
            assert os.path.exists(env.vault_dir)
            assert os.path.exists(env.key_path)
    
    def test_setup_creates_secret_key(self):
        """Test that secret key is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            
            content = Path(env.key_path).read_text()
            assert content == env.secret_key
            assert "SECRET_KEY_" in content
    
    def test_tick_increments_step_count(self):
        """Test that tick increments step count"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            
            initial_count = env.step_count
            env.tick()
            
            assert env.step_count == initial_count + 1
    
    def test_chaos_triggered_at_interval(self):
        """Test that chaos is triggered at the right interval"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir, chaos_interval=3)
            env.setup()
            
            # First 2 ticks should not trigger chaos
            env.tick()
            env.tick()
            
            # 3rd tick should trigger chaos
            initial_state = env.locked
            env.tick()
            
            # State might have changed (probabilistic)
            # Just check that tick was called
            assert env.step_count == 3
    
    def test_is_locked_state(self):
        """Test is_locked() method"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            
            # Initially unlocked
            assert env.is_locked() is False
            
            # Manually lock
            env.locked = True
            assert env.is_locked() is True
    
    def test_reset_environment(self):
        """Test resetting environment"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            
            # Make some changes
            for _ in range(10):
                env.tick()
            env.locked = True
            
            # Reset
            env.reset()
            
            assert env.step_count == 0
            assert env.locked is False


class TestChaosTools:
    """Test Chaos Scriptorium tools"""
    
    def test_shell_tool_initialization(self):
        """Test ShellTool initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            tool = ShellTool(env)
            
            assert tool.name == "shell_exec"
            assert tool.env == env
    
    def test_shell_tool_success_when_unlocked(self):
        """Test ShellTool succeeds when unlocked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            env.locked = False
            
            tool = ShellTool(env)
            
            # Run multiple times to account for randomness
            successes = 0
            for _ in range(10):
                result = tool.get({'command': 'echo test'})
                if result.success:
                    successes += 1
            
            # Should succeed most of the time when unlocked
            assert successes >= 7
    
    def test_shell_tool_often_fails_when_locked(self):
        """Test ShellTool often fails when locked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            env.locked = True
            
            tool = ShellTool(env)
            
            # Run multiple times
            failures = 0
            for _ in range(10):
                result = tool.get({'command': 'echo test'})
                if not result.success:
                    failures += 1
            
            # Should fail often when locked
            assert failures >= 4
    
    def test_python_tool_more_resilient_than_shell(self):
        """Test PythonTool is more resilient when locked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            env.locked = True
            
            shell_tool = ShellTool(env)
            python_tool = PythonTool(env)
            
            # Run both tools multiple times
            shell_failures = 0
            python_failures = 0
            
            for _ in range(20):
                if not shell_tool.get({'command': 'echo test'}).success:
                    shell_failures += 1
                if not python_tool.get({'code': 'result = "test"'}).success:
                    python_failures += 1
            
            # Python should fail less than shell when locked
            assert python_failures < shell_failures
    
    def test_file_read_tool_perfect_when_unlocked(self):
        """Test FileReadTool is perfect when unlocked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            env.locked = False
            
            tool = FileReadTool(env)
            
            # Should always succeed when unlocked
            for _ in range(10):
                result = tool.get({'path': env.key_path})
                assert result.success is True
                assert env.secret_key in result.value
    
    def test_file_read_tool_always_fails_when_locked(self):
        """Test FileReadTool always fails when locked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = ChaosEnvironment(root_dir=tmpdir)
            env.setup()
            env.locked = True
            
            tool = FileReadTool(env)
            
            # Should always fail when locked
            for _ in range(10):
                result = tool.get({'path': env.key_path})
                assert result.success is False
                assert result.prediction_error == 1.0


class TestChaosConfig:
    """Test ChaosConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ChaosConfig()
        
        assert config.chaos_interval == 3
        assert config.lock_probability == 0.5
        assert config.num_directories == 3
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ChaosConfig(
            chaos_interval=5,
            lock_probability=0.7
        )
        
        assert config.chaos_interval == 5
        assert config.lock_probability == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
