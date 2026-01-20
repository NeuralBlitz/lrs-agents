"""Test that the package can be imported."""

def test_import_lrs():
    """Test that lrs package can be imported."""
    import lrs
    assert lrs.__version__ == "0.2.1"


def test_import_core():
    """Test that core modules can be imported without dependencies."""
    from lrs.core import precision
    from lrs.core import lens
    from lrs.core import registry
    from lrs.core import free_energy
    
    assert precision is not None
    assert lens is not None
    assert registry is not None
    assert free_energy is not None


def test_create_lrs_agent_lazy():
    """Test that create_lrs_agent is lazily imported."""
    try:
        from lrs import create_lrs_agent
        assert create_lrs_agent is not None
    except ImportError as e:
        # If langgraph not installed, should get helpful message
        assert "langgraph" in str(e).lower()
