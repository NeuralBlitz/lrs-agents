"""Test that the package can be imported."""

def test_import_lrs():
    """Test that lrs package can be imported."""
    import lrs
    assert lrs.__version__ == "0.2.0"


def test_import_core():
    """Test that core modules can be imported."""
    from lrs.core import precision
    from lrs.core import lens
    from lrs.core import registry
    from lrs.core import free_energy
    
    assert precision is not None
    assert lens is not None
    assert registry is not None
    assert free_energy is not None


def test_create_lrs_agent():
    """Test that create_lrs_agent is available."""
    from lrs import create_lrs_agent
    assert create_lrs_agent is not None
