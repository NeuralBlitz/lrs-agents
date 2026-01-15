# Contributing to LRS-Agents

Thank you for your interest in contributing to LRS-Agents! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/NeuralBlitz/lrs-agents/issues) to avoid duplicates
2. Use the bug report template
3. Include minimal reproducible examples
4. Provide system information (OS, Python version, package versions)

### Suggesting Features

We welcome feature suggestions! Please:
1. Check [existing discussions](https://github.com/NeuralBlitz/lrs-agents/discussions)
2. Explain the use case and benefits
3. Consider implementation complexity
4. Be open to discussion and iteration

### Pull Requests

1. **Fork the repository** and create a feature branch from `develop`
2. **Install development dependencies**: `pip install -e ".[dev,test]"`
3. **Make your changes** following our coding standards
4. **Write or update tests** to maintain 95%+ coverage
5. **Run the test suite**: `pytest tests/ -v`
6. **Run linting**: `ruff check lrs tests && black lrs tests`
7. **Update documentation** if needed
8. **Submit the PR** with a clear description

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/lrs-agents.git
cd lrs-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lrs --cov-report=html

# Run specific test file
pytest tests/test_precision.py -v

# Run tests by marker
pytest -m unit  # or -m integration
```

### Code Quality

```bash
# Format code
black lrs tests

# Lint code
ruff check lrs tests

# Type check
mypy lrs

# Run all checks
pre-commit run --all-files
```

## Coding Standards

### Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://github.com/psf/black) for formatting (line length 100)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Use type hints for all function signatures

### Example

```python
from typing import Optional
from pydantic import BaseModel


class PrecisionParameters(BaseModel):
    """Tracks confidence in predictions using Beta distribution.
    
    Args:
        alpha: Success parameter (default: 1.0)
        beta: Failure parameter (default: 1.0)
        learning_rate_gain: Rate for increasing precision (default: 0.1)
        learning_rate_loss: Rate for decreasing precision (default: 0.2)
    """
    
    alpha: float = 1.0
    beta: float = 1.0
    learning_rate_gain: float = 0.1
    learning_rate_loss: float = 0.2
    
    def update(self, prediction_error: float) -> None:
        """Update precision based on prediction error.
        
        Args:
            prediction_error: Error magnitude in [0, 1]
        
        Raises:
            ValueError: If prediction_error not in valid range
        """
        if not 0 <= prediction_error <= 1:
            raise ValueError(f"Prediction error must be in [0,1], got {prediction_error}")
        
        delta = 1 - prediction_error
        self.alpha += self.learning_rate_gain * delta
        self.beta += self.learning_rate_loss * prediction_error
```

### Testing Standards

- Aim for 95%+ code coverage
- Write unit tests for all new functions
- Write integration tests for complex workflows
- Use descriptive test names: `test_precision_decreases_on_high_error`
- Use fixtures for common setup
- Mock external dependencies (LLMs, APIs)

```python
import pytest
from lrs.core.precision import PrecisionParameters


def test_precision_increases_on_success():
    """Precision should increase when prediction error is low."""
    precision = PrecisionParameters()
    initial = precision.value
    
    precision.update(prediction_error=0.1)
    
    assert precision.value > initial


def test_precision_decreases_on_failure():
    """Precision should decrease when prediction error is high."""
    precision = PrecisionParameters()
    initial = precision.value
    
    precision.update(prediction_error=0.9)
    
    assert precision.value < initial


@pytest.mark.parametrize("error,expected_direction", [
    (0.0, "increase"),
    (0.5, "stable"),
    (1.0, "decrease"),
])
def test_precision_update_direction(error, expected_direction):
    """Test precision update direction for various errors."""
    precision = PrecisionParameters()
    initial = precision.value
    
    precision.update(prediction_error=error)
    
    if expected_direction == "increase":
        assert precision.value > initial
    elif expected_direction == "decrease":
        assert precision.value < initial
    else:
        assert abs(precision.value - initial) < 0.01
```

### Documentation Standards

- Use NumPy-style docstrings
- Document all public APIs
- Include examples in docstrings
- Update relevant docs in `docs/source/`

## Project Structure

```
lrs-agents/
├── lrs/                        # Source code
│   ├── core/                   # Core components
│   │   ├── precision.py        # Precision tracking
│   │   ├── lens.py             # Tool lenses
│   │   ├── free_energy.py      # EFE calculation
│   │   └── registry.py         # Tool registry
│   ├── integration/            # Framework adapters
│   │   ├── langchain_adapter.py
│   │   ├── openai_assistants.py
│   │   └── autogpt_adapter.py
│   ├── monitoring/             # Monitoring tools
│   │   ├── dashboard.py
│   │   ├── tracker.py
│   │   └── structured_logging.py
│   └── agents/                 # Agent implementations
│       └── lrs_agent.py
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Shared fixtures
├── docs/                       # Documentation
│   └── source/
│       ├── getting_started/
│       ├── guides/
│       └── theory/
├── examples/                   # Usage examples
├── benchmarks/                 # Performance benchmarks
└── docker/                     # Docker configs
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch: `git checkout -b release/v0.2.0`
4. Run full test suite
5. Build docs and check for errors
6. Create PR to `main`
7. After merge, tag release: `git tag v0.2.0`
8. Push tag: `git push origin v0.2.0`
9. GitHub Actions will automatically publish to PyPI

## Getting Help

- **Documentation**: [lrs-agents.readthedocs.io](https://lrs-agents.readthedocs.io)
- **Discussions**: [GitHub Discussions](https://github.com/NeuralBlitz/lrs-agents/discussions)
- **Discord**: [Join our community](https://discord.gg/lrs-agents)
- **Email**: contact@lrs-agents.dev

## Recognition

Contributors are recognized in:
- `README.md` contributors section
- Release notes
- `AUTHORS.md` file

Thank you for contributing to LRS-Agents! 🎉