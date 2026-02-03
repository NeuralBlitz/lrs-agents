# AGENTS.md

This file contains guidelines for agentic coding agents working on the LRS-Agents repository.

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=lrs

# Run single test file
pytest tests/test_precision.py -v

# Run specific test function
pytest tests/test_precision.py::TestPrecisionParameters::test_initialization -v

# Run tests by marker
pytest tests/ -m "unit" -v          # Unit tests only
pytest tests/ -m "integration" -v    # Integration tests only
pytest tests/ -m "not slow" -v      # Exclude slow tests

# Run tests with coverage reports
pytest tests/ --cov=lrs --cov-report=term-missing --cov-report=html
```

### Code Quality
```bash
# Format code
black lrs tests

# Lint code (auto-fixes issues)
ruff check lrs tests --fix

# Type checking
mypy lrs --ignore-missing-imports --strict

# Security checking
bandit -r lrs -c pyproject.toml

# Run all pre-commit hooks
pre-commit run --all-files
```

### Installation
```bash
# Development installation
pip install -e ".[dev,test]"

# Full installation with all optional dependencies
pip install -e ".[all]"
```

## Code Style Guidelines

### Import Style
- Use `isort` formatting (handled automatically by black)
- Group imports: standard library, third-party, local application
- Use absolute imports for local modules: `from lrs.core.precision import PrecisionParameters`
- Avoid wildcard imports (`from module import *`)

### Formatting
- Line length: 100 characters (configured in black and ruff)
- Use black for formatting
- Use ruff for linting and auto-fixing
- Indentation: 4 spaces (no tabs)

### Type Hints
- Use type hints for all function signatures and class attributes
- Use `Optional[T]` for nullable types
- Use `Union[T, U]` or `T | U` (Python 3.10+) for unions
- Use `Dict[K, V]`, `List[T]`, etc. from typing module
- Use `@dataclass` for data containers with type hints

### Naming Conventions
- Classes: PascalCase (e.g., `PrecisionParameters`, `ExecutionResult`)
- Functions/variables: snake_case (e.g., `calculate_free_energy`, `prediction_error`)
- Private members: prefix with underscore (e.g., `_update_precision`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_LEARNING_RATE`)
- File names: snake_case (e.g., `precision.py`, `free_energy.py`)

### Error Handling
- Use specific exception types, avoid bare `except:`
- Wrap external dependencies in try/catch blocks
- Use `ExecutionResult` for tool execution outcomes
- Log errors with context when appropriate
- Raise meaningful exceptions with descriptive messages

### Documentation
- Use docstrings for all public classes, functions, and methods
- Follow Google-style or NumPy-style docstring format
- Include examples in docstrings for complex functionality
- Use inline comments sparingly, only for non-obvious logic
- Write descriptive commit messages following conventional format

### Class Design
- Use `@dataclass` for simple data containers
- Use composition over inheritance where possible
- Implement `__repr__` methods for debugging complex objects
- Use abstract base classes (`ABC`) for interfaces
- Keep classes focused and single-responsibility

### Testing Patterns
- Test files: `test_*.py` in `tests/` directory
- Test classes: `TestClassName`
- Test functions: `test_function_name`
- Use descriptive test names that explain what is being tested
- Use pytest fixtures for common setup
- Mock external dependencies using `unittest.mock` or `pytest-mock`
- Aim for 95%+ code coverage

### Performance Guidelines
- Use numpy for numerical operations
- Avoid unnecessary loops in favor of vectorized operations
- Profile before optimizing
- Consider caching for expensive computations
- Use appropriate data structures for the use case

### Mathematical Code
- Follow the notation used in the Active Inference literature
- Use descriptive variable names (e.g., `precision_gamma` not just `g`)
- Include mathematical formulas in docstrings when relevant
- Validate numerical inputs (e.g., probabilities in [0,1])
- Handle edge cases (zero division, negative inputs)

### Integration Patterns
- Use `ToolLens` for tool abstractions
- Implement proper error propagation in tool chains
- Use the registry pattern for tool management
- Follow the precision tracking patterns for learning
- Use the free energy calculation patterns for decision making

### Git Workflow
- Create feature branches from `develop`
- Write clear, conventional commit messages
- Include tests with all new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting PRs

### Code Review Checklist
- [ ] Code follows style guidelines (black/ruff pass)
- [ ] All type hints are correct and mypy passes
- [ ] Tests cover new/modified functionality
- [ ] Documentation is updated
- [ ] No security vulnerabilities (bandit passes)
- [ ] Performance implications are considered
- [ ] Error handling is robust
- [ ] Mathematical formulations are correct

## Repository Structure

```
lrs/
├── core/           # Core Active Inference implementation
│   ├── precision.py    # Precision tracking (Beta distributions)
│   ├── free_energy.py  # Free energy calculations
│   ├── lens.py         # Tool abstraction and composition
│   └── registry.py     # Tool management
├── integration/    # Framework integrations
├── monitoring/     # Dashboard and logging
├── multi_agent/    # Multi-agent coordination
└── benchmarks/     # Benchmark implementations

tests/              # Test suite (95%+ coverage required)
docs/               # Documentation source
examples/           # Working examples
```

## Key Concepts to Understand

1. **Active Inference**: Agents minimize prediction error through action selection
2. **Precision Tracking**: Confidence in predictions using Beta distributions
3. **Free Energy**: G = Epistemic Value - Pragmatic Value for policy selection
4. **Tool Lenses**: Bidirectional tool abstractions with composition
5. **Hierarchical Precision**: Multi-level confidence tracking

When working on this codebase, maintain the mathematical rigor and theoretical foundation while ensuring practical usability.