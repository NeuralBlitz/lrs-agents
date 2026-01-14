# Contributing to LRS-Agents

Thank you for your interest in contributing to LRS-Agents! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/NeuralBlitz/lrs-agents/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Minimal code example

### Suggesting Features

1. Check [existing feature requests](https://github.com/NeuralBlitz/lrs-agents/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
2. Create new issue with:
   - Clear use case
   - Proposed API/interface
   - Why this fits LRS-Agents' scope

### Pull Requests

#### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/lrs-agents
cd lrs-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
