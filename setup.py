"""Setup script for lrs-agents."""

from setuptools import setup

# Configuration is in pyproject.toml
setup(
    entry_points={
        'console_scripts': [
            'lrs=lrs.cli:main',
            'lrs-agents=lrs.cli:main',
        ],
    },
)
