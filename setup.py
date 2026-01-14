"""
Setup file for backward compatibility.

Modern build uses pyproject.toml, but this file is kept for
compatibility with tools that don't support PEP 517.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
