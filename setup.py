"""Setup script for lrs-agents."""

from setuptools import setup, find_packages

setup(
    name="lrs-agents",
    version="0.2.0",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    package_data={
        "lrs": ["py.typed"],
    },
    python_requires=">=3.9",
)
