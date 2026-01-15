"""Setup script for lrs-agents."""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lrs-agents",
    version="0.2.0",
    author="LRS-Agents Contributors",
    description="Resilient AI agents via Active Inference and Predictive Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeuralBlitz/lrs-agents",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    package_data={
        "lrs": ["py.typed"],
    },
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.20",
        "langgraph>=0.0.20",
        "pydantic>=2.5.0",
        "typing-extensions>=4.9.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",  # Required for Beta distribution
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.12.0",
        ],
        "dev": [
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "pre-commit>=3.5.0",
        ],
        "monitoring": [
            "streamlit>=1.28.0",
            "plotly>=5.18.0",
            "matplotlib>=3.7.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
