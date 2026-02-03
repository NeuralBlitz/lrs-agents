#!/usr/bin/env python3
"""Integration script for LRS-Agents with OpenCode."""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import lrs

    print(f"LRS-Agents version: {lrs.__version__}")
    print("LRS-Agents is available and ready to use!")

    # Example usage
    if hasattr(lrs, "create_lrs_agent"):
        print("create_lrs_agent function is available")
    else:
        print("create_lrs_agent function not found (may need additional dependencies)")

except ImportError as e:
    print(f"Error importing LRS-Agents: {e}")
    sys.exit(1)
