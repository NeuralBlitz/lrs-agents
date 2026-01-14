#!/bin/bash
# Fix documentation build issues

set -e

echo "🔧 Fixing documentation issues..."

# Create _static directory
mkdir -p source/_static
touch source/_static/.gitkeep
echo "✓ Created _static directory"

# Install package in development mode
cd ..
pip install -e .
echo "✓ Installed LRS-Agents package"

# Install documentation dependencies
cd docs
pip install -r requirements.txt
echo "✓ Installed documentation dependencies"

# Try building
echo "🏗️  Building documentation..."
make clean
make html

echo "✅ Documentation built successfully!"
echo "📖 Open build/html/index.html to view"
