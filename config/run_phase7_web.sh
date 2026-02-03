#!/bin/bash

# Phase 7: Autonomous Code Generation Web Interface Runner
# ========================================================
# Run script for the AI-powered application creation web interface

echo "🚀 Starting Phase 7: Autonomous Code Generation Web Interface"
echo "🧠 OpenCode ↔ LRS-Agents Cognitive AI Hub"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "phase7_web_interface.py" ]; then
    echo "❌ Error: phase7_web_interface.py not found. Please run from the project root."
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install -q flask

# Start the web interface
echo "🌐 Starting Autonomous Code Generation Web Interface..."
echo "📱 Open http://localhost:5001 in your browser"
echo "🧠 Describe applications in natural language and get complete code!"
echo ""
echo "Example descriptions you can try:"
echo "• Create a REST API for managing tasks with categories and priorities"
echo "• Build a web dashboard for visualizing sales data with charts"
echo "• Make a CLI tool for processing images and applying filters"
echo "• Develop a user authentication system with JWT tokens"
echo ""

python3 phase7_web_interface.py