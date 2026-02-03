#!/bin/bash

echo "🚀 Starting OpenCode ↔ LRS-Agents Cognitive AI Hub..."
echo "🌐 Server will be available at: https://$REPL_SLUG.$REPL_OWNER.repl.co"
echo "🧠 Cognitive Demo: Click '🚀 Cognitive Demo' button"
echo "============================================================"

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -q -r requirements.txt

# Start the server
python3 server.py