#!/bin/bash

echo "🛠️  Setting up OpenCode ↔ LRS-Agents Cognitive AI Hub..."
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the project root directory."
    exit 1
fi

echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the server, run:"
echo "   source venv/bin/activate"
echo "   python3 server.py"
echo ""
echo "🌐 Then visit: https://your-replit-url"
echo "🧠 Click '🚀 Cognitive Demo' to experience AI code analysis!"