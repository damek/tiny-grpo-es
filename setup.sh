#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up tiny-grpo environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv is already installed"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    uv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📚 Installing requirements..."
uv pip install -r requirements.txt

# Install flash-attn separately with no-build-isolation flag
echo "⚡ Installing flash-attn..."
uv pip install flash-attn --no-build-isolation

echo "login to hugging face"
huggingface-cli login

echo "✨ Setup complete! Activate the environment with: source .venv/bin/activate"
echo "🏃 Then run: uv run train.py" 
