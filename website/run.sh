#!/bin/bash
# Quick start script for TuiML Hub platform (using uv)

echo "======================================"
echo "TuiML Hub Platform Launcher"
echo "======================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies (creates venv and installs if needed)
if [ ! -d ".venv" ]; then
    echo "Syncing dependencies with uv..."
    uv sync
    echo "✓ Dependencies synced"
    echo ""
fi

# Check if database exists
if [ ! -f "tuiml_hub.db" ]; then
    echo "Database not found. Initializing..."
    uv run init_db.py
    echo ""
fi

# Start the server
echo "Starting TuiML Hub server..."
echo "Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
