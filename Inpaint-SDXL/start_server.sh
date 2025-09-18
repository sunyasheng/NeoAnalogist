#!/bin/bash

# SDXL Inpainting API Server Startup Script

echo "Starting SDXL Inpainting API Server..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if requirements are installed
if ! python -c "import torch, diffusers, fastapi" &> /dev/null; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Set default parameters
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8402}

echo "Server will start on $HOST:$PORT"
echo "Make sure you have CUDA available for best performance"
echo ""

# Start the server
python api.py --host $HOST --port $PORT
