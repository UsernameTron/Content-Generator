#!/bin/bash

# C. Pete Connor Model - Overnight Training Launcher
# Optimized for Apple Silicon M4 Pro

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set memory optimization environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Create logs directory if not exists
mkdir -p logs

# Print header
echo "============================================================"
echo "   C. Pete Connor Model - Overnight Training Launcher"
echo "             Optimized for Apple Silicon M4 Pro"
echo "============================================================"
echo ""
echo "Training will be logged to Weights & Biases"
echo "Visit: https://wandb.ai/$(whoami)/pete-connor-cx-ai-expert"
echo ""
echo "Local logs will be saved to: ./logs/"
echo "Model outputs will be saved to: ./outputs/finetune/"
echo ""
echo "The training process has been configured to run overnight"
echo "using memory-optimized settings for Apple Silicon."
echo ""
echo "Press Ctrl+C to stop training at any time."
echo "============================================================"
echo ""

# Check if W&B is initialized
if [ -z "$WANDB_API_KEY" ]; then
    # Load from .env file if it exists
    if [ -f ".env" ]; then
        echo "Loading Weights & Biases API key from .env file..."
        export $(grep -v '^#' .env | xargs)
    fi
    
    # If still not set, prompt the user
    if [ -z "$WANDB_API_KEY" ]; then
        echo "Weights & Biases API key not found!"
        echo "Please enter your W&B API key (or press Enter to continue without W&B):"
        read -r WANDB_API_KEY
        export WANDB_API_KEY
    fi
fi

# Launch the simplified training script
echo "Starting optimized training for Apple Silicon..."
python apple_silicon_training.py

# Keep terminal open after completion
echo ""
echo "Training process has completed."
echo "Press any key to close this window..."
read -n 1
