#!/bin/bash
# Desktop launcher for C. Pete Connor Content Generator (model-based version)

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  C. Pete Connor Content Generator - Model-based Version  "
echo "=========================================================="

# Setup virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install or update requirements
echo "Checking dependencies..."
pip install -U pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/training
mkdir -p outputs/finetune

# Setup data files if needed
echo "Setting up data files..."
python setup_data.py

# Check for W&B API key
if [ ! -f .env ] || ! grep -q "WANDB_API_KEY" .env; then
    echo "Setting up W&B monitoring..."
    python setup_wandb_training.py
fi

# Check if model exists
if [ ! -d "outputs/finetune/final" ]; then
    echo ""
    echo "========================================================"
    echo "  Notice: Fine-tuned model not found"
    echo "========================================================"
    echo "The fine-tuned model is not available. You have two options:"
    echo ""
    echo "1. Run the app with template-based fallback generation"
    echo "2. Run the fine-tuning process first (may take hours)"
    echo ""
    echo "For option 2, close this app and run:"
    echo "./run_finetune.command"
    echo ""
    read -p "Press Enter to continue with template-based fallback..."
fi

# Start the Streamlit app
echo "Starting C. Pete Connor Content Generator..."
cd src
streamlit run app_model.py

# Deactivate virtual environment when done
deactivate
