#!/bin/bash
# Run script for fine-tuning C. Pete Connor style model

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Setup virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install or update requirements
echo "Installing dependencies..."
pip install -U pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/training
mkdir -p outputs/finetune

# Setup W&B for training monitoring
echo "Setting up W&B for training monitoring..."
python setup_wandb_training.py

# Prepare training data
echo "Preparing training data from writing style..."
python prepare_training_data.py

# Start fine-tuning
echo "Starting model fine-tuning process..."
echo "This may take a while depending on your hardware..."
python finetune_model.py

# Deactivate virtual environment
deactivate

echo ""
echo "==================================================="
echo "Fine-tuning process complete!"
echo "Check the outputs/finetune directory for the trained model."
echo "Check W&B dashboard for training metrics and progress."
echo "==================================================="
