#!/bin/bash

# Healthcare Performance Enhancement System Launcher
# This script activates the virtual environment and runs the enhancement system

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages if not already installed
pip install -q pandas matplotlib seaborn rich

# Run the enhancement system
python run_enhancements.py

# Keep terminal open until user presses a key
echo ""
echo "Press any key to close this window..."
read -n 1
