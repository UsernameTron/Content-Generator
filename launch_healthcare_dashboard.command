#!/bin/bash
# Launch script for Healthcare Learning System Dashboard

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Ensure required packages are installed
echo "Checking dependencies..."
python3 -c "import matplotlib, rich" 2>/dev/null || {
    echo "Installing required packages..."
    pip3 install matplotlib rich
}

# Initialize data structure if needed
echo "Initializing data structure..."
python3 scripts/initialize_dashboard_data.py

# Launch the dashboard
echo "Launching Healthcare Learning System Dashboard..."
python3 scripts/interactive_learning_dashboard.py

# Handle exit
echo "Dashboard closed. Press any key to exit."
read -n 1
