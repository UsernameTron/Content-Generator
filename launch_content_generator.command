#\!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Launch the application
python app.py
