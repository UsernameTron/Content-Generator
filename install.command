#!/bin/bash
# Installation script for C. Pete Connor Content Generator

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  C. Pete Connor Content Generator - Installation         "
echo "=========================================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

echo "Python detected: $(python3 --version)"
echo

# Setup virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Update pip and install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -U pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check your internet connection and try again."
    exit 1
fi

# Run setup script
echo "Setting up data files and configurations..."
python setup_data.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to set up data files. Please check the error messages above."
    exit 1
fi

# Create desktop shortcuts
echo "Creating desktop shortcuts..."
DESKTOP_SHORTCUT="/Users/cpconnor/Desktop/PeteConnorContentGenerator.command"

cat > "$DESKTOP_SHORTCUT" << EOL
#!/bin/bash
# Desktop shortcut for C. Pete Connor Content Generator

# Path to the project directory
PROJECT_DIR="$DIR"

# Change to the project directory
cd "\$PROJECT_DIR"

# Execute the launcher
./launch_model_app.command
EOL

chmod +x "$DESKTOP_SHORTCUT"
chmod +x "$DIR/launch_model_app.command"
chmod +x "$DIR/run_finetune.command"

echo
echo "=========================================================="
echo "  Installation Complete!                                 "
echo "=========================================================="
echo
echo "You can now use the Content Generator in two ways:"
echo
echo "1. Launch the app directly:"
echo "   ./launch_model_app.command"
echo
echo "2. Use the desktop shortcut:"
echo "   ~/Desktop/PeteConnorContentGenerator.command"
echo
echo "If you want to fine-tune the model:"
echo "   ./run_finetune.command"
echo
echo "Enjoy generating content in C. Pete Connor's style!"
echo "=========================================================="

# Ask if user wants to launch the app now
read -p "Would you like to launch the app now? (y/n): " choice
case "$choice" in 
  y|Y ) ./launch_model_app.command;;
  * ) echo "You can launch the app later using the desktop shortcut.";;
esac
