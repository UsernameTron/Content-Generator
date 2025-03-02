#!/usr/bin/env python3
"""
Setup script for CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric
"""

import os
import sys
import subprocess
import shutil
import nltk
from pathlib import Path


def main():
    """
    Run the setup process for CANDOR application
    """
    print("\n" + "="*80)
    print(" CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric")
    print(" Setup Script")
    print("="*80 + "\n")
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print("⚠️  Warning: Python 3.9+ is recommended. You have Python {}.{}.{}".format(
            py_version.major, py_version.minor, py_version.micro))
    else:
        print("✅ Python version check passed: {}.{}.{}".format(
            py_version.major, py_version.minor, py_version.micro))
    
    # Create directories if they don't exist
    print("\nEnsuring directory structure...")
    directories = [
        "src/models",
        "src/processors",
        "src/adapters",
        "data",
        "outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Verified directory: {directory}")
    
    # Install Python dependencies
    print("\nChecking Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Installed dependencies from requirements.txt")
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Failed to install some dependencies. Check requirements.txt")
    
    # Download NLTK data
    print("\nDownloading NLTK resources...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("✅ Downloaded NLTK resources")
    except Exception as e:
        print(f"⚠️  Warning: Failed to download some NLTK resources: {str(e)}")
    
    # Create desktop launcher if it doesn't exist
    desktop_launcher = os.path.expanduser("~/Desktop/CANDOR_Launcher.command")
    
    if not os.path.exists(desktop_launcher):
        print("\nCreating desktop launcher...")
        
        launcher_script = f"""#!/bin/bash

# CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric
# Desktop Launcher

# Print welcome message
echo "========================================================"
echo "  CANDOR: Satirical Cross-Platform Content Generator"
echo "========================================================"
echo "Initializing..."

# Navigate to the project directory
cd {os.path.dirname(os.path.abspath(__file__))}

# Check if Python virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the application
echo "Starting CANDOR application..."
python app.py

# Keep terminal open after app closes for error inspection
echo "Application closed. Press any key to exit."
read -n 1 -s
"""
        
        with open(desktop_launcher, 'w') as f:
            f.write(launcher_script)
        
        # Make the launcher executable
        os.chmod(desktop_launcher, 0o755)
        print(f"✅ Created desktop launcher: {desktop_launcher}")
    else:
        print(f"✅ Desktop launcher already exists: {desktop_launcher}")
    
    print("\n" + "="*80)
    print(" Setup Complete!")
    print("="*80)
    print("\nLaunch the application using the desktop shortcut or by running:")
    print("python app.py\n")


if __name__ == "__main__":
    main()
