#!/bin/bash

# Multi-Platform Content Generator Desktop Launcher
# This script launches the app from the desktop, ensuring all dependencies are available

# Define colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}   Multi-Platform Content Generator Launcher${NC}"
echo -e "${BLUE}======================================================${NC}"

# Set the project directory path (hardcoded for reliability)
PROJECT_DIR="/Users/cpconnor/CascadeProjects/multi-platform-content-generator"

# Change to the project directory
echo -e "${YELLOW}Changing to project directory...${NC}"
cd "$PROJECT_DIR" || {
    echo -e "${RED}Error: Could not change to project directory at $PROJECT_DIR${NC}"
    echo "Please check if the path is correct."
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate || {
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
}

# Ensure PyQt6 is installed
echo -e "${YELLOW}Checking PyQt6 installation...${NC}"
pip show PyQt6 > /dev/null 2>&1 || {
    echo -e "${YELLOW}Installing PyQt6...${NC}"
    pip install PyQt6
}

# Display launch message
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}Launching Multi-Platform Content Generator...${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "${YELLOW}The application will start shortly.${NC}"
echo -e "${YELLOW}Press Ctrl+C in this terminal to stop the application.${NC}"
echo

# Launch the application
python app.py

# Deactivate virtual environment when done
deactivate

echo -e "${GREEN}Application closed.${NC}"
# Keep terminal open until user presses a key
read -n 1 -s -r -p "Press any key to exit..."