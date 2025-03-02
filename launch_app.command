#!/bin/bash

# Multi-Platform Content Generator Launcher
# This script launches the content generator application with all necessary setups

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Define colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}   Multi-Platform Content Generator Launcher${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "${YELLOW}Starting setup process...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install or update requirements
echo -e "${YELLOW}Installing/updating dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}"

# Run data setup
echo -e "${YELLOW}Setting up data...${NC}"
python setup_data.py
echo -e "${GREEN}Data setup complete.${NC}"

# Synchronize data files
echo -e "${YELLOW}Synchronizing data files...${NC}"
python sync_data.py
echo -e "${GREEN}Data synchronization complete.${NC}"

# Check if .env file exists for W&B configuration
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Weights & Biases not configured. Would you like to set it up now? (y/n)${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python setup_wandb.py
    else
        echo -e "${YELLOW}Skipping W&B setup. You can run setup_wandb.py later if needed.${NC}"
    fi
else
    echo -e "${GREEN}W&B configuration found.${NC}"
fi

# Launch the PyQt6 desktop application
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}Launching Multi-Platform Content Generator...${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "${YELLOW}The desktop application will start shortly.${NC}"
echo -e "${YELLOW}Press Ctrl+C in this terminal to stop the application.${NC}"
echo

python app.py

# Deactivate virtual environment when done
deactivate
