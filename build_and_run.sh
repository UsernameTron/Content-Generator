#!/bin/bash

# Build and run script for Multi-Platform Content Generator
# This script builds and runs the containerized application

# Define colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}   Multi-Platform Content Generator Deployment${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is required but not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is required but not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating a default one...${NC}"
    echo "WANDB_API_KEY=" > .env
    echo -e "${YELLOW}Please edit .env to add your Weights & Biases API key if needed.${NC}"
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed. See error messages above.${NC}"
    exit 1
fi

# Run the container
echo -e "${YELLOW}Starting the application...${NC}"
docker-compose up -d

# Check if container is running
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Application is now running!${NC}"
    echo -e "${BLUE}======================================================${NC}"
    echo -e "Access the application through the appropriate client."
    echo -e "To stop the application, run: docker-compose down"
    echo -e "${BLUE}======================================================${NC}"
else
    echo -e "${RED}Failed to start the application. See error messages above.${NC}"
    exit 1
fi

exit 0