#!/bin/bash

# Set script directory as working directory
cd "$(dirname "$0")"

# Display welcome message
echo "====================================================="
echo "   C. Pete Connor Model Evaluation Visualizer        "
echo "====================================================="
echo ""

# Create visualization environment if needed
if [ ! -d "viz_env" ]; then
    echo "Creating visualization environment..."
    python3 -m venv viz_env
    source viz_env/bin/activate
    pip install -U pip
    echo "Installing dependencies..."
    pip install numpy pandas matplotlib seaborn
else
    source viz_env/bin/activate
fi

# Find available result files
echo "Available evaluation result files:"
find . -name "*.json" -not -path "*/\.*" | grep -v "package.json" | nl

# Ask for results file
read -p "Enter the number of the results file to visualize (or full path): " RESULTS_CHOICE

# Handle the case where user enters a number
if [[ "$RESULTS_CHOICE" =~ ^[0-9]+$ ]]; then
    RESULTS_FILE=$(find . -name "*.json" -not -path "*/\.*" | grep -v "package.json" | sed -n "${RESULTS_CHOICE}p")
    if [ -z "$RESULTS_FILE" ]; then
        echo "Invalid selection. Please try again."
        read -n 1 -p "Press any key to exit..."
        exit 1
    fi
else
    RESULTS_FILE=$RESULTS_CHOICE
fi

# Verify the file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Results file not found: $RESULTS_FILE"
    read -n 1 -p "Press any key to exit..."
    exit 1
fi

echo ""
echo "Selected results file: $RESULTS_FILE"
echo ""

# Create output directory
read -p "Output directory for visualizations (default: visualizations): " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-visualizations}

# Generate visualizations
echo ""
echo "Generating visualizations..."
echo "============================"
viz_env/bin/python scripts/visualize_metrics.py --results "$RESULTS_FILE" --output "$OUTPUT_DIR" --html

# Open HTML report if it exists
if [ -f "$OUTPUT_DIR/evaluation_report.html" ]; then
    echo "Opening evaluation report..."
    open "$OUTPUT_DIR/evaluation_report.html"
else
    echo "Visualization generation failed."
    read -n 1 -p "Press any key to exit..."
    exit 1
fi

echo ""
echo "Visualization complete! Press any key to close..."
read -n 1
