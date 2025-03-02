#!/bin/bash

# Set script directory as working directory
cd "$(dirname "$0")"

# Display welcome message
echo "====================================================="
echo "     C. Pete Connor Model Evaluation Framework       "
echo "====================================================="
echo ""

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip
    echo "Ensuring all required dependencies are installed..."
    pip install torch transformers peft wandb rich psutil numpy pandas matplotlib seaborn
else
    source venv/bin/activate
fi

# Configure memory optimizations for Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Ask whether to validate the framework first
read -p "Would you like to validate the evaluation framework? (y/N): " VALIDATE
if [[ $VALIDATE == "y" || $VALIDATE == "Y" ]]; then
    echo "Validating evaluation framework..."
    python test_evaluation_framework.py
    if [ $? -ne 0 ]; then
        echo "Validation failed. Please fix the issues before running the evaluation."
        read -n 1 -p "Press any key to exit..."
        exit 1
    fi
    echo "Validation successful!"
    echo ""
fi

# Find available adapters
echo "Available adapter paths:"
find . -type d -name "adapter_*" -o -name "*_adapter" | grep -v "__pycache__" | nl

# Ask for adapter path if not provided
read -p "Enter the path to the adapter folder: " ADAPTER_PATH

# Ask for batch size
read -p "Number of questions/scenarios per evaluator (default: 1): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-1}

# Ask for WandB integration
read -p "Enable Weights & Biases integration? (y/N): " USE_WANDB
if [[ $USE_WANDB == "y" || $USE_WANDB == "Y" ]]; then
    WANDB_ARG="--use-wandb"
else
    WANDB_ARG=""
fi

# Ask if the user wants to skip any evaluation types
read -p "Skip cross-reference evaluation? (y/N): " SKIP_CROSS
if [[ $SKIP_CROSS == "y" || $SKIP_CROSS == "Y" ]]; then
    CROSS_ARG="--skip-cross-reference"
else
    CROSS_ARG=""
fi

read -p "Skip counterfactual reasoning evaluation? (y/N): " SKIP_COUNTER
if [[ $SKIP_COUNTER == "y" || $SKIP_COUNTER == "Y" ]]; then
    COUNTER_ARG="--skip-counterfactual"
else
    COUNTER_ARG=""
fi

# Ask for device
read -p "Device to use for evaluation (cpu/mps, default: mps): " DEVICE
DEVICE=${DEVICE:-mps}

# Run the evaluation
echo ""
echo "Starting model evaluation..."
echo "=============================="
python comprehensive_evaluate.py --model_path "$ADAPTER_PATH" --device "$DEVICE" --batch_size "${BATCH_SIZE:-1}" $WANDB_ARG $CROSS_ARG $COUNTER_ARG --save_results "evaluation_results.json"

# Ask if user wants to visualize results
echo ""
read -p "Would you like to visualize the evaluation results? (Y/n): " VISUALIZE
if [[ $VISUALIZE != "n" && $VISUALIZE != "N" ]]; then
    # Create visualization environment if needed
    if [ ! -d "viz_env" ]; then
        echo "Creating visualization environment..."
        python3 -m venv viz_env
        source viz_env/bin/activate
        pip install numpy pandas matplotlib seaborn
    else
        source viz_env/bin/activate
    fi
    
    echo "Generating visualizations and HTML report..."
    viz_env/bin/python scripts/visualize_metrics.py --results evaluation_results.json --html
    
    # Open HTML report
    if [ -f "visualizations/evaluation_report.html" ]; then
        echo "Opening evaluation report..."
        open visualizations/evaluation_report.html
    else
        echo "Visualization generation failed."
    fi
    
    # Switch back to main environment
    source venv/bin/activate
fi

echo ""
echo "Evaluation complete! Press any key to close..."
read -n 1
