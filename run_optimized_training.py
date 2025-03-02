#!/usr/bin/env python3
"""
Optimized Training Runner for Apple Silicon

This script sets specific environment variables and configurations
to optimize training on Apple Silicon devices and prevent MPS memory errors.
"""
import os
import sys
import subprocess
from pathlib import Path

# Set environment variables for MPS (Metal Performance Shaders)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit for memory allocations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for unsupported operations
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")  # Ensure proper caching

# Set extremely conservative batch sizes via environment variables
os.environ["TRAINING_BATCH_SIZE"] = "1"  # Smallest possible batch size
os.environ["EVAL_BATCH_SIZE"] = "1"  # Smallest possible evaluation batch size
os.environ["GRADIENT_ACCUMULATION_STEPS"] = "16"  # Increase gradient accumulation instead

def main():
    print("=" * 80)
    print("Starting optimized training for Apple Silicon")
    print("Using high watermark ratio: 0.0 (disables upper limit for memory)")
    print("Enabled MPS fallback for unsupported operations")
    print("Using minimal batch sizes and higher gradient accumulation")
    print("=" * 80)
    
    # Execute the training script
    cmd = [sys.executable, "run_overnight_training.py"]
    
    # Add any command-line arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the training script with all our environment variables set
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
