#!/usr/bin/env python3
"""
Prepare a small validation dataset from the training data.
"""
import json
import logging
import os
import random
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def prepare_validation_data(input_file, output_file, num_examples=20):
    """
    Prepare a validation dataset from training data by selecting random examples.
    
    Args:
        input_file: Path to the training data file
        output_file: Path to save the validation data
        num_examples: Number of examples to include in the validation set
    """
    logger.info(f"Preparing validation data from {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Training data file does not exist: {input_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read all lines from the input file
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    logger.info(f"Read {len(lines)} examples from training data")
    
    if len(lines) == 0:
        logger.error("No examples found in training data")
        return
    
    # Select random examples for validation
    if num_examples > len(lines):
        num_examples = len(lines)
        logger.warning(f"Requested more examples than available, using all {num_examples} examples")
    
    validation_lines = random.sample(lines, num_examples)
    
    # Ensure each line contains valid JSON
    validated_lines = []
    for line in validation_lines:
        try:
            json.loads(line)
            validated_lines.append(line)
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
    
    logger.info(f"Selected {len(validated_lines)} examples for validation data")
    
    # Write to output file
    with open(output_file, "w") as f:
        f.writelines(validated_lines)
    
    logger.info(f"Validation data saved to {output_file}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Default paths
    input_file = "data/training_data.jsonl"
    output_file = "data/validation_data.jsonl"
    
    prepare_validation_data(input_file, output_file)
    
    logger.info("Done!")
