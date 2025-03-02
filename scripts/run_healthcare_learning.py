#!/usr/bin/env python3
"""
Run healthcare contradiction detection continuous learning pipeline.
This script uses the HealthcareContinuousLearning class to improve the model
based on evaluation results.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from rich.logging import RichHandler
from visualize_metrics import HealthcareContinuousLearning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-learning")

def main():
    """Main function to run healthcare continuous learning."""
    parser = argparse.ArgumentParser(description="Run healthcare continuous learning")
    parser.add_argument("--eval-results", 
                       type=str, 
                       default="data/healthcare/evaluation/latest_results.json",
                       help="Path to evaluation results JSON")
    parser.add_argument("--examples", 
                       type=int, 
                       default=20,
                       help="Number of new examples to generate")
    parser.add_argument("--data-dir", 
                       type=str, 
                       default="data/healthcare",
                       help="Path to healthcare data directory")
    args = parser.parse_args()
    
    # Initialize continuous learning
    learner = HealthcareContinuousLearning(
        data_dir=args.data_dir
    )
    
    # Run learning cycle
    logger.info(f"Running continuous learning with {args.eval_results}")
    
    # Check if evaluation results exist
    eval_path = Path(args.eval_results)
    if not eval_path.exists():
        logger.error(f"Evaluation results not found: {args.eval_results}")
        return 1
    
    # Run learning cycle
    result = learner.run_continuous_learning_cycle(
        evaluation_results_path=str(eval_path),
        examples_to_generate=args.examples
    )
    
    # Output summary
    print("\n--- Continuous Learning Cycle Summary ---")
    print(f"Examples Generated: {result['metrics'].get('examples_generated', 0)}")
    print(f"Improvement Areas: {result['metrics'].get('improvement_areas', 0)}")
    print(f"Current Accuracy: {result['metrics'].get('current_accuracy', 0):.2f}")
    
    print("\n--- Steps Status ---")
    for step in result.get("steps", []):
        print(f"{step['name']}: {step['status']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
