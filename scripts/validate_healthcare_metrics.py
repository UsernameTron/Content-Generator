#!/usr/bin/env python3
"""
Healthcare Metrics Validation Script

This script provides a CLI interface to validate healthcare metrics using the HealthcareMetricsValidator.
It allows validating current metrics against baseline and target values, and generates a validation report.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enhancement_module.healthcare_metrics_validator import HealthcareMetricsValidator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Validate Healthcare Metrics')
    
    parser.add_argument('--current', '-c', required=True,
                        help='Path to current metrics JSON file')
    
    parser.add_argument('--baseline', '-b',
                        help='Path to baseline metrics JSON file (optional)')
    
    parser.add_argument('--target', '-t',
                        help='Path to target metrics JSON file (optional)')
    
    parser.add_argument('--config', 
                        help='Path to configuration file (optional)')
    
    parser.add_argument('--output', '-o',
                        help='Path to output validation report (optional, defaults to stdout)')
    
    parser.add_argument('--json-output', '-j',
                        help='Path to output validation results as JSON (optional)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def load_metrics_file(file_path: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing metrics data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics from {file_path}: {e}")
        sys.exit(1)


def write_output(content: str, output_path: Optional[str] = None) -> None:
    """
    Write content to output file or stdout.
    
    Args:
        content: Content to write
        output_path: Optional path to output file
    """
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write(content)
            logger.info(f"Report written to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write to {output_path}: {e}")
            print(content)
    else:
        print(content)


def main() -> None:
    """
    Main entry point for the validation script.
    """
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load metrics data
    current_metrics = load_metrics_file(args.current)
    baseline_metrics = load_metrics_file(args.baseline) if args.baseline else None
    target_metrics = load_metrics_file(args.target) if args.target else None
    
    # Create validator
    validator = HealthcareMetricsValidator(args.config)
    
    # Validate metrics
    logger.info("Validating healthcare metrics...")
    validation_results = validator.validate_metrics(
        current_metrics, baseline_metrics, target_metrics
    )
    
    # Generate report
    report = validator.generate_validation_report(validation_results)
    
    # Write report
    write_output(report, args.output)
    
    # Write JSON results if requested
    if args.json_output:
        try:
            with open(args.json_output, 'w') as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"JSON results written to {args.json_output}")
        except Exception as e:
            logger.error(f"Failed to write JSON results to {args.json_output}: {e}")
    
    # Exit with appropriate status code
    sys.exit(0 if validation_results['overall']['valid'] else 1)


if __name__ == "__main__":
    main()
