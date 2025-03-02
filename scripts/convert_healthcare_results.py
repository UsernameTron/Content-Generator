#!/usr/bin/env python3
"""
Convert healthcare evaluation results to the format expected by the visualization script.
This script takes healthcare evaluation results and reformats them for visualization.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-results-converter")

def convert_healthcare_results(input_path, output_path=None, healthcare_metrics_path=None):
    """
    Convert healthcare evaluation results to visualization format.
    
    Args:
        input_path: Path to healthcare evaluation results JSON file
        output_path: Path to save converted results (default: input path with _viz suffix)
        healthcare_metrics_path: Optional path to healthcare metrics from contradiction integrator
    
    Returns:
        Path to converted results file
    """
    # Set default output path if not provided
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = Path(input_path).parent / f"{input_stem}_viz.json"
    
    # Load healthcare evaluation results
    with open(input_path, 'r') as f:
        healthcare_results = json.load(f)
        
    # Load healthcare metrics if provided
    healthcare_metrics = None
    if healthcare_metrics_path and Path(healthcare_metrics_path).exists():
        logger.info(f"Loading healthcare metrics from {healthcare_metrics_path}")
        try:
            with open(healthcare_metrics_path, 'r') as f:
                healthcare_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading healthcare metrics: {e}")
    
    # Create visualization data structure
    viz_data = {
        "healthcare": healthcare_results,
        "metadata": healthcare_results.get("metadata", {}),
        "summary": {
            "title": "Healthcare Cross-Reference Evaluation",
            "description": "Evaluation of model performance on healthcare cross-referencing tasks",
            "date": healthcare_results.get("metadata", {}).get("evaluation_timestamp", 
                                                            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    }
    
    # Add performance summary data
    performance_summary = {}
    
    # Add contradiction detection metrics
    if "contradiction_detection" in healthcare_results and healthcare_results["contradiction_detection"]:
        contradiction_metrics = healthcare_results["contradiction_detection"]
        accuracy = contradiction_metrics.get("accuracy", 0.0)
        performance_summary["contradiction_detection"] = {
            "accuracy": accuracy,
            "target": 0.75,
            "gap": accuracy - 0.75
        }
    
    # Add evidence ranking metrics
    if "evidence_ranking" in healthcare_results and healthcare_results["evidence_ranking"]:
        evidence_metrics = healthcare_results["evidence_ranking"]
        accuracy = evidence_metrics.get("accuracy", 0.0)
        performance_summary["evidence_ranking"] = {
            "accuracy": accuracy,
            "target": 0.80,
            "gap": accuracy - 0.80
        }
    
    # Calculate overall score (normalized to 0-5 scale)
    if performance_summary:
        accuracies = [metrics.get("accuracy", 0) for metrics in performance_summary.values()]
        overall_score = np.mean(accuracies) * 5  # Scale to 0-5
        
        # Add healthcare domain data
        viz_data["healthcare"]["overall_score"] = overall_score
        viz_data["healthcare"]["benchmark"] = 3.5  # Target benchmark
    
    # Add domain-specific performance if available
    domains = [k for k in healthcare_results.keys() if k not in ["metadata", "summary", "overall"]]
    for domain in domains:
        if domain in healthcare_results and healthcare_results[domain]:
            viz_data[domain] = healthcare_results[domain]
    
    # Integrate healthcare contradiction metrics if available
    if healthcare_metrics:
        # Add contradiction analysis
        if "contradiction_analysis" in healthcare_metrics:
            viz_data["contradiction_analysis"] = healthcare_metrics["contradiction_analysis"]
            
        # Add temporal metrics
        if "temporal_metrics" in healthcare_metrics:
            viz_data["temporal_metrics"] = healthcare_metrics["temporal_metrics"]
            
        # Add terminology metrics
        if "terminology_metrics" in healthcare_metrics:
            viz_data["terminology_metrics"] = healthcare_metrics["terminology_metrics"]
            
        # Add credibility metrics
        if "credibility_metrics" in healthcare_metrics:
            viz_data["credibility_metrics"] = healthcare_metrics["credibility_metrics"]
            
        # Add contradiction performance if available
        if "contradiction_performance" in healthcare_metrics:
            viz_data["contradiction_performance"] = healthcare_metrics["contradiction_performance"]
            
        # Add timestamp from healthcare metrics
        if "timestamp" in healthcare_metrics:
            viz_data["metadata"]["healthcare_metrics_timestamp"] = healthcare_metrics["timestamp"]
    
    # Save visualization data
    with open(output_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    logger.info(f"Converted healthcare results saved to {output_path}")
    return output_path

def main():
    """Main function to run the healthcare results conversion."""
    parser = argparse.ArgumentParser(description="Convert healthcare evaluation results for visualization")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to healthcare evaluation results JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save converted results (default: input path with _viz suffix)")
    parser.add_argument("--healthcare-metrics", type=str, default=None,
                        help="Optional path to healthcare metrics from contradiction integrator")
    
    args = parser.parse_args()
    
    try:
        output_path = convert_healthcare_results(args.input, args.output, args.healthcare_metrics)
        print(f"Successfully converted results: {output_path}")
    except Exception as e:
        print(f"Error converting healthcare results: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
