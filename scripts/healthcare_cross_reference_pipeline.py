#!/usr/bin/env python3
"""
Healthcare Cross-Reference Pipeline.
End-to-end pipeline for healthcare cross-reference evaluation and visualization.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from rich.logging import RichHandler
from datetime import datetime

# Import our modules
from healthcare_evaluation import HealthcareEvaluator
from convert_healthcare_results import convert_healthcare_results
from visualize_metrics import MetricsVisualizer
from healthcare_contradiction_integration import HealthcareContradictionIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-pipeline")

def run_pipeline(model_path, adapter_path, data_dir, output_dir, device="mps", 
                 skip_evaluation=False, skip_visualization=False, metrics_history=None, visualization_types=None):
    """
    Run the healthcare cross-reference pipeline.
    
    Args:
        model_path: Path to the base model
        adapter_path: Path to the LoRA adapter (if applicable)
        data_dir: Directory containing evaluation data
        output_dir: Directory to save evaluation results and visualizations
        device: Device to run evaluation on (cpu, cuda, mps)
        skip_evaluation: Skip the evaluation step (use existing results)
        skip_visualization: Skip the visualization step
        metrics_history: Path to healthcare metrics history file for tracking over time
        visualization_types: List of visualization types to generate
        
    Returns:
        Dictionary containing paths to all generated files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup results paths
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Use fixed result path if skipping evaluation
    if skip_evaluation:
        results_path = output_dir / "healthcare_eval_latest.json"
    else:
        results_path = output_dir / f"healthcare_eval_{timestamp}.json"
        
    viz_data_path = output_dir / f"healthcare_eval_{timestamp}_viz.json"
    viz_output_dir = output_dir / f"visualizations_{timestamp}"
    
    paths = {
        "results": str(results_path),
        "viz_data": str(viz_data_path),
        "viz_output": str(viz_output_dir)
    }
    
    # Step 1: Run healthcare evaluation (if not skipped)
    if not skip_evaluation:
        logger.info(f"Running healthcare evaluation with model: {model_path}")
        logger.info(f"Adapter: {adapter_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Data directory: {data_dir}")
        
        # Initialize evaluator
        evaluator = HealthcareEvaluator(model_path, adapter_path, device)
        
        # Run evaluation
        results = evaluator.run_all_evaluations(data_dir, output_dir)
        
        # Update results path to actual path used
        for file in output_dir.glob("healthcare_eval_*.json"):
            if "viz" not in file.name:  # Skip viz files
                results_path = file
                paths["results"] = str(results_path)
                break
    else:
        logger.info(f"Skipping evaluation, using existing results: {results_path}")
    
    # Step 2: Convert results for visualization
    logger.info(f"Converting healthcare results for visualization")
    viz_data_path = convert_healthcare_results(results_path)
    paths["viz_data"] = str(viz_data_path)
    
    # Step 3: Generate visualizations (if not skipped)
    if not skip_visualization:
        logger.info(f"Generating visualizations")
        
        # Create visualizer
        visualizer = MetricsVisualizer(viz_data_path, viz_output_dir, visualization_types=visualization_types)
        
        # Generate all visualizations
        visualizer.generate_all_visualizations()
        
        # Track metrics over time if history file is provided
        if metrics_history:
            logger.info(f"Tracking metrics over time with history file: {metrics_history}")
            visualizer.track_healthcare_metrics_over_time(metrics_history_path=metrics_history)
        
        # Generate HTML report with all visualizations
        visualizer.generate_html_report()
        
        logger.info(f"Visualizations saved to {viz_output_dir}")
    
    # Step 4: Process contradiction dataset and generate healthcare-specific metrics
    contradiction_dataset_path = Path(data_dir) / "contradiction_dataset" / "medical_contradictions.json"
    healthcare_metrics_path = output_dir / "healthcare_contradiction_metrics.json"
    
    logger.info(f"Processing healthcare contradiction dataset...")
    integrator = HealthcareContradictionIntegrator(
        contradiction_dataset_path=contradiction_dataset_path,
        eval_results_path=results_path
    )
    
    # Generate and save healthcare metrics
    healthcare_metrics = integrator.save_metrics(healthcare_metrics_path)
    logger.info(f"Healthcare contradiction metrics saved to {healthcare_metrics_path}")
    
    return {
        "status": "success",
        "evaluation_result": paths["results"],
        "healthcare_metrics": str(healthcare_metrics_path),
        "visualization_data": paths["viz_data"],
        "visualizations_dir": paths["viz_output"] if not skip_visualization else None
    }

def main():
    """Main function to run the healthcare cross-reference pipeline."""
    parser = argparse.ArgumentParser(description="Run healthcare cross-reference pipeline")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b",
                        help="Path or name of base model")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter weights")
    parser.add_argument("--data", type=str, required=True,
                        help="Directory containing evaluation data")
    parser.add_argument("--output", type=str, default="output/healthcare",
                        help="Directory to save evaluation results and visualizations")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run evaluation on")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip evaluation step, use existing results")
    parser.add_argument("--skip-visualization", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--metrics-history", type=str, default=None,
                        help="Path to healthcare metrics history file for tracking over time")
    parser.add_argument("--visualization-types", type=str, nargs='+', 
                        help="List of visualization types to generate. If not specified, all visualizations will be generated.")
    
    args = parser.parse_args()
    
    try:
        paths = run_pipeline(
            model_path=args.model,
            adapter_path=args.adapter,
            data_dir=args.data,
            output_dir=args.output,
            device=args.device,
            skip_evaluation=args.skip_evaluation,
            skip_visualization=args.skip_visualization,
            metrics_history=args.metrics_history,
            visualization_types=args.visualization_types
        )
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Results: {paths['evaluation_result']}")
        
        if not args.skip_visualization:
            logger.info(f"Visualizations: {paths['visualizations_dir']}")
            
        # Open HTML report if it exists
        html_report = Path(paths['visualizations_dir']) / "evaluation_report.html"
        if html_report.exists():
            logger.info(f"HTML report: {html_report}")
            
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
