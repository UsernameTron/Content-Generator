#!/usr/bin/env python3
"""
Healthcare Contradiction Detection Enhancement Module.

This script implements targeted improvements for the healthcare contradiction detection system
with a focus on enhancing recall through edge case training and automated error analysis.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-contradiction-enhancement")

class ContradictionEnhancer:
    """
    Enhances healthcare contradiction detection through targeted interventions.
    Focuses on improving recall through edge case training and error analysis.
    """
    
    def __init__(
        self, 
        contradiction_dataset_path: str, 
        eval_results_path: str,
        output_dir: str,
        edge_cases_path: Optional[str] = None
    ):
        """
        Initialize the contradiction enhancer.
        
        Args:
            contradiction_dataset_path: Path to the contradiction dataset JSON file
            eval_results_path: Path to evaluation results JSON
            output_dir: Directory to save enhancement outputs
            edge_cases_path: Optional path to additional edge cases dataset
        """
        self.contradiction_dataset_path = Path(contradiction_dataset_path)
        self.eval_results_path = Path(eval_results_path)
        self.output_dir = Path(output_dir)
        self.edge_cases_path = Path(edge_cases_path) if edge_cases_path else None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load contradiction dataset
        if self.contradiction_dataset_path.exists():
            with open(self.contradiction_dataset_path, 'r') as f:
                self.contradiction_data = json.load(f)
            logger.info(f"Loaded {len(self.contradiction_data)} contradiction examples")
        else:
            raise FileNotFoundError(f"Contradiction dataset not found: {contradiction_dataset_path}")
        
        # Load evaluation results
        if self.eval_results_path.exists():
            with open(self.eval_results_path, 'r') as f:
                self.eval_data = json.load(f)
            logger.info(f"Loaded evaluation results from {eval_results_path}")
        else:
            raise FileNotFoundError(f"Evaluation results not found: {eval_results_path}")
        
        # Load edge cases if provided
        self.edge_cases = []
        if self.edge_cases_path and self.edge_cases_path.exists():
            with open(self.edge_cases_path, 'r') as f:
                self.edge_cases = json.load(f)
            logger.info(f"Loaded {len(self.edge_cases)} edge cases")
        
        # Initialize enhancement metrics
        self.enhancement_metrics = {
            "timestamp": datetime.now().isoformat(),
            "baseline_metrics": self._extract_baseline_metrics(),
            "enhancement_targets": {},
            "error_analysis": {},
            "edge_case_coverage": {},
            "projected_improvements": {}
        }
    
    def _extract_baseline_metrics(self) -> Dict[str, Any]:
        """Extract baseline contradiction detection metrics from evaluation results."""
        try:
            return self.eval_data.get("healthcare", {}).get("contradiction_detection", {})
        except (KeyError, TypeError):
            logger.warning("Could not extract baseline contradiction metrics")
            return {}
    
    def identify_error_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in contradiction detection errors.
        
        Returns:
            Dictionary with error pattern analysis
        """
        # Extract contradiction categories and their metrics
        categories = self.enhancement_metrics["baseline_metrics"].get("categories", {})
        
        # Identify categories with lowest accuracy
        sorted_categories = sorted(
            [(cat, data["accuracy"]) for cat, data in categories.items()],
            key=lambda x: x[1]
        )
        
        # Identify error patterns
        error_patterns = {
            "lowest_performing_categories": sorted_categories[:2],
            "recall_issues": []
        }
        
        # Analyze recall issues
        baseline_recall = self.enhancement_metrics["baseline_metrics"].get("recall", 0)
        if baseline_recall < 0.85:
            error_patterns["recall_issues"].append({
                "metric": "recall",
                "current_value": baseline_recall,
                "target_value": min(baseline_recall + 0.1, 0.95),
                "gap": min(baseline_recall + 0.1, 0.95) - baseline_recall
            })
        
        # Check for domain-specific issues
        for item in self.contradiction_data:
            domain = item.get("domain", "unknown")
            if domain not in error_patterns:
                error_patterns[domain] = {"count": 0, "examples": []}
            
            error_patterns[domain]["count"] += 1
            if len(error_patterns[domain]["examples"]) < 3:  # Limit examples for brevity
                error_patterns[domain]["examples"].append(item.get("id", "unknown"))
        
        self.enhancement_metrics["error_analysis"] = error_patterns
        return error_patterns
    
    def analyze_edge_cases(self) -> Dict[str, Any]:
        """
        Analyze edge cases to identify coverage gaps.
        
        Returns:
            Dictionary with edge case analysis
        """
        # Combine main dataset with edge cases if available
        all_cases = self.contradiction_data + self.edge_cases if self.edge_cases else self.contradiction_data
        
        # Analyze distribution of contradiction types
        categories = {}
        domains = {}
        complexity_levels = {"simple": 0, "moderate": 0, "complex": 0}
        
        for item in all_cases:
            # Extract category
            category = item.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
            
            # Extract domain
            domain = item.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1
            
            # Estimate complexity (simplified approach)
            statements = [item.get("statement1", ""), item.get("statement2", "")]
            avg_length = sum(len(s.split()) for s in statements) / 2
            
            if avg_length < 10:
                complexity_levels["simple"] += 1
            elif avg_length < 20:
                complexity_levels["moderate"] += 1
            else:
                complexity_levels["complex"] += 1
        
        # Identify coverage gaps
        coverage_gaps = []
        
        # Check domain coverage
        domain_counts = sorted([(d, c) for d, c in domains.items()], key=lambda x: x[1])
        if domain_counts and domain_counts[0][1] < 5:
            coverage_gaps.append({
                "type": "domain_coverage",
                "domain": domain_counts[0][0],
                "count": domain_counts[0][1],
                "recommendation": f"Add more examples for {domain_counts[0][0]} domain"
            })
        
        # Check complexity coverage
        if complexity_levels["complex"] / len(all_cases) < 0.2:
            coverage_gaps.append({
                "type": "complexity_coverage",
                "level": "complex",
                "count": complexity_levels["complex"],
                "percentage": complexity_levels["complex"] / len(all_cases),
                "recommendation": "Add more complex contradiction examples"
            })
        
        edge_case_analysis = {
            "categories": categories,
            "domains": domains,
            "complexity_levels": complexity_levels,
            "coverage_gaps": coverage_gaps,
            "total_examples": len(all_cases)
        }
        
        self.enhancement_metrics["edge_case_coverage"] = edge_case_analysis
        return edge_case_analysis
    
    def generate_enhancement_plan(self) -> Dict[str, Any]:
        """
        Generate a plan for enhancing contradiction detection.
        
        Returns:
            Dictionary with enhancement plan
        """
        baseline_metrics = self.enhancement_metrics["baseline_metrics"]
        error_analysis = self.enhancement_metrics["error_analysis"]
        edge_case_coverage = self.enhancement_metrics["edge_case_coverage"]
        
        # Set target metrics
        target_metrics = {
            "accuracy": min(baseline_metrics.get("accuracy", 0.8) + 0.05, 0.95),
            "precision": baseline_metrics.get("precision", 0.85),  # Maintain precision
            "recall": min(baseline_metrics.get("recall", 0.79) + 0.1, 0.92),  # Focus on recall improvement
            "f1_score": min(baseline_metrics.get("f1_score", 0.82) + 0.07, 0.93)
        }
        
        # Generate specific interventions
        interventions = []
        
        # Intervention 1: Edge case training
        interventions.append({
            "name": "edge_case_training",
            "description": "Enhance training with edge cases focusing on low-performing categories",
            "target_metrics": ["recall", "accuracy"],
            "expected_improvement": {
                "recall": 0.05,
                "accuracy": 0.03
            },
            "implementation_steps": [
                "Identify lowest performing contradiction categories",
                "Generate additional training examples for these categories",
                "Implement specialized training cycles focusing on edge cases",
                "Validate improvements with targeted evaluation"
            ]
        })
        
        # Intervention 2: Error analysis pipeline
        interventions.append({
            "name": "error_analysis_pipeline",
            "description": "Implement automated error analysis to identify and address systematic errors",
            "target_metrics": ["recall", "f1_score"],
            "expected_improvement": {
                "recall": 0.03,
                "f1_score": 0.02
            },
            "implementation_steps": [
                "Develop error categorization system",
                "Implement automated error detection and logging",
                "Create feedback loop for continuous improvement",
                "Generate periodic error analysis reports"
            ]
        })
        
        # Intervention 3: Domain-specific enhancements
        if error_analysis.get("lowest_performing_categories"):
            lowest_category = error_analysis["lowest_performing_categories"][0][0]
            interventions.append({
                "name": "domain_specific_enhancement",
                "description": f"Enhance detection capabilities for {lowest_category} contradictions",
                "target_metrics": ["accuracy", "recall"],
                "expected_improvement": {
                    "accuracy": 0.04,
                    "recall": 0.04
                },
                "implementation_steps": [
                    f"Analyze error patterns in {lowest_category} category",
                    "Develop specialized detection rules for this category",
                    "Implement category-specific preprocessing steps",
                    "Evaluate category-specific performance improvements"
                ]
            })
        
        # Calculate projected improvements
        projected_metrics = {
            metric: baseline_metrics.get(metric, 0) for metric in target_metrics
        }
        
        for intervention in interventions:
            for metric, improvement in intervention["expected_improvement"].items():
                if metric in projected_metrics:
                    projected_metrics[metric] += improvement
        
        # Cap projected metrics at reasonable values
        for metric in projected_metrics:
            projected_metrics[metric] = min(projected_metrics[metric], 0.95)
        
        enhancement_plan = {
            "target_metrics": target_metrics,
            "interventions": interventions,
            "projected_metrics": projected_metrics,
            "implementation_timeline": {
                "phase1": "Edge case training implementation",
                "phase2": "Error analysis pipeline development",
                "phase3": "Domain-specific enhancements",
                "phase4": "Comprehensive evaluation and refinement"
            }
        }
        
        self.enhancement_metrics["enhancement_targets"] = enhancement_plan
        return enhancement_plan
    
    def visualize_projected_improvements(self, output_path: Optional[str] = None) -> str:
        """
        Create visualization of projected improvements.
        
        Args:
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        baseline = self.enhancement_metrics["baseline_metrics"]
        projected = self.enhancement_metrics["enhancement_targets"]["projected_metrics"]
        
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        baseline_values = [baseline.get(m, 0) for m in metrics]
        projected_values = [projected.get(m, 0) for m in metrics]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Set width of bars
        barWidth = 0.3
        
        # Set positions of bars on X axis
        r1 = np.arange(len(metrics))
        r2 = [x + barWidth for x in r1]
        
        # Create bars
        plt.bar(r1, baseline_values, width=barWidth, edgecolor='grey', label='Current')
        plt.bar(r2, projected_values, width=barWidth, edgecolor='grey', label='Projected')
        
        # Add labels and title
        plt.xlabel('Metrics', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.title('Projected Improvement in Contradiction Detection')
        plt.xticks([r + barWidth/2 for r in range(len(metrics))], metrics)
        plt.ylim(0, 1.0)
        
        # Add a grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on bars
        for i, v in enumerate(baseline_values):
            plt.text(r1[i], v + 0.01, f'{v:.2f}', ha='center')
            
        for i, v in enumerate(projected_values):
            plt.text(r2[i], v + 0.01, f'{v:.2f}', ha='center')
        
        # Add legend
        plt.legend()
        
        # Save figure
        if output_path is None:
            output_path = self.output_dir / "contradiction_improvement_projection.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved projected improvement visualization to {output_path}")
        return str(output_path)
    
    def save_enhancement_plan(self, output_path: Optional[str] = None) -> str:
        """
        Save the enhancement plan to a JSON file.
        
        Args:
            output_path: Optional path to save the enhancement plan
            
        Returns:
            Path to the saved enhancement plan
        """
        if output_path is None:
            output_path = self.output_dir / "contradiction_enhancement_plan.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.enhancement_metrics, f, indent=2)
        
        logger.info(f"Saved enhancement plan to {output_path}")
        return str(output_path)
    
    def run_enhancement_analysis(self) -> Dict[str, Any]:
        """
        Run the complete enhancement analysis workflow.
        
        Returns:
            Dictionary with enhancement metrics and file paths
        """
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            # Step 1: Identify error patterns
            task1 = progress.add_task("[green]Identifying error patterns...", total=1)
            self.identify_error_patterns()
            progress.update(task1, completed=1)
            
            # Step 2: Analyze edge cases
            task2 = progress.add_task("[green]Analyzing edge cases...", total=1)
            self.analyze_edge_cases()
            progress.update(task2, completed=1)
            
            # Step 3: Generate enhancement plan
            task3 = progress.add_task("[green]Generating enhancement plan...", total=1)
            self.generate_enhancement_plan()
            progress.update(task3, completed=1)
            
            # Step 4: Create visualizations
            task4 = progress.add_task("[green]Creating visualizations...", total=1)
            viz_path = self.visualize_projected_improvements()
            progress.update(task4, completed=1)
            
            # Step 5: Save enhancement plan
            task5 = progress.add_task("[green]Saving enhancement plan...", total=1)
            plan_path = self.save_enhancement_plan()
            progress.update(task5, completed=1)
        
        return {
            "enhancement_metrics": self.enhancement_metrics,
            "visualization_path": viz_path,
            "plan_path": plan_path
        }


def main():
    """Main function to run the contradiction enhancement analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Healthcare Contradiction Enhancement")
    parser.add_argument("--contradiction-dataset", required=True, help="Path to contradiction dataset JSON")
    parser.add_argument("--eval-results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to save enhancement outputs")
    parser.add_argument("--edge-cases", help="Optional path to edge cases dataset")
    
    args = parser.parse_args()
    
    try:
        # Initialize the enhancer
        enhancer = ContradictionEnhancer(
            contradiction_dataset_path=args.contradiction_dataset,
            eval_results_path=args.eval_results,
            output_dir=args.output_dir,
            edge_cases_path=args.edge_cases
        )
        
        # Run enhancement analysis
        results = enhancer.run_enhancement_analysis()
        
        logger.info("Healthcare contradiction enhancement analysis completed successfully")
        logger.info(f"Enhancement plan saved to: {results['plan_path']}")
        logger.info(f"Visualization saved to: {results['visualization_path']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in healthcare contradiction enhancement: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
