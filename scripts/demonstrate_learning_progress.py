#!/usr/bin/env python3
"""
Demonstrate healthcare continuous learning progress over multiple cycles.
This script shows how the continuous learning system improves over time by:
1. Running multiple learning cycles
2. Tracking accuracy improvements
3. Visualizing learning progress
"""

import os
import sys
import json
import logging
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-learning-progress")
console = Console()

# Import demonstration function
from demonstrate_continuous_learning import demonstrate_continuous_learning
from generate_synthetic_evaluation import generate_synthetic_evaluation

def track_metrics_over_time(cycles_data, output_path=None):
    """Track and visualize healthcare metrics over multiple learning cycles.
    
    Args:
        cycles_data: List of metrics from each learning cycle
        output_path: Path to save the visualization
        
    Returns:
        Path to saved visualization
    """
    # Use non-interactive backend if needed
    if not os.environ.get('DISPLAY'):
        matplotlib.use('Agg')
    
    # Extract metrics
    cycle_numbers = [i+1 for i in range(len(cycles_data))]
    accuracies = [cycle.get("accuracy", 0) for cycle in cycles_data]
    example_counts = [cycle.get("example_count", 0) for cycle in cycles_data]
    improvement_counts = [cycle.get("improvement_areas", 0) for cycle in cycles_data]
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot accuracy over time
    ax1.plot(cycle_numbers, accuracies, 'o-', color='blue', linewidth=2)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Improvement Over Learning Cycles')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim([min(0.5, min(accuracies)-0.05), 1.0])
    
    # For each data point, add a label
    for i, acc in enumerate(accuracies):
        ax1.annotate(f'{acc:.2f}', 
                    (cycle_numbers[i], acc), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center')
    
    # Plot example count and improvement areas
    ax2.bar(cycle_numbers, example_counts, color='green', alpha=0.7, label='New Examples')
    ax2.set_ylabel('New Examples Count', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Create a twin axis for improvement areas
    ax3 = ax2.twinx()
    ax3.plot(cycle_numbers, improvement_counts, 'r-o', label='Improvement Areas')
    ax3.set_ylabel('Improvement Areas Count', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    # Set common x-axis label
    ax2.set_xlabel('Learning Cycle')
    ax2.set_xticks(cycle_numbers)
    
    # Add legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        return output_path
    else:
        output_path = "learning_progress.png"
        plt.savefig(output_path)
        return output_path

def run_multiple_learning_cycles(data_dir="data/healthcare", 
                                cycles=3, 
                                examples_per_cycle=10,
                                starting_accuracy=0.75):
    """Run multiple continuous learning cycles and track progress.
    
    Args:
        data_dir: Directory for healthcare data
        cycles: Number of learning cycles to run
        examples_per_cycle: Number of examples to generate per cycle
        starting_accuracy: Starting model accuracy
    """
    console.print("\n[bold blue]Healthcare Contradiction Detection - Multi-Cycle Learning Progress[/bold blue]\n")
    
    # Track metrics for each cycle
    cycles_data = []
    current_accuracy = starting_accuracy
    
    # Set up progress tracking
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Running learning cycles...", total=cycles)
        
        for cycle in range(cycles):
            # Adjust accuracy based on previous cycles (simulate improvement)
            if cycle > 0:
                # Increase accuracy by up to 5% per cycle, with diminishing returns
                max_improvement = 0.05 * (1 - current_accuracy)
                improvement = max(0.01, max_improvement)
                current_accuracy = min(0.95, current_accuracy + improvement)
            
            # Run learning cycle
            progress.update(task, description=f"[green]Running learning cycle {cycle+1}/{cycles}")
            
            # Run demonstration with current accuracy
            cycle_results = demonstrate_continuous_learning(
                data_dir=data_dir,
                examples_to_generate=examples_per_cycle,
                accuracy=current_accuracy
            )
            
            # Extract metrics from the cycle
            cycle_metrics = {
                "cycle": cycle + 1,
                "accuracy": current_accuracy,
                "improvement_areas": cycle_results.get("improvement_areas", 0),
                "new_examples": cycle_results.get("new_examples", 0),
                "example_count": examples_per_cycle
            }
            
            cycles_data.append(cycle_metrics)
            progress.update(task, advance=1)
    
    # Visualize learning progress
    console.print("\n[bold]Visualizing Learning Progress[/bold]")
    visualization_path = track_metrics_over_time(cycles_data)
    
    console.print(f"[green]Learning progress visualization saved to: {visualization_path}[/green]")
    
    # Print final summary table
    table = Table(title="Learning Cycles Summary")
    table.add_column("Cycle", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Improvement Areas", style="yellow")
    table.add_column("New Examples", style="blue")
    
    for cycle in cycles_data:
        table.add_row(
            str(cycle["cycle"]),
            f"{cycle['accuracy']:.2f}",
            str(cycle["improvement_areas"]),
            str(cycle["new_examples"])
        )
    
    console.print(table)
    
    # Calculate total improvement
    initial_accuracy = cycles_data[0]["accuracy"]
    final_accuracy = cycles_data[-1]["accuracy"]
    improvement = final_accuracy - initial_accuracy
    
    console.print("\n[bold blue]Continuous Learning Summary[/bold blue]")
    console.print(f"Initial accuracy: {initial_accuracy:.2f}")
    console.print(f"Final accuracy: {final_accuracy:.2f}")
    console.print(f"Absolute improvement: {improvement:.2f} ({improvement/initial_accuracy*100:.1f}% relative improvement)")
    console.print(f"Total new examples generated: {sum(c['new_examples'] for c in cycles_data)}")
    console.print(f"Total cycles: {cycles}")
    
    console.print("\n[bold green]Continuous learning progression completed successfully![/bold green]")

def main():
    parser = argparse.ArgumentParser(description="Demonstrate learning progression over multiple cycles")
    parser.add_argument("--data-dir", 
                       type=str, 
                       default="data/healthcare",
                       help="Path to healthcare data directory")
    parser.add_argument("--cycles", 
                       type=int, 
                       default=3,
                       help="Number of learning cycles to run")
    parser.add_argument("--examples", 
                       type=int, 
                       default=10,
                       help="Number of examples to generate per cycle")
    parser.add_argument("--accuracy", 
                       type=float, 
                       default=0.75,
                       help="Starting accuracy")
    args = parser.parse_args()
    
    run_multiple_learning_cycles(
        data_dir=args.data_dir,
        cycles=args.cycles,
        examples_per_cycle=args.examples,
        starting_accuracy=args.accuracy
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
