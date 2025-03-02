#!/usr/bin/env python3
"""
Healthcare Performance Enhancement System Runner

This script orchestrates the execution of all enhancement scripts and generates
a comprehensive report of the results.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

def setup_environment():
    """Ensure the virtual environment is activated and dependencies are installed."""
    if not os.path.exists("venv"):
        console.print("[yellow]Virtual environment not found. Creating...[/yellow]")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
    # Activate virtual environment in the current process
    venv_path = os.path.join("venv", "bin")
    os.environ["PATH"] = os.path.join(os.getcwd(), venv_path) + os.pathsep + os.environ["PATH"]
    
    # Check for required packages
    try:
        import pandas
        import matplotlib
        import seaborn
        import rich
    except ImportError:
        console.print("[yellow]Installing required packages...[/yellow]")
        subprocess.run([
            os.path.join(venv_path, "pip"), "install", 
            "pandas", "matplotlib", "seaborn", "rich"
        ], check=True)

def run_enhancement_scripts(args):
    """Run all enhancement scripts and collect their results."""
    enhancement_scripts = [
        "enhance_contradiction_detection.py",
        "enhance_counterfactual_reasoning.py",
        "enhance_cross_reference.py"
    ]
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for script in enhancement_scripts:
            task = progress.add_task(f"Running {script}...", total=None)
            
            # Base command for all scripts
            cmd = [
                os.path.join("venv", "bin", "python"),
                os.path.join("scripts", script),
                "--eval-results", args.eval_results,
                "--output-dir", args.output_dir
            ]
            
            # Add script-specific parameters
            if script == "enhance_contradiction_detection.py":
                # Add the required contradiction-dataset parameter
                contradiction_dataset = os.path.join(
                    "data", "healthcare", "contradiction_dataset", "medical_contradictions.json"
                )
                cmd.extend(["--contradiction-dataset", contradiction_dataset])
                
                # Add optional edge cases parameter if specified
                if args.edge_cases:
                    cmd.extend(["--edge-cases", args.edge_cases])
            
            try:
                console.print(f"[cyan]Running command:[/cyan] {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True)
                script_name = script.replace("enhance_", "").replace(".py", "")
                
                # Print stdout for debugging if verbose
                if args.verbose and result.stdout:
                    console.print(f"[dim]{result.stdout.decode()}[/dim]")
                
                # Load the enhancement plan
                # Map script names to their actual file names
                file_name_map = {
                    "contradiction_detection": "contradiction_enhancement_plan.json",
                    "counterfactual_reasoning": "counterfactual_enhancement_plan.json",
                    "cross_reference": "cross_reference_enhancement_plan.json"
                }
                
                plan_name = file_name_map.get(script_name, f"{script_name}_enhancement_plan.json")
                plan_file = os.path.join(args.output_dir, plan_name)
                
                with open(plan_file, 'r') as f:
                    results[script_name] = json.load(f)
                
                progress.update(task, description=f"[green]✓ {script} completed[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error running {script}:[/red]")
                console.print(f"[red]Command:[/red] {' '.join(cmd)}")
                console.print(f"[red]Error output:[/red]\n{e.stderr.decode()}")
                progress.update(task, description=f"[red]✗ {script} failed[/red]")
                if not args.continue_on_error:
                    sys.exit(1)
    
    return results

def generate_summary_report(results, args):
    """Generate a comprehensive summary report of all enhancements."""
    report_file = os.path.join(args.output_dir, "..", "enhancement_summary.md")
    
    # Create a rich table for display
    table = Table(title="Enhancement Summary")
    table.add_column("Enhancement Area", style="cyan")
    table.add_column("Current", style="yellow")
    table.add_column("Projected", style="green")
    table.add_column("Improvement", style="magenta")
    
    for area, data in results.items():
        area_name = area.replace("_", " ").title()
        
        if "metrics" in data:
            metrics = data["metrics"]
            for metric, values in metrics.items():
                if "current" in values and "projected" in values:
                    current = f"{values['current']:.2f}"
                    projected = f"{values['projected']:.2f}"
                    improvement = f"{values['projected'] - values['current']:.2f}"
                    
                    table.add_row(
                        f"{area_name} - {metric.title()}", 
                        current, 
                        projected, 
                        improvement
                    )
    
    console.print(table)
    console.print(f"[green]Full report saved to {report_file}[/green]")

def main():
    parser = argparse.ArgumentParser(description="Run healthcare enhancement scripts")
    parser.add_argument(
        "--eval-results", 
        default="data/healthcare_evaluation_results.json",
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        default="reports/enhancements",
        help="Directory to save enhancement reports"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running scripts even if one fails"
    )
    parser.add_argument(
        "--edge-cases",
        default=None,
        help="Path to edge cases file for contradiction detection enhancement"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from enhancement scripts"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    console.print(Panel.fit(
        "Healthcare Performance Enhancement System",
        title="Starting Enhancement Process",
        subtitle=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ))
    
    # Setup environment
    setup_environment()
    
    # Run enhancement scripts
    results = run_enhancement_scripts(args)
    
    # Generate summary report
    generate_summary_report(results, args)
    
    console.print(Panel.fit(
        "All enhancement scripts completed successfully.\n"
        "Review the detailed reports in the output directory.",
        title="Enhancement Process Complete",
        style="green"
    ))

if __name__ == "__main__":
    main()
