"""
Create WandB Dashboard

This script creates a dashboard in Weights & Biases to visualize fine-tuning metrics.
"""

import os
import argparse
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

def check_wandb_installed():
    """Check if wandb is installed."""
    try:
        import wandb
        return True, wandb.__version__
    except ImportError:
        return False, None

def create_dashboard(project_name, run_name=None, dashboard_name=None):
    """Create a WandB dashboard for fine-tuning metrics."""
    try:
        import wandb
        from wandb.sdk.artifacts.public_api import ArtifactNotLoggedError
        
        # Check if logged in
        try:
            api = wandb.Api()
            user = api.viewer()
            console.print(f"[bold green]✓ Logged in as: {user['username']}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]× Not logged in to WandB: {e}[/bold red]")
            console.print("Please run 'wandb login' or fix your authentication with fix_wandb_auth.py")
            return False
        
        # Check if project exists
        try:
            project = api.project(user["username"], project_name)
            console.print(f"[bold green]✓ Found project: {project_name}[/bold green]")
        except Exception as e:
            console.print(f"[bold yellow]× Project not found: {project_name}[/bold yellow]")
            console.print("A new project will be created when you start training.")
        
        # Create dashboard name if not provided
        if not dashboard_name:
            dashboard_name = f"{project_name} Fine-Tuning Metrics"
        
        # Try to create dashboard
        console.print(f"[bold]Creating dashboard: {dashboard_name}[/bold]")
        
        # Check for existing runs
        runs = list(api.runs(f"{user['username']}/{project_name}"))
        
        if not runs:
            console.print("[bold yellow]× No runs found in this project[/bold yellow]")
            console.print("Dashboard will be created with panels but no data will be shown until runs are available.")
        else:
            console.print(f"[bold green]✓ Found {len(runs)} runs in project[/bold green]")
            
            # Filter runs by name if run_name is provided
            if run_name:
                runs = [run for run in runs if run_name in run.name]
                console.print(f"[bold]Filtered to {len(runs)} runs with name containing '{run_name}'[/bold]")
        
        # Create dashboard with panels
        dashboard = api.create_dashboard(
            project=project_name,
            name=dashboard_name,
            description="Fine-tuning metrics dashboard for ML model training"
        )
        
        # Add panels to dashboard
        panels = [
            # Training Loss Panel
            {
                "name": "Training Loss",
                "type": "line",
                "query": {"keys": ["loss"], "groupby": "run.name"},
                "height": 6,
                "width": 12
            },
            # Learning Rate Panel
            {
                "name": "Learning Rate",
                "type": "line",
                "query": {"keys": ["learning_rate"], "groupby": "run.name"},
                "height": 6,
                "width": 12
            },
            # Training vs Validation Loss
            {
                "name": "Train vs Validation Loss",
                "type": "line",
                "query": {"keys": ["loss", "eval/loss"], "groupby": "run.name"},
                "height": 6,
                "width": 12
            },
            # Steps per Second
            {
                "name": "Training Speed",
                "type": "line",
                "query": {"keys": ["train/train_runtime", "train/train_samples_per_second"], "groupby": "run.name"},
                "height": 6,
                "width": 12
            },
            # GPU Memory
            {
                "name": "GPU Memory",
                "type": "line",
                "query": {"keys": ["gpu", "gpu_mem"], "groupby": "run.name"},
                "height": 6,
                "width": 12
            },
            # Run Summary
            {
                "name": "Run Summary",
                "type": "table",
                "query": {"keys": ["run.name", "train/epoch", "train/train_runtime", "train/train_samples_per_second"]},
                "height": 6,
                "width": 24
            }
        ]
        
        for panel_config in panels:
            dashboard.add_panel(
                name=panel_config["name"],
                panel_type=panel_config["type"],
                query=panel_config["query"],
                height=panel_config["height"],
                width=panel_config["width"]
            )
        
        dashboard_url = f"https://wandb.ai/{user['username']}/{project_name}/dashboards/{dashboard.id}"
        console.print(f"[bold green]✓ Dashboard created successfully![/bold green]")
        console.print(f"[bold]Dashboard URL:[/bold] {dashboard_url}")
        
        return True, dashboard_url
    except Exception as e:
        console.print(f"[bold red]× Error creating dashboard: {e}[/bold red]")
        return False, None

def get_project_from_config():
    """Get project name from finetune_config.json if available."""
    try:
        import json
        config_path = Path("finetune_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                wandb_config = config.get("wandb_config", {})
                project = wandb_config.get("project")
                run_name = wandb_config.get("name")
                return project, run_name
        return None, None
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return None, None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create WandB Dashboard for Fine-Tuning Metrics")
    parser.add_argument("--project", type=str, help="WandB project name")
    parser.add_argument("--run-name", type=str, help="Filter for specific run names (substring match)")
    parser.add_argument("--dashboard-name", type=str, help="Name for the dashboard")
    args = parser.parse_args()
    
    console.rule("[bold]Create WandB Dashboard[/bold]")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if wandb is installed
    installed, version = check_wandb_installed()
    if not installed:
        console.print("[bold red]× WandB is not installed[/bold red]")
        console.print("Please install WandB with: pip install wandb")
        return
    
    console.print(f"[bold green]✓ WandB is installed (version: {version})[/bold green]")
    
    # Get project from config if not provided
    project_from_config, run_name_from_config = get_project_from_config()
    
    project_name = args.project or project_from_config
    if not project_name:
        console.print("[bold red]× No project name provided[/bold red]")
        console.print("Please provide a project name with --project or in finetune_config.json")
        return
    
    run_name = args.run_name or run_name_from_config
    
    # Create dashboard
    success, dashboard_url = create_dashboard(
        project_name=project_name,
        run_name=run_name,
        dashboard_name=args.dashboard_name
    )
    
    if success:
        console.print(Panel(
            f"[bold green]WandB Dashboard created successfully![/bold green]\n\n"
            f"URL: {dashboard_url}\n\n"
            "You can view your training metrics in real-time as fine-tuning progresses.",
            title="Dashboard Created",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]Failed to create WandB Dashboard.[/bold red]\n"
            "Please check your WandB configuration and try again.",
            title="Creation Failed",
            border_style="red"
        ))

if __name__ == "__main__":
    main()
