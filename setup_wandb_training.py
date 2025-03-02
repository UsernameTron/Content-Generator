"""
Weights & Biases setup for model training monitoring.

This script configures W&B for monitoring the fine-tuning process, 
setting up appropriate dashboards and metrics tracking.
"""

import os
import sys
import json
import logging
import argparse
import getpass
from dotenv import load_dotenv, set_key
from pathlib import Path
import wandb
from rich.console import Console
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

def load_or_create_env_file(env_file=".env"):
    """
    Load or create .env file for storing W&B API key.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Path to .env file
    """
    env_path = Path(env_file)
    if not env_path.exists():
        logger.info(f"Creating new .env file at {env_path}")
        env_path.touch()
    else:
        logger.info(f"Loading existing .env file from {env_path}")
        load_dotenv(env_path)
    
    return env_path

def setup_wandb_key(env_path):
    """
    Set up W&B API key.
    
    Args:
        env_path: Path to .env file
        
    Returns:
        True if set up successfully, False otherwise
    """
    # Check if API key is already set
    existing_key = os.environ.get("WANDB_API_KEY")
    if existing_key:
        logger.info("W&B API key already set")
        console.print("W&B API key is already configured.")
        confirm = input("Do you want to use the existing key? (y/n): ").lower()
        if confirm in ["y", "yes"]:
            return True
    
    # Prompt for API key
    console.print(Panel(
        "[bold yellow]Weights & Biases Setup for Training Monitoring[/bold yellow]\n\n"
        "To monitor model training, you need a Weights & Biases API key.\n"
        "Get your key from: [link]https://wandb.ai/authorize[/link]\n\n"
        "Your key will be stored in a local .env file and won't be shared."
    ))
    
    api_key = getpass.getpass("Enter your W&B API key: ")
    
    if not api_key:
        console.print("[bold red]No API key provided. W&B monitoring will not be available.[/bold red]")
        return False
    
    # Set key in .env file
    try:
        set_key(str(env_path), "WANDB_API_KEY", api_key)
        os.environ["WANDB_API_KEY"] = api_key
        console.print("[bold green]W&B API key saved successfully![/bold green]")
        return True
    except Exception as e:
        logger.error(f"Error saving W&B API key: {e}")
        console.print(f"[bold red]Error saving W&B API key: {e}[/bold red]")
        return False

def setup_wandb_project(project_name, team_name=None):
    """
    Set up W&B project for model training.
    
    Args:
        project_name: Name of W&B project
        team_name: Optional team name for the project
        
    Returns:
        True if set up successfully, False otherwise
    """
    # Initialize W&B
    try:
        if team_name:
            wandb.init(project=project_name, entity=team_name)
        else:
            wandb.init(project=project_name)
        
        logger.info(f"W&B project initialized: {project_name}")
        console.print(f"[bold green]W&B project '{project_name}' initialized successfully![/bold green]")
        
        # Create and save default dashboard
        create_default_training_dashboard(project_name, team_name)
        
        # Finish the run (we're just setting up the project)
        wandb.finish()
        return True
    except Exception as e:
        logger.error(f"Error initializing W&B project: {e}")
        console.print(f"[bold red]Error initializing W&B project: {e}[/bold red]")
        return False

def create_default_training_dashboard(project_name, team_name=None):
    """
    Create default W&B dashboard for training monitoring.
    
    Args:
        project_name: Name of W&B project
        team_name: Optional team name for the project
    """
    try:
        # Define dashboard panels
        dashboard_panels = {
            "title": "C. Pete Connor Style Training Monitor",
            "description": "Monitoring dashboard for fine-tuning the C. Pete Connor satirical tech expert style model",
            "panels": [
                {
                    "title": "Training Loss",
                    "type": "line",
                    "metrics": ["total_loss", "base_loss", "style_penalty", "style_reward"],
                    "viewType": "individual"
                },
                {
                    "title": "Learning Rate",
                    "type": "line",
                    "metrics": ["learning_rate"]
                },
                {
                    "title": "Style Metrics",
                    "type": "line",
                    "metrics": ["style_penalty", "style_reward"],
                    "viewType": "individual"
                },
                {
                    "title": "GPU Memory",
                    "type": "line",
                    "metrics": ["gpu_memory_allocated", "gpu_memory_reserved"]
                },
                {
                    "title": "Training Samples",
                    "type": "text-table",
                    "metrics": ["generated_samples"]
                }
            ]
        }
        
        # Save dashboard configuration
        dashboard_path = Path("wandb_dashboard_config.json")
        with open(dashboard_path, "w") as f:
            json.dump(dashboard_panels, f, indent=2)
        
        logger.info(f"Saved default dashboard configuration to {dashboard_path}")
        console.print(f"[green]Default training dashboard configuration saved to {dashboard_path}[/green]")
        
        # Note: W&B doesn't have a direct API to create dashboards programmatically
        # We're saving the configuration so it can be referenced later
        console.print(
            "\n[yellow]Note: To create this dashboard in the W&B UI:[/yellow]\n"
            "1. Go to your project: https://wandb.ai/" + 
            (f"{team_name}/" if team_name else "") + 
            f"{project_name}\n"
            "2. Click 'Create Dashboard'\n"
            "3. Use the configurations saved in wandb_dashboard_config.json\n"
        )
    except Exception as e:
        logger.error(f"Error creating dashboard configuration: {e}")
        console.print(f"[bold red]Error creating dashboard configuration: {e}[/bold red]")

def test_wandb_connection():
    """
    Test W&B connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to initialize W&B in offline mode to test API key
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="test_connection")
        wandb.finish()
        
        # If successful, reset WANDB_MODE
        os.environ.pop("WANDB_MODE", None)
        
        logger.info("W&B connection test successful")
        console.print("[bold green]W&B connection test successful![/bold green]")
        return True
    except Exception as e:
        logger.error(f"W&B connection test failed: {e}")
        console.print(f"[bold red]W&B connection test failed: {e}[/bold red]")
        return False

def setup_wandb_training():
    """
    Main function to set up W&B for training monitoring.
    
    Returns:
        True if set up successfully, False otherwise
    """
    # Load or create .env file
    env_path = load_or_create_env_file()
    
    # Set up W&B API key
    if not setup_wandb_key(env_path):
        return False
    
    # Test W&B connection
    if not test_wandb_connection():
        return False
    
    # Set up W&B project
    project_name = "pete-connor-style-tuning"
    setup_wandb_project(project_name)
    
    # Save W&B configuration
    config = {
        "project": project_name,
        "log_model": True,
        "watch_model": True,
        "log_freq": 10,
        "save_code": True
    }
    
    config_path = Path("wandb_training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved W&B configuration to {config_path}")
    console.print(f"[green]W&B configuration saved to {config_path}[/green]")
    
    # Success message
    console.print(Panel(
        "[bold green]W&B Training Monitoring Setup Complete![/bold green]\n\n"
        f"Project: {project_name}\n"
        "Dashboard: Configuration saved to wandb_dashboard_config.json\n"
        "Configuration: Saved to wandb_training_config.json\n\n"
        "You're now ready to monitor model training with W&B!"
    ))
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up W&B for model training monitoring")
    args = parser.parse_args()
    
    setup_wandb_training()
