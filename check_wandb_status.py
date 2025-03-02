"""
Check Weights & Biases (WandB) Status

This script checks the status of WandB integration for the fine-tuning process,
verifies API key configuration, and ensures proper synchronization of metrics.
"""

import os
import logging
import json
import argparse
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def check_environment():
    """Check environment variables and .env file configuration."""
    # Load environment variables from .env file
    load_dotenv()
    
    console.rule("[bold]Checking Environment Configuration[/bold]")
    
    # Check for WandB API key
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        console.print("[bold red]ERROR: WANDB_API_KEY not found in environment variables.[/bold red]")
        console.print("Please add your WandB API key to the .env file:")
        console.print(Panel("WANDB_API_KEY=your_api_key_here", title=".env file", border_style="yellow"))
        return False
    elif wandb_key == "your_api_key_here":
        console.print("[bold yellow]WARNING: WANDB_API_KEY appears to be a placeholder.[/bold yellow]")
        console.print("Please replace with your actual WandB API key in the .env file.")
        return False
    else:
        key_preview = wandb_key[:4] + "..." + wandb_key[-4:] if len(wandb_key) > 8 else "***"
        console.print(f"[bold green]✓[/bold green] WANDB_API_KEY found: {key_preview}")
    
    # Check WandB mode
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode == "offline":
        console.print("[bold yellow]NOTE: WandB is configured to run in offline mode.[/bold yellow]")
        console.print("Metrics will be saved locally but not synced to wandb.ai.")
    elif wandb_mode == "disabled":
        console.print("[bold red]WARNING: WandB is disabled.[/bold red]")
        console.print("No metrics will be collected.")
        return False
    else:
        console.print("[bold green]✓[/bold green] WandB is configured for online tracking.")
    
    return True

def check_wandb_installation():
    """Check if wandb is installed and properly configured."""
    console.rule("[bold]Checking WandB Installation[/bold]")
    
    try:
        import wandb
        console.print(f"[bold green]✓[/bold green] WandB installed (version: {wandb.__version__})")
        
        # Just check if we're logged in - don't make complex API calls
        try:
            # Run a login check - this will use the API key from the environment
            is_logged_in = wandb.login(anonymous="never")
            
            if is_logged_in:
                console.print(f"[bold green]✓[/bold green] Successfully authenticated with WandB")
                return True
            else:
                console.print("[bold red]ERROR: WandB authentication failed[/bold red]")
                return False
        except Exception as e:
            console.print(f"[bold red]ERROR: Failed to authenticate with WandB: {str(e)}[/bold red]")
            return False
    except ImportError:
        console.print("[bold red]ERROR: WandB not installed.[/bold red]")
        console.print("Install with: pip install wandb")
        return False
    except Exception as e:
        console.print(f"[bold red]ERROR: {str(e)}[/bold red]")
        return False

def check_config_files():
    """Check config files for WandB settings."""
    console.rule("[bold]Checking Configuration Files[/bold]")
    
    # Check finetune_config.json
    config_path = Path("finetune_config.json")
    if not config_path.exists():
        console.print("[bold red]ERROR: finetune_config.json not found.[/bold red]")
        return False
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Check WandB config
        if "wandb_config" not in config:
            console.print("[bold red]ERROR: wandb_config section missing in finetune_config.json[/bold red]")
            return False
        
        wandb_config = config["wandb_config"]
        console.print("[bold green]✓[/bold green] Found WandB configuration in finetune_config.json:")
        console.print(f"  Project: [cyan]{wandb_config.get('project')}[/cyan]")
        console.print(f"  Run name: [cyan]{wandb_config.get('name')}[/cyan]")
        
        # Check training args
        training_args = config.get("training_config", {}).get("training_args", {})
        if "report_to" not in training_args:
            console.print("[bold yellow]WARNING: 'report_to' field missing in training configuration.[/bold yellow]")
        elif "wandb" not in training_args["report_to"]:
            console.print("[bold red]ERROR: 'wandb' not in 'report_to' field in training configuration.[/bold red]")
            console.print("Update training_args to include: \"report_to\": \"wandb\"")
            return False
        else:
            console.print("[bold green]✓[/bold green] Training configuration correctly references WandB")
        
        return True
    except json.JSONDecodeError:
        console.print("[bold red]ERROR: Invalid JSON in finetune_config.json[/bold red]")
        return False
    except Exception as e:
        console.print(f"[bold red]ERROR: {str(e)}[/bold red]")
        return False

def check_active_runs():
    """Check for active WandB runs."""
    console.rule("[bold]Checking Active WandB Runs[/bold]")
    
    try:
        import wandb
        api = wandb.Api()
        
        # Load config to get project name
        with open("finetune_config.json", "r") as f:
            config = json.load(f)
        
        project_name = config.get("wandb_config", {}).get("project")
        if not project_name:
            console.print("[bold yellow]WARNING: Could not determine project name from config.[/bold yellow]")
            return False
        
        try:
            # Get active runs - use a hardcoded entity from our previous test
            # We know from our test this is the correct entity name
            entity = "cpeteconnor-fiverr"
            
            try:
                runs = api.runs(f"{entity}/{project_name}")
                active_runs = [run for run in runs if run.state == "running"]
                
                if active_runs:
                    console.print(f"[bold green]✓[/bold green] Found {len(active_runs)} active run(s):")
                    for run in active_runs:
                        console.print(f"  • [cyan]{run.name}[/cyan] (started: {run.created_at})")
                        console.print(f"    URL: {run.url}")
                else:
                    console.print("[bold yellow]NOTE: No active runs found for this project.[/bold yellow]")
                    
                    # Check for recent runs
                    recent_runs = list(runs)[:5]  # Get up to 5 recent runs
                    if recent_runs:
                        console.print("[bold]Recent runs:[/bold]")
                        for run in recent_runs:
                            console.print(f"  • [cyan]{run.name}[/cyan] ({run.state}, created: {run.created_at})")
                            console.print(f"    URL: {run.url}")
            except Exception as e:
                console.print(f"[bold yellow]WARNING: Could not retrieve runs: {str(e)}[/bold yellow]")
                
            return True
        except Exception as e:
            console.print(f"[bold yellow]WARNING: Error checking active runs: {str(e)}[/bold yellow]")
            return False
    except ImportError:
        console.print("[bold red]ERROR: WandB not installed. Cannot check active runs.[/bold red]")
        return False

def force_sync():
    """Force sync local WandB files."""
    console.rule("[bold]Forcing WandB Sync[/bold]")
    
    try:
        import wandb
        console.print("Attempting to sync local runs...")
        result = wandb.sync()
        console.print(f"[bold green]✓[/bold green] Sync completed. Result: {result}")
        return True
    except ImportError:
        console.print("[bold red]ERROR: WandB not installed. Cannot force sync.[/bold red]")
        return False
    except Exception as e:
        console.print(f"[bold red]ERROR: {str(e)}[/bold red]")
        return False

def check_wandb_directories():
    """Check for WandB local directories."""
    console.rule("[bold]Checking Local WandB Files[/bold]")
    
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        console.print("[bold yellow]NOTE: No 'wandb' directory found.[/bold yellow]")
        console.print("This may indicate that no WandB runs have been initiated yet.")
        return True
    
    console.print(f"[bold green]✓[/bold green] Found local WandB directory: {wandb_dir}")
    
    # Check for run files
    run_dirs = list(wandb_dir.glob("run-*"))
    if not run_dirs:
        console.print("[bold yellow]NOTE: No run directories found.[/bold yellow]")
        return True
    
    console.print(f"[bold green]✓[/bold green] Found {len(run_dirs)} local run directories.")
    
    # Check for offline runs
    offline_runs = list(wandb_dir.glob("offline-run-*"))
    if offline_runs:
        console.print(f"[bold yellow]NOTE: Found {len(offline_runs)} offline runs that need syncing.[/bold yellow]")
        console.print("You can sync these runs by running: wandb sync wandb/offline-run-*")
    
    return True

def main():
    """Main function to check WandB status."""
    parser = argparse.ArgumentParser(description="Check WandB integration status")
    parser.add_argument("--force-sync", action="store_true", help="Force sync local WandB files")
    args = parser.parse_args()
    
    console.print("[bold]WandB Status Check[/bold]")
    console.print("This tool checks your Weights & Biases configuration to ensure proper metric tracking.")
    
    # Run checks
    env_check = check_environment()
    install_check = check_wandb_installation()
    config_check = check_config_files()
    dir_check = check_wandb_directories()
    run_check = check_active_runs()
    
    # Summary
    console.rule("[bold]Summary[/bold]")
    checks = {
        "Environment": env_check,
        "Installation": install_check,
        "Config Files": config_check,
        "Local Files": dir_check,
        "Active Runs": run_check
    }
    
    for name, result in checks.items():
        status = "[bold green]PASS[/bold green]" if result else "[bold red]FAIL[/bold red]"
        console.print(f"{name}: {status}")
    
    if all(checks.values()):
        console.print("\n[bold green]All checks passed! WandB integration looks good.[/bold green]")
    else:
        console.print("\n[bold yellow]Some checks failed. Please fix the issues above.[/bold yellow]")
    
    # Force sync if requested
    if args.force_sync:
        force_sync()
    
    # Next steps
    console.rule("[bold]Next Steps[/bold]")
    if not env_check:
        console.print("1. Set up your WandB API key in the .env file")
    if not install_check:
        console.print("1. Install WandB: pip install wandb")
        console.print("2. Run wandb login and follow the instructions")
    
    if not all(checks.values()):
        console.print("\nAfter addressing these issues, run this script again to verify fixes.")
    else:
        console.print("Your WandB integration is properly configured.")
        console.print("To view your runs, visit: https://wandb.ai")

if __name__ == "__main__":
    main()
