"""
WandB Authentication Fix

This script helps fix authentication issues with Weights & Biases.
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

def check_wandb_installation():
    """Check if wandb is installed."""
    try:
        import wandb
        version = wandb.__version__
        console.print(f"✓ WandB is installed (version: {version})")
        return True
    except ImportError:
        console.print("[bold red]× WandB is not installed[/bold red]")
        return False

def install_wandb():
    """Install wandb package."""
    console.print("[bold]Installing wandb...[/bold]")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "wandb"], check=True)
        console.print("[bold green]✓ Successfully installed wandb[/bold green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]× Failed to install wandb: {e}[/bold red]")
        return False

def check_env_file():
    """Check if .env file exists and has WANDB_API_KEY."""
    env_path = Path(".env")
    if not env_path.exists():
        console.print("[bold yellow]× .env file not found[/bold yellow]")
        return False
    
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    
    if not api_key:
        console.print("[bold yellow]× WANDB_API_KEY not found in .env file[/bold yellow]")
        return False
    
    if api_key.startswith("YOUR_WANDB_API_KEY") or "<" in api_key or ">" in api_key:
        console.print("[bold yellow]× WANDB_API_KEY appears to be a placeholder[/bold yellow]")
        return False
    
    # Show masked API key for privacy
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    console.print(f"[bold green]✓ WANDB_API_KEY found: {masked_key}[/bold green]")
    return True

def wandb_login():
    """Run wandb login command."""
    try:
        import wandb
        
        # First try with API key from environment
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            try:
                console.print("[bold]Attempting to log in with API key from environment...[/bold]")
                wandb.login(key=api_key)
                console.print("[bold green]✓ Successfully logged in with API key from environment[/bold green]")
                return True
            except Exception as e:
                console.print(f"[bold yellow]× Failed to log in with API key from environment: {e}[/bold yellow]")
        
        # If that fails, try interactive login
        console.print("[bold]Starting interactive login...[/bold]")
        result = subprocess.run(["wandb", "login"], capture_output=True, text=True)
        
        if "Logged in" in result.stdout:
            console.print("[bold green]✓ Successfully logged in to wandb[/bold green]")
            
            # Try to extract API key from wandb/settings
            try:
                wandb_dir = Path.home() / ".config" / "wandb"
                if wandb_dir.exists():
                    settings_file = wandb_dir / "settings"
                    if settings_file.exists():
                        with open(settings_file, "r") as f:
                            settings = json.load(f)
                            if "api_key" in settings:
                                new_api_key = settings["api_key"]
                                # Update .env file with new API key
                                if Confirm.ask("Do you want to update your .env file with the new API key?"):
                                    dotenv_path = Path(".env")
                                    set_key(str(dotenv_path), "WANDB_API_KEY", new_api_key)
                                    console.print("[bold green]✓ Updated .env file with new API key[/bold green]")
            except Exception as e:
                console.print(f"[bold yellow]× Failed to update .env file: {e}[/bold yellow]")
            
            return True
        else:
            console.print(f"[bold red]× Failed to log in to wandb:[/bold red]\n{result.stderr}")
            return False
    except Exception as e:
        console.print(f"[bold red]× Error during login: {e}[/bold red]")
        return False

def update_api_key_in_env():
    """Update API key in .env file."""
    console.print(Panel("To fix your WandB authentication, you need to provide your API key.", 
                       title="WandB API Key Setup", 
                       subtitle="Follow instructions at https://wandb.ai/authorize"))
    
    console.print("[bold]1. Go to https://wandb.ai/authorize to get your API key[/bold]")
    console.print("[bold]2. Copy your API key and paste it below[/bold]")
    
    api_key = Prompt.ask("Enter your WandB API key", password=True)
    
    if not api_key:
        console.print("[bold red]× No API key provided[/bold red]")
        return False
    
    # Update .env file
    try:
        dotenv_path = Path(".env")
        if not dotenv_path.exists():
            with open(dotenv_path, "w") as f:
                f.write("# Environment Variables\n")
        
        set_key(str(dotenv_path), "WANDB_API_KEY", api_key)
        console.print("[bold green]✓ Updated .env file with new API key[/bold green]")
        
        # Reload environment variables
        load_dotenv(override=True)
        return True
    except Exception as e:
        console.print(f"[bold red]× Failed to update .env file: {e}[/bold red]")
        return False

def sync_offline_runs():
    """Sync offline runs to wandb server."""
    try:
        # Check for offline runs
        wandb_dir = Path("wandb")
        if not wandb_dir.exists():
            console.print("[bold yellow]No wandb directory found[/bold yellow]")
            return
        
        offline_runs = list(wandb_dir.glob("offline-run-*"))
        if not offline_runs:
            console.print("[bold yellow]No offline runs found[/bold yellow]")
            return
        
        console.print(f"[bold]Found {len(offline_runs)} offline runs[/bold]")
        
        if Confirm.ask("Do you want to sync these offline runs to the WandB server?"):
            console.print("[bold]Syncing offline runs...[/bold]")
            
            for run_dir in offline_runs:
                console.print(f"Syncing {run_dir.name}...")
                try:
                    result = subprocess.run(["wandb", "sync", str(run_dir)], 
                                         capture_output=True, 
                                         text=True)
                    
                    if "Synced" in result.stdout:
                        console.print(f"[bold green]✓ Successfully synced {run_dir.name}[/bold green]")
                    else:
                        console.print(f"[bold yellow]× Failed to sync {run_dir.name}:[/bold yellow]\n{result.stderr}")
                except Exception as e:
                    console.print(f"[bold red]× Error syncing {run_dir.name}: {e}[/bold red]")
            
            console.print("[bold green]Finished syncing offline runs[/bold green]")
    except Exception as e:
        console.print(f"[bold red]× Error syncing offline runs: {e}[/bold red]")

def verify_authentication():
    """Verify WandB authentication."""
    try:
        import wandb
        
        console.print("[bold]Verifying WandB authentication...[/bold]")
        
        api = wandb.Api()
        
        # Try to get user info
        try:
            user = api.viewer()
            console.print(f"[bold green]✓ Successfully authenticated as: {user['username']}[/bold green]")
            return True
        except Exception as e:
            console.print(f"[bold red]× Authentication verification failed: {e}[/bold red]")
            return False
    except Exception as e:
        console.print(f"[bold red]× Error verifying authentication: {e}[/bold red]")
        return False

def main():
    """Main function."""
    console.rule("[bold]WandB Authentication Fix[/bold]")
    
    # Check wandb installation
    is_installed = check_wandb_installation()
    if not is_installed:
        if Confirm.ask("Do you want to install wandb?"):
            is_installed = install_wandb()
            if not is_installed:
                console.print("[bold red]× Failed to install wandb. Exiting...[/bold red]")
                return
    
    # Check .env file
    env_valid = check_env_file()
    if not env_valid:
        if Confirm.ask("Do you want to update your API key in the .env file?"):
            env_valid = update_api_key_in_env()
    
    # Try to login
    console.rule("[bold]WandB Login[/bold]")
    login_successful = wandb_login()
    
    if login_successful:
        # Verify authentication
        auth_verified = verify_authentication()
        
        if auth_verified:
            # Sync offline runs if authenticated
            console.rule("[bold]Sync Offline Runs[/bold]")
            sync_offline_runs()
            
            console.print(Panel(
                "[bold green]WandB authentication has been successfully fixed![/bold green]\n"
                "You can now use WandB for tracking your fine-tuning process."
                "\n\nTo restart training with proper WandB integration, run:\n"
                "python fix_wandb_and_continue_training.py",
                title="Success",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[bold yellow]WandB authentication could not be verified.[/bold yellow]\n"
                "Please try again or check your API key.",
                title="Authentication Issue",
                border_style="yellow"
            ))
    else:
        console.print(Panel(
            "[bold red]Failed to log in to WandB.[/bold red]\n"
            "Please try again with a valid API key or visit https://wandb.ai/authorize to get a new one.",
            title="Login Failed",
            border_style="red"
        ))

if __name__ == "__main__":
    main()
