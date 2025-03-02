"""
Fix WandB Integration and Continue Training

This script:
1. Fixes WandB integration issues
2. Checks the status of current fine-tuning processes
3. Provides options to continue training with proper logging
"""

import os
import sys
import logging
import subprocess
import json
import time
import argparse
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich import print

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def install_requirements():
    """Install required packages for WandB integration."""
    console.rule("[bold]Installing Required Packages[/bold]")
    
    requirements = ["wandb", "python-dotenv"]
    
    for package in requirements:
        try:
            console.print(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"[bold green]✓[/bold green] Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]ERROR: Failed to install {package}[/bold red]")
            console.print(f"Error: {e.stderr}")
            return False
    
    return True

def check_current_processes():
    """Check if fine-tuning processes are already running."""
    console.rule("[bold]Checking Current Fine-Tuning Processes[/bold]")
    
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        finetune_processes = []
        for line in result.stdout.splitlines():
            if "finetune_model.py" in line and "python" in line.lower():
                finetune_processes.append(line)
        
        if finetune_processes:
            console.print(f"[bold yellow]Found {len(finetune_processes)} running fine-tuning processes:[/bold yellow]")
            for i, process in enumerate(finetune_processes, 1):
                parts = process.split()
                pid = parts[1] if len(parts) > 1 else "Unknown"
                start_time = parts[9] if len(parts) > 9 else "Unknown"
                console.print(f"{i}. PID: {pid}, Started: {start_time}")
            
            return finetune_processes
        else:
            console.print("[bold green]No fine-tuning processes currently running.[/bold green]")
            return []
    except Exception as e:
        console.print(f"[bold red]ERROR: Failed to check processes: {str(e)}[/bold red]")
        return []

def stop_processes(processes):
    """Stop running fine-tuning processes."""
    if not processes:
        return True
    
    should_stop = Confirm.ask("[bold yellow]Do you want to stop existing fine-tuning processes?[/bold yellow]")
    if not should_stop:
        console.print("[bold yellow]Keeping existing processes running.[/bold yellow]")
        console.print("Note: WandB integration might not work correctly until these processes are restarted.")
        return False
    
    console.rule("[bold]Stopping Fine-Tuning Processes[/bold]")
    
    for process in processes:
        try:
            parts = process.split()
            pid = parts[1] if len(parts) > 1 else None
            
            if pid:
                console.print(f"Stopping process with PID {pid}...")
                subprocess.run(["kill", pid], check=True)
                console.print(f"[bold green]✓[/bold green] Successfully stopped process {pid}")
        except Exception as e:
            console.print(f"[bold red]ERROR: Failed to stop process: {str(e)}[/bold red]")
            return False
    
    # Verify all processes are stopped
    time.sleep(2)  # Give processes time to shut down
    remaining = check_current_processes()
    if remaining:
        console.print("[bold yellow]WARNING: Some processes are still running.[/bold yellow]")
        return False
    
    return True

def check_wandb_setup():
    """Check and setup WandB integration."""
    console.rule("[bold]Checking WandB Setup[/bold]")
    
    try:
        # Run the check_wandb_status.py script
        console.print("Running WandB status check...")
        result = subprocess.run(
            [sys.executable, "check_wandb_status.py"],
            capture_output=True,
            text=True
        )
        
        # Check if the script found any issues
        if "All checks passed!" in result.stdout:
            console.print("[bold green]✓[/bold green] WandB setup is correct.")
            return True
        else:
            console.print("[bold yellow]WandB setup requires attention.[/bold yellow]")
            console.print(result.stdout)
            
            # Ask for WandB API key if needed
            if "WANDB_API_KEY not found" in result.stdout or "WANDB_API_KEY appears to be a placeholder" in result.stdout:
                key = Prompt.ask(
                    "[bold]Enter your WandB API key[/bold] (or press Enter to use offline mode)",
                    password=True,
                    default=""
                )
                
                if key:
                    # Update .env file with the provided key
                    env_path = Path(".env")
                    if env_path.exists():
                        env_content = env_path.read_text()
                        if "WANDB_API_KEY=" in env_content:
                            new_content = env_content.replace(
                                "WANDB_API_KEY=your_api_key_here", 
                                f"WANDB_API_KEY={key}"
                            ).replace(
                                "WANDB_API_KEY=", 
                                f"WANDB_API_KEY={key}"
                            )
                            env_path.write_text(new_content)
                        else:
                            with open(env_path, "a") as f:
                                f.write(f"\nWANDB_API_KEY={key}\n")
                        console.print("[bold green]✓[/bold green] Updated .env file with WandB API key.")
                    else:
                        with open(env_path, "w") as f:
                            f.write(f"# Environment variables\nWANDB_API_KEY={key}\n")
                        console.print("[bold green]✓[/bold green] Created .env file with WandB API key.")
                else:
                    # Set to offline mode
                    env_path = Path(".env")
                    if env_path.exists():
                        env_content = env_path.read_text()
                        if "WANDB_MODE=" not in env_content:
                            with open(env_path, "a") as f:
                                f.write("\nWANDB_MODE=offline\n")
                        console.print("[bold yellow]Set WandB to offline mode in .env file.[/bold yellow]")
                    else:
                        with open(env_path, "w") as f:
                            f.write("# Environment variables\nWANDB_MODE=offline\n")
                        console.print("[bold yellow]Created .env file with WandB in offline mode.[/bold yellow]")
            
            return False
    except Exception as e:
        console.print(f"[bold red]ERROR: Failed to check WandB setup: {str(e)}[/bold red]")
        return False

def restart_training():
    """Restart the fine-tuning process with proper WandB integration."""
    console.rule("[bold]Restarting Fine-Tuning[/bold]")
    
    should_restart = Confirm.ask("[bold]Do you want to restart fine-tuning with proper WandB integration?[/bold]")
    if not should_restart:
        console.print("[bold yellow]Not restarting fine-tuning. You can manually run it later.[/bold yellow]")
        return
    
    # Check if we have python in the venv
    venv_python = Path("venv/bin/python")
    if not venv_python.exists():
        console.print("[bold red]ERROR: Virtual environment not found at venv/bin/python[/bold red]")
        console.print("Please check your virtual environment setup.")
        return
    
    try:
        console.print("[bold]Starting fine-tuning process...[/bold]")
        process = subprocess.Popen(
            ["venv/bin/python", "finetune_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        console.print(f"[bold green]✓[/bold green] Started fine-tuning process with PID {process.pid}")
        console.print("[bold]Showing initial output (press Ctrl+C to stop viewing):[/bold]")
        
        try:
            for i, line in enumerate(iter(process.stdout.readline, '')):
                print(line, end='')
                if i > 20:  # Show first 20 lines then stop
                    console.print("...")
                    console.print("[bold]Process continues running in the background.[/bold]")
                    break
        except KeyboardInterrupt:
            console.print("\n[bold]Stopped viewing output.[/bold]")
            console.print("[bold]Process continues running in the background.[/bold]")
        
        console.print("\n[bold green]Fine-tuning process is now running in the background.[/bold green]")
        console.print(f"You can monitor it with: ps aux | grep {process.pid}")
        console.print("To view logs, check: tail -f logs/finetune.log")
        console.print("To check WandB status later: python check_wandb_status.py")
    except Exception as e:
        console.print(f"[bold red]ERROR: Failed to restart training: {str(e)}[/bold red]")

def main():
    """Main function to fix WandB integration and continue training."""
    parser = argparse.ArgumentParser(description="Fix WandB integration and continue training")
    args = parser.parse_args()
    
    console.print("[bold]Fix WandB Integration and Continue Training[/bold]")
    
    # Step 1: Install required packages
    install_success = install_requirements()
    if not install_success:
        console.print("[bold red]Failed to install required packages. Exiting.[/bold red]")
        return
    
    # Step 2: Check current processes
    processes = check_current_processes()
    
    # Step 3: Stop processes if needed and user confirms
    if processes:
        stop_success = stop_processes(processes)
        if not stop_success and Confirm.ask("[bold yellow]Continue anyway?[/bold yellow]", default=False) is False:
            console.print("[bold]Exiting without making changes.[/bold]")
            return
    
    # Step 4: Check and fix WandB setup
    wandb_ready = check_wandb_setup()
    if not wandb_ready:
        console.print("[bold yellow]WandB setup is not optimal but we can continue.[/bold yellow]")
    
    # Step 5: Restart training if needed
    restart_training()
    
    console.rule("[bold]Process Complete[/bold]")
    console.print("You can now monitor your training via WandB or check the local logs.")

if __name__ == "__main__":
    main()
