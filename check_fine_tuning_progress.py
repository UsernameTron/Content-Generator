"""
Fine-Tuning Progress Checker

This script checks the progress of the fine-tuning process by examining
the model output directory and provides a summary of training status.
"""

import os
import json
import time
import glob
import logging
from pathlib import Path
from datetime import datetime
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.logging import RichHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def check_process_status(process_name="finetune_model.py"):
    """Check if the fine-tuning process is running."""
    try:
        import subprocess
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        processes = []
        for line in result.stdout.splitlines():
            if process_name in line and "python" in line.lower():
                parts = line.split()
                pid = parts[1] if len(parts) > 1 else "Unknown"
                cpu = parts[2] if len(parts) > 2 else "Unknown"
                mem = parts[3] if len(parts) > 3 else "Unknown"
                start = parts[9] if len(parts) > 9 else "Unknown"
                
                processes.append({
                    "pid": pid,
                    "cpu": cpu,
                    "memory": mem,
                    "start_time": start
                })
        
        return processes
    except Exception as e:
        logger.error(f"Error checking process status: {str(e)}")
        return []

def check_checkpoint_files(output_dir):
    """Check for checkpoint files in the output directory."""
    try:
        # Check for different file patterns
        patterns = [
            "*.bin",           # Training args
            "*.safetensors",   # Model weights
            "*.json",          # Config files
            "*/checkpoint-*",  # Checkpoint directories
        ]
        
        files = {}
        for pattern in patterns:
            path_pattern = os.path.join(output_dir, pattern)
            matches = glob.glob(path_pattern)
            files[pattern] = matches
        
        return files
    except Exception as e:
        logger.error(f"Error checking checkpoint files: {str(e)}")
        return {}

def check_training_progress(output_dir):
    """Check training progress based on checkpoint directories."""
    try:
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        
        # Sort checkpoints by step number
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        
        checkpoints = []
        for checkpoint in checkpoint_dirs:
            step = int(checkpoint.split("-")[-1])
            
            # Check if this checkpoint has a completed model
            has_model = os.path.exists(os.path.join(checkpoint, "adapter_model.safetensors"))
            
            # Check if this checkpoint has training state
            has_training_state = os.path.exists(os.path.join(checkpoint, "trainer_state.json"))
            
            # Get trainer state if available
            trainer_state = {}
            if has_training_state:
                try:
                    with open(os.path.join(checkpoint, "trainer_state.json"), "r") as f:
                        trainer_state = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load trainer state: {str(e)}")
            
            # Get loss from trainer state if available
            loss = trainer_state.get("log_history", [{}])[-1].get("loss", "N/A")
            
            # Get timestamp if available
            timestamp = None
            if trainer_state.get("log_history"):
                for entry in reversed(trainer_state.get("log_history", [])):
                    if "epoch" in entry:
                        timestamp = entry.get("timestamp", None)
                        break
            
            checkpoints.append({
                "step": step,
                "directory": checkpoint,
                "has_model": has_model,
                "has_training_state": has_training_state,
                "loss": loss,
                "timestamp": timestamp
            })
        
        return checkpoints
    except Exception as e:
        logger.error(f"Error checking training progress: {str(e)}")
        return []

def check_final_model(output_dir):
    """Check if the final model exists and is complete."""
    try:
        final_dir = os.path.join(output_dir, "final")
        
        if not os.path.exists(final_dir):
            return {
                "exists": False,
                "message": "Final model directory does not exist"
            }
        
        # Check for essential files
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(final_dir, f))]
        
        if missing_files:
            return {
                "exists": True,
                "complete": False,
                "missing_files": missing_files,
                "message": f"Final model exists but is missing files: {', '.join(missing_files)}"
            }
        
        # Check size of adapter model
        adapter_size = os.path.getsize(os.path.join(final_dir, "adapter_model.safetensors"))
        
        return {
            "exists": True,
            "complete": True,
            "size_mb": adapter_size / (1024 * 1024),
            "directory": final_dir,
            "message": "Final model exists and appears complete"
        }
    except Exception as e:
        logger.error(f"Error checking final model: {str(e)}")
        return {"exists": False, "message": f"Error checking final model: {str(e)}"}

def format_time(timestamp):
    """Format timestamp for display."""
    if not timestamp:
        return "N/A"
    
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(timestamp)

def monitor_training(output_dir, interval=10, max_checks=30):
    """Monitor training progress over time."""
    console.rule("[bold]Monitoring Fine-Tuning Progress[/bold]")
    console.print(f"Monitoring output directory: {output_dir}")
    console.print(f"Checking every {interval} seconds for {max_checks} iterations")
    console.print("Press Ctrl+C to stop monitoring\n")
    
    try:
        for i in track(range(max_checks), description="Monitoring..."):
            # Check process status
            processes = check_process_status()
            
            if processes:
                console.print(f"[bold green]Fine-tuning process is running[/bold green]")
                for process in processes:
                    console.print(f"PID: {process['pid']}, CPU: {process['cpu']}%, Memory: {process['memory']}%")
            else:
                console.print("[bold yellow]No fine-tuning process found[/bold yellow]")
            
            # Check checkpoints
            checkpoints = check_training_progress(output_dir)
            
            if checkpoints:
                latest = checkpoints[-1]
                console.print(f"Latest checkpoint: Step {latest['step']}, Loss: {latest['loss']}")
                
                # Create a table of checkpoints
                if len(checkpoints) > 0:
                    table = Table(title="Training Checkpoints")
                    table.add_column("Step", style="cyan")
                    table.add_column("Loss", style="magenta")
                    table.add_column("Time", style="green")
                    table.add_column("Status", style="yellow")
                    
                    for cp in checkpoints[-5:]:  # Show last 5 checkpoints
                        status = "Complete" if cp["has_model"] else "Partial"
                        table.add_row(
                            str(cp["step"]),
                            str(cp["loss"]),
                            format_time(cp["timestamp"]),
                            status
                        )
                    
                    console.print(table)
            else:
                console.print("[bold yellow]No checkpoints found yet[/bold yellow]")
            
            # Check final model
            final_model = check_final_model(output_dir)
            if final_model["exists"] and final_model.get("complete", False):
                console.print(Panel(
                    f"[bold green]Final model is complete![/bold green]\nSize: {final_model.get('size_mb', 0):.2f} MB\nLocation: {final_model.get('directory', 'N/A')}",
                    title="Final Model Status",
                    border_style="green"
                ))
            elif final_model["exists"]:
                console.print(Panel(
                    f"[bold yellow]Final model exists but may be incomplete[/bold yellow]\n{final_model.get('message', '')}",
                    title="Final Model Status",
                    border_style="yellow"
                ))
            else:
                console.print(Panel(
                    "[bold yellow]Final model does not exist yet[/bold yellow]",
                    title="Final Model Status",
                    border_style="yellow"
                ))
            
            console.rule()
            
            # Wait before next check
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[bold]Monitoring stopped by user[/bold]")
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")

def main():
    """Main function to check fine-tuning progress."""
    parser = argparse.ArgumentParser(description="Check fine-tuning progress")
    parser.add_argument("--output-dir", default="./outputs/finetune", help="Output directory for fine-tuning")
    parser.add_argument("--monitor", action="store_true", help="Continuously monitor progress")
    parser.add_argument("--interval", type=int, default=10, help="Interval between checks when monitoring (seconds)")
    parser.add_argument("--max-checks", type=int, default=30, help="Maximum number of checks when monitoring")
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        console.print(f"[bold red]Output directory {output_dir} does not exist[/bold red]")
        return
    
    console.rule("[bold]Fine-Tuning Progress Check[/bold]")
    console.print(f"Checking output directory: {output_dir}")
    
    # Check process status
    processes = check_process_status()
    if processes:
        console.print(f"[bold green]Fine-tuning process is running[/bold green]")
        for process in processes:
            console.print(f"PID: {process['pid']}, CPU: {process['cpu']}%, Memory: {process['memory']}%, Started: {process['start_time']}")
    else:
        console.print("[bold yellow]No fine-tuning process found[/bold yellow]")
    
    # Check for checkpoint files
    files = check_checkpoint_files(output_dir)
    console.print("\n[bold]Checkpoint Files:[/bold]")
    for pattern, matches in files.items():
        if matches:
            console.print(f"Found {len(matches)} files matching {pattern}")
    
    # Check training progress
    checkpoints = check_training_progress(output_dir)
    if checkpoints:
        console.print(f"\n[bold]Found {len(checkpoints)} checkpoints[/bold]")
        
        table = Table(title="Training Checkpoints")
        table.add_column("Step", style="cyan")
        table.add_column("Loss", style="magenta")
        table.add_column("Time", style="green")
        table.add_column("Status", style="yellow")
        
        for checkpoint in checkpoints:
            status = "Complete" if checkpoint["has_model"] else "Partial"
            table.add_row(
                str(checkpoint["step"]),
                str(checkpoint["loss"]),
                format_time(checkpoint["timestamp"]),
                status
            )
        
        console.print(table)
        
        # Estimate progress
        if len(checkpoints) > 1:
            first_step = checkpoints[0]["step"]
            latest_step = checkpoints[-1]["step"]
            step_diff = latest_step - first_step
            
            # Get timestamp difference if available
            if checkpoints[0]["timestamp"] and checkpoints[-1]["timestamp"]:
                time_diff = checkpoints[-1]["timestamp"] - checkpoints[0]["timestamp"]
                time_per_step = time_diff / step_diff if step_diff > 0 else 0
                
                # Assuming max_steps from config is 1000
                max_steps = 500  # Based on your configuration
                steps_remaining = max_steps - latest_step
                estimated_time = steps_remaining * time_per_step
                
                console.print(f"\n[bold]Progress:[/bold] {latest_step}/{max_steps} steps ({latest_step/max_steps*100:.1f}%)")
                console.print(f"[bold]Estimated time remaining:[/bold] {estimated_time/60:.1f} minutes")
    else:
        console.print("\n[bold yellow]No checkpoints found[/bold yellow]")
    
    # Check final model
    final_model = check_final_model(output_dir)
    console.print("\n[bold]Final Model Status:[/bold]")
    if final_model["exists"] and final_model.get("complete", False):
        console.print(f"[bold green]Final model is complete![/bold green]")
        console.print(f"Size: {final_model.get('size_mb', 0):.2f} MB")
        console.print(f"Location: {final_model.get('directory', 'N/A')}")
    elif final_model["exists"]:
        console.print(f"[bold yellow]Final model exists but may be incomplete[/bold yellow]")
        console.print(final_model.get("message", ""))
    else:
        console.print(f"[bold yellow]Final model does not exist yet[/bold yellow]")
    
    # Monitor if requested
    if args.monitor:
        monitor_training(output_dir, args.interval, args.max_checks)

if __name__ == "__main__":
    main()
