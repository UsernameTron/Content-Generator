#!/usr/bin/env python3
"""
Training Status Check Utility for C. Pete Connor Model

This script checks the status of the fine-tuning process and provides insights on:
1. Training progress and estimated completion time
2. Current loss values and learning rates
3. Memory usage and potential issues
4. Weights & Biases integration status
"""

import os
import re
import time
import glob
import json
import psutil
import argparse
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

# Initialize rich console
console = Console()

def get_training_log_file():
    """Find the most recent training log file"""
    log_files = glob.glob("logs/training_*.log")
    if not log_files:
        return None
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files[0]

def parse_training_logs(log_file):
    """Parse training logs to extract metrics"""
    if not log_file or not os.path.exists(log_file):
        return None
    
    metrics = {
        "start_time": None,
        "last_update": None,
        "current_step": 0,
        "total_steps": 0,
        "current_loss": None,
        "avg_loss": None,
        "learning_rate": None,
        "batch_size": None,
        "grad_accum_steps": None,
        "memory_usage": None,
        "errors": []
    }
    
    # Regular expressions for extracting information
    start_time_pattern = r"\[(.*?)\].*Starting training"
    step_pattern = r"Step\s+(\d+)/(\d+)"
    loss_pattern = r"loss\s*=\s*([\d\.]+)"
    avg_loss_pattern = r"avg_loss\s*=\s*([\d\.]+)"
    lr_pattern = r"learning_rate\s*=\s*([\d\.e\-]+)"
    memory_pattern = r"memory usage: ([\d\.]+)GB"
    error_pattern = r"ERROR.*?:(.*)"
    config_pattern = r"Training configuration: (.*)"
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
        # Extract start time
        for line in lines:
            start_match = re.search(start_time_pattern, line)
            if start_match:
                metrics["start_time"] = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
                break
        
        # Process the last 100 lines for recent info
        for line in lines[-100:]:
            # Extract steps
            step_match = re.search(step_pattern, line)
            if step_match:
                metrics["current_step"] = int(step_match.group(1))
                metrics["total_steps"] = int(step_match.group(2))
                
                # Update last update time from this line
                time_match = re.search(r"\[(.*?)\]", line)
                if time_match:
                    try:
                        metrics["last_update"] = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass
            
            # Extract loss
            loss_match = re.search(loss_pattern, line)
            if loss_match:
                metrics["current_loss"] = float(loss_match.group(1))
            
            # Extract average loss
            avg_loss_match = re.search(avg_loss_pattern, line)
            if avg_loss_match:
                metrics["avg_loss"] = float(avg_loss_match.group(1))
            
            # Extract learning rate
            lr_match = re.search(lr_pattern, line)
            if lr_match:
                metrics["learning_rate"] = float(lr_match.group(1))
            
            # Extract memory usage
            memory_match = re.search(memory_pattern, line)
            if memory_match:
                metrics["memory_usage"] = float(memory_match.group(1))
            
            # Extract configuration
            config_match = re.search(config_pattern, line)
            if config_match:
                try:
                    config = json.loads(config_match.group(1))
                    metrics["batch_size"] = config.get("batch_size")
                    metrics["grad_accum_steps"] = config.get("gradient_accumulation_steps")
                except:
                    pass
        
        # Extract errors (from the entire log)
        for line in lines:
            error_match = re.search(error_pattern, line)
            if error_match:
                metrics["errors"].append(error_match.group(1).strip())
    
    return metrics

def check_checkpoints():
    """Check existing checkpoints and their timestamps"""
    checkpoint_dirs = glob.glob("outputs/finetune/checkpoint-*")
    
    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        try:
            # Extract step number from directory name
            step = int(os.path.basename(checkpoint_dir).split("-")[1])
            # Get timestamp
            timestamp = datetime.fromtimestamp(os.path.getmtime(checkpoint_dir))
            
            checkpoints.append({
                "step": step,
                "timestamp": timestamp,
                "path": checkpoint_dir
            })
        except (ValueError, IndexError):
            continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x["step"])
    
    return checkpoints

def check_wandb_integration():
    """Check if W&B integration is active"""
    wandb_dir = os.path.join(os.getcwd(), "wandb")
    has_wandb = os.path.isdir(wandb_dir)
    
    wandb_run = None
    if has_wandb:
        run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
        if run_dirs:
            # Get the latest run
            run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            wandb_run = os.path.basename(run_dirs[0])
    
    return {
        "active": has_wandb,
        "run_id": wandb_run
    }

def estimate_completion_time(metrics, checkpoints):
    """Estimate when training will complete"""
    if not metrics or not metrics["start_time"] or not metrics["last_update"]:
        return None
    
    if metrics["current_step"] == 0 or metrics["total_steps"] == 0:
        return None
    
    # Calculate time per step
    if len(checkpoints) >= 2:
        # Use checkpoint timestamps for more accurate calculation
        first_checkpoint = checkpoints[0]
        last_checkpoint = checkpoints[-1]
        steps_diff = last_checkpoint["step"] - first_checkpoint["step"]
        time_diff = (last_checkpoint["timestamp"] - first_checkpoint["timestamp"]).total_seconds()
        
        if steps_diff > 0:
            time_per_step = time_diff / steps_diff
        else:
            return None
    else:
        # Use start time and last update time
        elapsed_time = (metrics["last_update"] - metrics["start_time"]).total_seconds()
        time_per_step = elapsed_time / metrics["current_step"] if metrics["current_step"] > 0 else 0
    
    # Calculate remaining time
    remaining_steps = metrics["total_steps"] - metrics["current_step"]
    remaining_seconds = remaining_steps * time_per_step
    
    # Calculate estimated completion time
    completion_time = metrics["last_update"] + timedelta(seconds=remaining_seconds)
    
    return {
        "time_per_step": time_per_step,
        "remaining_steps": remaining_steps,
        "remaining_time": timedelta(seconds=remaining_seconds),
        "completion_time": completion_time
    }

def get_system_status():
    """Get current system status"""
    memory = psutil.virtual_memory()
    
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "memory_percent": memory.percent,
        "process_count": len(psutil.Process().children())
    }

def check_training_process():
    """Check if training process is running"""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = process.info.get('cmdline', [])
        if cmdline and len(cmdline) > 1:
            cmdline_str = ' '.join(cmdline)
            if 'apple_silicon_training.py' in cmdline_str:
                return process.info['pid']
    return None

def display_training_status(metrics, checkpoints, wandb_status, time_estimate, system_status, is_running):
    """Display training status in a nice format"""
    console.rule("[bold blue]C. Pete Connor Model - Training Status Check")
    
    # Display overall status
    if is_running:
        rprint(Panel("[bold green]Training process is RUNNING", title="Process Status"))
    else:
        if metrics and metrics["current_step"] >= metrics["total_steps"]:
            rprint(Panel("[bold blue]Training has COMPLETED", title="Process Status"))
        else:
            rprint(Panel("[bold red]Training process is NOT RUNNING", title="Process Status"))
    
    # Display training progress
    if metrics:
        progress_table = Table(title="Training Progress")
        progress_table.add_column("Metric", style="cyan")
        progress_table.add_column("Value", style="green")
        
        progress_table.add_row("Current Step", f"{metrics['current_step']} / {metrics['total_steps']}")
        progress_table.add_row("Progress", f"{metrics['current_step'] / metrics['total_steps'] * 100:.1f}%" if metrics['total_steps'] > 0 else "Unknown")
        progress_table.add_row("Current Loss", f"{metrics['current_loss']:.4f}" if metrics['current_loss'] is not None else "N/A")
        progress_table.add_row("Average Loss", f"{metrics['avg_loss']:.4f}" if metrics['avg_loss'] is not None else "N/A")
        progress_table.add_row("Learning Rate", f"{metrics['learning_rate']:.6f}" if metrics['learning_rate'] is not None else "N/A")
        
        if metrics['start_time']:
            progress_table.add_row("Start Time", metrics['start_time'].strftime("%Y-%m-%d %H:%M:%S"))
        if metrics['last_update']:
            progress_table.add_row("Last Update", metrics['last_update'].strftime("%Y-%m-%d %H:%M:%S"))
        
        progress_table.add_row("Batch Size", str(metrics['batch_size']) if metrics['batch_size'] else "N/A")
        progress_table.add_row("Gradient Accumulation", str(metrics['grad_accum_steps']) if metrics['grad_accum_steps'] else "N/A")
        progress_table.add_row("Reported Memory Usage", f"{metrics['memory_usage']:.2f} GB" if metrics['memory_usage'] else "N/A")
        
        console.print(progress_table)
        
        # Show progress bar
        if metrics['current_step'] > 0 and metrics['total_steps'] > 0:
            with Progress(TextColumn("[progress.description]{task.description}"),
                         BarColumn(),
                         TextColumn("{task.percentage:>3.0f}%"),
                         TimeElapsedColumn()) as progress:
                progress.add_task("[cyan]Overall Progress", total=metrics['total_steps'], completed=metrics['current_step'])
    
    # Display time estimate
    if time_estimate:
        time_table = Table(title="Time Estimate")
        time_table.add_column("Metric", style="cyan")
        time_table.add_column("Value", style="green")
        
        time_table.add_row("Time per Step", f"{time_estimate['time_per_step']:.2f} seconds")
        time_table.add_row("Remaining Steps", str(time_estimate['remaining_steps']))
        time_table.add_row("Estimated Remaining Time", str(time_estimate['remaining_time']))
        time_table.add_row("Estimated Completion", time_estimate['completion_time'].strftime("%Y-%m-%d %H:%M:%S"))
        
        console.print(time_table)
    
    # Display checkpoint information
    if checkpoints:
        checkpoint_table = Table(title="Checkpoints")
        checkpoint_table.add_column("Step", style="cyan")
        checkpoint_table.add_column("Timestamp", style="green")
        checkpoint_table.add_column("Path", style="blue")
        
        # Only show latest 5 checkpoints if there are many
        display_checkpoints = checkpoints[-5:] if len(checkpoints) > 5 else checkpoints
        
        for checkpoint in display_checkpoints:
            checkpoint_table.add_row(
                str(checkpoint["step"]),
                checkpoint["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                checkpoint["path"]
            )
        
        console.print(checkpoint_table)
        console.print(f"Total checkpoints: {len(checkpoints)}")
    
    # Display Weights & Biases status
    wandb_panel = Panel(
        f"[bold green]Active: {wandb_status['active']}[/bold green]\n" +
        (f"Run ID: {wandb_status['run_id']}" if wandb_status['run_id'] else "No active run found"),
        title="Weights & Biases Integration"
    )
    console.print(wandb_panel)
    
    # Display system status
    system_table = Table(title="System Status")
    system_table.add_column("Metric", style="cyan")
    system_table.add_column("Value", style="green")
    
    system_table.add_row("CPU Usage", f"{system_status['cpu_percent']}%")
    system_table.add_row("Memory Usage", f"{system_status['memory_used_gb']:.2f} GB / {system_status['memory_total_gb']:.2f} GB ({system_status['memory_percent']}%)")
    
    # Add warning if memory usage is high
    if system_status['memory_percent'] > 85:
        system_table.add_row("Memory Warning", "[bold red]High memory usage detected! Risk of OOM error.[/bold red]")
    elif system_status['memory_percent'] > 70:
        system_table.add_row("Memory Warning", "[bold yellow]Elevated memory usage, monitor for issues.[/bold yellow]")
    
    system_table.add_row("Training Processes", str(system_status['process_count']))
    
    console.print(system_table)
    
    # Display errors if any
    if metrics and metrics["errors"]:
        console.print("[bold red]Errors Detected:[/bold red]")
        for i, error in enumerate(metrics["errors"][-5:], 1):  # Show last 5 errors
            console.print(f"{i}. {error}")
        
        if len(metrics["errors"]) > 5:
            console.print(f"...and {len(metrics['errors']) - 5} more errors. Check the log file for details.")

def main():
    parser = argparse.ArgumentParser(description="Check C. Pete Connor model training status")
    parser.add_argument("--log-file", type=str, help="Specific log file to analyze (default: most recent)")
    args = parser.parse_args()
    
    # Get log file
    log_file = args.log_file if args.log_file else get_training_log_file()
    
    if not log_file:
        console.print("[bold red]No training log files found. Has training started?[/bold red]")
        return
    
    # Check if training process is running
    training_pid = check_training_process()
    
    # Parse logs
    metrics = parse_training_logs(log_file)
    
    # Check checkpoints
    checkpoints = check_checkpoints()
    
    # Check W&B integration
    wandb_status = check_wandb_integration()
    
    # Get system status
    system_status = get_system_status()
    
    # Calculate estimated completion time
    time_estimate = estimate_completion_time(metrics, checkpoints)
    
    # Display status
    display_training_status(metrics, checkpoints, wandb_status, time_estimate, system_status, training_pid is not None)
    
    # Return success
    return 0

if __name__ == "__main__":
    main()
