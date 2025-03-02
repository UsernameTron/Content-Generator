"""
Log Capture Utility

This script captures logs from the fine-tuning process and saves them
to a log file for later analysis.
"""

import os
import time
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress
from rich import print

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def get_process_id(process_name):
    """Get process IDs for a given process name."""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        process_ids = []
        for line in result.stdout.splitlines():
            if process_name in line and "python" in line.lower():
                parts = line.split()
                pid = parts[1] if len(parts) > 1 else None
                if pid:
                    process_ids.append((pid, line))
        
        return process_ids
    except Exception as e:
        logger.error(f"Error getting process IDs: {str(e)}")
        return []

def capture_process_output(pid, duration=3600, interval=5):
    """Capture output from a process for the specified duration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"finetune_pid{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    console.print(f"[bold]Capturing output from process {pid} for {duration/60:.1f} minutes[/bold]")
    console.print(f"Log file: {log_file}")
    
    end_time = time.time() + duration
    captures = 0
    
    with Progress() as progress:
        task = progress.add_task("[green]Capturing logs...", total=duration)
        
        while time.time() < end_time:
            # Update progress
            progress.update(task, completed=min(time.time() - (end_time - duration), duration))
            
            try:
                # Get process status and output
                result = subprocess.run(
                    ["ps", "aux", "-p", pid], 
                    capture_output=True, 
                    text=True
                )
                
                if pid not in result.stdout:
                    logger.warning(f"Process {pid} is no longer running.")
                    break
                
                # Get process output using ps command
                output_result = subprocess.run(
                    ["ps", "u", "-p", pid], 
                    capture_output=True, 
                    text=True
                )
                
                # Append to log file
                with open(log_file, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n--- Capture at {timestamp} ---\n")
                    f.write(output_result.stdout)
                    f.write("\n")
                
                captures += 1
            except Exception as e:
                logger.error(f"Error capturing output: {str(e)}")
            
            time.sleep(interval)
    
    console.print(f"[bold green]Capture complete. Made {captures} captures over {duration/60:.1f} minutes.[/bold green]")
    console.print(f"Log saved to: {log_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Capture logs from fine-tuning process")
    parser.add_argument("--duration", type=int, default=3600, help="Duration to capture logs (seconds)")
    parser.add_argument("--interval", type=int, default=5, help="Interval between captures (seconds)")
    parser.add_argument("--pid", type=str, help="Specific process ID to monitor")
    args = parser.parse_args()
    
    console.rule("[bold]Fine-Tuning Log Capture[/bold]")
    
    if args.pid:
        process_ids = [(args.pid, f"User-specified process {args.pid}")]
    else:
        # Look for finetune_model.py processes
        process_ids = get_process_id("finetune_model.py")
    
    if not process_ids:
        console.print("[bold red]No fine-tuning processes found.[/bold red]")
        return
    
    console.print(f"[bold]Found {len(process_ids)} fine-tuning processes:[/bold]")
    for i, (pid, details) in enumerate(process_ids, 1):
        console.print(f"{i}. PID: {pid}")
        console.print(f"   {details}")
    
    if len(process_ids) == 1:
        selected_pid = process_ids[0][0]
    else:
        # If multiple processes, let user select one
        selection = input("\nEnter process number to monitor (or press Enter for the first one): ")
        try:
            index = int(selection) - 1 if selection.strip() else 0
            selected_pid = process_ids[index][0]
        except (ValueError, IndexError):
            console.print("[bold red]Invalid selection. Using the first process.[/bold red]")
            selected_pid = process_ids[0][0]
    
    capture_process_output(selected_pid, args.duration, args.interval)

if __name__ == "__main__":
    main()
