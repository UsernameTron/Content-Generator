#!/usr/bin/env python3
"""
Overnight Training Runner for C. Pete Connor Model

This script handles the overnight training process with monitoring, 
logging, and error handling to ensure continuous training without interruptions.
"""

import os
import sys
import time
import json
import logging
import traceback
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("overnight_training")
console = Console()

def setup_wandb_monitoring(config_path):
    """
    Setup Weights & Biases monitoring for the training run.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        wandb_run: The wandb run object
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        wandb_config = config.get("wandb_config", {})
        
        # Initialize Weights & Biases run for monitoring
        logger.info("Initializing Weights & Biases monitoring")
        wandb_run = wandb.init(
            project=wandb_config.get("project", "pete-connor-model"),
            name=wandb_config.get("name", f"overnight-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            tags=wandb_config.get("tags", ["overnight-training", "pete-connor-model"]),
            notes=wandb_config.get("notes", "Overnight continuous training run"),
            config=config,
            monitor_gym=False,
            save_code=True
        )
        
        logger.info(f"W&B Dashboard URL: {wandb_run.get_url()}")
        return wandb_run
    
    except Exception as e:
        logger.error(f"Error setting up W&B: {str(e)}")
        return None

def check_system_resources():
    """
    Check if system resources are available for training.
    
    Returns:
        bool: True if resources are available, False otherwise
    """
    # This is a simple check - you may want to add more sophisticated checks
    import psutil
    
    # Check memory usage
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 90:
        logger.warning(f"High memory usage detected: {memory_usage}%. Training may be unstable.")
    
    # Check disk space
    disk_usage = psutil.disk_usage('/').percent
    if disk_usage > 95:
        logger.warning(f"Low disk space detected: {disk_usage}% used. Training may run out of disk space.")
    
    # All checks passed
    return True

def monitor_process(process, wandb_run=None):
    """
    Monitor a training process and report status.
    
    Args:
        process: The subprocess to monitor
        wandb_run: The wandb run object for logging
    
    Returns:
        int: The exit code of the process
    """
    console.print("[bold green]Training in progress...")
    start_time = time.time()
    last_update = start_time
    
    while process.poll() is None:
        # Update status every 30 seconds
        current_time = time.time()
        if current_time - last_update > 30:
            elapsed = current_time - start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            status_msg = f"[green]Training in progress... Elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s"
            console.print(status_msg)
            
            # Log to W&B if available
            if wandb_run is not None:
                wandb_run.log({
                    "training_time_hours": elapsed / 3600,
                    "heartbeat": True
                })
            
            last_update = current_time
        
        # Sleep to reduce CPU usage from polling
        time.sleep(1)
    
    return process.returncode

def run_training(config_path):
    """
    Run the training process with monitoring and error handling.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        bool: True if training was successful, False otherwise
    """
    logger.info("Starting overnight training process")
    
    # Check system resources
    if not check_system_resources():
        logger.error("System resources check failed. Training aborted.")
        return False
    
    # Setup W&B monitoring
    wandb_run = setup_wandb_monitoring(config_path)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create training log file
    log_file = logs_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    try:
        # Run the training script as a subprocess
        logger.info(f"Starting training process with config: {config_path}")
        logger.info(f"Logs will be written to: {log_file}")
        
        with open(log_file, "w") as f:
            # Include a header in the log file
            f.write(f"=== C. Pete Connor Model Training Log - Started at {datetime.now().isoformat()} ===\n\n")
            
            # Start the training process
            cmd = [sys.executable, "finetune_model.py", "--config", config_path]
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor the process
            exit_code = monitor_process(process, wandb_run)
            
            # Check the result
            if exit_code == 0:
                success_message = "Training completed successfully!"
                logger.info(success_message)
                f.write(f"\n\n=== {success_message} Finished at {datetime.now().isoformat()} ===\n")
                
                if wandb_run is not None:
                    wandb.alert(
                        title="Training Complete",
                        text=success_message,
                        level=wandb.AlertLevel.INFO
                    )
                
                return True
            else:
                error_message = f"Training process exited with code {exit_code}"
                logger.error(error_message)
                f.write(f"\n\n=== ERROR: {error_message} at {datetime.now().isoformat()} ===\n")
                
                if wandb_run is not None:
                    wandb.alert(
                        title="Training Error",
                        text=error_message,
                        level=wandb.AlertLevel.ERROR
                    )
                
                return False
    
    except KeyboardInterrupt:
        logger.warning("Training process interrupted by user")
        if wandb_run is not None:
            wandb.alert(
                title="Training Interrupted",
                text="Training process was interrupted by the user",
                level=wandb.AlertLevel.WARNING
            )
        return False
    
    except Exception as e:
        error_message = f"Error running training: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        if wandb_run is not None:
            wandb.alert(
                title="Training Error",
                text=error_message,
                level=wandb.AlertLevel.ERROR
            )
        
        return False
    
    finally:
        # Clean up wandb
        if wandb_run is not None:
            wandb_run.finish()

def main():
    parser = argparse.ArgumentParser(description="Run overnight training for C. Pete Connor model")
    parser.add_argument("--config", type=str, default="finetune_config.json", help="Path to the training configuration file")
    args = parser.parse_args()
    
    console.rule("[bold green]C. Pete Connor Model - Overnight Training")
    console.print("[bold green]Preparing overnight training...")
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[bold red]Error: Config file {config_path} does not exist!")
        return 1
    
    # Run the training
    success = run_training(str(config_path))
    
    if success:
        console.rule("[bold green]Training Completed Successfully!")
    else:
        console.rule("[bold red]Training Completed with Errors")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
