"""
Test W&B Authentication and Logging

This script verifies W&B authentication and tests logging metrics.
"""

import os
import sys
import logging
import wandb
from rich.console import Console
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

def test_wandb_logging():
    """Test W&B authentication and logging."""
    # Load environment variables
    load_dotenv()
    
    console.rule("[bold]Testing W&B Authentication & Logging[/bold]")
    
    try:
        # Initialize W&B with test run
        wandb.init(
            project="pete-connor-cx-ai-expert",
            name="auth-test-run",
            notes="Testing W&B authentication and logging",
            tags=["test", "authentication"],
        )
        
        console.print("[bold green]✓ Successfully connected to W&B![/bold green]")
        
        # Log some test metrics
        for i in range(10):
            wandb.log({
                "test_metric": i * 2,
                "accuracy": 0.5 + i * 0.05,
                "loss": 1.0 - i * 0.1
            })
        
        console.print("[bold green]✓ Successfully logged test metrics![/bold green]")
        
        # Display run information
        run_url = wandb.run.get_url()
        console.print(f"[bold]View this run at:[/bold] {run_url}")
        
        # Finish the run
        wandb.finish()
        
        return True
    except Exception as e:
        console.print(f"[bold red]× Error testing W&B: {str(e)}[/bold red]")
        return False

if __name__ == "__main__":
    success = test_wandb_logging()
    if success:
        console.print("\n[bold green]✓ W&B authentication and logging test passed![/bold green]")
        console.print("[bold]Your W&B integration is working correctly![/bold]")
    else:
        console.print("\n[bold red]× W&B authentication and logging test failed.[/bold red]")
        console.print("[bold]Please check your W&B API key and configuration.[/bold]")
    
    sys.exit(0 if success else 1)
