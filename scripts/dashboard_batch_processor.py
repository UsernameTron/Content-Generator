#!/usr/bin/env python3
"""
Batch processor for healthcare learning dashboard.
Provides functionality for running multiple learning cycles sequentially.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.console import Console

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.demonstrate_continuous_learning import demonstrate_continuous_learning

# Configure logging
logger = logging.getLogger("batch-processor")
console = Console()

class BatchProcessor:
    """Batch processor for running multiple learning cycles."""
    
    def __init__(self, data_dir="data/healthcare"):
        """Initialize batch processor.
        
        Args:
            data_dir: Directory containing healthcare data
        """
        self.data_dir = Path(data_dir)
        self.history_path = self.data_dir / "learning_history.json"
        self.batch_history_path = self.data_dir / "batch_history.json"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize batch history if it doesn't exist
        if not self.batch_history_path.exists():
            with open(self.batch_history_path, 'w') as f:
                json.dump({"batches": []}, f, indent=2)
                
    def run_batch(self, cycles=5, examples_per_cycle=10, starting_accuracy=None, evaluation_frequency=1):
        """Run a batch of learning cycles.
        
        Args:
            cycles: Number of cycles to run
            examples_per_cycle: Number of examples to generate per cycle
            starting_accuracy: Starting accuracy (None to use auto-progression)
            evaluation_frequency: How often to perform full evaluation
            
        Returns:
            dict: Batch results
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_start_time = time.time()
        
        # Load history to get current accuracy if not specified
        if starting_accuracy is None:
            try:
                if self.history_path.exists():
                    with open(self.history_path, 'r') as f:
                        history = json.load(f)
                    events = history.get("events", [])
                    training_events = [e for e in events if e.get("type") == "training_update"]
                    
                    if training_events:
                        last_event = training_events[-1]
                        starting_accuracy = last_event.get("metrics", {}).get("current_accuracy", 0.75)
                    else:
                        starting_accuracy = 0.75
                else:
                    starting_accuracy = 0.75
            except Exception as e:
                logger.error(f"Error loading history: {str(e)}")
                starting_accuracy = 0.75
        
        console.print(f"\n[bold]Starting batch {batch_id} with {cycles} cycles[/bold]")
        console.print(f"Starting accuracy: {starting_accuracy:.2f}")
        console.print(f"Examples per cycle: {examples_per_cycle}")
        console.print(f"Evaluation frequency: Every {evaluation_frequency} cycles")
        
        # Track batch results
        batch_results = {
            "batch_id": batch_id,
            "start_time": datetime.now().isoformat(),
            "cycles": cycles,
            "examples_per_cycle": examples_per_cycle,
            "starting_accuracy": starting_accuracy,
            "cycle_results": [],
            "overall_improvement": 0.0
        }
        
        current_accuracy = starting_accuracy
        
        # Run cycles with progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"Running {cycles} learning cycles...", total=cycles)
            
            for cycle in range(1, cycles + 1):
                try:
                    cycle_start_time = time.time()
                    do_evaluation = (cycle % evaluation_frequency == 0) or (cycle == cycles)
                    
                    # Calculate dynamic improvement with diminishing returns
                    improvement = max(0.01, 0.05 * (1 - current_accuracy))
                    target_accuracy = min(0.95, current_accuracy + improvement)
                    
                    # Run learning cycle
                    results = demonstrate_continuous_learning(
                        data_dir=str(self.data_dir),
                        examples_to_generate=examples_per_cycle,
                        accuracy=target_accuracy,
                        run_evaluation=do_evaluation
                    )
                    
                    # Update accuracy for next cycle
                    current_accuracy = target_accuracy
                    
                    # Track cycle results
                    cycle_results = {
                        "cycle": cycle,
                        "accuracy": target_accuracy,
                        "examples_generated": examples_per_cycle,
                        "duration_seconds": time.time() - cycle_start_time,
                        "success": True
                    }
                    
                    batch_results["cycle_results"].append(cycle_results)
                    
                except Exception as e:
                    logger.error(f"Error in cycle {cycle}: {str(e)}")
                    cycle_results = {
                        "cycle": cycle,
                        "accuracy": current_accuracy,
                        "examples_generated": 0,
                        "duration_seconds": time.time() - cycle_start_time,
                        "success": False,
                        "error": str(e)
                    }
                    batch_results["cycle_results"].append(cycle_results)
                
                # Update progress
                progress.update(task, advance=1, description=f"Cycle {cycle}/{cycles} - Accuracy: {current_accuracy:.2f}")
        
        # Calculate overall improvement
        batch_results["overall_improvement"] = current_accuracy - starting_accuracy
        batch_results["end_time"] = datetime.now().isoformat()
        batch_results["duration_seconds"] = time.time() - batch_start_time
        
        # Save batch history
        try:
            with open(self.batch_history_path, 'r') as f:
                batch_history = json.load(f)
            
            batch_history["batches"].append(batch_results)
            
            with open(self.batch_history_path, 'w') as f:
                json.dump(batch_history, f, indent=2)
                
            logger.info(f"Saved batch results to {self.batch_history_path}")
        except Exception as e:
            logger.error(f"Error saving batch history: {str(e)}")
        
        console.print(f"\n[bold green]Batch {batch_id} completed![/bold green]")
        console.print(f"Overall improvement: {batch_results['overall_improvement']:.2f}")
        console.print(f"Final accuracy: {current_accuracy:.2f}")
        console.print(f"Total duration: {batch_results['duration_seconds']:.1f} seconds")
        
        return batch_results
        
    def get_batch_history(self):
        """Get history of all batches.
        
        Returns:
            list: Batch history
        """
        try:
            if self.batch_history_path.exists():
                with open(self.batch_history_path, 'r') as f:
                    batch_history = json.load(f)
                return batch_history.get("batches", [])
            return []
        except Exception as e:
            logger.error(f"Error loading batch history: {str(e)}")
            return []
            
if __name__ == "__main__":
    # Simple test if run directly
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Batch processor for healthcare learning")
    parser.add_argument("--data-dir", type=str, default="data/healthcare", help="Path to healthcare data directory")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to run")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples per cycle")
    parser.add_argument("--accuracy", type=float, help="Starting accuracy (optional)")
    parser.add_argument("--evaluation-frequency", type=int, default=1, help="Evaluation frequency")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(data_dir=args.data_dir)
    processor.run_batch(
        cycles=args.cycles,
        examples_per_cycle=args.examples,
        starting_accuracy=args.accuracy,
        evaluation_frequency=args.evaluation_frequency
    )
