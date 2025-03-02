#!/usr/bin/env python3
"""
Batch Testing Script for Healthcare Contradiction Detection

This script automates running tests with different configuration presets,
allowing for comprehensive testing across multiple configurations.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preset manager
from scripts.config_presets import ConfigPresetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("batch_tests")

# Console for rich output
console = Console()

class BatchTestRunner:
    """Runs batch tests with different configuration presets."""
    
    def __init__(self, base_dir=None):
        """Initialize the batch test runner.
        
        Args:
            base_dir: Base directory for the project
        """
        # Set up paths
        if base_dir is None:
            self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = Path(base_dir)
            
        self.config_path = self.base_dir / "config" / "dashboard_config.json"
        self.reports_dir = self.base_dir / "reports" / "batch_tests"
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize preset manager
        self.preset_manager = ConfigPresetManager()
        
    def run_tests_with_preset(self, preset_name, test_script=None, timeout=None):
        """Run tests with a specific preset.
        
        Args:
            preset_name: Name of the preset to use
            test_script: Path to the test script to run (default: scripts/run_tests.py)
            timeout: Timeout for the test run in seconds (default: None)
            
        Returns:
            dict: Test results
        """
        # Apply preset to configuration
        logger.info(f"Applying preset '{preset_name}'")
        if not self.preset_manager.apply_preset(preset_name, self.config_path):
            logger.error(f"Failed to apply preset '{preset_name}'")
            return {"success": False, "error": f"Failed to apply preset '{preset_name}'"}
        
        # Set default test script if not provided
        if test_script is None:
            test_script = self.base_dir / "scripts" / "run_tests.py"
        else:
            test_script = Path(test_script)
            
        # Check if test script exists
        if not test_script.exists():
            logger.error(f"Test script not found: {test_script}")
            return {"success": False, "error": f"Test script not found: {test_script}"}
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.reports_dir / f"test_results_{preset_name}_{timestamp}.json"
        
        # Run test script
        logger.info(f"Running tests with preset '{preset_name}'")
        try:
            # Build command
            cmd = [
                sys.executable,
                str(test_script),
                "--output", str(output_file),
                "--preset", preset_name
            ]
            
            # Run command
            start_time = time.time()
            process = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            # Check result
            if process.returncode != 0:
                logger.error(f"Test run failed with exit code {process.returncode}")
                logger.error(f"Error output: {process.stderr}")
                return {
                    "success": False,
                    "error": f"Test run failed with exit code {process.returncode}",
                    "stderr": process.stderr,
                    "stdout": process.stdout
                }
            
            # Load test results
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        results = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading test results: {str(e)}")
                    results = {}
            else:
                results = {}
                
            # Add metadata
            results["preset"] = preset_name
            results["timestamp"] = timestamp
            results["duration"] = end_time - start_time
            results["success"] = True
            
            logger.info(f"Tests completed successfully with preset '{preset_name}'")
            return results
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test run timed out after {timeout} seconds")
            return {"success": False, "error": f"Test run timed out after {timeout} seconds"}
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def run_batch_tests(self, presets, test_script=None, timeout=None):
        """Run tests with multiple presets.
        
        Args:
            presets: List of preset names to use
            test_script: Path to the test script to run (default: scripts/run_tests.py)
            timeout: Timeout for each test run in seconds (default: None)
            
        Returns:
            dict: Batch test results
        """
        # Validate presets
        available_presets = self.preset_manager.list_presets()
        available_preset_names = [p["name"] for p in available_presets]
        
        invalid_presets = [p for p in presets if p not in available_preset_names]
        if invalid_presets:
            logger.error(f"Invalid presets: {', '.join(invalid_presets)}")
            logger.info(f"Available presets: {', '.join(available_preset_names)}")
            return {"success": False, "error": f"Invalid presets: {', '.join(invalid_presets)}"}
        
        # Create batch results
        batch_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "presets": presets,
            "results": {}
        }
        
        # Run tests for each preset
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Running batch tests with {len(presets)} presets", total=len(presets))
            
            for preset in presets:
                progress.update(task, description=f"Testing with preset '{preset}'")
                
                # Run tests with preset
                results = self.run_tests_with_preset(preset, test_script, timeout)
                
                # Add results to batch results
                batch_results["results"][preset] = results
                
                # Update progress
                progress.update(task, advance=1)
                
        # Create summary
        summary = self._create_summary(batch_results)
        batch_results["summary"] = summary
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.reports_dir / f"batch_test_results_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(batch_results, f, indent=2)
            logger.info(f"Batch test results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving batch test results: {str(e)}")
            
        # Display summary
        self._display_summary(summary)
            
        return batch_results
        
    def _create_summary(self, batch_results):
        """Create a summary of batch test results.
        
        Args:
            batch_results: Batch test results
            
        Returns:
            dict: Summary of batch test results
        """
        summary = {
            "total_presets": len(batch_results["presets"]),
            "successful_presets": 0,
            "failed_presets": 0,
            "performance": {}
        }
        
        # Count successful and failed presets
        for preset, results in batch_results["results"].items():
            if results.get("success", False):
                summary["successful_presets"] += 1
            else:
                summary["failed_presets"] += 1
                
            # Extract performance metrics if available
            if results.get("success", False) and "metrics" in results:
                metrics = results["metrics"]
                
                # Add metrics to summary
                for metric, value in metrics.items():
                    if metric not in summary["performance"]:
                        summary["performance"][metric] = {}
                        
                    summary["performance"][metric][preset] = value
                    
        # Calculate best performing preset for each metric
        best_presets = {}
        for metric, values in summary["performance"].items():
            if values:
                best_preset = max(values.items(), key=lambda x: x[1])
                best_presets[metric] = {
                    "preset": best_preset[0],
                    "value": best_preset[1]
                }
                
        summary["best_presets"] = best_presets
        
        return summary
        
    def _display_summary(self, summary):
        """Display a summary of batch test results.
        
        Args:
            summary: Summary of batch test results
        """
        console.print("\n[bold]Batch Test Summary[/bold]")
        console.print(f"Total presets: {summary['total_presets']}")
        console.print(f"Successful presets: {summary['successful_presets']}")
        console.print(f"Failed presets: {summary['failed_presets']}")
        
        if summary.get("best_presets"):
            console.print("\n[bold]Best Performing Presets[/bold]")
            
            # Create table
            table = Table()
            table.add_column("Metric")
            table.add_column("Best Preset")
            table.add_column("Value")
            
            # Add rows
            for metric, data in summary["best_presets"].items():
                table.add_row(
                    metric,
                    data["preset"],
                    str(data["value"])
                )
                
            # Display table
            console.print(table)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run batch tests with different configuration presets")
    parser.add_argument("--presets", type=str, help="Comma-separated list of presets to use")
    parser.add_argument("--all", action="store_true", help="Use all available presets")
    parser.add_argument("--test-script", type=str, help="Path to the test script to run")
    parser.add_argument("--timeout", type=int, help="Timeout for each test run in seconds")
    
    args = parser.parse_args()
    
    # Create batch test runner
    runner = BatchTestRunner()
    
    # Determine presets to use
    if args.all:
        available_presets = runner.preset_manager.list_presets()
        presets = [p["name"] for p in available_presets]
    elif args.presets:
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    else:
        # Use default presets
        presets = ["comprehensive_testing", "quick_testing", "production"]
        
    if not presets:
        logger.error("No presets specified")
        return 1
        
    # Run batch tests
    runner.run_batch_tests(presets, args.test_script, args.timeout)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
