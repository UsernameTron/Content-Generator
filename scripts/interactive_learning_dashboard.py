#!/usr/bin/env python3
"""
Interactive dashboard for healthcare continuous learning system.
Provides visualization and management of learning cycles, metrics tracking,
and dataset exploration.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.markdown import Markdown
from rich.live import Live
from rich.prompt import Prompt, Confirm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_synthetic_evaluation import generate_synthetic_evaluation
from scripts.demonstrate_continuous_learning import demonstrate_continuous_learning

# Try to import optional modules
try:
    from scripts.dashboard_config_reader import DashboardConfigReader
    config_reader_available = True
except ImportError:
    config_reader_available = False

try:
    from scripts.dashboard_batch_processor import BatchProcessor
    batch_processor_available = True
except ImportError:
    batch_processor_available = False

try:
    from scripts.dashboard_dataset_importer import DatasetImporter
    dataset_importer_available = True
except ImportError:
    dataset_importer_available = False

try:
    from scripts.dashboard_performance_analyzer import PerformanceAnalyzer
    performance_analyzer_available = True
except ImportError:
    performance_analyzer_available = False

try:
    from scripts.advanced_testing import AdvancedTestingManager
    advanced_testing_available = True
except ImportError:
    advanced_testing_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("learning_dashboard.log"), logging.StreamHandler()]
)
logger = logging.getLogger("learning-dashboard")
console = Console()

class LearningDashboard:
    """Interactive dashboard for healthcare continuous learning system."""
    
    def __init__(self, data_dir="data/healthcare", mode="regular", config_path=None):
        """Initialize the dashboard.
        
        Args:
            data_dir: Directory containing healthcare data
            mode: Dashboard mode ('regular' or 'testing')
            config_path: Path to dashboard configuration file
        """
        self.data_dir = Path(data_dir)
        self.history_path = self.data_dir / "learning_history.json"
        self.training_path = self.data_dir / "training" / "healthcare_training.json"
        self.contradiction_path = self.data_dir / "contradiction_dataset" / "medical_contradictions.json"
        self.eval_dir = self.data_dir / "evaluation"
        self.mode = mode
        self.config_path = config_path or Path("dashboard_config.json")
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.eval_dir.mkdir(exist_ok=True, parents=True)
        (self.data_dir / "training").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "contradiction_dataset").mkdir(exist_ok=True, parents=True)
        
        # Initialize history if it doesn't exist
        if not self.history_path.exists():
            with open(self.history_path, 'w') as f:
                json.dump({"events": [], "metrics": {}}, f, indent=2)
        
        # Load learning history
        self.history = self._load_history()
        
        # Track current state
        self.current_eval_file = None
        self.current_accuracy = 0.75  # Default starting accuracy
        self.last_visualization = None
        
        # Load configuration if available
        self.config = self._load_config()
        
        # Initialize optional components based on configuration
        self._initialize_components()
        
    def _load_config(self):
        """Load dashboard configuration."""
        if config_reader_available:
            config_reader = DashboardConfigReader(config_path=self.config_path)
            return config_reader.get_config()
        else:
            # Default configuration
            return {
                "dashboard": {
                    "features": {
                        "batch_processing": True,
                        "dataset_import": True,
                        "performance_comparison": True,
                        "advanced_testing": True
                    },
                    "default_settings": {
                        "cycles": 5,
                        "batch_size": 10,
                        "evaluation_frequency": 2
                    }
                }
            }
            
    def _initialize_components(self):
        """Initialize optional dashboard components based on configuration."""
        features = self.config.get("dashboard", {}).get("features", {})
        
        # Initialize batch processor if enabled
        self.batch_processor = None
        if features.get("batch_processing", False) and batch_processor_available:
            self.batch_processor = BatchProcessor(data_dir=self.data_dir)
            
        # Initialize dataset importer if enabled
        self.dataset_importer = None
        if features.get("dataset_import", False) and dataset_importer_available:
            self.dataset_importer = DatasetImporter(data_dir=self.data_dir)
            
        # Initialize performance analyzer if enabled
        self.performance_analyzer = None
        if features.get("performance_comparison", False) and performance_analyzer_available:
            self.performance_analyzer = PerformanceAnalyzer(data_dir=self.data_dir)
            
        # Initialize advanced testing manager if enabled
        self.testing_manager = None
        if features.get("advanced_testing", False) and advanced_testing_available:
            self.testing_manager = AdvancedTestingManager(config_path=self.config_path)
            
    def _load_history(self):
        """Load learning history from disk."""
        try:
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            return {"events": [], "metrics": {}}
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return {"events": [], "metrics": {}}
    
    def _save_history(self):
        """Save learning history to disk."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Saved learning history to {self.history_path}")
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
    
    def _count_training_examples(self):
        """Count examples in the training dataset."""
        try:
            if self.training_path.exists():
                with open(self.training_path, 'r') as f:
                    return len(json.load(f))
            return 0
        except Exception as e:
            logger.error(f"Error counting examples: {str(e)}")
            return 0
    
    def _get_contradiction_categories(self):
        """Get counts of contradiction categories in dataset."""
        categories = {}
        try:
            if self.contradiction_path.exists():
                with open(self.contradiction_path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    category = item.get("category")
                    if category:
                        categories[category] = categories.get(category, 0) + 1
            return categories
        except Exception as e:
            logger.error(f"Error analyzing contradiction dataset: {str(e)}")
            return {}
    
    def _get_learning_metrics(self):
        """Extract learning metrics from history."""
        events = self.history.get("events", [])
        
        # Filter training events
        training_events = [e for e in events if e.get("type") == "training_update"]
        
        # Extract metrics
        timestamps = []
        accuracies = []
        example_counts = []
        
        for event in training_events:
            metrics = event.get("metrics", {})
            timestamps.append(event.get("timestamp", ""))
            accuracies.append(metrics.get("current_accuracy", 0))
            example_counts.append(metrics.get("examples_generated", 0))
        
        return {
            "timestamps": timestamps,
            "accuracies": accuracies,
            "example_counts": example_counts,
            "total_events": len(training_events)
        }
    
    def _visualize_learning_progress(self):
        """Generate visualization of learning progress."""
        metrics = self._get_learning_metrics()
        
        if not metrics["timestamps"]:
            return None
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Convert timestamps to sequence numbers
        x_values = list(range(1, len(metrics["timestamps"]) + 1))
        
        # Plot accuracy over time
        ax1.plot(x_values, metrics["accuracies"], 'o-', color='blue', linewidth=2)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Over Learning Events')
        ax1.grid(True, linestyle='--', alpha=0.7)
        if metrics["accuracies"]:
            min_acc = min(0.5, min(metrics["accuracies"]) - 0.05)
            ax1.set_ylim([min_acc, 1.0])
        
        # Plot example count
        ax2.bar(x_values, metrics["example_counts"], color='green', alpha=0.7)
        ax2.set_xlabel('Learning Event')
        ax2.set_ylabel('New Examples')
        ax2.set_title('Examples Generated per Learning Event')
        
        # Set integer ticks on x-axis
        ax2.set_xticks(x_values)
        
        # Format plot
        plt.tight_layout()
        
        # Save plot
        output_path = self.data_dir / "learning_progress.png"
        plt.savefig(output_path)
        plt.close(fig)
        
        self.last_visualization = str(output_path)
        return str(output_path)
    
    def run_learning_cycle(self, accuracy=None, examples=10):
        """Run a continuous learning cycle.
        
        Args:
            accuracy: Model accuracy to simulate (None to use last accuracy + improvement)
            examples: Number of examples to generate
        """
        # If accuracy not specified, calculate from history
        if accuracy is None:
            metrics = self._get_learning_metrics()
            if metrics["accuracies"]:
                last_accuracy = metrics["accuracies"][-1]
                # Apply diminishing returns improvement
                improvement = max(0.01, 0.05 * (1 - last_accuracy))
                accuracy = min(0.95, last_accuracy + improvement)
            else:
                accuracy = self.current_accuracy
        
        # Update current accuracy
        self.current_accuracy = accuracy
        
        # Run learning cycle
        try:
            results = demonstrate_continuous_learning(
                data_dir=str(self.data_dir),
                examples_to_generate=examples,
                accuracy=accuracy
            )
            
            # Refresh history after cycle
            self.history = self._load_history()
            
            return results
        except Exception as e:
            logger.error(f"Error running learning cycle: {str(e)}")
            return None

    def render_dashboard(self, live):
        """Render the main dashboard layout.
        
        Args:
            live: Live context for updating display
        """
        # Create main layout
        layout = Layout()
        
        # Split into sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=1)
        )
        
        # Split main area into left and right panels
        layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="actions", ratio=1)
        )
        
        # Create header
        header = Panel(
            Markdown("# Healthcare Contradiction Detection Learning System"),
            style="bold blue"
        )
        
        # Create metrics panel
        metrics_panel = self._render_metrics_panel()
        
        # Create actions panel
        actions_panel = self._render_actions_panel()
        
        # Create footer
        footer = Panel(
            "Press Ctrl+C to exit | Select an option from the Actions panel",
            style="dim"
        )
        
        # Update layout
        layout["header"] = header
        layout["metrics"] = metrics_panel
        layout["actions"] = actions_panel
        layout["footer"] = footer
        
        # Render layout
        live.update(layout)
    
    def _render_metrics_panel(self):
        """Render metrics panel with system statistics."""
        # Count examples
        training_count = self._count_training_examples()
        
        # Get learning metrics
        metrics = self._get_learning_metrics()
        
        # Get contradiction categories
        categories = self._get_contradiction_categories()
        
        # Create metrics table
        metrics_table = Table(title="System Statistics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        # Add rows
        metrics_table.add_row("Training Examples", str(training_count))
        metrics_table.add_row("Learning Events", str(metrics.get("total_events", 0)))
        if metrics.get("accuracies"):
            metrics_table.add_row("Current Accuracy", f"{metrics['accuracies'][-1]:.2f}")
            metrics_table.add_row("Initial Accuracy", f"{metrics['accuracies'][0]:.2f}" if metrics['accuracies'] else "N/A")
        else:
            metrics_table.add_row("Current Accuracy", f"{self.current_accuracy:.2f}")
            metrics_table.add_row("Initial Accuracy", "N/A")
        
        # Create categories table
        categories_table = Table(title="Contradiction Categories")
        categories_table.add_column("Category", style="cyan")
        categories_table.add_column("Count", style="green")
        
        for category, count in categories.items():
            categories_table.add_row(category, str(count))
        
        # Generate visualization if not exists
        if not self.last_visualization:
            self._visualize_learning_progress()
        
        # Create layout for metrics panel
        metrics_md = """
        ## System Metrics
        
        The continuous learning system tracks performance metrics and learning events
        over time. View the latest statistics and learning progress below.
        """
        
        # Combine all elements
        metrics_layout = Layout()
        metrics_layout.split(
            Layout(Markdown(metrics_md), size=6),
            Layout(metrics_table, size=10),
            Layout(categories_table)
        )
        
        return Panel(metrics_layout, title="System Metrics", border_style="blue")
    
    def _render_actions_panel(self):
        """Render actions panel with available operations."""
        # Create actions layout
        actions_md = """
        ## Available Actions
        
        Select an action to perform on the continuous learning system:
        
        1. **Run Learning Cycle** - Execute a single learning cycle
        2. **Visualize Progress** - Generate learning progress charts
        3. **View Learning History** - Examine past learning events
        4. **Generate New Evaluation** - Create synthetic evaluation data
        5. **Exit Dashboard** - Close the dashboard
        
        Enter your selection (1-5):
        """
        
        return Panel(Markdown(actions_md), title="Actions", border_style="green")
    
    def run_dashboard(self):
        """Run the interactive dashboard."""
        console = Console()
        
        console.print(Panel.fit(
            "[bold]Healthcare Contradiction Detection Learning Dashboard[/bold]\n"
            f"Mode: [cyan]{self.mode.upper()}[/cyan]",
            style="green"
        ))
        
        # Display different interface based on mode
        if self.mode == "testing":
            self._run_testing_dashboard()
        else:
            self._run_regular_dashboard()
            
    def _run_regular_dashboard(self):
        """Run the regular learning dashboard."""
        console = Console()
        
        with Live(self.render_dashboard(None), refresh_per_second=4) as live:
            while True:
                action = Prompt.ask(
                    "Select action", 
                    choices=["run", "batch", "import", "analyze", "compare", "visualize", "config", "exit"],
                    default="run"
                )
                
                if action == "exit":
                    break
                elif action == "run":
                    examples = int(Prompt.ask("Number of examples to generate", default="10"))
                    self.run_learning_cycle(examples=examples)
                    live.update(self.render_dashboard(live))
                elif action == "batch" and self.batch_processor:
                    cycles = int(Prompt.ask("Number of cycles to run", default="5"))
                    examples = int(Prompt.ask("Examples per cycle", default="10"))
                    self.batch_processor.run_batch(cycles=cycles, examples_per_cycle=examples)
                    live.update(self.render_dashboard(live))
                elif action == "import" and self.dataset_importer:
                    import_type = Prompt.ask("Import type", choices=["file", "example"], default="file")
                    if import_type == "file":
                        file_path = Prompt.ask("Path to JSON file")
                        self.dataset_importer.import_from_file(file_path)
                    else:
                        self.dataset_importer.add_interactive_example()
                    live.update(self.render_dashboard(live))
                elif action == "analyze" and self.performance_analyzer:
                    self.performance_analyzer.print_performance_summary()
                    if Confirm.ask("Generate detailed report?"):
                        report_path = self.performance_analyzer.generate_performance_report()
                        if report_path:
                            console.print(f"[green]Report generated: {report_path}[/green]")
                elif action == "compare" and self.performance_analyzer:
                    compare_type = Prompt.ask("Compare type", choices=["cycles", "categories", "domains"], default="cycles")
                    if compare_type == "cycles":
                        cycles_str = Prompt.ask("Cycles to compare (comma-separated indices)")
                        try:
                            cycles = [int(c.strip()) for c in cycles_str.split(",")]
                            comparison = self.performance_analyzer.compare_learning_cycles(cycles)
                            console.print(f"[bold]Comparison Results:[/bold]")
                            console.print(f"Overall improvement: {comparison['overall_improvement']:.2f}")
                        except Exception as e:
                            console.print(f"[red]Error comparing cycles: {str(e)}[/red]")
                elif action == "visualize":
                    self._visualize_learning_progress()
                    console.print(f"[green]Visualization saved to {self.last_visualization}[/green]")
                elif action == "config":
                    self._edit_configuration()
                    live.update(self.render_dashboard(live))
                
                # Update the display
                live.update(self.render_dashboard(live))
                
        console.print("[bold green]Dashboard closed. Thank you for using the Healthcare Learning System![/bold green]")
        
    def _run_testing_dashboard(self):
        """Run the testing dashboard interface."""
        console = Console()
        
        if not self.testing_manager:
            console.print("[red]Advanced testing is not available. Please check your configuration.[/red]")
            return
            
        console.print(Panel.fit("[bold]Healthcare Contradiction Detection Testing Dashboard[/bold]", style="blue"))
        
        while True:
            action = Prompt.ask(
                "Select testing action", 
                choices=["scenario", "edge", "stress", "regression", "create", "report", "compare", "config", "exit"],
                default="scenario"
            )
            
            if action == "exit":
                break
            elif action == "scenario":
                scenario = Prompt.ask("Scenario name (leave empty for all)", default="")
                scenario_name = scenario if scenario else None
                self.testing_manager.run_scenario_tests(scenario_name)
            elif action == "edge":
                self.testing_manager.run_edge_case_tests()
            elif action == "stress":
                batch_size = int(Prompt.ask("Batch size", default="100"))
                iterations = int(Prompt.ask("Iterations", default="5"))
                self.testing_manager.run_stress_test(batch_size, iterations)
            elif action == "regression":
                baseline = Prompt.ask("Baseline model (leave empty for default)", default="")
                baseline_model = baseline if baseline else None
                self.testing_manager.run_regression_test(baseline_model)
            elif action == "create":
                case_type = Prompt.ask("Test case type", choices=["scenario", "edge_case"], default="scenario")
                console.print("[yellow]Enter test case data in JSON format:[/yellow]")
                console.print("Example: {\"id\": \"test_case_1\", \"description\": \"Test case description\", \"test_cases\": []}")
                
                # In a real implementation, we would have a more user-friendly way to create test cases
                # For now, we'll just ask for JSON input
                try:
                    case_data_str = Prompt.ask("Test case data (JSON)")
                    case_data = json.loads(case_data_str)
                    self.testing_manager.create_test_case(case_type, case_data)
                except Exception as e:
                    console.print(f"[red]Error creating test case: {str(e)}[/red]")
            elif action == "report" and self.performance_analyzer:
                self.performance_analyzer.generate_performance_report()
            elif action == "compare" and self.performance_analyzer:
                compare_type = Prompt.ask("Compare type", choices=["cycles", "categories", "domains"], default="cycles")
                if compare_type == "cycles":
                    cycles_str = Prompt.ask("Cycles to compare (comma-separated indices)")
                    try:
                        cycles = [int(c.strip()) for c in cycles_str.split(",")]
                        comparison = self.performance_analyzer.compare_learning_cycles(cycles)
                        console.print(f"[bold]Comparison Results:[/bold]")
                        console.print(f"Overall improvement: {comparison['overall_improvement']:.2f}")
                    except Exception as e:
                        console.print(f"[red]Error comparing cycles: {str(e)}[/red]")
            elif action == "config":
                self._edit_configuration()
                
        console.print("[bold blue]Testing dashboard closed.[/bold blue]")
        
    def _edit_configuration(self):
        """Interactive configuration editor for modifying test parameters."""
        console = Console()
        
        console.print(Panel.fit("[bold]Configuration Editor[/bold]", style="cyan"))
        console.print("This editor allows you to modify dashboard configuration parameters.")
        
        # Load current configuration
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading configuration: {str(e)}[/red]")
            return
        
        # Create a backup of the current configuration
        backup_path = f"{self.config_path}.backup"
        try:
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            console.print(f"[green]Created backup at {backup_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create backup: {str(e)}[/yellow]")
        
        # Display configuration sections
        sections = list(config.keys())
        
        while True:
            console.print("\n[bold]Available Configuration Sections:[/bold]")
            for i, section in enumerate(sections, 1):
                console.print(f"{i}. {section}")
            console.print(f"{len(sections) + 1}. Save and Exit")
            console.print(f"{len(sections) + 2}. Exit Without Saving")
            
            choice = Prompt.ask(
                "Select section to edit", 
                choices=[str(i) for i in range(1, len(sections) + 3)],
                default="1"
            )
            
            # Handle save and exit
            if choice == str(len(sections) + 1):
                try:
                    with open(self.config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    console.print(f"[green]Configuration saved to {self.config_path}[/green]")
                    
                    # Reload configuration
                    self.config = self._load_config()
                    console.print("[green]Configuration reloaded successfully[/green]")
                    break
                except Exception as e:
                    console.print(f"[red]Error saving configuration: {str(e)}[/red]")
            
            # Handle exit without saving
            elif choice == str(len(sections) + 2):
                if Confirm.ask("Are you sure you want to exit without saving?"):
                    console.print("[yellow]Exiting without saving changes[/yellow]")
                    break
                continue
            
            # Edit selected section
            else:
                section_idx = int(choice) - 1
                section_name = sections[section_idx]
                self._edit_config_section(config, section_name)
    
    def _edit_config_section(self, config, section_name):
        """Edit a specific section of the configuration.
        
        Args:
            config: The configuration dictionary
            section_name: The name of the section to edit
        """
        console = Console()
        section = config[section_name]
        
        # Display section as a table
        table = Table(title=f"{section_name.title()} Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Type", style="yellow")
        
        # Flatten nested structure for easier editing
        flat_params = self._flatten_config(section, parent_key=section_name)
        
        # Display parameters
        for param, value in flat_params.items():
            param_name = param.replace(f"{section_name}.", "")
            param_type = type(value).__name__
            table.add_row(param_name, str(value), param_type)
        
        console.print(table)
        
        # Edit parameters
        while True:
            param_to_edit = Prompt.ask(
                "Enter parameter to edit (or 'back' to return)", 
                default="back"
            )
            
            if param_to_edit.lower() == "back":
                break
            
            # Construct full parameter path
            full_param = f"{section_name}.{param_to_edit}" if not param_to_edit.startswith(section_name) else param_to_edit
            
            if full_param in flat_params:
                current_value = flat_params[full_param]
                value_type = type(current_value).__name__
                
                console.print(f"Current value: [green]{current_value}[/green] (Type: {value_type})")
                
                # Get new value
                new_value_str = Prompt.ask("Enter new value", default=str(current_value))
                
                # Convert string to appropriate type
                try:
                    if isinstance(current_value, bool):
                        new_value = new_value_str.lower() in ("true", "yes", "y", "1")
                    elif isinstance(current_value, int):
                        new_value = int(new_value_str)
                    elif isinstance(current_value, float):
                        new_value = float(new_value_str)
                    elif isinstance(current_value, list):
                        # Parse as JSON if it starts with [ or as comma-separated list otherwise
                        if new_value_str.strip().startswith("["):
                            new_value = json.loads(new_value_str)
                        else:
                            new_value = [item.strip() for item in new_value_str.split(",")]
                    elif isinstance(current_value, dict):
                        # Parse as JSON
                        new_value = json.loads(new_value_str)
                    else:
                        new_value = new_value_str
                        
                    # Update the value in the nested structure
                    self._update_nested_config(config, full_param, new_value)
                    console.print(f"[green]Updated {param_to_edit} to {new_value}[/green]")
                    
                except Exception as e:
                    console.print(f"[red]Error setting value: {str(e)}[/red]")
            else:
                console.print(f"[red]Parameter '{param_to_edit}' not found in {section_name} section[/red]")
    
    def _flatten_config(self, config_section, parent_key=""):
        """Flatten a nested configuration section for easier editing.
        
        Args:
            config_section: The configuration section to flatten
            parent_key: The parent key for nested parameters
            
        Returns:
            A flattened dictionary with dot-notation keys
        """
        items = {}
        for key, value in config_section.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(self._flatten_config(value, new_key))
            else:
                items[new_key] = value
                
        return items
    
    def _update_nested_config(self, config, param_path, value):
        """Update a value in a nested configuration structure.
        
        Args:
            config: The configuration dictionary
            param_path: The parameter path in dot notation
            value: The new value to set
        """
        parts = param_path.split(".")
        current = config
        
        # Navigate to the nested location
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value

def main():
    parser = argparse.ArgumentParser(description="Interactive dashboard for healthcare learning system")
    parser.add_argument("--data-dir", 
                       type=str, 
                       default="data/healthcare",
                       help="Path to healthcare data directory")
    parser.add_argument("--mode",
                       type=str,
                       choices=["regular", "testing"],
                       default=os.environ.get("DASHBOARD_MODE", "regular"),
                       help="Dashboard mode: regular or testing")
    parser.add_argument("--config",
                       type=str,
                       default="dashboard_config.json",
                       help="Path to dashboard configuration file")
    parser.add_argument("--run-scheduled-tests",
                       action="store_true",
                       help="Run tests using current configuration")
    args = parser.parse_args()
    
    # Create dashboard instance
    dashboard = LearningDashboard(data_dir=args.data_dir, mode=args.mode, config_path=args.config)
    
    # Handle scheduled test run
    if args.run_scheduled_tests:
        run_scheduled_tests(dashboard)
    else:
        # Run interactive dashboard
        dashboard.run_dashboard()
    
    return 0

def run_scheduled_tests(dashboard):
    """Run tests in scheduled mode using current configuration.
    
    Args:
        dashboard: LearningDashboard instance
    """
    console = Console()
    console.print(Panel.fit("[bold]Running Scheduled Tests[/bold]", style="green"))
    
    # Load configuration
    try:
        with open(dashboard.config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")
        return
    
    # Get testing configuration
    testing_config = config.get("testing", {})
    batch_size = testing_config.get("batch_size", 64)
    
    # Get performance configuration
    performance_config = config.get("performance", {})
    reporting = performance_config.get("reporting", {})
    generate_report = reporting.get("generate_html", True)
    
    # Run tests based on dashboard mode
    console.print(f"[blue]Running tests in {dashboard.mode} mode with batch size {batch_size}[/blue]")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if dashboard.mode == "testing" and dashboard.testing_manager:
        # Run comprehensive test suite
        console.print("[yellow]Running comprehensive test suite...[/yellow]")
        dashboard.testing_manager.run_scenario_tests()
        dashboard.testing_manager.run_edge_case_tests()
        
        # Run stress test with configured batch size
        console.print(f"[yellow]Running stress test with batch size {batch_size}...[/yellow]")
        dashboard.testing_manager.run_stress_test(batch_size, iterations=3)
        
        # Run regression test
        console.print("[yellow]Running regression test...[/yellow]")
        dashboard.testing_manager.run_regression_test()
    else:
        # Run standard learning cycles
        cycles = 5  # Default number of cycles
        examples = batch_size
        
        console.print(f"[yellow]Running {cycles} learning cycles with {examples} examples each...[/yellow]")
        
        # Run batch processing if available
        if dashboard.batch_processor:
            dashboard.batch_processor.run_batch(cycles=cycles, examples_per_cycle=examples)
        else:
            # Manually run learning cycles
            for i in range(cycles):
                console.print(f"[blue]Running learning cycle {i+1}/{cycles}...[/blue]")
                dashboard.run_learning_cycle(examples=examples)
    
    # Generate performance report if analyzer is available
    if dashboard.performance_analyzer and generate_report:
        console.print("[yellow]Generating performance report...[/yellow]")
        report_path = dashboard.performance_analyzer.generate_performance_report()
        if report_path:
            console.print(f"[green]Report generated: {report_path}[/green]")
    
    # Generate visualization
    vis_path = dashboard._visualize_learning_progress()
    if vis_path:
        console.print(f"[green]Visualization saved to: {vis_path}[/green]")
    
    console.print(f"[bold green]Scheduled tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold green]")

if __name__ == "__main__":
    sys.exit(main())
