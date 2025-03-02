#!/usr/bin/env python3
"""
Metrics Dashboard for Healthcare Contradiction Detection

This module provides a real-time metrics dashboard for tracking key performance
indicators (KPIs) related to contradiction detection accuracy over time.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import threading
import signal
import curses
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich import box

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
try:
    from scripts.dashboard_performance_analyzer import PerformanceAnalyzer
    from scripts.config_reader import ConfigReader
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/metrics_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("metrics_dashboard")

# Console for rich output
console = Console()

class MetricsDashboard:
    """Real-time metrics dashboard for tracking contradiction detection KPIs."""
    
    def __init__(self, config_path="config/dashboard_config.json", refresh_interval=10):
        """Initialize the metrics dashboard.
        
        Args:
            config_path: Path to dashboard configuration file
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.config_path = config_path
        self.refresh_interval = refresh_interval
        self.running = False
        self.last_update = datetime.now()
        
        # Load configuration
        try:
            self.config_reader = ConfigReader(config_path)
            self.config = self.config_reader.get_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = {}
            
        # Set up paths
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = self.base_dir / self.config.get("data_directory", "data/healthcare")
        self.reports_dir = self.base_dir / self.config.get("reports_directory", "reports")
        
        # Initialize performance analyzer
        self.analyzer = PerformanceAnalyzer(self.data_dir, self.reports_dir)
        
        # Initialize metrics storage
        self.metrics = {
            "current": {},
            "historical": [],
            "alerts": [],
            "kpis": {
                "accuracy": {"value": 0, "trend": "stable", "target": 0.8},
                "improvement": {"value": 0, "trend": "stable", "target": 0.05},
                "examples": {"value": 0, "trend": "stable", "target": 100},
                "categories": {"value": 0, "trend": "stable", "target": 5},
                "domains": {"value": 0, "trend": "stable", "target": 3}
            }
        }
        
        # Load initial metrics
        self.load_metrics()
        
    def load_metrics(self):
        """Load metrics from performance analyzer."""
        try:
            # Get current metrics
            metrics = self.analyzer.get_performance_metrics()
            
            # Update current metrics
            self.metrics["current"] = metrics
            
            # Update KPIs
            self.metrics["kpis"]["accuracy"]["value"] = metrics.get("current_accuracy", 0)
            self.metrics["kpis"]["improvement"]["value"] = metrics.get("improvement", 0)
            self.metrics["kpis"]["examples"]["value"] = metrics.get("total_examples", 0)
            self.metrics["kpis"]["categories"]["value"] = len(metrics.get("category_accuracies", {}))
            self.metrics["kpis"]["domains"]["value"] = len(metrics.get("domain_accuracies", {}))
            
            # Update trends
            if self.metrics["historical"]:
                last_metrics = self.metrics["historical"][-1]
                
                for kpi, data in self.metrics["kpis"].items():
                    if kpi == "accuracy":
                        last_value = last_metrics.get("current_accuracy", 0)
                    elif kpi == "improvement":
                        last_value = last_metrics.get("improvement", 0)
                    elif kpi == "examples":
                        last_value = last_metrics.get("total_examples", 0)
                    elif kpi == "categories":
                        last_value = len(last_metrics.get("category_accuracies", {}))
                    elif kpi == "domains":
                        last_value = len(last_metrics.get("domain_accuracies", {}))
                    else:
                        last_value = 0
                        
                    current_value = data["value"]
                    
                    if current_value > last_value:
                        data["trend"] = "up"
                    elif current_value < last_value:
                        data["trend"] = "down"
                    else:
                        data["trend"] = "stable"
            
            # Add to historical data (keep last 100 entries)
            self.metrics["historical"].append(metrics)
            if len(self.metrics["historical"]) > 100:
                self.metrics["historical"] = self.metrics["historical"][-100:]
                
            # Generate alerts
            self._generate_alerts()
                
            self.last_update = datetime.now()
            logger.info("Metrics updated successfully")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            
    def _generate_alerts(self):
        """Generate alerts based on metrics."""
        # Clear existing alerts
        self.metrics["alerts"] = []
        
        # Check accuracy
        accuracy = self.metrics["kpis"]["accuracy"]["value"]
        target_accuracy = self.metrics["kpis"]["accuracy"]["target"]
        
        if accuracy < target_accuracy * 0.8:  # Below 80% of target
            self.metrics["alerts"].append({
                "level": "critical",
                "message": f"Accuracy ({accuracy:.2f}) is significantly below target ({target_accuracy:.2f})"
            })
        elif accuracy < target_accuracy:
            self.metrics["alerts"].append({
                "level": "warning",
                "message": f"Accuracy ({accuracy:.2f}) is below target ({target_accuracy:.2f})"
            })
            
        # Check improvement
        improvement = self.metrics["kpis"]["improvement"]["value"]
        target_improvement = self.metrics["kpis"]["improvement"]["target"]
        
        if improvement < 0:
            self.metrics["alerts"].append({
                "level": "critical",
                "message": f"Performance regression detected: {improvement:.2f}"
            })
        elif improvement < target_improvement * 0.5:  # Below 50% of target
            self.metrics["alerts"].append({
                "level": "warning",
                "message": f"Improvement ({improvement:.2f}) is below target ({target_improvement:.2f})"
            })
            
        # Check examples
        examples = self.metrics["kpis"]["examples"]["value"]
        target_examples = self.metrics["kpis"]["examples"]["target"]
        
        if examples < target_examples * 0.5:  # Below 50% of target
            self.metrics["alerts"].append({
                "level": "warning",
                "message": f"Number of examples ({examples}) is below target ({target_examples})"
            })
            
        # Check categories with low accuracy
        category_accuracies = self.metrics["current"].get("category_accuracies", {})
        for category, accuracies in category_accuracies.items():
            if accuracies and accuracies[-1] < 0.6:  # Below 60%
                self.metrics["alerts"].append({
                    "level": "warning",
                    "message": f"Low accuracy ({accuracies[-1]:.2f}) for category '{category}'"
                })
                
        # Check domains with low accuracy
        domain_accuracies = self.metrics["current"].get("domain_accuracies", {})
        for domain, accuracies in domain_accuracies.items():
            if accuracies and accuracies[-1] < 0.6:  # Below 60%
                self.metrics["alerts"].append({
                    "level": "warning",
                    "message": f"Low accuracy ({accuracies[-1]:.2f}) for domain '{domain}'"
                })
                
    def start(self):
        """Start the metrics dashboard."""
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start dashboard UI
        try:
            with Live(self._generate_layout(), refresh_per_second=4) as live:
                self.live = live
                while self.running:
                    live.update(self._generate_layout())
                    time.sleep(0.25)  # Update UI 4 times per second
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop the metrics dashboard."""
        self.running = False
        logger.info("Stopping metrics dashboard")
        
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown."""
        self.stop()
        
    def _update_loop(self):
        """Background thread for updating metrics."""
        while self.running:
            # Check if it's time to update
            if (datetime.now() - self.last_update).total_seconds() >= self.refresh_interval:
                self.load_metrics()
                
            time.sleep(1)
            
    def _generate_layout(self):
        """Generate dashboard layout."""
        layout = Layout()
        
        # Split layout into sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1)
        )
        
        # Split main section into columns
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left column into sections
        layout["left"].split(
            Layout(name="kpis", size=8),
            Layout(name="trends", ratio=1)
        )
        
        # Split right column into sections
        layout["right"].split(
            Layout(name="alerts", size=10),
            Layout(name="categories", ratio=1)
        )
        
        # Generate header
        header = Panel(
            f"[bold blue]Healthcare Contradiction Detection - Metrics Dashboard[/bold blue]\n"
            f"Last updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}",
            box=box.ROUNDED
        )
        layout["header"].update(header)
        
        # Generate KPIs table
        kpis_table = Table(title="Key Performance Indicators", box=box.ROUNDED)
        kpis_table.add_column("KPI")
        kpis_table.add_column("Value")
        kpis_table.add_column("Target")
        kpis_table.add_column("Status")
        
        for kpi, data in self.metrics["kpis"].items():
            # Format value
            if kpi in ["accuracy", "improvement"]:
                value = f"{data['value']:.2f}"
                target = f"{data['target']:.2f}"
            else:
                value = str(int(data['value']))
                target = str(int(data['target']))
                
            # Determine status
            if data['value'] >= data['target']:
                status = "[green]✓[/green]"
            elif data['value'] >= data['target'] * 0.8:
                status = "[yellow]⚠[/yellow]"
            else:
                status = "[red]✗[/red]"
                
            # Add trend indicator
            if data['trend'] == "up":
                value += " [green]↑[/green]"
            elif data['trend'] == "down":
                value += " [red]↓[/red]"
                
            # Format KPI name
            kpi_name = kpi.capitalize()
                
            kpis_table.add_row(kpi_name, value, target, status)
            
        layout["kpis"].update(kpis_table)
        
        # Generate alerts panel
        alerts = self.metrics["alerts"]
        if alerts:
            alerts_content = ""
            for alert in alerts:
                if alert["level"] == "critical":
                    alerts_content += f"[bold red]CRITICAL: {alert['message']}[/bold red]\n"
                else:
                    alerts_content += f"[yellow]WARNING: {alert['message']}[/yellow]\n"
        else:
            alerts_content = "[green]No active alerts[/green]"
            
        alerts_panel = Panel(alerts_content, title="Alerts", box=box.ROUNDED)
        layout["alerts"].update(alerts_panel)
        
        # Generate trends panel
        if self.metrics["historical"]:
            # Extract data for trend visualization
            historical = self.metrics["historical"][-20:]  # Last 20 entries
            accuracies = [h.get("current_accuracy", 0) for h in historical]
            
            # Create a simple ASCII chart
            max_acc = max(accuracies) if accuracies else 1
            min_acc = min(accuracies) if accuracies else 0
            range_acc = max(max_acc - min_acc, 0.1)  # Avoid division by zero
            
            chart_height = 10
            chart = []
            
            for i in range(chart_height):
                level = max_acc - (i / chart_height) * range_acc
                row = f"{level:.2f} |"
                
                for acc in accuracies:
                    if acc >= level:
                        row += "█"
                    else:
                        row += " "
                        
                chart.append(row)
                
            chart.append("      " + "-" * len(accuracies))
            chart.append("      " + "Time →")
            
            trends_content = "\n".join(chart)
        else:
            trends_content = "Insufficient data for trend visualization"
            
        trends_panel = Panel(trends_content, title="Accuracy Trend", box=box.ROUNDED)
        layout["trends"].update(trends_panel)
        
        # Generate categories panel
        categories_table = Table(title="Category Performance", box=box.ROUNDED)
        categories_table.add_column("Category")
        categories_table.add_column("Accuracy")
        categories_table.add_column("Improvement")
        
        category_accuracies = self.metrics["current"].get("category_accuracies", {})
        category_improvements = self.metrics["current"].get("category_improvement", {})
        
        # Sort categories by accuracy
        sorted_categories = sorted(
            [(k, v[-1] if v else 0) for k, v in category_accuracies.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for category, accuracy in sorted_categories[:10]:  # Show top 10
            improvement = category_improvements.get(category, 0)
            
            # Format accuracy and improvement
            if accuracy >= 0.8:
                acc_str = f"[green]{accuracy:.2f}[/green]"
            elif accuracy >= 0.6:
                acc_str = f"[yellow]{accuracy:.2f}[/yellow]"
            else:
                acc_str = f"[red]{accuracy:.2f}[/red]"
                
            if improvement > 0:
                imp_str = f"[green]+{improvement:.2f}[/green]"
            elif improvement < 0:
                imp_str = f"[red]{improvement:.2f}[/red]"
            else:
                imp_str = f"{improvement:.2f}"
                
            categories_table.add_row(category, acc_str, imp_str)
            
        layout["categories"].update(categories_table)
        
        return layout
        
    def export_metrics(self, output_path=None):
        """Export current metrics to a file.
        
        Args:
            output_path: Path to output file. If None, use default path.
            
        Returns:
            str: Path to exported metrics file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.reports_dir / f"metrics_export_{timestamp}.json"
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export metrics
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self.metrics
                }, f, indent=2)
            logger.info(f"Metrics exported to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None
            
def main():
    """Main function for running the metrics dashboard."""
    parser = argparse.ArgumentParser(description="Metrics Dashboard for Healthcare Contradiction Detection")
    parser.add_argument("--config", type=str, default="config/dashboard_config.json", help="Path to dashboard configuration file")
    parser.add_argument("--refresh", type=int, default=10, help="Dashboard refresh interval in seconds")
    parser.add_argument("--export-only", action="store_true", help="Export metrics without starting dashboard")
    parser.add_argument("--export-path", type=str, help="Path to export metrics file")
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = MetricsDashboard(args.config, args.refresh)
    
    # Export metrics if requested
    if args.export_only:
        export_path = dashboard.export_metrics(args.export_path)
        if export_path:
            print(f"Metrics exported to {export_path}")
        return
        
    # Start dashboard
    dashboard.start()
    
if __name__ == "__main__":
    main()
