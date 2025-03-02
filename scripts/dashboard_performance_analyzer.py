#!/usr/bin/env python3
"""
Performance analyzer for healthcare learning dashboard.
Provides functionality for analyzing and comparing model performance across learning cycles.
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Configure logging
logger = logging.getLogger("performance-analyzer")
console = Console()

class PerformanceAnalyzer:
    """Performance analyzer for measuring model improvement."""
    
    def __init__(self, data_dir="data/healthcare"):
        """Initialize performance analyzer.
        
        Args:
            data_dir: Directory containing healthcare data
        """
        self.data_dir = Path(data_dir)
        self.history_path = self.data_dir / "learning_history.json"
        self.batch_history_path = self.data_dir / "batch_history.json"
        self.eval_dir = self.data_dir / "evaluation"
        self.reports_dir = self.data_dir / "performance_reports"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.eval_dir.mkdir(exist_ok=True, parents=True)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        
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
            
    def _load_batch_history(self):
        """Load batch history from disk."""
        try:
            if self.batch_history_path.exists():
                with open(self.batch_history_path, 'r') as f:
                    return json.load(f)
            return {"batches": []}
        except Exception as e:
            logger.error(f"Error loading batch history: {str(e)}")
            return {"batches": []}
            
    def _get_training_events(self):
        """Get training events from history."""
        history = self._load_history()
        events = history.get("events", [])
        
        # Filter training events
        training_events = [e for e in events if e.get("type") == "training_update"]
        
        return training_events
        
    def _get_eval_events(self):
        """Get evaluation events from history."""
        history = self._load_history()
        events = history.get("events", [])
        
        # Filter evaluation events
        eval_events = [e for e in events if e.get("type") == "evaluation"]
        
        return eval_events
        
    def _get_batch_events(self):
        """Get batch processing events."""
        batch_history = self._load_batch_history()
        return batch_history.get("batches", [])
        
    def get_performance_metrics(self):
        """Get performance metrics from history.
        
        Returns:
            dict: Performance metrics
        """
        training_events = self._get_training_events()
        eval_events = self._get_eval_events()
        
        if not training_events:
            return {
                "total_events": 0,
                "total_examples": 0,
                "accuracy_history": [],
                "improvement": 0.0
            }
            
        # Extract metrics
        timestamps = []
        accuracies = []
        example_counts = []
        category_accuracies = {}
        domain_accuracies = {}
        
        for event in training_events:
            metrics = event.get("metrics", {})
            timestamps.append(event.get("timestamp", ""))
            accuracies.append(metrics.get("current_accuracy", 0))
            example_counts.append(metrics.get("examples_generated", 0))
            
            # Track category and domain accuracies
            if "category_accuracies" in metrics:
                for category, accuracy in metrics["category_accuracies"].items():
                    if category not in category_accuracies:
                        category_accuracies[category] = []
                    category_accuracies[category].append(accuracy)
                    
            if "domain_accuracies" in metrics:
                for domain, accuracy in metrics["domain_accuracies"].items():
                    if domain not in domain_accuracies:
                        domain_accuracies[domain] = []
                    domain_accuracies[domain].append(accuracy)
        
        # Calculate improvements
        initial_accuracy = accuracies[0] if accuracies else 0
        current_accuracy = accuracies[-1] if accuracies else 0
        improvement = current_accuracy - initial_accuracy
        
        # Calculate improvement by category and domain
        category_improvement = {}
        for category, acc_history in category_accuracies.items():
            if len(acc_history) >= 2:
                category_improvement[category] = acc_history[-1] - acc_history[0]
                
        domain_improvement = {}
        for domain, acc_history in domain_accuracies.items():
            if len(acc_history) >= 2:
                domain_improvement[domain] = acc_history[-1] - acc_history[0]
        
        return {
            "total_events": len(training_events),
            "total_examples": sum(example_counts),
            "accuracy_history": list(zip(timestamps, accuracies)),
            "example_count_history": list(zip(timestamps, example_counts)),
            "initial_accuracy": initial_accuracy,
            "current_accuracy": current_accuracy,
            "improvement": improvement,
            "category_accuracies": category_accuracies,
            "domain_accuracies": domain_accuracies,
            "category_improvement": category_improvement,
            "domain_improvement": domain_improvement
        }
        
    def generate_performance_report(self):
        """Generate comprehensive performance report.
        
        Returns:
            str: Path to report file
        """
        # Get metrics
        metrics = self.get_performance_metrics()
        
        if metrics["total_events"] == 0:
            console.print("[yellow]No training events found. Cannot generate report.[/yellow]")
            return None
            
        # Create report file
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"performance_report_{report_time}.json"
        
        # Add report metadata
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "summary": {
                "total_learning_events": metrics["total_events"],
                "total_examples_generated": metrics["total_examples"],
                "initial_accuracy": metrics["initial_accuracy"],
                "current_accuracy": metrics["current_accuracy"],
                "absolute_improvement": metrics["improvement"],
                "relative_improvement": (metrics["improvement"] / metrics["initial_accuracy"]) * 100 if metrics["initial_accuracy"] > 0 else 0,
                "top_improved_categories": sorted(metrics["category_improvement"].items(), key=lambda x: x[1], reverse=True)[:3],
                "top_improved_domains": sorted(metrics["domain_improvement"].items(), key=lambda x: x[1], reverse=True)[:3]
            }
        }
        
        # Identify improvement areas
        report["improvement_areas"] = self._identify_improvement_areas(metrics)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(metrics, report["improvement_areas"])
        
        # Save report
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved performance report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving performance report: {str(e)}")
            return None
            
        # Generate visualizations
        self.generate_performance_visualizations(report_time)
        
        # Generate HTML report
        html_report_path = self._generate_html_report(report, report_time)
        
        # Print report summary
        self._print_report_summary(report)
        
        return str(report_file)
        
    def _identify_improvement_areas(self, metrics):
        """Identify areas that need improvement.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            dict: Improvement areas
        """
        improvement_areas = {
            "low_performance_categories": [],
            "low_performance_domains": [],
            "stagnant_categories": [],
            "stagnant_domains": [],
            "regression_areas": []
        }
        
        # Identify low-performing categories (below 0.7 accuracy)
        for category, accuracies in metrics.get("category_accuracies", {}).items():
            if accuracies and accuracies[-1] < 0.7:
                improvement_areas["low_performance_categories"].append({
                    "name": category,
                    "current_accuracy": accuracies[-1],
                    "improvement": metrics.get("category_improvement", {}).get(category, 0)
                })
                
        # Identify low-performing domains (below 0.7 accuracy)
        for domain, accuracies in metrics.get("domain_accuracies", {}).items():
            if accuracies and accuracies[-1] < 0.7:
                improvement_areas["low_performance_domains"].append({
                    "name": domain,
                    "current_accuracy": accuracies[-1],
                    "improvement": metrics.get("domain_improvement", {}).get(domain, 0)
                })
                
        # Identify stagnant categories (improvement less than 0.05)
        for category, improvement in metrics.get("category_improvement", {}).items():
            if 0 <= improvement < 0.05:
                accuracies = metrics.get("category_accuracies", {}).get(category, [])
                improvement_areas["stagnant_categories"].append({
                    "name": category,
                    "current_accuracy": accuracies[-1] if accuracies else 0,
                    "improvement": improvement
                })
                
        # Identify stagnant domains (improvement less than 0.05)
        for domain, improvement in metrics.get("domain_improvement", {}).items():
            if 0 <= improvement < 0.05:
                accuracies = metrics.get("domain_accuracies", {}).get(domain, [])
                improvement_areas["stagnant_domains"].append({
                    "name": domain,
                    "current_accuracy": accuracies[-1] if accuracies else 0,
                    "improvement": improvement
                })
                
        # Identify regression areas (negative improvement)
        for category, improvement in metrics.get("category_improvement", {}).items():
            if improvement < 0:
                accuracies = metrics.get("category_accuracies", {}).get(category, [])
                improvement_areas["regression_areas"].append({
                    "name": category,
                    "type": "category",
                    "current_accuracy": accuracies[-1] if accuracies else 0,
                    "previous_accuracy": accuracies[0] if accuracies else 0,
                    "regression": abs(improvement)
                })
                
        for domain, improvement in metrics.get("domain_improvement", {}).items():
            if improvement < 0:
                accuracies = metrics.get("domain_accuracies", {}).get(domain, [])
                improvement_areas["regression_areas"].append({
                    "name": domain,
                    "type": "domain",
                    "current_accuracy": accuracies[-1] if accuracies else 0,
                    "previous_accuracy": accuracies[0] if accuracies else 0,
                    "regression": abs(improvement)
                })
                
        # Sort improvement areas
        for key in improvement_areas:
            if key == "regression_areas":
                improvement_areas[key] = sorted(improvement_areas[key], key=lambda x: x["regression"], reverse=True)
            else:
                improvement_areas[key] = sorted(improvement_areas[key], key=lambda x: x["current_accuracy"])
                
        return improvement_areas
        
    def _generate_recommendations(self, metrics, improvement_areas):
        """Generate recommendations based on metrics and improvement areas.
        
        Args:
            metrics: Performance metrics
            improvement_areas: Identified improvement areas
            
        Returns:
            list: Recommendations
        """
        recommendations = []
        
        # Recommendation for overall improvement
        if metrics["improvement"] < 0.1:
            recommendations.append({
                "priority": "high",
                "area": "overall",
                "recommendation": "Overall model performance improvement is low. Consider increasing the diversity of training examples."
            })
        
        # Recommendations for low-performing categories
        for item in improvement_areas["low_performance_categories"][:3]:  # Top 3 lowest
            recommendations.append({
                "priority": "high",
                "area": f"category:{item['name']}",
                "recommendation": f"Low accuracy ({item['current_accuracy']:.2f}) in category '{item['name']}'. Add more diverse examples for this category."
            })
            
        # Recommendations for low-performing domains
        for item in improvement_areas["low_performance_domains"][:3]:  # Top 3 lowest
            recommendations.append({
                "priority": "high",
                "area": f"domain:{item['name']}",
                "recommendation": f"Low accuracy ({item['current_accuracy']:.2f}) in domain '{item['name']}'. Add more specialized examples for this domain."
            })
            
        # Recommendations for stagnant categories
        for item in improvement_areas["stagnant_categories"][:3]:  # Top 3
            recommendations.append({
                "priority": "medium",
                "area": f"category:{item['name']}",
                "recommendation": f"Stagnant improvement ({item['improvement']:.2f}) in category '{item['name']}'. Try different example types for this category."
            })
            
        # Recommendations for regression areas
        for item in improvement_areas["regression_areas"]:
            recommendations.append({
                "priority": "critical",
                "area": f"{item['type']}:{item['name']}",
                "recommendation": f"Performance regression detected in {item['type']} '{item['name']}'. Accuracy decreased by {item['regression']:.2f}. Review recent changes and examples."
            })
            
        # General recommendations based on overall metrics
        if metrics["total_events"] < 5:
            recommendations.append({
                "priority": "medium",
                "area": "training",
                "recommendation": "Limited training history. Run more learning cycles to improve model performance."
            })
            
        if metrics["total_examples"] < 50:
            recommendations.append({
                "priority": "medium",
                "area": "data",
                "recommendation": "Limited training examples. Add more diverse examples to improve model robustness."
            })
            
        # Sort recommendations by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations = sorted(recommendations, key=lambda x: priority_order.get(x["priority"], 4))
        
        return recommendations
        
    def _generate_html_report(self, report, report_time):
        """Generate HTML report with visualizations.
        
        Args:
            report: Report data
            report_time: Report timestamp
            
        Returns:
            str: Path to HTML report
        """
        # Create HTML report directory
        html_dir = self.reports_dir / f"report_{report_time}"
        html_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy visualizations to report directory
        for viz_file in self.reports_dir.glob(f"*_{report_time}.png"):
            import shutil
            shutil.copy(viz_file, html_dir)
            
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Healthcare Contradiction Detection Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric {{ background-color: #e9ecef; padding: 10px; margin: 5px; border-radius: 5px; width: 200px; }}
                .metric h3 {{ margin-top: 0; }}
                .improvement {{ color: green; }}
                .regression {{ color: red; }}
                .visualization {{ margin: 20px 0; }}
                .recommendations {{ margin-top: 20px; }}
                .critical {{ background-color: #f8d7da; padding: 10px; margin: 5px; border-radius: 5px; }}
                .high {{ background-color: #fff3cd; padding: 10px; margin: 5px; border-radius: 5px; }}
                .medium {{ background-color: #d1ecf1; padding: 10px; margin: 5px; border-radius: 5px; }}
                .low {{ background-color: #d4edda; padding: 10px; margin: 5px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Healthcare Contradiction Detection Performance Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Learning Events</h3>
                        <p>{report["summary"]["total_learning_events"]}</p>
                    </div>
                    <div class="metric">
                        <h3>Examples Generated</h3>
                        <p>{report["summary"]["total_examples_generated"]}</p>
                    </div>
                    <div class="metric">
                        <h3>Initial Accuracy</h3>
                        <p>{report["summary"]["initial_accuracy"]:.2f}</p>
                    </div>
                    <div class="metric">
                        <h3>Current Accuracy</h3>
                        <p>{report["summary"]["current_accuracy"]:.2f}</p>
                    </div>
                    <div class="metric">
                        <h3>Absolute Improvement</h3>
                        <p class="{'improvement' if report["summary"]["absolute_improvement"] >= 0 else 'regression'}">
                            {report["summary"]["absolute_improvement"]:.2f}
                        </p>
                    </div>
                    <div class="metric">
                        <h3>Relative Improvement</h3>
                        <p class="{'improvement' if report["summary"]["relative_improvement"] >= 0 else 'regression'}">
                            {report["summary"]["relative_improvement"]:.1f}%
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="visualization">
                <h2>Performance Visualizations</h2>
                <img src="accuracy_trend_{report_time}.png" alt="Accuracy Trend" style="max-width: 100%;">
        """
        
        # Add category improvement visualization if it exists
        category_viz_path = self.reports_dir / f"category_improvement_{report_time}.png"
        if category_viz_path.exists():
            html_content += f"""
                <img src="category_improvement_{report_time}.png" alt="Category Improvement" style="max-width: 100%;">
            """
            
        html_content += """
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
        """
        
        # Add recommendations
        for rec in report["recommendations"]:
            html_content += f"""
                <div class="{rec['priority']}">
                    <h3>{rec['area']}</h3>
                    <p>{rec['recommendation']}</p>
                </div>
            """
            
        html_content += """
            </div>
            
            <div class="improvement-areas">
                <h2>Improvement Areas</h2>
        """
        
        # Add improvement areas
        improvement_areas = report["improvement_areas"]
        
        # Add regression areas
        if improvement_areas["regression_areas"]:
            html_content += """
                <h3>Regression Areas</h3>
                <ul>
            """
            for item in improvement_areas["regression_areas"]:
                html_content += f"""
                    <li class="regression">
                        {item['type'].capitalize()} '{item['name']}': Decreased from {item['previous_accuracy']:.2f} to {item['current_accuracy']:.2f} (-{item['regression']:.2f})
                    </li>
                """
            html_content += """
                </ul>
            """
            
        # Add low-performance categories
        if improvement_areas["low_performance_categories"]:
            html_content += """
                <h3>Low Performance Categories</h3>
                <ul>
            """
            for item in improvement_areas["low_performance_categories"]:
                html_content += f"""
                    <li>
                        Category '{item['name']}': Current accuracy {item['current_accuracy']:.2f} (Improvement: {item['improvement']:.2f})
                    </li>
                """
            html_content += """
                </ul>
            """
            
        # Add low-performance domains
        if improvement_areas["low_performance_domains"]:
            html_content += """
                <h3>Low Performance Domains</h3>
                <ul>
            """
            for item in improvement_areas["low_performance_domains"]:
                html_content += f"""
                    <li>
                        Domain '{item['name']}': Current accuracy {item['current_accuracy']:.2f} (Improvement: {item['improvement']:.2f})
                    </li>
                """
            html_content += """
                </ul>
            """
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        html_file = html_dir / "report.html"
        try:
            with open(html_file, 'w') as f:
                f.write(html_content)
            logger.info(f"Generated HTML report: {html_file}")
            return str(html_file)
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return None
            
    def _print_report_summary(self, report):
        """Print a summary of the performance report.
        
        Args:
            report: Report data
        """
        console.print("\n[bold]Performance Report Summary[/bold]")
        
        # Print summary metrics
        summary = report["summary"]
        console.print(f"Learning Events: {summary['total_learning_events']}")
        console.print(f"Examples Generated: {summary['total_examples_generated']}")
        console.print(f"Initial Accuracy: {summary['initial_accuracy']:.2f}")
        console.print(f"Current Accuracy: {summary['current_accuracy']:.2f}")
        console.print(f"Absolute Improvement: {summary['absolute_improvement']:.2f}")
        console.print(f"Relative Improvement: {summary['relative_improvement']:.1f}%")
        
        # Print top recommendations
        console.print("\n[bold]Top Recommendations:[/bold]")
        for i, rec in enumerate(report["recommendations"][:5]):  # Show top 5
            priority_color = {
                "critical": "red",
                "high": "yellow",
                "medium": "blue",
                "low": "green"
            }.get(rec["priority"], "white")
            
            console.print(f"[{priority_color}]{i+1}. {rec['recommendation']}[/{priority_color}]")
            
        # Print regression areas if any
        regression_areas = report["improvement_areas"]["regression_areas"]
        if regression_areas:
            console.print("\n[bold red]Regression Areas:[/bold red]")
            for area in regression_areas:
                console.print(f"[red]{area['type'].capitalize()} '{area['name']}': Decreased by {area['regression']:.2f}[/red]")
        
    def generate_performance_visualizations(self, report_time):
        """Generate visualizations for performance report.
        
        Args:
            report_time: Report timestamp
        """
        metrics = self.get_performance_metrics()
        
        if metrics["total_events"] == 0:
            return
            
        # Extract data for plotting
        timestamps = [t for t, _ in metrics["accuracy_history"]]
        accuracies = [a for _, a in metrics["accuracy_history"]]
        events = list(range(1, len(timestamps) + 1))
        
        # Create accuracy over time plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Accuracy plot
        ax1.plot(events, accuracies, 'o-', color='blue', linewidth=2)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Over Learning Events')
        ax1.grid(True, linestyle='--', alpha=0.7)
        if accuracies:
            min_acc = min(0.5, min(accuracies) - 0.05)
            ax1.set_ylim([min_acc, 1.0])
            
        # Add improvement annotation
        if len(accuracies) >= 2:
            improvement = accuracies[-1] - accuracies[0]
            ax1.annotate(f'Improvement: +{improvement:.2f}', 
                        xy=(events[-1], accuracies[-1]),
                        xytext=(events[-1] - 2, accuracies[-1] + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=12)
        
        # Example count plot
        example_counts = [c for _, c in metrics["example_count_history"]]
        ax2.bar(events, example_counts, color='green', alpha=0.7)
        ax2.set_xlabel('Learning Event')
        ax2.set_ylabel('New Examples')
        ax2.set_title('Examples Generated per Learning Event')
        
        # Set integer ticks on x-axis
        ax2.set_xticks(events)
        
        plt.tight_layout()
        
        # Save plot
        acc_plot_path = self.reports_dir / f"accuracy_trend_{report_time}.png"
        plt.savefig(acc_plot_path)
        plt.close(fig)
        
        # Create category improvement plot if data exists
        if metrics["category_improvement"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort categories by improvement
            categories = sorted(metrics["category_improvement"].items(), key=lambda x: x[1])
            cat_names = [cat for cat, _ in categories]
            improvements = [imp for _, imp in categories]
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(cat_names))
            bars = ax.barh(y_pos, improvements, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cat_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Improvement in Accuracy')
            ax.set_title('Improvement by Category')
            
            # Color bars based on improvement
            for i, bar in enumerate(bars):
                if improvements[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.tight_layout()
            
            # Save plot
            cat_plot_path = self.reports_dir / f"category_improvement_{report_time}.png"
            plt.savefig(cat_plot_path)
            plt.close(fig)
            
    def compare_learning_cycles(self, cycles=None):
        """Compare specific learning cycles or most recent ones.
        
        Args:
            cycles: List of cycle indices to compare, or None for last 5
            
        Returns:
            dict: Comparison results
        """
        training_events = self._get_training_events()
        
        if not training_events:
            return {"error": "No training events found"}
            
        # Select cycles to compare
        if cycles is None:
            # Default to last 5 cycles or all if less than 5
            cycles = list(range(max(0, len(training_events) - 5), len(training_events)))
        
        # Validate cycle indices
        valid_cycles = [i for i in cycles if 0 <= i < len(training_events)]
        if not valid_cycles:
            return {"error": "No valid cycle indices provided"}
            
        # Extract metrics for selected cycles
        selected_events = [training_events[i] for i in valid_cycles]
        
        # Compare metrics
        comparisons = []
        
        for event in selected_events:
            metrics = event.get("metrics", {})
            
            comparison = {
                "cycle": valid_cycles[selected_events.index(event)],
                "timestamp": event.get("timestamp", "")[:19],  # Trim timestamp
                "accuracy": metrics.get("current_accuracy", 0),
                "examples_generated": metrics.get("examples_generated", 0),
                "improvement_areas": metrics.get("improvement_areas", []),
                "category_accuracies": metrics.get("category_accuracies", {}),
                "domain_accuracies": metrics.get("domain_accuracies", {}),
            }
            
            comparisons.append(comparison)
            
        # Calculate differences between consecutive cycles
        cycle_differences = []
        
        for i in range(1, len(comparisons)):
            prev = comparisons[i-1]
            curr = comparisons[i]
            
            diff = {
                "cycles": f"{prev['cycle']} → {curr['cycle']}",
                "accuracy_change": curr["accuracy"] - prev["accuracy"],
                "examples_added": curr["examples_generated"],
            }
            
            # Calculate category accuracy changes
            category_changes = {}
            for category in set(list(prev["category_accuracies"].keys()) + list(curr["category_accuracies"].keys())):
                prev_acc = prev["category_accuracies"].get(category, 0)
                curr_acc = curr["category_accuracies"].get(category, 0)
                category_changes[category] = curr_acc - prev_acc
                
            diff["category_changes"] = category_changes
            
            # Calculate domain accuracy changes
            domain_changes = {}
            for domain in set(list(prev["domain_accuracies"].keys()) + list(curr["domain_accuracies"].keys())):
                prev_acc = prev["domain_accuracies"].get(domain, 0)
                curr_acc = curr["domain_accuracies"].get(domain, 0)
                domain_changes[domain] = curr_acc - prev_acc
                
            diff["domain_changes"] = domain_changes
            
            cycle_differences.append(diff)
            
        return {
            "cycles_compared": valid_cycles,
            "cycle_metrics": comparisons,
            "cycle_differences": cycle_differences,
            "overall_improvement": comparisons[-1]["accuracy"] - comparisons[0]["accuracy"] if comparisons else 0
        }
        
    def print_performance_summary(self):
        """Print performance summary to console."""
        metrics = self.get_performance_metrics()
        
        if metrics["total_events"] == 0:
            console.print("[yellow]No training events found. Cannot display performance summary.[/yellow]")
            return
            
        console.print("\n[bold]Performance Summary[/bold]")
        console.print(f"Total learning events: {metrics['total_events']}")
        console.print(f"Total examples generated: {metrics['total_examples']}")
        console.print(f"Initial accuracy: {metrics['initial_accuracy']:.2f}")
        console.print(f"Current accuracy: {metrics['current_accuracy']:.2f}")
        console.print(f"Overall improvement: {metrics['improvement']:.2f} ({(metrics['improvement'] / metrics['initial_accuracy']) * 100:.1f}% relative improvement)")
        
        # Most improved categories
        if metrics["category_improvement"]:
            console.print("\n[bold]Top Improved Categories:[/bold]")
            
            category_table = Table()
            category_table.add_column("Category", style="cyan")
            category_table.add_column("Initial", style="yellow")
            category_table.add_column("Current", style="green")
            category_table.add_column("Improvement", style="blue")
            
            # Sort by improvement
            sorted_categories = sorted(metrics["category_improvement"].items(), key=lambda x: x[1], reverse=True)
            
            for category, improvement in sorted_categories[:5]:  # Show top 5
                if category in metrics["category_accuracies"] and len(metrics["category_accuracies"][category]) >= 2:
                    initial = metrics["category_accuracies"][category][0]
                    current = metrics["category_accuracies"][category][-1]
                    category_table.add_row(
                        category,
                        f"{initial:.2f}",
                        f"{current:.2f}",
                        f"{improvement:.2f}"
                    )
                    
            console.print(category_table)
            
        # Most improved domains
        if metrics["domain_improvement"]:
            console.print("\n[bold]Top Improved Domains:[/bold]")
            
            domain_table = Table()
            domain_table.add_column("Domain", style="cyan")
            domain_table.add_column("Initial", style="yellow")
            domain_table.add_column("Current", style="green")
            domain_table.add_column("Improvement", style="blue")
            
            # Sort by improvement
            sorted_domains = sorted(metrics["domain_improvement"].items(), key=lambda x: x[1], reverse=True)
            
            for domain, improvement in sorted_domains[:5]:  # Show top 5
                if domain in metrics["domain_accuracies"] and len(metrics["domain_accuracies"][domain]) >= 2:
                    initial = metrics["domain_accuracies"][domain][0]
                    current = metrics["domain_accuracies"][domain][-1]
                    domain_table.add_row(
                        domain,
                        f"{initial:.2f}",
                        f"{current:.2f}",
                        f"{improvement:.2f}"
                    )
                    
            console.print(domain_table)
            
    def batch_export_reports(self, report_ids=None, output_format="html"):
        """Export multiple performance reports as a single comparative analysis.
        
        Args:
            report_ids: List of report IDs to include. If None, use the most recent reports.
            output_format: Output format (html or json)
            
        Returns:
            str: Path to the exported report file
        """
        # If no report IDs provided, get the most recent reports (up to 5)
        if report_ids is None:
            report_files = sorted(list(self.reports_dir.glob("performance_report_*.json")), 
                                key=lambda x: x.stat().st_mtime, reverse=True)
            report_ids = [f.stem.replace("performance_report_", "") for f in report_files[:5]]
            
        if not report_ids:
            console.print("[yellow]No reports found for batch export.[/yellow]")
            return None
            
        # Load all reports
        reports = []
        for report_id in report_ids:
            report_file = self.reports_dir / f"performance_report_{report_id}.json"
            if not report_file.exists():
                logger.warning(f"Report file not found: {report_file}")
                continue
                
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    reports.append(report)
            except Exception as e:
                logger.error(f"Error loading report {report_file}: {str(e)}")
                continue
                
        if not reports:
            console.print("[yellow]No valid reports found for batch export.[/yellow]")
            return None
            
        # Sort reports by timestamp
        reports = sorted(reports, key=lambda x: x.get("timestamp", ""))
        
        # Create batch report
        batch_report = {
            "timestamp": datetime.now().isoformat(),
            "report_count": len(reports),
            "report_ids": report_ids,
            "comparative_analysis": self._generate_comparative_analysis(reports),
            "individual_reports": reports
        }
        
        # Create export directory
        export_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.reports_dir / f"batch_export_{export_time}"
        export_dir.mkdir(exist_ok=True, parents=True)
        
        # Save batch report as JSON
        json_file = export_dir / "batch_report.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(batch_report, f, indent=2)
            logger.info(f"Saved batch report to {json_file}")
        except Exception as e:
            logger.error(f"Error saving batch report: {str(e)}")
            return None
            
        # Generate visualizations
        self._generate_comparative_visualizations(reports, export_dir, export_time)
        
        # Generate HTML report if requested
        if output_format.lower() == "html":
            html_file = self._generate_batch_html_report(batch_report, export_dir, export_time)
            if html_file:
                return str(html_file)
        
        return str(json_file)
        
    def _generate_comparative_analysis(self, reports):
        """Generate comparative analysis from multiple reports.
        
        Args:
            reports: List of reports to analyze
            
        Returns:
            dict: Comparative analysis
        """
        if not reports:
            return {}
            
        # Initialize analysis
        analysis = {
            "time_period": {
                "start": reports[0].get("timestamp", ""),
                "end": reports[-1].get("timestamp", "")
            },
            "accuracy_trend": [],
            "improvement_trend": [],
            "category_trends": {},
            "domain_trends": {},
            "top_improved_categories": [],
            "top_improved_domains": [],
            "regression_areas": [],
            "overall_improvement": 0
        }
        
        # Extract metrics from each report
        for report in reports:
            metrics = report.get("metrics", {})
            summary = report.get("summary", {})
            
            # Add to accuracy trend
            analysis["accuracy_trend"].append({
                "timestamp": report.get("timestamp", ""),
                "accuracy": metrics.get("current_accuracy", 0)
            })
            
            # Add to improvement trend
            analysis["improvement_trend"].append({
                "timestamp": report.get("timestamp", ""),
                "improvement": metrics.get("improvement", 0)
            })
            
            # Update category trends
            for category, accuracy in metrics.get("category_accuracies", {}).items():
                if category not in analysis["category_trends"]:
                    analysis["category_trends"][category] = []
                    
                if accuracy:
                    analysis["category_trends"][category].append({
                        "timestamp": report.get("timestamp", ""),
                        "accuracy": accuracy[-1] if accuracy else 0
                    })
                    
            # Update domain trends
            for domain, accuracy in metrics.get("domain_accuracies", {}).items():
                if domain not in analysis["domain_trends"]:
                    analysis["domain_trends"][domain] = []
                    
                if accuracy:
                    analysis["domain_trends"][domain].append({
                        "timestamp": report.get("timestamp", ""),
                        "accuracy": accuracy[-1] if accuracy else 0
                    })
        
        # Calculate overall improvement (from first to last report)
        if len(reports) >= 2:
            first_accuracy = reports[0].get("metrics", {}).get("current_accuracy", 0)
            last_accuracy = reports[-1].get("metrics", {}).get("current_accuracy", 0)
            analysis["overall_improvement"] = last_accuracy - first_accuracy
        
        # Identify top improved categories
        category_improvements = {}
        for category, trend in analysis["category_trends"].items():
            if len(trend) >= 2:
                improvement = trend[-1]["accuracy"] - trend[0]["accuracy"]
                category_improvements[category] = improvement
                
        analysis["top_improved_categories"] = sorted(
            [{"name": k, "improvement": v} for k, v in category_improvements.items()],
            key=lambda x: x["improvement"],
            reverse=True
        )[:5]  # Top 5
        
        # Identify top improved domains
        domain_improvements = {}
        for domain, trend in analysis["domain_trends"].items():
            if len(trend) >= 2:
                improvement = trend[-1]["accuracy"] - trend[0]["accuracy"]
                domain_improvements[domain] = improvement
                
        analysis["top_improved_domains"] = sorted(
            [{"name": k, "improvement": v} for k, v in domain_improvements.items()],
            key=lambda x: x["improvement"],
            reverse=True
        )[:5]  # Top 5
        
        # Identify regression areas
        for category, improvement in category_improvements.items():
            if improvement < 0:
                analysis["regression_areas"].append({
                    "name": category,
                    "type": "category",
                    "regression": abs(improvement)
                })
                
        for domain, improvement in domain_improvements.items():
            if improvement < 0:
                analysis["regression_areas"].append({
                    "name": domain,
                    "type": "domain",
                    "regression": abs(improvement)
                })
                
        analysis["regression_areas"] = sorted(
            analysis["regression_areas"],
            key=lambda x: x["regression"],
            reverse=True
        )
        
        return analysis
        
    def _generate_comparative_visualizations(self, reports, export_dir, export_time):
        """Generate visualizations for comparative analysis.
        
        Args:
            reports: List of reports
            export_dir: Directory to save visualizations
            export_time: Export timestamp
        """
        if not reports or len(reports) < 2:
            return
            
        # Extract data for plotting
        timestamps = []
        accuracies = []
        improvements = []
        
        for report in reports:
            timestamp = datetime.fromisoformat(report.get("timestamp", datetime.now().isoformat()))
            timestamps.append(timestamp)
            accuracies.append(report.get("metrics", {}).get("current_accuracy", 0))
            improvements.append(report.get("metrics", {}).get("improvement", 0))
            
        # Create accuracy trend plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, accuracies, 'o-', color='blue', label='Accuracy')
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Trend Across Reports')
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save accuracy trend plot
        acc_plot_path = export_dir / f"accuracy_trend_{export_time}.png"
        plt.savefig(acc_plot_path)
        plt.close(fig)
        
        # Create improvement trend plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, improvements, 'o-', color='green', label='Improvement')
        ax.set_xlabel('Time')
        ax.set_ylabel('Improvement')
        ax.set_title('Improvement Trend Across Reports')
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save improvement trend plot
        imp_plot_path = export_dir / f"improvement_trend_{export_time}.png"
        plt.savefig(imp_plot_path)
        plt.close(fig)
        
        # Create category comparison plot (for top categories)
        category_data = {}
        for report in reports:
            timestamp = datetime.fromisoformat(report.get("timestamp", datetime.now().isoformat()))
            for category, accuracies in report.get("metrics", {}).get("category_accuracies", {}).items():
                if category not in category_data:
                    category_data[category] = {"timestamps": [], "accuracies": []}
                    
                if accuracies:
                    category_data[category]["timestamps"].append(timestamp)
                    category_data[category]["accuracies"].append(accuracies[-1])
                    
        # Select top 5 categories with most data points
        top_categories = sorted(
            [(k, len(v["timestamps"])) for k, v in category_data.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if top_categories:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for category, _ in top_categories:
                data = category_data[category]
                ax.plot(data["timestamps"], data["accuracies"], 'o-', label=category)
                
            ax.set_xlabel('Time')
            ax.set_ylabel('Accuracy')
            ax.set_title('Category Accuracy Trends')
            ax.grid(True)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save category trend plot
            cat_plot_path = export_dir / f"category_trends_{export_time}.png"
            plt.savefig(cat_plot_path)
            plt.close(fig)
            
    def _generate_batch_html_report(self, batch_report, export_dir, export_time):
        """Generate HTML report for batch export.
        
        Args:
            batch_report: Batch report data
            export_dir: Directory to save the report
            export_time: Export timestamp
            
        Returns:
            str: Path to HTML report
        """
        comparative = batch_report["comparative_analysis"]
        reports = batch_report["individual_reports"]
        
        if not reports:
            return None
            
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Healthcare Contradiction Detection - Comparative Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric {{ background-color: #e9ecef; padding: 10px; margin: 5px; border-radius: 5px; width: 200px; }}
                .metric h3 {{ margin-top: 0; }}
                .improvement {{ color: green; }}
                .regression {{ color: red; }}
                .visualization {{ margin: 20px 0; }}
                .recommendations {{ margin-top: 20px; }}
                .critical {{ background-color: #f8d7da; padding: 10px; margin: 5px; border-radius: 5px; }}
                .high {{ background-color: #fff3cd; padding: 10px; margin: 5px; border-radius: 5px; }}
                .medium {{ background-color: #d1ecf1; padding: 10px; margin: 5px; border-radius: 5px; }}
                .low {{ background-color: #d4edda; padding: 10px; margin: 5px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Healthcare Contradiction Detection - Comparative Analysis</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Analysis Period: {comparative.get("time_period", {}).get("start", "")} to {comparative.get("time_period", {}).get("end", "")}</p>
            <p>Reports Analyzed: {batch_report["report_count"]}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Initial Accuracy</h3>
                        <p>{reports[0].get("metrics", {}).get("current_accuracy", 0):.2f}</p>
                    </div>
                    <div class="metric">
                        <h3>Final Accuracy</h3>
                        <p>{reports[-1].get("metrics", {}).get("current_accuracy", 0):.2f}</p>
                    </div>
                    <div class="metric">
                        <h3>Overall Improvement</h3>
                        <p class="{'improvement' if comparative.get('overall_improvement', 0) >= 0 else 'regression'}">
                            {comparative.get("overall_improvement", 0):.2f}
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="visualization">
                <h2>Performance Visualizations</h2>
                <img src="accuracy_trend_{export_time}.png" alt="Accuracy Trend" style="max-width: 100%;">
                <img src="improvement_trend_{export_time}.png" alt="Improvement Trend" style="max-width: 100%;">
        """
        
        # Add category trends visualization if it exists
        cat_viz_path = export_dir / f"category_trends_{export_time}.png"
        if cat_viz_path.exists():
            html_content += f"""
                <img src="category_trends_{export_time}.png" alt="Category Trends" style="max-width: 100%;">
            """
            
        html_content += """
            </div>
            
            <div class="top-improvements">
                <h2>Top Improvements</h2>
        """
        
        # Add top improved categories
        if comparative.get("top_improved_categories"):
            html_content += """
                <h3>Top Improved Categories</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Improvement</th>
                    </tr>
            """
            
            for item in comparative.get("top_improved_categories", []):
                html_content += f"""
                    <tr>
                        <td>{item.get("name", "")}</td>
                        <td class="{'improvement' if item.get('improvement', 0) >= 0 else 'regression'}">{item.get("improvement", 0):.2f}</td>
                    </tr>
                """
                
            html_content += """
                </table>
            """
            
        # Add top improved domains
        if comparative.get("top_improved_domains"):
            html_content += """
                <h3>Top Improved Domains</h3>
                <table>
                    <tr>
                        <th>Domain</th>
                        <th>Improvement</th>
                    </tr>
            """
            
            for item in comparative.get("top_improved_domains", []):
                html_content += f"""
                    <tr>
                        <td>{item.get("name", "")}</td>
                        <td class="{'improvement' if item.get('improvement', 0) >= 0 else 'regression'}">{item.get("improvement", 0):.2f}</td>
                    </tr>
                """
                
            html_content += """
                </table>
            """
            
        html_content += """
            </div>
        """
        
        # Add regression areas if any
        if comparative.get("regression_areas"):
            html_content += """
                <div class="regression-areas">
                    <h2>Regression Areas</h2>
                    <table>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Regression</th>
                        </tr>
            """
            
            for item in comparative.get("regression_areas", []):
                html_content += f"""
                    <tr>
                        <td>{item.get("name", "")}</td>
                        <td>{item.get("type", "").capitalize()}</td>
                        <td class="regression">{item.get("regression", 0):.2f}</td>
                    </tr>
                """
                
            html_content += """
                    </table>
                </div>
            """
            
        # Add report comparison table
        html_content += """
            <div class="report-comparison">
                <h2>Report Comparison</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Accuracy</th>
                        <th>Improvement</th>
                        <th>Examples</th>
                    </tr>
        """
        
        for report in reports:
            timestamp = datetime.fromisoformat(report.get("timestamp", "")).strftime("%Y-%m-%d %H:%M:%S")
            accuracy = report.get("metrics", {}).get("current_accuracy", 0)
            improvement = report.get("metrics", {}).get("improvement", 0)
            examples = report.get("metrics", {}).get("total_examples", 0)
            
            html_content += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{accuracy:.2f}</td>
                    <td class="{'improvement' if improvement >= 0 else 'regression'}">{improvement:.2f}</td>
                    <td>{examples}</td>
                </tr>
            """
            
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        html_file = export_dir / "comparative_analysis.html"
        try:
            with open(html_file, 'w') as f:
                f.write(html_content)
            logger.info(f"Generated batch HTML report: {html_file}")
            return str(html_file)
        except Exception as e:
            logger.error(f"Error generating batch HTML report: {str(e)}")
            return None
            
if __name__ == "__main__":
    # Simple test if run directly
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Performance analyzer for healthcare learning")
    parser.add_argument("--data-dir", type=str, default="data/healthcare", help="Path to healthcare data directory")
    parser.add_argument("--generate-report", action="store_true", help="Generate performance report")
    parser.add_argument("--compare-cycles", type=str, help="Compare specific learning cycles (comma-separated indices)")
    parser.add_argument("--batch-export", action="store_true", help="Export multiple performance reports as a single comparative analysis")
    parser.add_argument("--batch-report-ids", type=str, help="List of report IDs to include in batch export (comma-separated)")
    parser.add_argument("--batch-output-format", type=str, default="html", help="Output format for batch export (html or json)")
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(data_dir=args.data_dir)
    
    if args.generate_report:
        report_path = analyzer.generate_performance_report()
        if report_path:
            print(f"Performance report generated: {report_path}")
    elif args.compare_cycles:
        try:
            cycles = [int(i.strip()) for i in args.compare_cycles.split(",")]
            comparison = analyzer.compare_learning_cycles(cycles)
            print(f"Comparison of cycles {cycles}:")
            print(f"Overall improvement: {comparison['overall_improvement']:.2f}")
        except Exception as e:
            print(f"Error comparing cycles: {str(e)}")
    elif args.batch_export:
        report_ids = None
        if args.batch_report_ids:
            report_ids = [i.strip() for i in args.batch_report_ids.split(",")]
        batch_report_path = analyzer.batch_export_reports(report_ids, args.batch_output_format)
        if batch_report_path:
            print(f"Batch report generated: {batch_report_path}")
    else:
        analyzer.print_performance_summary()
