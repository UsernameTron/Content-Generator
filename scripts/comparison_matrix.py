#!/usr/bin/env python3
"""
Comparison Matrix Generator for Healthcare Contradiction Detection Dashboard
Generates visual comparison matrices of performance metrics across different test configurations
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class ComparisonMatrixGenerator:
    """Generates comparison matrices for test results across different configurations"""
    
    def __init__(self, results_dir=None, output_dir=None):
        """Initialize the comparison matrix generator
        
        Args:
            results_dir: Directory containing batch test results
            output_dir: Directory to save visualization outputs
        """
        self.results_dir = results_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports", "batch_tests"
        )
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports", "comparisons"
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data structures
        self.results_data = {}
        self.comparison_metrics = [
            "accuracy", "precision", "recall", "f1_score", 
            "contradiction_detection_rate", "processing_time"
        ]
        
    def load_results(self, batch_id=None):
        """Load test results from the results directory
        
        Args:
            batch_id: Specific batch ID to load, or latest if None
        
        Returns:
            Dictionary of loaded results
        """
        if not os.path.exists(self.results_dir):
            console.print(f"[red]Error: Results directory not found: {self.results_dir}[/red]")
            return {}
            
        # Find the latest batch if no specific ID provided
        if batch_id is None:
            batch_dirs = [d for d in os.listdir(self.results_dir) 
                         if os.path.isdir(os.path.join(self.results_dir, d))]
            if not batch_dirs:
                console.print("[yellow]No batch test results found.[/yellow]")
                return {}
            batch_id = sorted(batch_dirs)[-1]  # Get the latest batch
        
        batch_path = os.path.join(self.results_dir, batch_id)
        if not os.path.exists(batch_path):
            console.print(f"[red]Error: Batch ID not found: {batch_id}[/red]")
            return {}
            
        # Load all result files in the batch directory
        self.results_data = {}
        for filename in os.listdir(batch_path):
            if filename.endswith('.json'):
                preset_name = filename.replace('_results.json', '')
                with open(os.path.join(batch_path, filename), 'r') as f:
                    try:
                        self.results_data[preset_name] = json.load(f)
                    except json.JSONDecodeError:
                        console.print(f"[red]Error: Invalid JSON in {filename}[/red]")
        
        console.print(f"[green]Loaded results for {len(self.results_data)} presets from batch {batch_id}[/green]")
        return self.results_data
    
    def generate_comparison_table(self):
        """Generate a rich console table comparing key metrics across presets
        
        Returns:
            Rich Table object
        """
        if not self.results_data:
            console.print("[yellow]No results data loaded. Run load_results() first.[/yellow]")
            return None
            
        table = Table(title="Configuration Preset Performance Comparison")
        
        # Add columns
        table.add_column("Metric", style="cyan")
        for preset_name in sorted(self.results_data.keys()):
            table.add_column(preset_name, style="green")
            
        # Add rows for each metric
        for metric in self.comparison_metrics:
            row = [metric.replace('_', ' ').title()]
            for preset_name in sorted(self.results_data.keys()):
                # Get metric value, handling nested structures
                value = self._get_nested_value(self.results_data[preset_name], metric)
                
                # Format the value based on the metric type
                if metric == "processing_time":
                    formatted_value = f"{value:.2f}s" if value is not None else "N/A"
                elif isinstance(value, float):
                    formatted_value = f"{value:.4f}" if value is not None else "N/A"
                else:
                    formatted_value = str(value) if value is not None else "N/A"
                    
                row.append(formatted_value)
            table.add_row(*row)
            
        return table
    
    def _get_nested_value(self, data, key_path):
        """Get a value from a nested dictionary using a dot-separated path
        
        Args:
            data: Dictionary to extract value from
            key_path: Dot-separated path to the value
            
        Returns:
            The value at the specified path, or None if not found
        """
        if '.' in key_path:
            parts = key_path.split('.')
            current = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        else:
            return data.get(key_path, None)
    
    def generate_heatmap(self, normalize=True, filename=None):
        """Generate a heatmap visualization of metrics across presets
        
        Args:
            normalize: Whether to normalize values for better visualization
            filename: Output filename, or auto-generated if None
            
        Returns:
            Path to the saved heatmap image
        """
        if not self.results_data:
            console.print("[yellow]No results data loaded. Run load_results() first.[/yellow]")
            return None
            
        # Prepare data for heatmap
        metrics = []
        presets = sorted(self.results_data.keys())
        
        for metric in self.comparison_metrics:
            metric_values = []
            for preset_name in presets:
                value = self._get_nested_value(self.results_data[preset_name], metric)
                # Convert to float if possible, otherwise use 0
                try:
                    metric_values.append(float(value) if value is not None else 0.0)
                except (ValueError, TypeError):
                    metric_values.append(0.0)
            metrics.append(metric_values)
            
        # Create DataFrame for heatmap
        df = pd.DataFrame(metrics, 
                         index=[m.replace('_', ' ').title() for m in self.comparison_metrics],
                         columns=presets)
        
        # Normalize if requested
        if normalize:
            # Normalize each row (metric) to 0-1 scale
            for idx, row in enumerate(metrics):
                min_val = min(row)
                max_val = max(row)
                if max_val > min_val:  # Avoid division by zero
                    metrics[idx] = [(x - min_val) / (max_val - min_val) for x in row]
            
            # Update DataFrame with normalized values
            df = pd.DataFrame(metrics, 
                             index=[m.replace('_', ' ').title() for m in self.comparison_metrics],
                             columns=presets)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(df.values, cmap='viridis', aspect='auto')
        
        # Add labels and colorbar
        plt.colorbar(label='Normalized Performance' if normalize else 'Value')
        plt.xticks(np.arange(len(presets)), presets, rotation=45, ha='right')
        plt.yticks(np.arange(len(self.comparison_metrics)), 
                  [m.replace('_', ' ').title() for m in self.comparison_metrics])
        
        # Add value annotations
        for i in range(len(self.comparison_metrics)):
            for j in range(len(presets)):
                original_value = self._get_nested_value(
                    self.results_data[presets[j]], 
                    self.comparison_metrics[i]
                )
                
                if original_value is not None:
                    if isinstance(original_value, float):
                        text = f"{original_value:.3f}"
                    else:
                        text = str(original_value)
                    
                    # Choose text color based on background darkness
                    color = 'white' if df.values[i, j] > 0.5 else 'black'
                    plt.text(j, i, text, ha='center', va='center', color=color)
        
        plt.title('Configuration Preset Performance Comparison')
        plt.tight_layout()
        
        # Save the figure
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comparison_matrix_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Heatmap saved to: {output_path}[/green]")
        return output_path
    
    def generate_radar_chart(self, filename=None):
        """Generate a radar chart comparing presets across metrics
        
        Args:
            filename: Output filename, or auto-generated if None
            
        Returns:
            Path to the saved radar chart image
        """
        if not self.results_data:
            console.print("[yellow]No results data loaded. Run load_results() first.[/yellow]")
            return None
            
        # Filter metrics to only include those that make sense in a radar chart
        radar_metrics = [m for m in self.comparison_metrics if m != "processing_time"]
        
        # Prepare data for radar chart
        presets = sorted(self.results_data.keys())
        metrics_values = {}
        
        for preset_name in presets:
            values = []
            for metric in radar_metrics:
                value = self._get_nested_value(self.results_data[preset_name], metric)
                # Convert to float if possible, otherwise use 0
                try:
                    values.append(float(value) if value is not None else 0.0)
                except (ValueError, TypeError):
                    values.append(0.0)
            metrics_values[preset_name] = values
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add metric labels
        metric_labels = [m.replace('_', ' ').title() for m in radar_metrics]
        plt.xticks(angles[:-1], metric_labels, size=12)
        
        # Plot each preset
        colors = plt.cm.tab10(np.linspace(0, 1, len(presets)))
        for i, preset_name in enumerate(presets):
            values = metrics_values[preset_name]
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=preset_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Customize the chart
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Configuration Preset Performance Comparison', size=15)
        
        # Save the figure
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"radar_comparison_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Radar chart saved to: {output_path}[/green]")
        return output_path
    
    def generate_bar_chart(self, metric, filename=None):
        """Generate a bar chart for a specific metric across presets
        
        Args:
            metric: The metric to visualize
            filename: Output filename, or auto-generated if None
            
        Returns:
            Path to the saved bar chart image
        """
        if not self.results_data:
            console.print("[yellow]No results data loaded. Run load_results() first.[/yellow]")
            return None
            
        if metric not in self.comparison_metrics:
            console.print(f"[yellow]Metric '{metric}' not found in comparison metrics.[/yellow]")
            return None
            
        # Prepare data for bar chart
        presets = sorted(self.results_data.keys())
        values = []
        
        for preset_name in presets:
            value = self._get_nested_value(self.results_data[preset_name], metric)
            # Convert to float if possible, otherwise use 0
            try:
                values.append(float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                values.append(0.0)
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(presets, values, color=plt.cm.viridis(np.linspace(0, 1, len(presets))))
        
        # Add labels and title
        plt.xlabel('Configuration Preset')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Presets')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save the figure
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{metric}_comparison_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Bar chart for {metric} saved to: {output_path}[/green]")
        return output_path
    
    def generate_all_visualizations(self, batch_id=None):
        """Generate all visualizations for a batch of test results
        
        Args:
            batch_id: Specific batch ID to visualize, or latest if None
            
        Returns:
            Dictionary of paths to generated visualizations
        """
        # Load results if not already loaded
        if not self.results_data:
            self.load_results(batch_id)
            
        if not self.results_data:
            console.print("[yellow]No results data available to visualize.[/yellow]")
            return {}
            
        # Create timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate all visualizations
        visualizations = {}
        
        # Generate heatmap
        heatmap_path = self.generate_heatmap(
            normalize=True, 
            filename=f"heatmap_comparison_{timestamp}.png"
        )
        visualizations['heatmap'] = heatmap_path
        
        # Generate radar chart
        radar_path = self.generate_radar_chart(
            filename=f"radar_comparison_{timestamp}.png"
        )
        visualizations['radar'] = radar_path
        
        # Generate bar charts for each metric
        bar_charts = {}
        for metric in self.comparison_metrics:
            bar_path = self.generate_bar_chart(
                metric, 
                filename=f"{metric}_comparison_{timestamp}.png"
            )
            bar_charts[metric] = bar_path
        visualizations['bar_charts'] = bar_charts
        
        # Display comparison table in console
        table = self.generate_comparison_table()
        console.print(table)
        
        return visualizations

def main():
    """Main function to run the comparison matrix generator"""
    parser = argparse.ArgumentParser(
        description="Generate comparison matrices for healthcare contradiction detection test results"
    )
    parser.add_argument(
        "--batch-id", 
        help="Specific batch ID to visualize (default: latest batch)"
    )
    parser.add_argument(
        "--results-dir", 
        help="Directory containing batch test results"
    )
    parser.add_argument(
        "--output-dir", 
        help="Directory to save visualization outputs"
    )
    parser.add_argument(
        "--metric", 
        help="Generate bar chart for specific metric only"
    )
    parser.add_argument(
        "--heatmap-only", 
        action="store_true",
        help="Generate only the heatmap visualization"
    )
    parser.add_argument(
        "--radar-only", 
        action="store_true",
        help="Generate only the radar chart visualization"
    )
    
    args = parser.parse_args()
    
    # Create generator with specified directories
    generator = ComparisonMatrixGenerator(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    # Load results
    generator.load_results(args.batch_id)
    
    if args.metric:
        # Generate bar chart for specific metric
        generator.generate_bar_chart(args.metric)
    elif args.heatmap_only:
        # Generate only heatmap
        generator.generate_heatmap()
    elif args.radar_only:
        # Generate only radar chart
        generator.generate_radar_chart()
    else:
        # Generate all visualizations
        generator.generate_all_visualizations()
    
if __name__ == "__main__":
    main()
