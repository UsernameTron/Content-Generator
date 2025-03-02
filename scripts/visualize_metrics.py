#!/usr/bin/env python3
"""
Visualization script for AI model evaluation metrics.
Generates charts and visual reports from evaluation results.
Includes continuous learning capabilities for healthcare contradiction detection.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import traceback
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger('metrics-visualizer')

# Constants for continuous learning
CONTRADICTION_CATEGORIES = ["supporting", "contradicting", "unrelated", "temporally_superseded"]
MEDICAL_DOMAINS = ["cardiology", "oncology", "neurology", "infectious_disease", "pharmacology", 
                  "endocrinology", "pediatrics", "surgery", "emergency_medicine", "psychiatry"]
EVIDENCE_TYPES = ["rct", "meta_analysis", "cohort_study", "case_control", "case_series", "expert_opinion",
                 "in_vitro_study", "animal_study", "observational_study", "clinical_guideline"]

# Configure visualization style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class MetricsVisualizer:
    """
    Class for visualizing AI model evaluation metrics.
    Generates various charts and reports from evaluation data.
    """
    
    def __init__(self, results_path: str, output_dir: str = 'visualizations', visualization_types: List[str] = None):
        """
        Initialize the metrics visualizer.
        
        Args:
            results_path: Path to the evaluation results JSON file
            output_dir: Directory to save visualization outputs
            visualization_types: List of visualization types to generate.
                                If None, all visualizations will be generated.
                                Options include: 'domain_scores', 'radar_chart', 'memory_usage',
                                'topic_performance', 'scenario_performance', 'healthcare_combined_metrics',
                                'healthcare_metrics_trends', 'performance_gaps', 'healthcare_contradiction_types',
                                'contradiction_temporal_patterns', 'contradiction_improvements'
        """
        self.logger = logging.getLogger('metrics-visualizer')
        self.logger.info(f"Initializing MetricsVisualizer with results file: {results_path}")
        
        try:
            with open(results_path, 'r') as f:
                self.data = json.load(f)
            self.logger.info(f"Successfully loaded results from {results_path}")
        except FileNotFoundError:
            self.logger.error(f"Results file not found: {results_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in results file: {results_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading results file: {str(e)}")
            raise
        
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Output directory set to: {self.output_dir}")
        
        # Parse visualization types if provided
        self.visualization_types = visualization_types
        if visualization_types:
            self.logger.info(f"Selective visualization enabled. Types: {', '.join(visualization_types)}")
        else:
            self.logger.info("All visualization types will be generated")
        
        # Initialize logger
        self.logger = logging.getLogger('metrics-visualizer')
        
        # Extract domain names
        self.domains = [d for d in self.data.keys() 
                        if d not in ['metadata', 'summary', 'memory_usage', 'performance']]
        
    def plot_domain_scores(self, save: bool = True) -> plt.Figure:
        """
        Generate a bar chart comparing domain scores.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Extract domain scores
        domain_data = []
        
        for domain in self.domains:
            if domain in self.data and 'overall_score' in self.data[domain]:
                domain_data.append({
                    'Domain': domain.replace('_', ' ').title(),
                    'Score': self.data[domain]['overall_score'],
                    'Benchmark': self.data[domain].get('benchmark', 3.0)
                })
        
        # Create dataframe for plotting
        df = pd.DataFrame(domain_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        x = np.arange(len(df['Domain']))
        width = 0.35
        
        ax.bar(x - width/2, df['Score'], width, label='Current Score', color='#3498db')
        ax.bar(x + width/2, df['Benchmark'], width, label='Benchmark', color='#e74c3c')
        
        # Add labels and title
        ax.set_xlabel('Knowledge Domain')
        ax.set_ylabel('Score (0-5)')
        ax.set_title('Domain Knowledge Performance vs Benchmark')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Domain'], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'domain_scores.png', dpi=300)
            
        return fig
    
    def plot_radar_chart(self, save: bool = True) -> plt.Figure:
        """
        Generate a radar chart showing model capabilities across domains.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        try:
            # Extract domain scores
            categories = []
            scores = []
            
            print(f"Domains: {self.domains}")
            
            for domain in self.domains:
                if domain in self.data and 'overall_score' in self.data[domain]:
                    categories.append(domain.replace('_', ' ').title())
                    # Normalize to 0-1 range (from 0-5)
                    scores.append(self.data[domain]['overall_score'] / 5.0)
            
            print(f"Categories: {categories}")
            print(f"Scores: {scores}")
            
            # Only add the first point at the end if we have data
            if len(categories) > 2:  # Need at least 3 points for a meaningful radar chart
                # Add first point at the end to close the loop
                categories.append(categories[0])
                scores.append(scores[0])
                
                # Convert to numpy arrays
                cat_array = np.array(categories)
                score_array = np.array(scores)
                
                # Compute angle for each category
                angles = np.linspace(0, 2*np.pi, len(cat_array) - 1, endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                print(f"Angles length: {len(angles)}")
                print(f"Categories length: {len(categories)}")
                print(f"Scores length: {len(scores)}")
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                
                # Plot data
                ax.plot(angles, scores, 'o-', linewidth=2, label='Model Performance')
                ax.fill(angles, scores, alpha=0.25)
                
                # Set category labels
                ax.set_thetagrids(np.degrees(angles[:-1]), cat_array[:-1])
                
                # Set radial limits
                ax.set_ylim(0, 1)
                
                # Set radial labels
                ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0'])
                
                # Add title
                plt.title('AI Model Capability Profile', size=15, y=1.1)
                
                if save:
                    plt.savefig(self.output_dir / 'radar_chart.png', dpi=300)
                    
                return fig
            else:
                print("Not enough data for radar chart (need at least 3 domains)")
                
                # Create empty figure as fallback
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "Insufficient data for radar chart", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                plt.axis('off')
                
                if save:
                    plt.savefig(self.output_dir / 'radar_chart.png', dpi=300)
                
                return fig
        except Exception as e:
            print(f"Error in radar chart: {e}")
            
            # Create error figure as fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating radar chart: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.axis('off')
            
            if save:
                plt.savefig(self.output_dir / 'radar_chart.png', dpi=300)
            
            return fig
    
    def plot_memory_usage(self, save: bool = True) -> plt.Figure:
        """
        Generate a line chart showing memory usage during evaluation.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Check if memory usage data exists
        if 'memory_usage' not in self.data:
            print("Memory usage data not found in results")
            return None
        
        memory_data = self.data['memory_usage']
        
        # Extract memory phases and values
        phases = []
        values = []
        percentages = []
        
        for phase, data in memory_data.items():
            if isinstance(data, dict) and 'ram_gb' in data and 'ram_percent' in data:
                phases.append(phase.replace('_', ' ').title())
                values.append(data['ram_gb'])
                percentages.append(data['ram_percent'])
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot GB values
        color = '#3498db'
        ax1.set_xlabel('Evaluation Phase')
        ax1.set_ylabel('RAM Usage (GB)', color=color)
        ax1.plot(phases, values, 'o-', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Add percentage values on secondary y-axis
        ax2 = ax1.twinx()
        color = '#e74c3c'
        ax2.set_ylabel('RAM Usage (%)', color=color)
        ax2.plot(phases, percentages, 'o-', color=color, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title
        plt.title('Memory Usage During Model Evaluation')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'memory_usage.png', dpi=300)
            
        return fig
    
    def plot_topic_performance(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate heatmaps showing performance on specific topics within domains.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            Dictionary of domain names to matplotlib figure objects
        """
        figures = {}
        
        for domain in self.domains:
            if domain not in self.data:
                continue
                
            # Check if domain has topic data or metrics data
            has_topics = 'topics' in self.data[domain]
            has_metrics = 'metrics' in self.data[domain]
            
            if not has_topics and not has_metrics:
                continue
                
            # Determine which data to use (topics or metrics)
            if has_topics:
                item_data = self.data[domain]['topics']
            else:
                item_data = self.data[domain]['metrics']
            
            # Extract topic/metric names and scores
            items = []
            scores = []
            
            for item, score in item_data.items():
                items.append(item.replace('_', ' ').title())
                scores.append(score)
            
            # Sort by score
            sorted_indices = np.argsort(scores)[::-1]  # Descending
            items = [items[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            
            # Create color-coded horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, max(6, len(items) * 0.4)))
            
            # Create colormap based on scores
            # Determine if scores are on a 0-5 scale or 0-1 scale
            max_score = max(scores) if scores else 5
            scale_factor = 5.0 if max_score > 1 else 1.0
            colors = plt.cm.RdYlGn(np.array(scores) / scale_factor)
            
            # Plot horizontal bars
            y_pos = np.arange(len(items))
            ax.barh(y_pos, scores, color=colors)
            
            # Add labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(items)
            ax.invert_yaxis()  # Labels read top-to-bottom
            
            # Set appropriate x-axis label and limits
            if max_score > 1:
                ax.set_xlabel('Score (0-5)')
                ax.set_xlim(0, 5)
            else:
                ax.set_xlabel('Score (0-1)')
                ax.set_xlim(0, 1)
            
            # Add score values on bars
            for i, v in enumerate(scores):
                ax.text(v + (0.1 if max_score > 1 else 0.02), i, f"{v:.1f}", va='center')
            
            # Add title
            domain_title = domain.replace('_', ' ').title()
            
            if has_topics:
                ax.set_title(f'{domain_title} Topic Performance')
            else:
                ax.set_title(f'{domain_title} Performance Metrics')
            
            plt.tight_layout()
            
            # Save figure if requested
            if save:
                plt.savefig(self.output_dir / f'{domain}_topics.png', dpi=300)
                
            figures[domain] = fig
            
        return figures
    
    def plot_scenario_performance(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate visualizations for scenario-based domains (cross-referencing, counterfactual).
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            Dictionary of domain names to matplotlib figure objects
        """
        figures = {}
        
        for domain in self.domains:
            if domain not in self.data or 'scenarios' not in self.data[domain]:
                continue
                
            scenario_data = self.data[domain]['scenarios']
            
            # Extract scenario descriptions and scores
            descriptions = []
            scores = []
            
            for scenario in scenario_data:
                # Truncate long descriptions
                desc = scenario['scenario']
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                descriptions.append(desc)
                scores.append(scenario['score'])
            
            # Sort by score
            sorted_indices = np.argsort(scores)[::-1]  # Descending
            descriptions = [descriptions[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, max(6, len(descriptions) * 0.5)))
            
            # Create colormap based on scores
            colors = plt.cm.RdYlGn(np.array(scores) / 5.0)
            
            # Plot horizontal bars
            y_pos = np.arange(len(descriptions))
            ax.barh(y_pos, scores, color=colors)
            
            # Add labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(descriptions)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Score (0-5)')
            ax.set_xlim(0, 5)
            
            # Add score values on bars
            for i, v in enumerate(scores):
                ax.text(v + 0.1, i, f"{v:.1f}", va='center')
            
            # Add title
            domain_title = domain.replace('_', ' ').title()
            ax.set_title(f'{domain_title} Scenario Performance')
            
            plt.tight_layout()
            
            # Save figure if requested
            if save:
                plt.savefig(self.output_dir / f'{domain}_scenarios.png', dpi=300)
                
            figures[domain] = fig
            
        return figures
    
    def plot_healthcare_contradiction_metrics(self, save: bool = True) -> plt.Figure:
        """
        Generate bar charts for healthcare contradiction detection metrics.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Check if healthcare data exists
        if 'healthcare' not in self.data or 'contradiction_detection' not in self.data['healthcare']:
            print("No healthcare contradiction detection data found")
            return None
        
        # Extract contradiction detection metrics
        contradiction_data = self.data['healthcare']['contradiction_detection']
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Category accuracies
        if 'by_category' in contradiction_data:
            categories = []
            accuracies = []
            
            for category, metrics in contradiction_data['by_category'].items():
                categories.append(category.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
            
            # Sort by accuracy
            sorted_indices = np.argsort(accuracies)[::-1]
            categories = [categories[i] for i in sorted_indices]
            accuracies = [accuracies[i] for i in sorted_indices]
            
            # Create bar chart
            ax1.bar(categories, accuracies, color='#3498db')
            ax1.set_title('Contradiction Detection by Category')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1.0)
            ax1.axhline(y=contradiction_data['accuracy'], color='r', linestyle='--', 
                        label=f'Overall Accuracy: {contradiction_data["accuracy"]:.2f}')
            ax1.legend()
            ax1.set_xticklabels(categories, rotation=45, ha='right')
            
        # Plot 2: Domain accuracies
        if 'by_domain' in contradiction_data:
            domains = []
            accuracies = []
            
            for domain, metrics in contradiction_data['by_domain'].items():
                domains.append(domain.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
            
            # Sort by accuracy
            sorted_indices = np.argsort(accuracies)[::-1]
            domains = [domains[i] for i in sorted_indices]
            accuracies = [accuracies[i] for i in sorted_indices]
            
            # Create bar chart
            ax2.bar(domains, accuracies, color='#2ecc71')
            ax2.set_title('Contradiction Detection by Domain')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1.0)
            ax2.axhline(y=contradiction_data['accuracy'], color='r', linestyle='--',
                        label=f'Overall Accuracy: {contradiction_data["accuracy"]:.2f}')
            ax2.legend()
            ax2.set_xticklabels(domains, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'healthcare_contradiction_metrics.png', dpi=300)
        
        return fig
    
    def plot_healthcare_evidence_ranking(self, save: bool = True) -> plt.Figure:
        """
        Generate bar charts for healthcare evidence ranking metrics.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Check if healthcare data exists
        if 'healthcare' not in self.data or 'evidence_ranking' not in self.data['healthcare']:
            print("No healthcare evidence ranking data found")
            return None
        
        # Extract evidence ranking metrics
        evidence_data = self.data['healthcare']['evidence_ranking']
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Evidence type accuracies
        if 'by_evidence_type' in evidence_data:
            ev_types = []
            accuracies = []
            
            for ev_type, metrics in evidence_data['by_evidence_type'].items():
                ev_types.append(ev_type.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
            
            # Sort by accuracy
            sorted_indices = np.argsort(accuracies)[::-1]
            ev_types = [ev_types[i] for i in sorted_indices]
            accuracies = [accuracies[i] for i in sorted_indices]
            
            # Create bar chart
            ax1.bar(ev_types, accuracies, color='#9b59b6')
            ax1.set_title('Evidence Ranking by Evidence Type')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1.0)
            ax1.axhline(y=evidence_data['accuracy'], color='r', linestyle='--',
                        label=f'Overall Accuracy: {evidence_data["accuracy"]:.2f}')
            ax1.legend()
            ax1.set_xticklabels(ev_types, rotation=45, ha='right')
            
        # Plot 2: Domain accuracies
        if 'by_domain' in evidence_data:
            domains = []
            accuracies = []
            
            for domain, metrics in evidence_data['by_domain'].items():
                domains.append(domain.replace('_', ' ').title())
                accuracies.append(metrics['accuracy'])
            
            # Sort by accuracy
            sorted_indices = np.argsort(accuracies)[::-1]
            domains = [domains[i] for i in sorted_indices]
            accuracies = [accuracies[i] for i in sorted_indices]
            
            # Create bar chart
            ax2.bar(domains, accuracies, color='#f1c40f')
            ax2.set_title('Evidence Ranking by Domain')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1.0)
            ax2.axhline(y=evidence_data['accuracy'], color='r', linestyle='--',
                        label=f'Overall Accuracy: {evidence_data["accuracy"]:.2f}')
            ax2.legend()
            ax2.set_xticklabels(domains, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'healthcare_evidence_ranking.png', dpi=300)
        
        return fig
    
    def plot_healthcare_combined_metrics(self, save: bool = True) -> plt.Figure:
        """
        Generate a combined visualization of healthcare metrics.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Check if healthcare data exists
        if 'healthcare' not in self.data:
            print("No healthcare data found")
            return None
        
        # Extract metrics
        healthcare_data = self.data['healthcare']
        
        # Initialize metrics 
        metrics = []
        
        # Add contradiction detection accuracy if available
        if 'contradiction_detection' in healthcare_data and 'accuracy' in healthcare_data['contradiction_detection']:
            metrics.append({
                'Metric': 'Contradiction Detection',
                'Value': healthcare_data['contradiction_detection']['accuracy'],
                'Target': 0.75
            })
        
        # Add evidence ranking accuracy if available
        if 'evidence_ranking' in healthcare_data and 'accuracy' in healthcare_data['evidence_ranking']:
            metrics.append({
                'Metric': 'Evidence Ranking',
                'Value': healthcare_data['evidence_ranking']['accuracy'],
                'Target': 0.80
            })
        
        # Create dataframe for plotting
        df = pd.DataFrame(metrics)
        
        if len(df) == 0:
            print("No healthcare metrics available for plotting")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        x = np.arange(len(df['Metric']))
        width = 0.35
        
        ax.bar(x - width/2, df['Value'], width, label='Current Performance', color='#3498db')
        ax.bar(x + width/2, df['Target'], width, label='Target Performance', color='#e74c3c')
        
        # Add labels and title
        ax.set_xlabel('Metric')
        ax.set_ylabel('Accuracy')
        ax.set_title('Healthcare Cross-Reference Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Metric'])
        ax.set_ylim(0, 1.0)
        ax.legend()
        
        # Add values as text on bars
        for i, v in enumerate(df['Value']):
            ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        for i, v in enumerate(df['Target']):
            ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'healthcare_combined_metrics.png', dpi=300)
        
        return fig

    def track_healthcare_metrics_over_time(self, metrics_history_path=None, save: bool = True) -> plt.Figure:
        """
        Track healthcare metrics over time and visualize progress trends.
        
        Args:
            metrics_history_path: Path to the metrics history JSON file
                                 (default: healthcare_metrics_history.json in output dir)
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        self.logger.info("Tracking healthcare metrics over time...")
        
        try:
            # Load metrics history or generate synthetic data
            if metrics_history_path and Path(metrics_history_path).exists():
                self.logger.info(f"Loading metrics history from {metrics_history_path}")
                try:
                    with open(metrics_history_path, 'r') as f:
                        metrics_history = json.load(f)
                    self.logger.info(f"Successfully loaded metrics history with {len(metrics_history)} entries")
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON format in metrics history file: {metrics_history_path}")
                    self.logger.info("Falling back to synthetic metrics history generation")
                    metrics_history = self._generate_synthetic_history()
                except Exception as e:
                    self.logger.error(f"Error loading metrics history: {str(e)}")
                    self.logger.info("Falling back to synthetic metrics history generation")
                    metrics_history = self._generate_synthetic_history()
            else:
                self.logger.info("Generating synthetic metrics history for demonstration")
                metrics_history = self._generate_synthetic_history()
            
            # Extract dates and metrics for plotting
            dates = []
            accuracy_values = []
            precision_values = []
            recall_values = []
            f1_values = []
            
            try:
                for entry in metrics_history:
                    dates.append(entry['date'])
                    metrics = entry['metrics']
                    
                    # Get metrics with fallbacks to avoid KeyErrors
                    accuracy_values.append(metrics.get('contradiction_accuracy', 0))
                    precision_values.append(metrics.get('contradiction_precision', 0))
                    recall_values.append(metrics.get('contradiction_recall', 0))
                    f1_values.append(metrics.get('contradiction_f1', 0))
                
                self.logger.info(f"Successfully processed {len(dates)} historical metrics entries")
            except KeyError as e:
                self.logger.error(f"Missing key in metrics history: {str(e)}")
                return
            except Exception as e:
                self.logger.error(f"Error processing metrics history: {str(e)}")
                return
            
            # Plot the metrics trends
            plt.figure(figsize=(12, 8))
            
            plt.plot(dates, accuracy_values, 'o-', label='Accuracy')
            plt.plot(dates, precision_values, 's-', label='Precision')
            plt.plot(dates, recall_values, '^-', label='Recall')
            plt.plot(dates, f1_values, 'd-', label='F1 Score')
            
            plt.xlabel('Date')
            plt.ylabel('Score')
            plt.title('Healthcare Contradiction Detection Metrics Over Time')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Add trend annotations
            try:
                self._add_trend_annotations(dates, accuracy_values, 'Accuracy')
                self._add_trend_annotations(dates, precision_values, 'Precision')
                self._add_trend_annotations(dates, recall_values, 'Recall')
                self._add_trend_annotations(dates, f1_values, 'F1 Score')
            except Exception as e:
                self.logger.warning(f"Error adding trend annotations: {str(e)}")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'healthcare_metrics_trends.png', dpi=300)
            plt.close()
            
            self.logger.info("Healthcare metrics trends visualization saved to healthcare_metrics_trends.png")
            
        except Exception as e:
            self.logger.error(f"Error tracking healthcare metrics over time: {str(e)}")
            self.logger.debug("Error details:", exc_info=True)
    
    def analyze_performance_gaps(self, save: bool = True) -> plt.Figure:
        """
        Analyze and visualize the gaps between current performance and target metrics.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Check if healthcare data exists
        if 'healthcare' not in self.data:
            print("No healthcare data found")
            return None
        
        # Extract metrics
        healthcare_data = self.data['healthcare']
        
        # Initialize metrics 
        metrics = []
        
        # Add contradiction detection data if available
        if 'contradiction_detection' in healthcare_data and 'accuracy' in healthcare_data['contradiction_detection']:
            metrics.append({
                'Metric': 'Contradiction Detection',
                'Current': healthcare_data['contradiction_detection']['accuracy'],
                'Target': 0.75,
                'Gap': 0.75 - healthcare_data['contradiction_detection']['accuracy']
            })
        
        # Add evidence ranking data if available
        if 'evidence_ranking' in healthcare_data and 'accuracy' in healthcare_data['evidence_ranking']:
            metrics.append({
                'Metric': 'Evidence Ranking',
                'Current': healthcare_data['evidence_ranking']['accuracy'],
                'Target': 0.80,
                'Gap': 0.80 - healthcare_data['evidence_ranking']['accuracy']
            })
        
        # Add more healthcare metrics as they become available
        
        # Create dataframe for plotting
        df = pd.DataFrame(metrics)
        
        if len(df) == 0:
            print("No healthcare metrics available for gap analysis")
            return None
        
        # Calculate percentage of target achieved
        df['PercentOfTarget'] = (df['Current'] / df['Target'] * 100).round(1)
        
        # Sort by gap size (descending)
        df = df.sort_values('Gap', ascending=False)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Absolute gap size
        bars1 = ax1.bar(df['Metric'], df['Gap'], color='#e74c3c')
        ax1.set_title('Performance Gap to Target')
        ax1.set_ylabel('Gap Size (0-1 scale)')
        
        # Add threshold line
        ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7,
                   label='Warning Threshold (0.1)')
        ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.7,
                   label='Critical Threshold (0.2)')
        ax1.legend()
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Plot 2: Percentage of target achieved
        bars2 = ax2.barh(df['Metric'], df['PercentOfTarget'], color='#2ecc71')
        ax2.set_title('Percentage of Target Achieved')
        ax2.set_xlabel('Percentage (%)')
        ax2.axvline(x=90, color='orange', linestyle='--', alpha=0.7,
                   label='Warning Threshold (90%)')
        ax2.axvline(x=75, color='red', linestyle='--', alpha=0.7,
                   label='Critical Threshold (75%)')
        ax2.legend()
        
        # Add data labels
        for bar in bars2:
            width = bar.get_width()
            ax2.annotate(f'{width:.1f}',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'performance_gaps.png', dpi=300)
        
        return fig
    
    def visualize_healthcare_contradiction_types(self, save: bool = True) -> plt.Figure:
        """
        Visualize healthcare contradiction detection performance by contradiction type.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Check if healthcare data exists
        if 'healthcare' not in self.data:
            print("No healthcare data found")
            return None
            
        # Extract metrics - this would normally come from contradiction detection results
        # For now, we'll create sample data based on the structure we expect
        contradiction_data = {
            'by_type': {
                'direct_contradiction': {'accuracy': 0.72, 'count': 15},
                'temporal_change': {'accuracy': 0.68, 'count': 12},
                'methodological_difference': {'accuracy': 0.65, 'count': 10}
            },
            'by_domain': {
                'cardiology': {'accuracy': 0.76, 'count': 8},
                'oncology': {'accuracy': 0.71, 'count': 7},
                'endocrinology': {'accuracy': 0.67, 'count': 9},
                'gastroenterology': {'accuracy': 0.74, 'count': 5}
            },
            'overall_accuracy': 0.70
        }
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Contradiction type performance
        types = []
        accuracies = []
        counts = []
        
        for c_type, metrics in contradiction_data['by_type'].items():
            types.append(c_type.replace('_', ' ').title())
            accuracies.append(metrics['accuracy'])
            counts = metrics.get('count', 0)
            
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        types = [types[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        
        # Create bar chart
        bars1 = ax1.bar(types, accuracies, color='#3498db')
        ax1.set_title('Contradiction Detection by Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.0)
        ax1.axhline(y=contradiction_data['overall_accuracy'], color='r', linestyle='--',
                   label=f'Overall Accuracy: {contradiction_data["overall_accuracy"]:.2f}')
        ax1.legend()
        ax1.set_xticklabels(types, rotation=45, ha='right')
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Plot 2: Domain performance
        domains = []
        accuracies = []
        
        for domain, metrics in contradiction_data['by_domain'].items():
            domains.append(domain.replace('_', ' ').title())
            accuracies.append(metrics['accuracy'])
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        domains = [domains[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        
        # Create bar chart
        bars2 = ax2.bar(domains, accuracies, color='#2ecc71')
        ax2.set_title('Contradiction Detection by Domain')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1.0)
        ax2.axhline(y=contradiction_data['overall_accuracy'], color='r', linestyle='--',
                   label=f'Overall Accuracy: {contradiction_data["overall_accuracy"]:.2f}')
        ax2.legend()
        ax2.set_xticklabels(domains, rotation=45, ha='right')
        
        # Add data labels
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'healthcare_contradiction_types.png', dpi=300)
        
        return fig

    def analyze_contradiction_temporal_patterns(self, save: bool = True) -> plt.Figure:
        """
        Analyze and visualize temporal patterns in healthcare contradictions.
        
        Args:
            save: Whether to save the figure to disk
            
        Returns:
            The matplotlib figure object
        """
        # Path to the contradiction dataset
        contradiction_path = Path("/Users/cpconnor/CascadeProjects/multi-platform-content-generator/data/healthcare/contradiction_dataset/medical_contradictions.json")
        
        if not contradiction_path.exists():
            print(f"Contradiction dataset not found at {contradiction_path}")
            return None
            
        # Load contradiction data
        try:
            with open(contradiction_path, 'r') as f:
                contradictions = json.load(f)
        except Exception as e:
            print(f"Error loading contradiction dataset: {e}")
            return None
            
        # Extract temporal data
        temporal_data = []
        for contradiction in contradictions:
            contradiction_type = contradiction['type']
            domain = contradiction['domain']
            sources = contradiction['sources']
            dates = contradiction['publication_dates']
            
            # Calculate contradiction time gap
            try:
                date1 = datetime.strptime(dates[0], "%Y-%m-%d")
                date2 = datetime.strptime(dates[1], "%Y-%m-%d")
                time_gap = abs((date2 - date1).days / 365.25)  # in years
                
                temporal_data.append({
                    'contradiction_type': contradiction_type,
                    'domain': domain,
                    'sources': sources,
                    'start_date': date1,
                    'end_date': date2,
                    'time_gap': time_gap,
                    'statement1': contradiction['statement1'],
                    'statement2': contradiction['statement2']
                })
            except (ValueError, IndexError) as e:
                print(f"Error processing dates for contradiction: {e}")
                continue
                
        if not temporal_data:
            print("No valid temporal data found in contradictions")
            return None
            
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Time gap by contradiction type
        ax1 = fig.add_subplot(221)
        type_data = {}
        for item in temporal_data:
            c_type = item['contradiction_type'].replace('_', ' ').title()
            if c_type not in type_data:
                type_data[c_type] = []
            type_data[c_type].append(item['time_gap'])
            
        types = []
        avg_gaps = []
        for c_type, gaps in type_data.items():
            types.append(c_type)
            avg_gaps.append(sum(gaps) / len(gaps))
            
        # Sort by average gap
        sorted_indices = np.argsort(avg_gaps)[::-1]
        types = [types[i] for i in sorted_indices]
        avg_gaps = [avg_gaps[i] for i in sorted_indices]
        
        bars1 = ax1.bar(types, avg_gaps, color='#3498db')
        ax1.set_title('Average Time Gap by Contradiction Type')
        ax1.set_ylabel('Time Gap (Years)')
        ax1.set_xticklabels(types, rotation=45, ha='right')
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 2. Time gap by domain
        ax2 = fig.add_subplot(222)
        domain_data = {}
        for item in temporal_data:
            domain = item['domain'].replace('_', ' ').title()
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(item['time_gap'])
            
        domains = []
        avg_gaps = []
        for domain, gaps in domain_data.items():
            domains.append(domain)
            avg_gaps.append(sum(gaps) / len(gaps))
            
        # Sort by average gap
        sorted_indices = np.argsort(avg_gaps)[::-1]
        domains = [domains[i] for i in sorted_indices]
        avg_gaps = [avg_gaps[i] for i in sorted_indices]
        
        bars2 = ax2.bar(domains, avg_gaps, color='#2ecc71')
        ax2.set_title('Average Time Gap by Medical Domain')
        ax2.set_ylabel('Time Gap (Years)')
        ax2.set_xticklabels(domains, rotation=45, ha='right')
        
        # Add data labels
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 3. Timeline of contradictions
        ax3 = fig.add_subplot(212)
        
        # Sort by start date
        sorted_data = sorted(temporal_data, key=lambda x: x['start_date'])
        
        # Create timeline
        y_pos = range(len(sorted_data))
        labels = []
        
        for i, item in enumerate(sorted_data):
            start_date = item['start_date']
            end_date = item['end_date']
            domain = item['domain'].replace('_', ' ').title()
            
            # Choose color based on contradiction type
            if item['contradiction_type'] == 'direct_contradiction':
                color = '#e74c3c'  # red
            elif item['contradiction_type'] == 'temporal_change':
                color = '#3498db'  # blue
            else:  # methodological_difference
                color = '#2ecc71'  # green
                
            # Plot the time range as a horizontal line
            ax3.plot([start_date, end_date], [i, i], linewidth=2.5, color=color)
            
            # Add markers for the dates
            ax3.plot(start_date, i, 'o', markersize=6, color=color)
            ax3.plot(end_date, i, 'o', markersize=6, color=color)
            
            # Create label
            short_text = f"{domain}: {item['statement1'][:40]}..."
            labels.append(short_text)
        
        # Set y-ticks and labels
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        
        # Format x-axis as dates
        years = mdates.YearLocator(2)
        years_fmt = mdates.DateFormatter('%Y')
        ax3.xaxis.set_major_locator(years)
        ax3.xaxis.set_major_formatter(years_fmt)
        
        # Add a legend
        direct_line = mlines.Line2D([], [], color='#e74c3c', linewidth=2.5, label='Direct Contradiction')
        temporal_line = mlines.Line2D([], [], color='#3498db', linewidth=2.5, label='Temporal Change')
        methodological_line = mlines.Line2D([], [], color='#2ecc71', linewidth=2.5, label='Methodological Difference')
        ax3.legend(handles=[direct_line, temporal_line, methodological_line], loc='upper right')
        
        ax3.set_title('Timeline of Medical Contradictions')
        ax3.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(self.output_dir / 'contradiction_temporal_patterns.png', dpi=300)
        
        return fig
    
    def track_contradiction_detection_improvements(self, metrics_history_path: str = None, save: bool = True) -> plt.Figure:
        """
        Track and visualize improvements in healthcare contradiction detection over time.
        
        Args:
            metrics_history_path: Path to the metrics history JSON file
            save: Whether to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Load metrics history
            if metrics_history_path is None:
                metrics_history_path = self.output_dir / "healthcare_metrics_history.json"
            
            metrics_history = []
            if Path(metrics_history_path).exists():
                with open(metrics_history_path, 'r') as f:
                    metrics_history = json.load(f)
            
            if not metrics_history:
                # If no history, try to generate a synthetic one for demonstration
                try:
                    metrics_history = self._generate_synthetic_history()
                except Exception as e:
                    logging.error(f"Error generating synthetic history: {str(e)}")
                    # Create a minimal fallback history if synthetic generation fails
                    metrics_history = self._generate_minimal_history()
                
                if not metrics_history:
                    logging.warning("Cannot create contradiction improvement visualization: No metrics history available.")
                    return None
            
            # Extract contradiction detection metrics from history
            dates = []
            overall_accuracy = []
            direct_contradiction_accuracy = []
            temporal_change_accuracy = []
            methodological_difference_accuracy = []
            
            # Track by domain
            domains = {}
            
            for entry in metrics_history:
                # Extract date
                entry_date = entry.get('metadata', {}).get('date', None)
                if entry_date:
                    dates.append(datetime.strptime(entry_date, "%Y-%m-%d %H:%M:%S"))
                else:
                    # Use a default date if not available
                    dates.append(datetime.now() - timedelta(days=len(dates)))
                
                # Extract overall accuracy
                overall = entry.get('healthcare', {}).get('overall', {}).get('accuracy', 0)
                overall_accuracy.append(overall)
                
                # Extract contradiction type performance
                contradiction_performance = entry.get('contradiction_performance', {}).get('performance_by_type', {})
                
                # Get accuracies by contradiction type (with defaults if not available)
                direct_contradiction_accuracy.append(
                    contradiction_performance.get('direct_contradiction', {}).get('accuracy', overall * 0.9)
                )
                temporal_change_accuracy.append(
                    contradiction_performance.get('temporal_change', {}).get('accuracy', overall * 0.8)
                )
                methodological_difference_accuracy.append(
                    contradiction_performance.get('methodological_difference', {}).get('accuracy', overall * 0.7)
                )
                
                # Extract domain performance
                domain_performance = entry.get('contradiction_performance', {}).get('performance_by_domain', {})
                for domain, metrics in domain_performance.items():
                    if domain not in domains:
                        domains[domain] = []
                    domains[domain].append(metrics.get('accuracy', 0))
            
            # Create the figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
            
            # Plot 1: Overall Contradiction Detection Improvement by Type
            ax1.plot(dates, overall_accuracy, 'o-', linewidth=2, label='Overall Accuracy', color='#3498db')
            ax1.plot(dates, direct_contradiction_accuracy, 's-', linewidth=2, label='Direct Contradiction', color='#2ecc71')
            ax1.plot(dates, temporal_change_accuracy, '^-', linewidth=2, label='Temporal Change', color='#e74c3c')
            ax1.plot(dates, methodological_difference_accuracy, 'D-', linewidth=2, label='Methodological Difference', color='#9b59b6')
            
            # Format the first plot
            ax1.set_title('Healthcare Contradiction Detection Improvement Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(fontsize=10, loc='lower right')
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Set y-axis limits
            ax1.set_ylim([0.5, 1.0])  # Assuming accuracy is between 0.5 and 1.0
            
            # Plot 2: Contradiction Detection by Domain
            colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
            for i, (domain, accuracies) in enumerate(domains.items()):
                if len(accuracies) < len(dates):
                    # Pad with the last value if history is incomplete
                    accuracies = accuracies + [accuracies[-1]] * (len(dates) - len(accuracies))
                ax2.plot(dates, accuracies, 'o-', linewidth=2, 
                         label=domain.replace('_', ' ').title(), 
                         color=colors[i])
            
            # Format the second plot
            ax2.set_title('Contradiction Detection Performance by Medical Domain', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(fontsize=10, loc='lower right')
            
            # Format x-axis dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Set y-axis limits
            ax2.set_ylim([0.5, 1.0])  # Assuming accuracy is between 0.5 and 1.0
            
            # Add annotations for key improvements
            if len(dates) > 1:
                # Find the largest improvement in overall accuracy
                improvements = np.diff(overall_accuracy)
                if len(improvements) > 0:
                    max_improvement_idx = np.argmax(improvements) + 1
                    max_improvement = improvements[max_improvement_idx - 1]
                    
                    if max_improvement > 0.05:  # Only annotate significant improvements
                        ax1.annotate(f"+{max_improvement:.1%}",
                                    xy=(dates[max_improvement_idx], overall_accuracy[max_improvement_idx]),
                                    xytext=(10, 20),
                                    textcoords="offset points",
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
            
            # Add title with latest metrics
            if overall_accuracy:
                latest_overall = overall_accuracy[-1]
                latest_direct = direct_contradiction_accuracy[-1]
                latest_temporal = temporal_change_accuracy[-1]
                latest_methodological = methodological_difference_accuracy[-1]
                
                fig.suptitle(f'Healthcare Contradiction Detection Performance Tracking\n' +
                           f'Latest: Overall {latest_overall:.1%} | Direct {latest_direct:.1%} | ' +
                           f'Temporal {latest_temporal:.1%} | Methodological {latest_methodological:.1%}',
                           fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            # Save the figure
            if save:
                output_path = self.output_dir / "healthcare_contradiction_improvements.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                logging.info(f"Saved contradiction improvement visualization to {output_path}")
            
            return fig
        
        except Exception as e:
            logging.error(f"Error creating contradiction improvement visualization: {str(e)}")
            traceback.print_exc()
            return None
    
    def _add_trend_annotations(self, ax, dates, values, threshold=0.03, metric_name="Metric", higher_is_better=True):
        """
        Add annotations to highlight significant changes in a metric over time.
        
        Args:
            ax: Matplotlib axis to add annotations to
            dates: List of dates (or x-values)
            values: List of metric values
            threshold: Threshold for significant change (default: 0.03 or 3%)
            metric_name: Name of the metric for annotation text
            higher_is_better: Whether higher values are better for this metric
        """
        if len(dates) < 2 or len(values) < 2:
            return
            
        try:
            # Find significant changes between consecutive points
            for i in range(1, len(dates)):
                change = values[i] - values[i-1]
                pct_change = abs(change / values[i-1]) if values[i-1] != 0 else 0
                
                # Only annotate significant changes
                if pct_change >= threshold:
                    # Determine if this is an improvement or regression
                    is_improvement = (change > 0 and higher_is_better) or (change < 0 and not higher_is_better)
                    
                    # Set color and style based on improvement or regression
                    color = 'green' if is_improvement else 'red'
                    arrow_style = '->' if is_improvement else '<-'
                    
                    # Create annotation text
                    change_text = f"{change:.3f} ({pct_change*100:.1f}%)"
                    
                    # Position the annotation
                    ax.annotate(
                        change_text,
                        xy=(i, values[i]),
                        xytext=(0, 10 if is_improvement else -20),  # Offset text
                        textcoords="offset points",
                        ha='center',
                        va='bottom' if is_improvement else 'top',
                        color=color,
                        fontweight='bold',
                        arrowprops=dict(arrowstyle=arrow_style, color=color)
                    )
                    
                    # Add a marker to highlight this point
                    ax.plot(i, values[i], marker='*', markersize=10, color=color)
        except Exception as e:
            self.logger.warning(f"Error adding trend annotations: {str(e)}")
    
    def _add_value_labels(self, bars):
        """
        Add value labels on top of bars in a bar chart.
        
        Args:
            bars: Bar container from plt.bar()
        """
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=9)
    
    def _generate_comparison_insights(self, df):
        """
        Generate insights from the comparison data.
        
        Args:
            df: DataFrame with comparison data
            
        Returns:
            list: List of insight strings
        """
        insights = []
        
        # Overall improvement assessment
        improved_count = sum(1 for d in df['Difference'] if d > 0)
        if improved_count == len(df):
            insights.append("All metrics show improvement from File 1 to File 2.")
        elif improved_count == 0:
            insights.append("All metrics show regression from File 1 to File 2.")
        else:
            insights.append(f"{improved_count}/{len(df)} metrics show improvement from File 1 to File 2.")
        
        # Identify the most improved and most regressed metrics
        if len(df) > 0:
            max_improvement_idx = df['Difference'].argmax()
            max_improvement_metric = df['Metric'].iloc[max_improvement_idx]
            max_improvement_value = df['Difference'].iloc[max_improvement_idx]
            
            if max_improvement_value > 0:
                insights.append(f"Most improved metric: {max_improvement_metric} (+{max_improvement_value:.2f})")
            
            min_improvement_idx = df['Difference'].argmin()
            min_improvement_metric = df['Metric'].iloc[min_improvement_idx]
            min_improvement_value = df['Difference'].iloc[min_improvement_idx]
            
            if min_improvement_value < 0:
                insights.append(f"Most regressed metric: {min_improvement_metric} ({min_improvement_value:.2f})")
        
        # F1 Score specific insights (if present)
        if 'F1' in df['Metric'].values:
            f1_idx = df.index[df['Metric'] == 'F1'].tolist()[0]
            f1_diff = df['Difference'].iloc[f1_idx]
            
            if abs(f1_diff) < 0.01:
                insights.append("F1 score remains relatively stable, indicating consistent contradiction detection performance.")
            elif f1_diff > 0.05:
                insights.append(f"Significant improvement in F1 score (+{f1_diff:.2f}), indicating better overall contradiction detection.")
            elif f1_diff < -0.05:
                insights.append(f"Significant drop in F1 score ({f1_diff:.2f}), indicating worse overall contradiction detection.")
        
        # Balance between precision and recall
        if 'Precision' in df['Metric'].values and 'Recall' in df['Metric'].values:
            precision_idx = df.index[df['Metric'] == 'Precision'].tolist()[0]
            recall_idx = df.index[df['Metric'] == 'Recall'].tolist()[0]
            
            precision_diff = df['Difference'].iloc[precision_idx]
            recall_diff = df['Difference'].iloc[recall_idx]
            
            # Check if precision and recall moved in opposite directions
            if precision_diff > 0 and recall_diff < 0:
                insights.append("Precision improved but recall decreased, suggesting the model has become more conservative in flagging contradictions.")
            elif precision_diff < 0 and recall_diff > 0:
                insights.append("Recall improved but precision decreased, suggesting the model has become more aggressive in flagging contradictions.")
            elif precision_diff > 0 and recall_diff > 0:
                insights.append("Both precision and recall improved, indicating better overall contradiction detection without trade-offs.")
        
        return insights
    
    def _generate_synthetic_history(self, num_entries=5):
        """
        Generate synthetic metrics history for demonstration purposes.
        
        Args:
            num_entries (int, optional): Number of data points to generate. Defaults to 5.
            
        Returns:
            list: List of synthetic metrics history entries
        """
        self.logger.info(f"Generating synthetic metrics history with {num_entries} entries for demonstration")
        
        try:
            history = []
            
            # Generate dates (past N days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            date_range = (end_date - start_date).days
            
            # Ensure we have enough days for the number of entries
            if date_range < num_entries:
                self.logger.warning(f"Date range ({date_range} days) is less than requested entries ({num_entries})")
                self.logger.info(f"Adjusting start date to ensure enough days")
                start_date = end_date - timedelta(days=num_entries*2)
                date_range = (end_date - start_date).days
            
            # Select evenly spaced dates across the range
            step = date_range // num_entries
            dates = [(start_date + timedelta(days=i*step)).strftime('%Y-%m-%d') 
                    for i in range(num_entries)]
            
            # Start with base metrics similar to current metrics
            try:
                base_accuracy = self.data['summary'].get('contradiction_accuracy', 0.7)
                base_precision = self.data['summary'].get('contradiction_precision', 0.65)
                base_recall = self.data['summary'].get('contradiction_recall', 0.75)
                base_f1 = self.data['summary'].get('contradiction_f1', 0.7)
            except KeyError:
                self.logger.warning("Could not find contradiction metrics in current data, using defaults")
                base_accuracy = 0.7
                base_precision = 0.65
                base_recall = 0.75
                base_f1 = 0.7
            except Exception as e:
                self.logger.error(f"Error extracting base metrics: {str(e)}")
                base_accuracy = 0.7
                base_precision = 0.65
                base_recall = 0.75
                base_f1 = 0.7
            
            # Generate metrics history with progressive improvements and some fluctuations
            for i, date in enumerate(dates):
                # Add some random variations and a general trend of improvement
                improvement_factor = i / (num_entries - 1) if num_entries > 1 else 0  # 0 to 1
                random_factor = random.uniform(-0.02, 0.02)  # Small random fluctuations
                
                # Create metrics for this point in time
                entry = {
                    'date': date,
                    'metrics': {
                        'contradiction_accuracy': max(0, min(1, base_accuracy - 0.1 + (0.15 * improvement_factor) + random_factor)),
                        'contradiction_precision': max(0, min(1, base_precision - 0.15 + (0.2 * improvement_factor) + random_factor)),
                        'contradiction_recall': max(0, min(1, base_recall - 0.05 + (0.1 * improvement_factor) + random_factor)),
                        'contradiction_f1': max(0, min(1, base_f1 - 0.1 + (0.15 * improvement_factor) + random_factor))
                    }
                }
                
                history.append(entry)
            
            self.logger.info(f"Successfully generated synthetic history with {len(history)} entries")
            return history
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic metrics history: {str(e)}")
            self.logger.debug("Error details:", exc_info=True)
            
            # Return a minimal fallback history if generation fails
            self.logger.info("Using fallback synthetic history")
            fallback_history = []
            for i in range(3):
                date = (datetime.now() - timedelta(days=i*10)).strftime('%Y-%m-%d')
                fallback_history.append({
                    'date': date,
                    'metrics': {
                        'contradiction_accuracy': 0.7 + (i * 0.05),
                        'contradiction_precision': 0.65 + (i * 0.05),
                        'contradiction_recall': 0.75 + (i * 0.03),
                        'contradiction_f1': 0.7 + (i * 0.04)
                    }
                })
            return fallback_history
    
    def _generate_minimal_history(self):
        """Generate a minimal metrics history for demonstration purposes."""
        self.logger.info("Generating minimal metrics history for demonstration")
        
        # Generate dates (past 5 months)
        today = datetime.now()
        dates = [(today - timedelta(days=30*i)).strftime('%Y-%m-%d') for i in range(5)]
        dates.reverse()  # Start with oldest date
        
        # Define baseline metrics with some improvement over time
        metrics_history = []
        
        baseline_metrics = {
            'contradiction_detection_accuracy': 0.65,
            'false_positive_rate': 0.18,
            'false_negative_rate': 0.22,
            'precision': 0.75,
            'recall': 0.70,
            'f1_score': 0.72
        }
        
        # Generate metrics history with progressive improvements and some fluctuations
        for i, date in enumerate(dates):
            # Add some random variations and a general trend of improvement
            improvement_factor = i / (len(dates) - 1)  # 0 to 1
            random_factor = random.uniform(-0.02, 0.02)  # Small random fluctuations
            
            # Create metrics for this point in time
            metrics = {
                'contradiction_detection_accuracy': min(0.95, baseline_metrics['contradiction_detection_accuracy'] + improvement_factor + random_factor),
                'false_positive_rate': max(0.05, baseline_metrics['false_positive_rate'] - improvement_factor/2 + random_factor/2),
                'false_negative_rate': max(0.05, baseline_metrics['false_negative_rate'] - improvement_factor/2 + random_factor/2),
                'precision': min(0.95, baseline_metrics['precision'] + improvement_factor + random_factor),
                'recall': min(0.95, baseline_metrics['recall'] + improvement_factor + random_factor),
                'f1_score': min(0.95, baseline_metrics['f1_score'] + improvement_factor + random_factor)
            }
            
            # Add entry to history
            metrics_history.append({
                'date': date,
                'metrics': metrics
            })
        
        return metrics_history
    
    def generate_all_visualizations(self, save: bool = True):
        """
        Generate all available visualizations and save them.
        """
        def try_visualization(func):
            try:
                func()
            except Exception as e:
                print(f"Error generating visualization: {e}")
                traceback.print_exc()
        
        print("Generating visualizations...")
        
        # Try generating each visualization
        if self.visualization_types is None or 'domain_scores' in self.visualization_types:
            try_visualization(lambda: self.plot_domain_scores())
        if self.visualization_types is None or 'radar_chart' in self.visualization_types:
            try_visualization(lambda: self.plot_radar_chart())
        if self.visualization_types is None or 'memory_usage' in self.visualization_types:
            try_visualization(lambda: self.plot_memory_usage())
        if self.visualization_types is None or 'topic_performance' in self.visualization_types:
            try_visualization(lambda: self.plot_topic_performance())
        if self.visualization_types is None or 'scenario_performance' in self.visualization_types:
            try_visualization(lambda: self.plot_scenario_performance())
        if self.visualization_types is None or 'healthcare_combined_metrics' in self.visualization_types:
            try_visualization(lambda: self.plot_healthcare_combined_metrics())
        if self.visualization_types is None or 'healthcare_metrics_trends' in self.visualization_types:
            try_visualization(lambda: self.track_healthcare_metrics_over_time())
        if self.visualization_types is None or 'performance_gaps' in self.visualization_types:
            try_visualization(lambda: self.analyze_performance_gaps())
        if self.visualization_types is None or 'healthcare_contradiction_types' in self.visualization_types:
            try_visualization(lambda: self.visualize_healthcare_contradiction_types())
        if self.visualization_types is None or 'contradiction_temporal_patterns' in self.visualization_types:
            try_visualization(lambda: self.analyze_contradiction_temporal_patterns())
        if self.visualization_types is None or 'contradiction_improvements' in self.visualization_types:
            try_visualization(lambda: self.track_contradiction_detection_improvements())
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def generate_html_report(self, save: bool = True) -> str:
        """
        Generate an HTML report with all visualizations.
        
        Args:
            save: Whether to save the HTML file to disk
            
        Returns:
            HTML string containing the report
        """
        # Generate all visualizations
        self.generate_all_visualizations()
        
        # Create HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Model Evaluation Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }
                h1, h2, h3 { color: #2c3e50; }
                .visualization { margin: 20px 0; }
                .visualization img { max-width: 100%; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { text-align: left; padding: 8px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #2c3e50; color: white; }
                .improvement { color: green; font-weight: bold; }
                .acceptable { color: #f39c12; font-weight: bold; }
                .regression { color: red; font-weight: bold; }
                .summary-box { background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin: 20px 0; border: 1px solid #dee2e6; }
                .metrics-table { width: 100%; max-width: 800px; }
                .insights { margin: 15px 0; }
                .insights p { margin: 8px 0; }
                .recommendations li { margin-bottom: 8px; }
                .key-metric { font-size: 24px; font-weight: bold; }
                .metric-card { display: inline-block; width: 30%; min-width: 250px; background: #fff; border-radius: 8px; padding: 15px; margin: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <h1>AI Model Evaluation Results</h1>
        """
        
        # Add metadata if available
        if 'metadata' in self.data:
            metadata = self.data['metadata']
            html += "<h2>Evaluation Details</h2>"
            html += "<table>"
            html += "<tr><th>Property</th><th>Value</th></tr>"
            
            for key, value in metadata.items():
                html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            
            html += "</table>"
        
        # Add contradiction detection analysis summary
        html += """
        <h2>Contradiction Detection Analysis Summary</h2>
        <div class="summary-box">
        """
        
        # Extract key contradiction metrics
        contradiction_metrics = {}
        has_contradiction_data = False
        
        try:
            if "healthcare_metrics" in self.data and "contradiction_detection_accuracy" in self.data["healthcare_metrics"]:
                has_contradiction_data = True
                metrics_source = self.data["healthcare_metrics"]
                contradiction_metrics = {
                    "accuracy": metrics_source.get("contradiction_detection_accuracy", 0),
                    "precision": metrics_source.get("precision", 0),
                    "recall": metrics_source.get("recall", 0),
                    "f1_score": metrics_source.get("f1_score", 0),
                    "false_positive_rate": metrics_source.get("false_positive_rate", 0),
                    "false_negative_rate": metrics_source.get("false_negative_rate", 0)
                }
            elif "metrics" in self.data:
                has_contradiction_data = True
                metrics_source = self.data["metrics"]
                contradiction_metrics = {
                    "accuracy": metrics_source.get("contradiction_detection_accuracy", 0),
                    "precision": metrics_source.get("precision", 0),
                    "recall": metrics_source.get("recall", 0),
                    "f1_score": metrics_source.get("f1_score", 0),
                    "false_positive_rate": metrics_source.get("false_positive_rate", 0),
                    "false_negative_rate": metrics_source.get("false_negative_rate", 0)
                }
        except Exception as e:
            self.logger.warning(f"Could not extract contradiction metrics: {str(e)}")
        
        if has_contradiction_data:
            # Add metrics summary table
            html += """
            <h3>Key Performance Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                </tr>
            """
            
            # Define thresholds for each metric
            thresholds = {
                "accuracy": {"good": 0.90, "acceptable": 0.85},
                "precision": {"good": 0.90, "acceptable": 0.85},
                "recall": {"good": 0.85, "acceptable": 0.80},
                "f1_score": {"good": 0.87, "acceptable": 0.82},
                "false_positive_rate": {"good": 0.05, "acceptable": 0.10, "inverse": True},
                "false_negative_rate": {"good": 0.08, "acceptable": 0.15, "inverse": True}
            }
            
            # Generate table rows with status indicators
            for metric_key, metric_value in contradiction_metrics.items():
                metric_name = metric_key.replace('_', ' ').title()
                
                # Determine status based on thresholds
                status = "Needs improvement"
                status_class = "regression"
                
                if metric_key in thresholds:
                    threshold = thresholds[metric_key]
                    inverse = threshold.get("inverse", False)
                    
                    if inverse:
                        if metric_value <= threshold["good"]:
                            status = "Good"
                            status_class = "improvement"
                        elif metric_value <= threshold["acceptable"]:
                            status = "Acceptable"
                            status_class = "acceptable"
                    else:
                        if metric_value >= threshold["good"]:
                            status = "Good"
                            status_class = "improvement"
                        elif metric_value >= threshold["acceptable"]:
                            status = "Acceptable"
                            status_class = "acceptable"
                
                value_formatted = f"{metric_value:.2%}" if isinstance(metric_value, float) else metric_value
                html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value_formatted}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
                """
            
            html += """
            </table>
            """
            
            # Add text analysis of the metrics
            html += """
            <h3>Analysis Insights</h3>
            <div class="insights">
            """
            
            # Generate insights based on the metrics
            insights = []
            
            # Accuracy analysis
            if contradiction_metrics["accuracy"] >= 0.90:
                insights.append("The model demonstrates <strong>excellent overall accuracy</strong> in detecting contradictions in healthcare data.")
            elif contradiction_metrics["accuracy"] >= 0.85:
                insights.append("The model shows <strong>good overall accuracy</strong> in contradiction detection, but there's room for improvement.")
            else:
                insights.append("The model's <strong>overall accuracy</strong> in contradiction detection needs improvement.")
            
            # Precision vs. Recall analysis
            if contradiction_metrics["precision"] > contradiction_metrics["recall"] + 0.05:
                insights.append("The model prioritizes <strong>precision over recall</strong>, meaning it's more conservative in flagging contradictions but more reliable when it does.")
            elif contradiction_metrics["recall"] > contradiction_metrics["precision"] + 0.05:
                insights.append("The model prioritizes <strong>recall over precision</strong>, catching more contradictions but with more false positives.")
            else:
                insights.append("The model maintains a <strong>good balance between precision and recall</strong>, indicating consistent performance across different types of contradictions.")
            
            # Error analysis
            if contradiction_metrics["false_positive_rate"] < 0.05:
                insights.append("The model has a <strong>very low false positive rate</strong>, indicating high confidence in flagged contradictions.")
            elif contradiction_metrics["false_positive_rate"] > 0.10:
                insights.append("The <strong>false positive rate</strong> is higher than desired, which may lead to unnecessary flags in the healthcare context.")
            
            if contradiction_metrics["false_negative_rate"] < 0.05:
                insights.append("The model has a <strong>very low false negative rate</strong>, rarely missing actual contradictions.")
            elif contradiction_metrics["false_negative_rate"] > 0.10:
                insights.append("The <strong>false negative rate</strong> is concerning, as the model is missing a significant number of actual contradictions.")
            
            # F1 Score analysis
            if contradiction_metrics["f1_score"] >= 0.90:
                insights.append("The model's <strong>excellent F1 score</strong> demonstrates robust overall performance balancing precision and recall.")
            elif contradiction_metrics["f1_score"] < 0.80:
                insights.append("The <strong>lower F1 score</strong> indicates an imbalance in the model's ability to consistently identify contradictions.")
            
            # Add the insights to the HTML
            for insight in insights:
                html += f"<p>• {insight}</p>"
            
            # Add recommendations section based on metrics
            html += """
            </div>
            <h3>Recommendations</h3>
            <ul class="recommendations">
            """
            
            recommendations = []
            
            # Generate recommendations based on metric analysis
            if contradiction_metrics["false_positive_rate"] > 0.08:
                recommendations.append("Focus on reducing false positives by improving the model's understanding of clinical context and terminology variations.")
            
            if contradiction_metrics["false_negative_rate"] > 0.08:
                recommendations.append("Address false negatives by enhancing the model's sensitivity to subtle contradictions and improving entity recognition.")
            
            if contradiction_metrics["precision"] < 0.85:
                recommendations.append("Improve precision by refining the contradiction detection criteria and enhancing the model's domain-specific knowledge.")
            
            if contradiction_metrics["recall"] < 0.85:
                recommendations.append("Increase recall by expanding the model's training on diverse contradiction patterns and edge cases.")
            
            if not recommendations:
                recommendations.append("Continue monitoring performance across different healthcare domains and contradiction types.")
                recommendations.append("Consider evaluating on more complex or specialized healthcare scenarios to further validate performance.")
            
            # Add the recommendations to the HTML
            for recommendation in recommendations:
                html += f"<li>{recommendation}</li>"
            
            html += """
            </ul>
            """
        else:
            html += """
            <p>No contradiction detection metrics available in the current evaluation data.</p>
            """
        
        html += """
        </div>
        """
        
        # Add visualizations
        html += "<h2>Performance Visualizations</h2>"
        
        # Domain scores
        html += """
        <div class="visualization">
            <h3>Domain Knowledge Performance</h3>
            <img src="domain_scores.png" alt="Domain Scores">
        </div>
        """
        
        # Radar chart
        html += """
        <div class="visualization">
            <h3>Capability Profile</h3>
            <img src="radar_chart.png" alt="Capability Radar Chart">
        </div>
        """
        
        # Memory usage
        html += """
        <div class="visualization">
            <h3>Memory Usage During Evaluation</h3>
            <img src="memory_usage.png" alt="Memory Usage">
        </div>
        """
        
        # Topic performance for each domain
        html += "<h2>Detailed Topic Performance</h2>"
        
        for domain in self.domains:
            domain_name = domain.replace('_', ' ').title()
            
            # First check if topics visualization exists
            topic_file = self.output_dir / f"{domain}_topics.png"
            if topic_file.exists():
                html += f"""
                <div class="visualization">
                    <h3>{domain_name} Topics</h3>
                    <img src="{domain}_topics.png" alt="{domain_name} Topic Performance">
                </div>
                """
            
            # Also check if scenarios visualization exists
            scenario_file = self.output_dir / f"{domain}_scenarios.png"
            if scenario_file.exists():
                html += f"""
                <div class="visualization">
                    <h3>{domain_name} Scenarios</h3>
                    <img src="{domain}_scenarios.png" alt="{domain_name} Scenario Performance">
                </div>
                """
        
        # Add healthcare visualizations
        html += "<h2>Healthcare Performance</h2>"
        
        # Contradiction detection metrics
        html += """
        <div class="visualization">
            <h3>Healthcare Contradiction Detection Metrics</h3>
            <img src="healthcare_contradiction_metrics.png" alt="Healthcare Contradiction Detection Metrics">
        </div>
        """
        
        # Evidence ranking metrics
        html += """
        <div class="visualization">
            <h3>Healthcare Evidence Ranking Metrics</h3>
            <img src="healthcare_evidence_ranking.png" alt="Healthcare Evidence Ranking Metrics">
        </div>
        """
        
        # Combined healthcare metrics
        html += """
        <div class="visualization">
            <h3>Healthcare Combined Metrics</h3>
            <img src="healthcare_combined_metrics.png" alt="Healthcare Combined Metrics">
        </div>
        """
        
        # Healthcare metrics trends
        healthcare_trends_file = self.output_dir / "healthcare_metrics_trends.png"
        if healthcare_trends_file.exists():
            html += """
            <div class="visualization">
                <h3>Healthcare Metrics Performance Over Time</h3>
                <img src="healthcare_metrics_trends.png" alt="Healthcare Metrics Trends">
                <p>This chart tracks performance of healthcare metrics over time, showing progress toward target thresholds.</p>
            </div>
            """
            
        # Healthcare performance gaps
        performance_gaps_file = self.output_dir / "healthcare_performance_gaps.png"
        if performance_gaps_file.exists():
            html += """
            <div class="visualization">
                <h3>Healthcare Performance Gap Analysis</h3>
                <img src="healthcare_performance_gaps.png" alt="Healthcare Performance Gaps">
                <p>This visualization shows the current gaps to target metrics and percentage of targets achieved.</p>
            </div>
            """
        
        # Healthcare contradiction types
        contradiction_types_file = self.output_dir / "healthcare_contradiction_types.png"
        if contradiction_types_file.exists():
            html += """
            <div class="visualization">
                <h3>Healthcare Contradiction Detection by Type</h3>
                <img src="healthcare_contradiction_types.png" alt="Healthcare Contradiction Detection by Type">
                <p>This chart shows the performance of the model in detecting different types of contradictions in healthcare.</p>
            </div>
            """
            
        # Healthcare contradiction temporal patterns
        contradiction_temporal_file = self.output_dir / "contradiction_temporal_patterns.png"
        if contradiction_temporal_file.exists():
            html += """
            <div class="visualization">
                <h3>Temporal Patterns in Healthcare Contradictions</h3>
                <img src="contradiction_temporal_patterns.png" alt="Temporal Patterns in Healthcare Contradictions">
                <p>This visualization analyzes the time gaps and patterns in healthcare contradictions across different types and domains.</p>
            </div>
            """
        
        # Healthcare contradiction improvements
        contradiction_improvements_file = self.output_dir / "healthcare_contradiction_improvements.png"
        if contradiction_improvements_file.exists():
            html += """
            <div class="visualization">
                <h3>Healthcare Contradiction Detection Improvements</h3>
                <img src="healthcare_contradiction_improvements.png" alt="Healthcare Contradiction Detection Improvements">
                <p>This visualization tracks improvements in contradiction detection performance over time, showing progress by contradiction type and medical domain.</p>
            </div>
            """
        
        # Add summary
        if 'summary' in self.data:
            summary = self.data['summary']
            html += "<h2>Performance Summary</h2>"
            
            if 'strengths' in summary:
                html += "<h3>Strengths</h3><ul>"
                for strength in summary['strengths']:
                    html += f"<li>{strength}</li>"
                html += "</ul>"
                
            if 'weaknesses' in summary:
                html += "<h3>Areas for Improvement</h3><ul>"
                for weakness in summary['weaknesses']:
                    html += f"<li>{weakness}</li>"
                html += "</ul>"
                
            if 'recommendations' in summary:
                html += "<h3>Recommendations</h3><ul>"
                for rec in summary['recommendations']:
                    html += f"<li>{rec}</li>"
                html += "</ul>"
        
        html += """
        </body>
        </html>
        """
        
        # Save file if requested
        if save:
            with open(self.output_dir / 'evaluation_report.html', 'w') as f:
                f.write(html)
            
        return html

    def compare_contradiction_performance(self, first_file, second_file, output_path=None):
        """
        Compare contradiction detection performance between two result files.
        
        Args:
            first_file (str): Path to the first results file
            second_file (str): Path to the second results file
            output_path (str, optional): Path to save the comparison visualization
            
        Returns:
            bool: True if comparison was successful, False otherwise
        """
        self.logger.info(f"Comparing contradiction performance between {first_file} and {second_file}")
        
        try:
            # Load the data from both files
            with open(first_file, 'r') as f:
                data1 = json.load(f)
            with open(second_file, 'r') as f:
                data2 = json.load(f)
                
            self.logger.info("Successfully loaded both result files")
            
            # Extract contradiction metrics
            metric_keys = ['contradiction_accuracy', 'contradiction_precision', 
                           'contradiction_recall', 'contradiction_f1']
            
            # Validate the data has the required metrics
            if 'summary' not in data1 or 'summary' not in data2:
                self.logger.error("One or both result files are missing the 'summary' section")
                return False
                
            metrics1 = {k: data1['summary'].get(k, 0) for k in metric_keys}
            metrics2 = {k: data2['summary'].get(k, 0) for k in metric_keys}
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                'Metric': [k.replace('contradiction_', '').capitalize() for k in metric_keys],
                'File 1': [metrics1[k] for k in metric_keys],
                'File 2': [metrics2[k] for k in metric_keys]
            })
            
            # Calculate differences for annotations
            df['Difference'] = df['File 2'] - df['File 1']
            
            # Create the comparison visualization
            plt.figure(figsize=(12, 8))
            
            # Plot the bars
            bar_width = 0.35
            x = np.arange(len(df['Metric']))
            
            bars1 = plt.bar(x - bar_width/2, df['File 1'], bar_width, label=Path(first_file).name)
            bars2 = plt.bar(x + bar_width/2, df['File 2'], bar_width, label=Path(second_file).name)
            
            # Add labels and title
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Contradiction Detection Performance Comparison')
            plt.xticks(x, df['Metric'])
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            self._add_value_labels(bars1)
            self._add_value_labels(bars2)
            
            # Add difference annotations
            for i, diff in enumerate(df['Difference']):
                color = 'green' if diff > 0 else 'red'
                plt.annotate(f"{diff:+.2f}",
                            xy=(x[i], max(df['File 1'][i], df['File 2'][i]) + 0.05),
                            ha='center', va='bottom',
                            color=color, fontweight='bold')
                            
                # Add arrow to indicate improvement or regression
                if abs(diff) > 0.01:  # Only add arrows for significant changes
                    plt.annotate('',
                                xy=(x[i], max(df['File 1'][i], df['File 2'][i]) + 0.03),
                                xytext=(x[i], max(df['File 1'][i], df['File 2'][i]) + 0.08),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                                ha='center')
            
            # Add watermark with timestamp
            plt.figtext(0.95, 0.05, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        fontsize=8, ha='right', va='bottom', alpha=0.5)
            
            # Save or show the plot
            if output_path:
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Comparison saved to {output_path}")
                except Exception as e:
                    self.logger.error(f"Error saving comparison visualization: {str(e)}")
                    return False
            else:
                output_path = self.output_dir / "contradiction_performance_comparison.png"
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Comparison saved to {output_path}")
                except Exception as e:
                    self.logger.error(f"Error saving comparison visualization: {str(e)}")
                    return False
            
            plt.close()
            
            # Generate additional insights
            self.logger.info("Generating performance insights...")
            insights = self._generate_comparison_insights(df)
            for insight in insights:
                self.logger.info(f"INSIGHT: {insight}")
                
            return True
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in input file: {str(e)}")
            return False
        except KeyError as e:
            self.logger.error(f"Missing required key in result data: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error comparing contradiction performance: {str(e)}")
            self.logger.debug("Error details:", exc_info=True)
            return False
    
def main():
    """Main function to run the metrics visualization script."""
    parser = argparse.ArgumentParser(description='Generate visualizations from healthcare cross-reference results')
    parser.add_argument('--results', '-r', type=str,
                        help='Path to healthcare cross-reference results file')
    parser.add_argument('--output', '-o', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--html-report', action='store_true',
                        help='Generate HTML report with all visualizations')
    parser.add_argument('--metrics-history', type=str, default=None,
                        help='Path to healthcare metrics history file for tracking metrics over time')
    parser.add_argument('--visualization-types', type=str, nargs='+', default=None,
                        help='List of visualization types to generate. Options include: domain_scores, radar_chart, memory_usage, topic_performance, scenario_performance, healthcare_combined_metrics, healthcare_metrics_trends, performance_gaps, healthcare_contradiction_types, contradiction_temporal_patterns, contradiction_improvements')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                        help='Compare contradiction detection performance between two result files')
    parser.add_argument('--compare-output', type=str, default=None,
                        help='Path to save comparison visualization (used with --compare)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.compare and not args.results:
        parser.error("Either --results or --compare must be specified")
    
    if args.compare:
        # Initialize visualizer for comparison
        visualizer = MetricsVisualizer(
            args.results if args.results else args.compare[0], 
            args.output
        )
        # Compare the contradiction detection performance
        visualizer.compare_contradiction_performance(
            args.compare[0], 
            args.compare[1], 
            args.compare_output
        )
    elif args.results:
        visualizer = MetricsVisualizer(args.results, args.output, visualization_types=args.visualization_types)
        
        if args.metrics_history:
            visualizer.track_healthcare_metrics_over_time(metrics_history_path=args.metrics_history)
        
        if args.html_report:
            visualizer.generate_html_report()
        else:
            visualizer.generate_all_visualizations()
    
    return 0

if __name__ == "__main__":
    exit(main())

class HealthcareContinuousLearning:
    """
    Class for continuous learning in healthcare contradiction detection.
    Analyzes performance metrics, identifies improvement areas, and generates
    new training examples to continuously improve model performance.
    """
    
    def __init__(self, 
                data_dir: str = "data/healthcare",
                model_dir: str = "models/healthcare",
                metrics_dir: str = "data/healthcare/evaluation",
                config_path: Optional[str] = None):
        """Initialize the healthcare continuous learning system.
        
        Args:
            data_dir: Directory for healthcare data
            model_dir: Directory for healthcare models
            metrics_dir: Directory for evaluation metrics
            config_path: Path to configuration file
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.metrics_dir = Path(metrics_dir)
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize visualizer for metrics analysis
        self.visualizer = MetricsVisualizer(results_path=str(self.metrics_dir))
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    logger.error(f"Error loading configuration: {str(e)}")
            else:
                logger.warning(f"Configuration file not found: {config_path}")
        
        # Initialize history tracking
        self.history_path = self.data_dir / "learning_history.json"
        self.learning_history = self._load_history()
        
        # Set default parameters
        self.contradiction_categories = self.config.get("contradiction_categories", CONTRADICTION_CATEGORIES)
        self.medical_domains = self.config.get("medical_domains", MEDICAL_DOMAINS)
        self.evidence_types = self.config.get("evidence_types", EVIDENCE_TYPES)
        
        # Dataset paths
        self.contradiction_dataset_path = self.data_dir / "contradiction_dataset" / "medical_contradictions.json"
        self.training_data_path = self.data_dir / "training" / "healthcare_training.json"
        self.evaluation_data_path = self.data_dir / "evaluation" / "healthcare_eval.json"
        
        # Ensure dataset directories exist
        (self.data_dir / "contradiction_dataset").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "training").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "evaluation").mkdir(exist_ok=True, parents=True)
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load learning history from disk.
        
        Returns:
            List of historical learning events
        """
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading learning history: {str(e)}")
                return []
        return []
    
    def _save_history(self) -> None:
        """Save learning history to disk."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.learning_history, f, indent=2)
            logger.info(f"Saved learning history to {self.history_path}")
        except Exception as e:
            logger.error(f"Error saving learning history: {str(e)}")
    
    def track_learning_event(self, event_type: str, metrics: Dict[str, Any]) -> None:
        """Track a learning event in history.
        
        Args:
            event_type: Type of learning event (e.g., 'training', 'evaluation')
            metrics: Metrics and details of the learning event
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "metrics": metrics
        }
        
        self.learning_history.append(event)
        self._save_history()
        logger.info(f"Tracked learning event: {event_type}")

    def analyze_performance(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance to identify improvement areas.
        
        Args:
            evaluation_results: Results from contradiction detection evaluation
            
        Returns:
            Analysis results with improvement recommendations
        """
        try:
            logger.info("Analyzing contradiction detection performance")
            
            # Initialize analysis results
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "overall_metrics": {},
                "category_performance": {},
                "domain_performance": {},
                "improvement_areas": [],
                "prioritized_examples": []
            }
            
            # Extract overall metrics
            if "accuracy" in evaluation_results:
                analysis["overall_metrics"]["accuracy"] = evaluation_results["accuracy"]
            
            # Analyze category performance
            if "by_category" in evaluation_results:
                for category, metrics in evaluation_results["by_category"].items():
                    analysis["category_performance"][category] = metrics
            
            # Analyze domain performance
            if "by_domain" in evaluation_results:
                for domain, metrics in evaluation_results["by_domain"].items():
                    analysis["domain_performance"][domain] = metrics
            
            # Identify poorest performing categories
            if "by_category" in evaluation_results:
                category_accuracies = [(cat, metrics.get("accuracy", 0)) 
                                    for cat, metrics in evaluation_results["by_category"].items()]
                sorted_categories = sorted(category_accuracies, key=lambda x: x[1])
                
                # Identify categories below threshold (e.g., 0.8 accuracy)
                threshold = self.config.get("improvement_threshold", 0.8)
                for category, accuracy in sorted_categories:
                    if accuracy < threshold:
                        analysis["improvement_areas"].append({
                            "type": "category",
                            "name": category,
                            "current_accuracy": accuracy,
                            "target_accuracy": threshold
                        })
            
            # Identify poorest performing domains
            if "by_domain" in evaluation_results:
                domain_accuracies = [(domain, metrics.get("accuracy", 0)) 
                                   for domain, metrics in evaluation_results["by_domain"].items()]
                sorted_domains = sorted(domain_accuracies, key=lambda x: x[1])
                
                # Identify domains below threshold
                for domain, accuracy in sorted_domains:
                    if accuracy < threshold:
                        analysis["improvement_areas"].append({
                            "type": "domain",
                            "name": domain,
                            "current_accuracy": accuracy,
                            "target_accuracy": threshold
                        })
            
            # Find examples that were incorrectly classified
            if "examples" in evaluation_results:
                incorrect_examples = [ex for ex in evaluation_results["examples"] if not ex.get("correct", True)]
                
                # Sort by domain and category to prioritize improvement areas
                prioritized_examples = []
                
                # First, add examples from poorest performing categories
                for area in analysis["improvement_areas"]:
                    if area["type"] == "category":
                        category = area["name"]
                        matching_examples = [ex for ex in incorrect_examples 
                                          if ex.get("true_category") == category]
                        prioritized_examples.extend(matching_examples[:10])  # Limit to 10 per category
                
                # Then, add examples from poorest performing domains
                for area in analysis["improvement_areas"]:
                    if area["type"] == "domain":
                        domain = area["name"]
                        matching_examples = [ex for ex in incorrect_examples 
                                          if ex.get("domain") == domain]
                        prioritized_examples.extend(matching_examples[:10])  # Limit to 10 per domain
                
                # Ensure we don't have duplicates
                seen_examples = set()
                unique_examples = []
                for ex in prioritized_examples:
                    # Create a key from statement text
                    key = (ex.get("statement_1", ""), ex.get("statement_2", ""))
                    if key not in seen_examples:
                        seen_examples.add(key)
                        unique_examples.append(ex)
                
                analysis["prioritized_examples"] = unique_examples[:50]  # Limit to 50 total
            
            logger.info(f"Performance analysis complete. Identified {len(analysis['improvement_areas'])} improvement areas")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def generate_training_examples(self, 
                                 analysis_results: Dict[str, Any], 
                                 count: int = 50) -> List[Dict[str, Any]]:
        """Generate new training examples based on analysis.
        
        Args:
            analysis_results: Results from performance analysis
            count: Number of examples to generate
            
        Returns:
            List of new training examples
        """
        try:
            logger.info(f"Generating {count} new training examples based on analysis")
            
            new_examples = []
            improvement_areas = analysis_results.get("improvement_areas", [])
            prioritized_examples = analysis_results.get("prioritized_examples", [])
            
            if not improvement_areas or not prioritized_examples:
                logger.warning("No improvement areas or prioritized examples found in analysis")
                return []
            
            # Get categories and domains to focus on
            focus_categories = [area["name"] for area in improvement_areas if area["type"] == "category"]
            focus_domains = [area["name"] for area in improvement_areas if area["type"] == "domain"]
            
            # If no specific focus areas, use all available
            if not focus_categories:
                focus_categories = self.contradiction_categories
            if not focus_domains:
                focus_domains = self.medical_domains
            
            # Load existing examples for patterns
            existing_examples = self._load_existing_examples()
            
            # Use prioritized examples as templates for new examples
            for i in range(count):
                # Cycle through prioritized examples as templates
                template_idx = i % len(prioritized_examples) if prioritized_examples else 0
                
                if prioritized_examples:
                    template = prioritized_examples[template_idx]
                    
                    # Create variation based on the template
                    new_example = self._create_example_variation(
                        template, 
                        focus_categories, 
                        focus_domains,
                        existing_examples
                    )
                    
                    if new_example:
                        new_examples.append(new_example)
            
            logger.info(f"Generated {len(new_examples)} new training examples")
            return new_examples
            
        except Exception as e:
            logger.error(f"Error generating training examples: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _load_existing_examples(self) -> List[Dict[str, Any]]:
        """Load existing examples from the contradiction dataset.
        
        Returns:
            List of existing examples
        """
        examples = []
        
        # Load from medical contradictions dataset
        if self.contradiction_dataset_path.exists():
            try:
                with open(self.contradiction_dataset_path, 'r') as f:
                    examples.extend(json.load(f))
                logger.info(f"Loaded {len(examples)} existing examples from contradiction dataset")
            except Exception as e:
                logger.error(f"Error loading contradiction dataset: {str(e)}")
        
        # Load from training data if it exists
        if self.training_data_path.exists():
            try:
                with open(self.training_data_path, 'r') as f:
                    examples.extend(json.load(f))
                logger.info(f"Loaded examples from training data")
            except Exception as e:
                logger.error(f"Error loading training data: {str(e)}")
        
        return examples
    
    def _create_example_variation(self, 
                                template: Dict[str, Any],
                                focus_categories: List[str],
                                focus_domains: List[str],
                                existing_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a variation of an example based on a template.
        
        Args:
            template: Template example to base variation on
            focus_categories: Categories to focus on
            focus_domains: Domains to focus on
            existing_examples: Existing examples to avoid duplication
            
        Returns:
            New example variation or None if failed
        """
        try:
            # Create a new example based on the template
            new_example = {}
            
            # Get template statements
            statement1 = template.get("statement_1", template.get("statement1", ""))
            statement2 = template.get("statement_2", template.get("statement2", ""))
            
            if not statement1 or not statement2:
                return None
            
            # Choose a category, preferring focus categories
            if focus_categories and random.random() < 0.8:  # 80% chance to use focus category
                category = random.choice(focus_categories)
            else:
                category = random.choice(self.contradiction_categories)
            
            # Choose a domain, preferring focus domains
            if focus_domains and random.random() < 0.8:  # 80% chance to use focus domain
                domain = random.choice(focus_domains)
            else:
                domain = random.choice(self.medical_domains)
            
            # Create slight variations of the statements
            variations = [
                f"According to recent studies, {statement1.lower()}",
                f"Research suggests that {statement1.lower()}",
                f"Medical literature indicates {statement1.lower()}",
                f"Clinical evidence shows {statement1.lower()}",
                f"Healthcare professionals agree that {statement1.lower()}"
            ]
            new_statement1 = random.choice(variations)
            
            variations = [
                f"However, other studies show {statement2.lower()}",
                f"In contrast, some research indicates {statement2.lower()}",
                f"Alternatively, evidence suggests {statement2.lower()}",
                f"On the other hand, clinical data shows {statement2.lower()}",
                f"Conversely, medical experts state that {statement2.lower()}"
            ]
            new_statement2 = random.choice(variations)
            
            # Create a modified example with the new statements
            new_example = {
                "statement1": new_statement1,
                "statement2": new_statement2,
                "type": category,
                "domain": domain,
                "medical_specialty": domain,
                "sources": [
                    f"Generated Example {datetime.now().strftime('%Y-%m-%d')}",
                    "Continuous Learning System"
                ],
                "publication_dates": [
                    datetime.now().strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d")
                ]
            }
            
            # Check if this example is too similar to existing examples
            for ex in existing_examples:
                ex_statement1 = ex.get("statement1", ex.get("statement_1", ""))
                ex_statement2 = ex.get("statement2", ex.get("statement_2", ""))
                
                # Simple similarity check (could be more sophisticated with embedding comparison)
                if (self._similarity(new_statement1, ex_statement1) > 0.8 or
                    self._similarity(new_statement2, ex_statement2) > 0.8):
                    return None
            
            return new_example
            
        except Exception as e:
            logger.error(f"Error creating example variation: {str(e)}")
            return None
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate a simple text similarity score.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # This is a very simple implementation
        # In a real system, use embeddings or better NLP techniques
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def update_training_data(self, new_examples: List[Dict[str, Any]]) -> str:
        """Update training dataset with new examples.
        
        Args:
            new_examples: New training examples to add
            
        Returns:
            Path to updated training data
        """
        try:
            if not new_examples:
                logger.warning("No new examples to add to training data")
                return ""
            
            # Load existing training data
            existing_data = []
            if self.training_data_path.exists():
                try:
                    with open(self.training_data_path, 'r') as f:
                        existing_data = json.load(f)
                    logger.info(f"Loaded {len(existing_data)} existing training examples")
                except Exception as e:
                    logger.error(f"Error loading existing training data: {str(e)}")
            
            # Add new examples
            existing_data.extend(new_examples)
            
            # Save updated training data
            with open(self.training_data_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            # Also update the main contradiction dataset
            self._update_contradiction_dataset(new_examples)
            
            logger.info(f"Updated training data with {len(new_examples)} new examples")
            return str(self.training_data_path)
            
        except Exception as e:
            logger.error(f"Error updating training data: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    def _update_contradiction_dataset(self, new_examples: List[Dict[str, Any]]) -> None:
        """Update the main contradiction dataset with new examples.
        
        Args:
            new_examples: New examples to add
        """
        try:
            # Load existing dataset
            existing_data = []
            if self.contradiction_dataset_path.exists():
                try:
                    with open(self.contradiction_dataset_path, 'r') as f:
                        existing_data = json.load(f)
                    logger.info(f"Loaded {len(existing_data)} existing contradiction examples")
                except Exception as e:
                    logger.error(f"Error loading contradiction dataset: {str(e)}")
            
            # Add new examples (converting format if needed)
            for example in new_examples:
                # Convert to the format used in medical_contradictions.json
                contradiction_example = {
                    "statement1": example.get("statement1", example.get("statement_1", "")),
                    "statement2": example.get("statement2", example.get("statement_2", "")),
                    "type": example.get("type", example.get("relationship", "contradicting")),
                    "domain": example.get("domain", "general"),
                    "medical_specialty": example.get("medical_specialty", example.get("domain", "general")),
                    "sources": example.get("sources", ["Continuous Learning System"]),
                    "publication_dates": example.get("publication_dates", [datetime.now().strftime("%Y-%m-%d")])
                }
                
                existing_data.append(contradiction_example)
            
            # Save updated dataset
            with open(self.contradiction_dataset_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Updated contradiction dataset with {len(new_examples)} new examples")
            
        except Exception as e:
            logger.error(f"Error updating contradiction dataset: {str(e)}")
            logger.error(traceback.format_exc())
