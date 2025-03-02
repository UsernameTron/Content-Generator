#!/usr/bin/env python3
"""
Path-based relationship visualization utility.

This module generates network visualizations of path-based relationships
extracted from context data.
"""

import sys
import os
import logging
import argparse
import json
from typing import Dict, List, Any, Optional

# Configure matplotlib for non-interactive backend (useful for headless environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancement_module.context_analyzer import ContextAnalyzer
from enhancement_module.reasoning_core import extract_path_relationships

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PathVisualizer:
    """Visualization class for path-based relationship graphs."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Optional directory to save visualizations
        """
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'output'
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_relationship_graph(
        self, 
        relationships: List[Dict[str, str]],
        title: str = "Path-Based Relationships",
        filename: str = "relationship_graph.png"
    ) -> nx.DiGraph:
        """
        Create a directed graph from path-based relationships.
        
        Args:
            relationships: List of relationships extracted with extract_path_relationships
            title: Title for the graph
            filename: Filename to save the visualization
            
        Returns:
            NetworkX directed graph
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for rel in relationships:
            source = self._clean_node_name(rel['source'])
            target = self._clean_node_name(rel['target'])
            rel_type = rel['type']
            
            # Add nodes if they don't exist
            if not G.has_node(source):
                G.add_node(source)
            if not G.has_node(target):
                G.add_node(target)
            
            # Add edge with relationship type as attribute
            G.add_edge(source, target, relation=rel_type)
        
        # Plot the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # For reproducibility
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, alpha=0.7)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add edge labels (relationship types)
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add title and adjust layout
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved relationship graph to {output_path}")
        
        # Return the graph for further analysis
        return G
    
    def visualize_from_context(
        self, 
        context_data: Any,
        title: str = "Context Relationships",
        filename: str = "context_relationships.png"
    ) -> nx.DiGraph:
        """
        Create visualization from context data.
        
        Args:
            context_data: Either a list of context strings or a hierarchy dict
            title: Title for the graph
            filename: Filename to save the visualization
            
        Returns:
            NetworkX directed graph
        """
        flat_context = None
        
        # Convert hierarchy to flat context if needed
        if isinstance(context_data, dict):
            analyzer = ContextAnalyzer()
            flat_context = analyzer.flatten_context_hierarchy(context_data)
        elif isinstance(context_data, list):
            flat_context = context_data
        else:
            raise ValueError("context_data must be a dict (hierarchy) or list (flat context)")
        
        # Extract relationships
        relationships = extract_path_relationships(flat_context)
        
        # Create the graph
        return self.create_relationship_graph(relationships, title, filename)
    
    def _clean_node_name(self, name: str) -> str:
        """
        Clean node names for better visualization.
        
        Args:
            name: Original node name from the relationship
            
        Returns:
            Cleaned node name
        """
        # Replace array indices with cleaner representation
        name = name.replace('[', '_').replace(']', '')
        
        # Shorten values paths
        if 'values.' in name:
            name = name.replace('values.', 'val_')
        
        return name

def visualize_test_data():
    """Create visualizations from test data."""
    # Create a sample hierarchical context
    test_hierarchy = {
        'metadata': ['test_metric_id: M12345', 'date: 2025-03-15'],
        'values': {
            'current': 0.85,
            'target': 0.90,
            'baseline': 0.76,
            'historical': [0.74, 0.78, 0.80, 0.82]
        },
        'trends': ['Increasing by 2% quarterly', 'Accelerating improvement'],
        'components': ['Provider Communication', 'Facility Quality', 'Wait Times'],
        'factors': ['Staff Training', 'Process Optimization', 'Facility Updates'],
        'notes': ['Recent improvement influenced by new training program']
    }
    
    # Create visualizer
    visualizer = PathVisualizer()
    
    # Create visualization from hierarchy
    visualizer.visualize_from_context(
        test_hierarchy,
        title="Healthcare Metrics Relationships",
        filename="healthcare_metrics_relationships.png"
    )
    
    logger.info("Created visualization from test data")

def main():
    """Main function for CLI interface."""
    parser = argparse.ArgumentParser(description="Visualize path-based relationships")
    parser.add_argument(
        '--input', '-i', 
        help='Input JSON file (can be hierarchy or flat context list)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output filename'
    )
    parser.add_argument(
        '--title', '-t',
        default="Path-Based Relationships",
        help='Title for the visualization'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Use test data instead of input file'
    )
    parser.add_argument(
        '--output-dir', '-d',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    if args.test:
        visualize_test_data()
        return
        
    if not args.input:
        logger.error("Either --input or --test must be specified")
        return
    
    try:
        # Load input data
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        # Create visualizer
        visualizer = PathVisualizer(args.output_dir)
        
        # Create visualization
        visualizer.visualize_from_context(
            data,
            title=args.title,
            filename=args.output or "relationship_graph.png"
        )
        
        logger.info("Visualization complete")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
