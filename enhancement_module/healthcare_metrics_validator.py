#!/usr/bin/env python3
"""
Healthcare Metrics Validation Module

This module provides specialized validation functionality for healthcare performance metrics 
by leveraging the PathEncoder to enhance contextual understanding and relationship detection
between different healthcare performance indicators.
"""

import json
import logging
import os
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np

from enhancement_module.path_encoder import PathEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthcareMetricsValidator:
    """
    Specialized validator for healthcare metrics that leverages path-based relationship
    encoding for enhanced context analysis and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the HealthcareMetricsValidator.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.encoder = PathEncoder()
        self.config = self._load_config(config_path)
        self.metrics_categories = {
            'traditional': ['accuracy', 'precision', 'recall', 'f1_score'],
            'customer_experience': ['response_time', 'satisfaction', 'usability'],
            'artificial_intelligence': ['reasoning', 'knowledge_integration', 'adaptability']
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from a file if provided, otherwise use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        default_config = {
            'relevance_threshold': 0.65,
            'validation_threshold': 0.75,
            'relationship_confidence': 0.70,
            'metrics_baseline_path': './data/metrics_baseline.json',
            'metrics_target_path': './data/metrics_targets.json'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    return {**default_config, **user_config}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def load_metrics_data(self, metrics_path: str) -> Dict[str, Any]:
        """
        Load metrics data from a JSON file.
        
        Args:
            metrics_path: Path to metrics JSON file
            
        Returns:
            Dictionary containing metrics data
        """
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics data from {metrics_path}: {e}")
            return {}
    
    def encode_metrics(self, metrics_data: Dict[str, Any]) -> List[str]:
        """
        Encode metrics data using path-based relationship encoding.
        
        Args:
            metrics_data: Dictionary containing metrics data
            
        Returns:
            List of encoded paths
        """
        return self.encoder.encode(metrics_data)
    
    def detect_relationships(self, encoded_metrics: List[str]) -> List[str]:
        """
        Detect relationships between different metrics paths.
        
        Args:
            encoded_metrics: List of encoded metrics paths
            
        Returns:
            List of relationship encodings
        """
        relationships = []
        
        # Define key metric relationships for healthcare domain
        relationship_patterns = [
            # Customer Experience relationships
            ('customer_experience.satisfaction', 'customer_experience.response_time', 'correlation_high'),
            ('customer_experience.usability', 'customer_experience.satisfaction', 'direct_influence'),
            
            # AI metrics relationships
            ('artificial_intelligence.reasoning', 'artificial_intelligence.knowledge_integration', 'correlation_medium'),
            ('artificial_intelligence.adaptability', 'artificial_intelligence.reasoning', 'enhancement'),
            
            # Cross-category relationships
            ('traditional.accuracy', 'customer_experience.satisfaction', 'foundation'),
            ('artificial_intelligence.reasoning', 'traditional.precision', 'enhancement'),
            ('customer_experience.response_time', 'traditional.recall', 'inverse_correlation')
        ]
        
        # Generate relationships based on patterns
        for src_path, target_path, relation_type in relationship_patterns:
            # Verify paths exist in encoded metrics
            src_exists = any(src_path in path for path in encoded_metrics)
            target_exists = any(target_path in path for path in encoded_metrics)
            
            if src_exists and target_exists:
                relationship = self.encoder.encode_relationships(src_path, target_path, relation_type)
                relationships.append(relationship)
                logger.debug(f"Added relationship: {relationship}")
        
        # Detect additional relationships based on value correlations
        # (Implementation would analyze numerical correlations between metrics)
        
        return relationships
    
    def validate_metrics(self, current_metrics: Dict[str, Any], 
                         baseline_metrics: Optional[Dict[str, Any]] = None,
                         target_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate current metrics against baseline and target values.
        
        Args:
            current_metrics: Dictionary of current metrics values
            baseline_metrics: Optional dictionary of baseline metrics
            target_metrics: Optional dictionary of target metrics
            
        Returns:
            Dictionary containing validation results
        """
        # Encode the metrics data
        encoded_current = self.encode_metrics(current_metrics)
        
        # Detect relationships
        relationships = self.detect_relationships(encoded_current)
        encoded_current.extend(relationships)
        
        # Load baseline and target if not provided
        if baseline_metrics is None and os.path.exists(self.config['metrics_baseline_path']):
            baseline_metrics = self.load_metrics_data(self.config['metrics_baseline_path'])
        
        if target_metrics is None and os.path.exists(self.config['metrics_target_path']):
            target_metrics = self.load_metrics_data(self.config['metrics_target_path'])
        
        # Initialize validation results
        validation_results = {
            'overall': {
                'valid': True,
                'score': 0.0,
                'improvement_from_baseline': 0.0,
                'distance_to_target': 0.0
            },
            'categories': {},
            'metrics': {},
            'relationships': {
                'valid': True,
                'inconsistencies': []
            }
        }
        
        # Validate each metrics category
        category_scores = []
        for category, metrics in self.metrics_categories.items():
            category_result = self._validate_category(
                category, metrics, current_metrics, baseline_metrics, target_metrics
            )
            validation_results['categories'][category] = category_result
            category_scores.append(category_result['score'])
        
        # Calculate overall score
        if category_scores:
            validation_results['overall']['score'] = np.mean(category_scores)
            
        # Validate relationships between metrics
        self._validate_relationships(validation_results, current_metrics)
        
        return validation_results
    
    def _validate_category(self, category: str, metrics: List[str], 
                          current: Dict[str, Any],
                          baseline: Optional[Dict[str, Any]],
                          target: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a specific category of metrics.
        
        Args:
            category: Category name
            metrics: List of metrics in the category
            current: Current metrics values
            baseline: Baseline metrics values
            target: Target metrics values
            
        Returns:
            Dictionary containing validation results for the category
        """
        result = {
            'valid': True,
            'score': 0.0,
            'improvement_from_baseline': 0.0,
            'distance_to_target': 0.0,
            'metrics': {}
        }
        
        metric_scores = []
        
        for metric in metrics:
            metric_path = f"{category}.{metric}"
            
            # Extract values using path notation
            current_value = self._get_nested_value(current, metric_path)
            baseline_value = self._get_nested_value(baseline, metric_path) if baseline else None
            target_value = self._get_nested_value(target, metric_path) if target else None
            
            # Skip if current value is missing
            if current_value is None:
                continue
                
            metric_result = {
                'current': current_value,
                'baseline': baseline_value,
                'target': target_value,
                'valid': True
            }
            
            # Calculate improvement from baseline
            if baseline_value is not None:
                improvement = current_value - baseline_value
                metric_result['improvement'] = improvement
                metric_result['percent_improvement'] = (improvement / baseline_value) * 100 if baseline_value else 0
            
            # Calculate distance to target
            if target_value is not None:
                distance = target_value - current_value
                metric_result['distance_to_target'] = distance
                metric_result['percent_to_target'] = (current_value / target_value) * 100 if target_value else 0
                
                # Validate if metric meets minimum threshold
                threshold = self.config['validation_threshold'] * target_value
                metric_result['valid'] = current_value >= threshold
                if not metric_result['valid']:
                    result['valid'] = False
            
            # Add to results
            result['metrics'][metric] = metric_result
            metric_scores.append(current_value)
        
        # Calculate category score
        if metric_scores:
            result['score'] = np.mean(metric_scores)
            
            # Calculate category improvement
            if baseline and any(self._get_nested_value(baseline, f"{category}.{m}") for m in metrics):
                baseline_scores = [self._get_nested_value(baseline, f"{category}.{m}") for m in metrics]
                baseline_scores = [s for s in baseline_scores if s is not None]
                if baseline_scores:
                    baseline_avg = np.mean(baseline_scores)
                    result['improvement_from_baseline'] = result['score'] - baseline_avg
            
            # Calculate category distance to target
            if target and any(self._get_nested_value(target, f"{category}.{m}") for m in metrics):
                target_scores = [self._get_nested_value(target, f"{category}.{m}") for m in metrics]
                target_scores = [s for s in target_scores if s is not None]
                if target_scores:
                    target_avg = np.mean(target_scores)
                    result['distance_to_target'] = target_avg - result['score']
        
        return result
    
    def _validate_relationships(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Validate relationships between metrics values.
        
        Args:
            results: Validation results to update
            metrics: Current metrics values
        """
        inconsistencies = []
        
        # Define expected relationships
        expected_relationships = [
            # If satisfaction is high, response time should be high
            {
                'path1': 'customer_experience.satisfaction',
                'path2': 'customer_experience.response_time',
                'relation': 'correlation',
                'threshold': 0.1
            },
            # If reasoning is high, knowledge_integration should be high
            {
                'path1': 'artificial_intelligence.reasoning',
                'path2': 'artificial_intelligence.knowledge_integration',
                'relation': 'correlation',
                'threshold': 0.15
            }
        ]
        
        for relation in expected_relationships:
            path1 = relation['path1']
            path2 = relation['path2']
            
            val1 = self._get_nested_value(metrics, path1)
            val2 = self._get_nested_value(metrics, path2)
            
            if val1 is not None and val2 is not None:
                if relation['relation'] == 'correlation':
                    # Check if values are correlated within threshold
                    if abs(val1 - val2) > relation['threshold']:
                        inconsistencies.append({
                            'type': 'correlation_violation',
                            'path1': path1,
                            'path2': path2,
                            'value1': val1,
                            'value2': val2,
                            'threshold': relation['threshold'],
                            'difference': abs(val1 - val2)
                        })
        
        results['relationships']['inconsistencies'] = inconsistencies
        results['relationships']['valid'] = len(inconsistencies) == 0
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get a value from a nested dictionary using a dot-notation path.
        
        Args:
            data: Dictionary to search
            path: Dot-notation path (e.g., 'category.metric')
            
        Returns:
            Value at the specified path or None if not found
        """
        if data is None:
            return None
            
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Dictionary containing validation results
            
        Returns:
            Formatted validation report
        """
        report = []
        report.append("=== Healthcare Metrics Validation Report ===\n")
        
        # Overall result
        overall = validation_results['overall']
        report.append(f"Overall Validation: {'PASSED' if overall['valid'] else 'FAILED'}")
        report.append(f"Overall Score: {overall['score']:.2f}\n")
        
        # Category results
        report.append("--- Category Results ---")
        for category, result in validation_results['categories'].items():
            report.append(f"\n{category.replace('_', ' ').title()}:")
            report.append(f"  Status: {'PASSED' if result['valid'] else 'FAILED'}")
            report.append(f"  Score: {result['score']:.2f}")
            
            if 'improvement_from_baseline' in result and result['improvement_from_baseline'] != 0:
                improvement = result['improvement_from_baseline']
                direction = "improvement" if improvement > 0 else "decline"
                report.append(f"  Baseline {direction}: {abs(improvement):.2f}")
            
            if 'distance_to_target' in result and result['distance_to_target'] != 0:
                report.append(f"  Distance to target: {result['distance_to_target']:.2f}")
            
            # Individual metrics
            report.append("\n  Metrics:")
            for metric, metric_result in result['metrics'].items():
                status = '✅' if metric_result.get('valid', True) else '❌'
                current = metric_result.get('current', 'N/A')
                report.append(f"    {status} {metric}: {current:.2f}")
                
                if 'improvement' in metric_result:
                    improvement = metric_result['improvement']
                    direction = "↑" if improvement > 0 else "↓"
                    report.append(f"      Baseline {direction}: {abs(improvement):.2f} ({metric_result.get('percent_improvement', 0):.1f}%)")
                
                if 'distance_to_target' in metric_result:
                    report.append(f"      Target gap: {metric_result['distance_to_target']:.2f} ({metric_result.get('percent_to_target', 0):.1f}%)")
        
        # Relationship validation
        relationships = validation_results.get('relationships', {})
        report.append("\n--- Relationship Validation ---")
        status = '✅' if relationships.get('valid', True) else '❌'
        report.append(f"{status} Relationships: {'CONSISTENT' if relationships.get('valid', True) else 'INCONSISTENT'}")
        
        inconsistencies = relationships.get('inconsistencies', [])
        if inconsistencies:
            report.append("\nInconsistencies:")
            for i, issue in enumerate(inconsistencies):
                report.append(f"  {i+1}. {issue['type']} between {issue['path1']} and {issue['path2']}")
                report.append(f"     Values: {issue['value1']:.2f} vs {issue['value2']:.2f} (diff: {issue['difference']:.2f}, threshold: {issue['threshold']:.2f})")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    validator = HealthcareMetricsValidator()
    
    # Sample healthcare metrics data
    current_metrics = {
        "traditional": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.88,
            "f1_score": 0.885
        },
        "customer_experience": {
            "response_time": 0.91,
            "satisfaction": 0.86,
            "usability": 0.87
        },
        "artificial_intelligence": {
            "reasoning": 0.85,
            "knowledge_integration": 0.88,
            "adaptability": 0.86
        }
    }
    
    # Sample baseline
    baseline_metrics = {
        "traditional": {
            "accuracy": 0.88,
            "precision": 0.83,
            "recall": 0.82,
            "f1_score": 0.825
        },
        "customer_experience": {
            "response_time": 0.85,
            "satisfaction": 0.79,
            "usability": 0.82
        },
        "artificial_intelligence": {
            "reasoning": 0.74,
            "knowledge_integration": 0.78,
            "adaptability": 0.75
        }
    }
    
    # Sample targets
    target_metrics = {
        "traditional": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.925
        },
        "customer_experience": {
            "response_time": 0.93,
            "satisfaction": 0.88,
            "usability": 0.89
        },
        "artificial_intelligence": {
            "reasoning": 0.86,
            "knowledge_integration": 0.89,
            "adaptability": 0.86
        }
    }
    
    # Validate metrics
    validation_results = validator.validate_metrics(
        current_metrics, baseline_metrics, target_metrics
    )
    
    # Generate and print report
    report = validator.generate_validation_report(validation_results)
    print(report)
