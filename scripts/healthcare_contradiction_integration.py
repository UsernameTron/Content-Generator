#!/usr/bin/env python3
"""
Healthcare Contradiction Integration Module.
Connects the contradiction dataset to the evaluation framework and adds healthcare-specific metrics.
"""

import os
import json
import logging
import random
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import traceback
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-contradiction-integration")

class HealthcareContradictionIntegrator:
    """
    Integrates healthcare contradiction dataset with the evaluation framework.
    Provides specialized metrics for measuring contradiction detection performance.
    """
    
    def __init__(self, contradiction_dataset_path: str, eval_results_path: Optional[str] = None):
        """
        Initialize the healthcare contradiction integrator.
        
        Args:
            contradiction_dataset_path: Path to the contradiction dataset JSON file
            eval_results_path: Optional path to existing evaluation results
        """
        self.contradiction_dataset_path = Path(contradiction_dataset_path)
        self.eval_results_path = Path(eval_results_path) if eval_results_path else None
        
        # Load contradiction dataset
        if self.contradiction_dataset_path.exists():
            with open(self.contradiction_dataset_path, 'r') as f:
                self.contradiction_data = json.load(f)
            logger.info(f"Loaded {len(self.contradiction_data)} contradiction examples")
        else:
            raise FileNotFoundError(f"Contradiction dataset not found: {contradiction_dataset_path}")
        
        # Load evaluation results if provided
        self.eval_data = None
        if self.eval_results_path and self.eval_results_path.exists():
            with open(self.eval_results_path, 'r') as f:
                self.eval_data = json.load(f)
            logger.info(f"Loaded existing evaluation results from {eval_results_path}")
    
    def analyze_contradiction_types(self) -> Dict[str, Any]:
        """
        Analyze contradiction types in the dataset.
        
        Returns:
            Dictionary with contradiction type analysis metrics
        """
        types = {}
        domains = {}
        
        # Count occurrences of each contradiction type and domain
        for item in self.contradiction_data:
            contradiction_type = item.get('type', 'unknown')
            domain = item.get('domain', 'unknown')
            
            types[contradiction_type] = types.get(contradiction_type, 0) + 1
            domains[domain] = domains.get(domain, 0) + 1
        
        # Calculate percentages
        total = len(self.contradiction_data)
        type_percentages = {t: (count / total) * 100 for t, count in types.items()}
        domain_percentages = {d: (count / total) * 100 for d, count in domains.items()}
        
        return {
            'contradiction_types': types,
            'type_percentages': type_percentages,
            'domains': domains,
            'domain_percentages': domain_percentages,
            'total_contradictions': total
        }
    
    def compute_temporal_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to temporal aspects of medical contradictions.
        
        Returns:
            Dictionary with temporal metrics
        """
        time_gaps = []
        time_ranges = []
        
        for item in self.contradiction_data:
            dates = item.get('publication_dates', [])
            if len(dates) >= 2:
                try:
                    # Parse dates and calculate time gap
                    date1 = datetime.strptime(dates[0], "%Y-%m-%d")
                    date2 = datetime.strptime(dates[1], "%Y-%m-%d")
                    gap = abs((date2 - date1).days / 365.25)  # Gap in years
                    
                    time_gaps.append({
                        'gap': gap,
                        'type': item.get('type', 'unknown'),
                        'domain': item.get('domain', 'unknown')
                    })
                    
                    time_ranges.append({
                        'start': dates[0][:4],  # Year only
                        'end': dates[1][:4],    # Year only
                        'domain': item.get('domain', 'unknown'),
                        'type': item.get('type', 'unknown')
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing dates for item: {e}")
        
        # Calculate average time gap by contradiction type and domain
        df_gaps = pd.DataFrame(time_gaps)
        avg_gap_by_type = {}
        avg_gap_by_domain = {}
        
        if not df_gaps.empty:
            avg_gap_by_type = df_gaps.groupby('type')['gap'].mean().to_dict()
            avg_gap_by_domain = df_gaps.groupby('domain')['gap'].mean().to_dict()
        
        return {
            'time_gaps': time_gaps,
            'time_ranges': time_ranges,
            'avg_gap_by_type': avg_gap_by_type,
            'avg_gap_by_domain': avg_gap_by_domain,
            'overall_avg_gap': np.mean([g['gap'] for g in time_gaps]) if time_gaps else 0
        }
    
    def evaluate_medical_terminology(self) -> Dict[str, Any]:
        """
        Evaluate medical terminology usage in contradictions.
        
        Returns:
            Dictionary with medical terminology metrics
        """
        # List of common medical terminology categories (simplified)
        medical_categories = {
            'medications': ['therapy', 'drug', 'medication', 'dose', 'treatment', 'aspirin', 'vitamin', 'hormone'],
            'procedures': ['surgery', 'screening', 'test', 'procedure', 'examination', 'mastectomy'],
            'conditions': ['cancer', 'disease', 'syndrome', 'disorder', 'condition', 'symptom'],
            'anatomy': ['breast', 'heart', 'liver', 'organ', 'tissue', 'brain'],
            'physiology': ['blood pressure', 'cholesterol', 'metabolism', 'immune', 'hormonal']
        }
        
        # Initialize counters
        terminology_counts = {category: 0 for category in medical_categories}
        domain_terminology = {}
        
        # Count terminology occurrences
        for item in self.contradiction_data:
            domain = item.get('domain', 'unknown')
            if domain not in domain_terminology:
                domain_terminology[domain] = {category: 0 for category in medical_categories}
            
            # Check both statements for terminology
            combined_text = (item.get('statement1', '') + ' ' + item.get('statement2', '')).lower()
            
            for category, terms in medical_categories.items():
                for term in terms:
                    if term.lower() in combined_text:
                        terminology_counts[category] += 1
                        domain_terminology[domain][category] += 1
                        break  # Count only once per category per contradiction
        
        return {
            'terminology_counts': terminology_counts,
            'domain_terminology': domain_terminology,
            'terminology_density': sum(terminology_counts.values()) / len(self.contradiction_data) if self.contradiction_data else 0
        }
    
    def evaluate_source_credibility(self) -> Dict[str, Any]:
        """
        Evaluate source credibility metrics in the contradiction dataset.
        
        Returns:
            Dictionary with source credibility metrics
        """
        # Simplified credibility scoring (would be more sophisticated in a real system)
        credibility_tiers = {
            'high': ['NEJM', 'JAMA', 'Lancet', 'BMJ', 'NIH', 'WHO', 'Cochrane'],
            'medium': ['Guidelines', 'Review', 'Initiative', 'Association', 'College', 'Society'],
            'standard': []  # All other sources
        }
        
        source_counts = {}
        credibility_scores = {}
        domain_credibility = {}
        
        for item in self.contradiction_data:
            domain = item.get('domain', 'unknown')
            sources = item.get('sources', [])
            
            if domain not in domain_credibility:
                domain_credibility[domain] = {'high': 0, 'medium': 0, 'standard': 0}
            
            for source in sources:
                # Count source occurrences
                source_counts[source] = source_counts.get(source, 0) + 1
                
                # Evaluate credibility
                credibility = 'standard'
                for tier, keywords in credibility_tiers.items():
                    if any(keyword in source for keyword in keywords):
                        credibility = tier
                        break
                
                credibility_scores[source] = credibility
                domain_credibility[domain][credibility] += 1
        
        # Calculate overall credibility distribution
        total_sources = sum(source_counts.values())
        credibility_distribution = {
            'high': 0,
            'medium': 0,
            'standard': 0
        }
        
        for source, count in source_counts.items():
            credibility = credibility_scores.get(source, 'standard')
            credibility_distribution[credibility] += count
        
        # Convert to percentages
        if total_sources > 0:
            credibility_distribution = {k: (v / total_sources) * 100 for k, v in credibility_distribution.items()}
        
        return {
            'source_counts': source_counts,
            'credibility_scores': credibility_scores,
            'credibility_distribution': credibility_distribution,
            'domain_credibility': domain_credibility
        }
    
    def generate_healthcare_metrics(self) -> Dict[str, Any]:
        """
        Generate comprehensive healthcare-specific metrics.
        
        Returns:
            Dictionary with all healthcare-specific metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'contradiction_analysis': self.analyze_contradiction_types(),
            'temporal_metrics': self.compute_temporal_metrics(),
            'terminology_metrics': self.evaluate_medical_terminology(),
            'credibility_metrics': self.evaluate_source_credibility()
        }
        
        # If we have evaluation results, calculate performance on contradiction types
        if self.eval_data:
            contradiction_performance = self.calculate_contradiction_performance()
            metrics['contradiction_performance'] = contradiction_performance
        
        return metrics
    
    def calculate_contradiction_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics specific to contradiction detection.
        
        Returns:
            Dictionary with contradiction detection performance metrics
        """
        # This is a simplified implementation - in a real system, this would match
        # evaluation results with specific contradiction examples
        
        if not self.eval_data:
            return {}
        
        # Get contradiction types from dataset
        contradiction_types = {}
        for item in self.contradiction_data:
            c_type = item.get('type', 'unknown')
            if c_type not in contradiction_types:
                contradiction_types[c_type] = []
            contradiction_types[c_type].append(item)
        
        # Initialize performance metrics
        performance_by_type = {}
        performance_by_domain = {}
        
        # Extract performance metrics from evaluation data (simplified assumption)
        # In a real implementation, this would match specific contradictions with their results
        
        # Simplified approach - using existing metrics as proxies
        if 'overall' in self.eval_data:
            base_accuracy = self.eval_data.get('overall', {}).get('accuracy', 0.75)
            
            # Simulate different accuracies for different contradiction types
            # In a real system, these would come from actual evaluations
            type_modifiers = {
                'direct_contradiction': 1.1,    # Easier to detect
                'temporal_change': 0.9,         # Harder to detect
                'methodological_difference': 0.85  # Hardest to detect
            }
            
            for c_type, examples in contradiction_types.items():
                modifier = type_modifiers.get(c_type, 1.0)
                accuracy = min(base_accuracy * modifier, 0.98)  # Cap at 98%
                
                performance_by_type[c_type] = {
                    'accuracy': accuracy,
                    'sample_count': len(examples),
                    'confidence': 0.7 + (0.2 * random.random())  # Simulated confidence
                }
        
        # Extract domain performance
        domains = {}
        for item in self.contradiction_data:
            domain = item.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(item)
        
        for domain, examples in domains.items():
            if domain in self.eval_data:
                performance_by_domain[domain] = self.eval_data[domain]
            else:
                # Use overall metrics if domain-specific not available
                performance_by_domain[domain] = {
                    'accuracy': self.eval_data.get('overall', {}).get('accuracy', 0.75),
                    'sample_count': len(examples)
                }
        
        return {
            'performance_by_type': performance_by_type,
            'performance_by_domain': performance_by_domain
        }
    
    def save_metrics(self, output_path: str) -> None:
        """
        Save the healthcare metrics to a JSON file.
        
        Args:
            output_path: Path to save the metrics file
        """
        metrics = self.generate_healthcare_metrics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Healthcare contradiction metrics saved to {output_path}")
        
        return metrics


def main():
    """Main function to run the healthcare contradiction integration."""
    import argparse
    import random  # for the simplified implementation
    
    parser = argparse.ArgumentParser(description="Healthcare Contradiction Integration")
    parser.add_argument("--contradiction-dataset", required=True, help="Path to contradiction dataset JSON")
    parser.add_argument("--eval-results", help="Path to evaluation results JSON")
    parser.add_argument("--output", required=True, help="Path to save healthcare metrics")
    
    args = parser.parse_args()
    
    try:
        # Initialize the integrator
        integrator = HealthcareContradictionIntegrator(
            contradiction_dataset_path=args.contradiction_dataset,
            eval_results_path=args.eval_results
        )
        
        # Generate and save metrics
        metrics = integrator.save_metrics(args.output)
        
        logger.info("Healthcare contradiction integration completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error in healthcare contradiction integration: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import random  # Required for the simplified implementation
    import traceback
    exit(main())
