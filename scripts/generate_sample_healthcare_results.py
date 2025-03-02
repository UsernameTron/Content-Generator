#!/usr/bin/env python3
"""
Generate sample healthcare evaluation results for testing visualizations.
Creates realistic-looking evaluation data without requiring model execution.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_sample_results(output_path, performance_level="medium"):
    """
    Generate sample healthcare evaluation results.
    
    Args:
        output_path: Path to save sample results
        performance_level: Performance level to simulate (low, medium, high)
        
    Returns:
        Path to generated results file
    """
    # Set base accuracy based on performance level
    if performance_level == "low":
        base_accuracy = 0.55
    elif performance_level == "medium":
        base_accuracy = 0.68
    elif performance_level == "high":
        base_accuracy = 0.82
    else:
        base_accuracy = 0.68  # Default to medium
    
    # Define sample categories and domains
    contradiction_categories = [
        "direct_contradiction", 
        "partial_contradiction", 
        "temporal_conflict", 
        "demographic_conflict",
        "dosage_conflict",
        "treatment_guidance"
    ]
    
    healthcare_domains = [
        "cardiology", 
        "oncology", 
        "neurology", 
        "pediatrics",
        "emergency_medicine",
        "internal_medicine"
    ]
    
    evidence_types = [
        "clinical_trial",
        "meta_analysis",
        "case_study",
        "expert_opinion",
        "clinical_guideline",
        "observational_study"
    ]
    
    # Generate contradiction detection results
    contradiction_detection = {
        "accuracy": base_accuracy + np.random.uniform(-0.05, 0.05),
        "by_category": {},
        "by_domain": {},
        "examples": []
    }
    
    # Generate category metrics
    for category in contradiction_categories:
        # Add some variation to accuracy by category
        category_accuracy = min(1.0, max(0.0, 
            base_accuracy + np.random.normal(0, 0.12)))
        
        total = np.random.randint(15, 50)
        correct = int(total * category_accuracy)
        
        contradiction_detection["by_category"][category] = {
            "accuracy": category_accuracy,
            "total": total,
            "correct": correct
        }
    
    # Generate domain metrics
    for domain in healthcare_domains:
        # Add some variation to accuracy by domain
        domain_accuracy = min(1.0, max(0.0, 
            base_accuracy + np.random.normal(0, 0.08)))
        
        total = np.random.randint(20, 60)
        correct = int(total * domain_accuracy)
        
        contradiction_detection["by_domain"][domain] = {
            "accuracy": domain_accuracy,
            "total": total,
            "correct": correct
        }
    
    # Generate evidence ranking results
    evidence_ranking = {
        "accuracy": base_accuracy + np.random.uniform(-0.03, 0.07),
        "by_evidence_type": {},
        "by_domain": {},
        "examples": []
    }
    
    # Generate evidence type metrics
    for ev_type in evidence_types:
        # Add some variation to accuracy by evidence type
        ev_accuracy = min(1.0, max(0.0, 
            base_accuracy + np.random.normal(0, 0.10)))
        
        total = np.random.randint(10, 40)
        correct = int(total * ev_accuracy)
        
        evidence_ranking["by_evidence_type"][ev_type] = {
            "accuracy": ev_accuracy,
            "total": total,
            "correct": correct
        }
    
    # Generate domain metrics for evidence ranking
    for domain in healthcare_domains:
        # Add some variation to accuracy by domain
        domain_accuracy = min(1.0, max(0.0, 
            base_accuracy + np.random.normal(0, 0.08)))
        
        total = np.random.randint(15, 45)
        correct = int(total * domain_accuracy)
        
        evidence_ranking["by_domain"][domain] = {
            "accuracy": domain_accuracy,
            "total": total,
            "correct": correct
        }
    
    # Create sample examples (just a few for demonstration)
    for i in range(5):
        category = np.random.choice(contradiction_categories)
        domain = np.random.choice(healthcare_domains)
        is_correct = np.random.random() > 0.3  # 70% correct
        
        contradiction_detection["examples"].append({
            "task": f"Sample contradiction task {i+1}",
            "statement_1": f"Sample medical statement 1 for {domain}",
            "statement_2": f"Sample medical statement 2 for {domain} - {'contradicting' if is_correct else 'supporting'}",
            "expected": "contradiction" if is_correct else "non_contradiction",
            "predicted": "contradiction" if is_correct else "non_contradiction",
            "correct": is_correct,
            "category": category,
            "domain": domain
        })
    
    for i in range(5):
        ev_type1 = np.random.choice(evidence_types)
        ev_type2 = np.random.choice([t for t in evidence_types if t != ev_type1])
        domain = np.random.choice(healthcare_domains)
        is_correct = np.random.random() > 0.3  # 70% correct
        
        evidence_ranking["examples"].append({
            "task": f"Sample evidence ranking task {i+1}",
            "evidence_1": {
                "description": f"Sample {ev_type1} evidence for {domain}",
                "type": ev_type1
            },
            "evidence_2": {
                "description": f"Sample {ev_type2} evidence for {domain}",
                "type": ev_type2
            },
            "stronger_evidence": ev_type1 if is_correct else ev_type2,
            "predicted_stronger": "evidence_1" if is_correct else "evidence_2",
            "correct": is_correct,
            "domain": domain
        })
    
    # Create complete results
    results = {
        "contradiction_detection": contradiction_detection,
        "evidence_ranking": evidence_ranking,
        "metadata": {
            "model_path": "sample/healthcare/model",
            "adapter_path": "sample/healthcare/adapter",
            "device": "mps",
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Sample healthcare results saved to {output_path}")
    
    return output_path

def main():
    """Main function to generate sample healthcare results."""
    parser = argparse.ArgumentParser(description="Generate sample healthcare evaluation results")
    parser.add_argument("--output", type=str, default="output/sample_healthcare_eval.json",
                        help="Path to save sample results")
    parser.add_argument("--performance", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Performance level to simulate")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        output_path = generate_sample_results(output_path, args.performance)
        print(f"Successfully generated sample results: {output_path}")
    except Exception as e:
        print(f"Error generating sample results: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
