#!/usr/bin/env python3
"""
Generate synthetic evaluation results for testing the continuous learning system.
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

def generate_synthetic_evaluation(output_path, accuracy=0.75):
    """Generate synthetic evaluation results for contradiction detection.
    
    Args:
        output_path: Path to save the evaluation results
        accuracy: Overall accuracy to simulate
    """
    # Categories and domains
    categories = ["supporting", "contradicting", "unrelated", "temporally_superseded"]
    domains = ["cardiology", "oncology", "neurology", "infectious_disease", "pharmacology"]
    
    # Generate category and domain accuracies
    category_metrics = {}
    for category in categories:
        # Vary accuracies around the overall accuracy
        cat_accuracy = max(0.1, min(0.95, accuracy + random.uniform(-0.2, 0.2)))
        category_metrics[category] = {
            "correct": int(100 * cat_accuracy),
            "total": 100,
            "accuracy": cat_accuracy
        }
    
    domain_metrics = {}
    for domain in domains:
        # Vary accuracies around the overall accuracy
        dom_accuracy = max(0.1, min(0.95, accuracy + random.uniform(-0.2, 0.2)))
        domain_metrics[domain] = {
            "correct": int(100 * dom_accuracy),
            "total": 100,
            "accuracy": dom_accuracy
        }
    
    # Generate example results
    examples = []
    for i in range(100):
        # Generate more incorrect examples to make learning interesting
        is_correct = random.random() < accuracy
        
        # Select random category and domain
        category = random.choice(categories)
        domain = random.choice(domains)
        
        # Generate random prediction
        if is_correct:
            predicted_category = category
        else:
            # Choose a different category for incorrect prediction
            other_categories = [c for c in categories if c != category]
            predicted_category = random.choice(other_categories)
        
        # Generate example statements
        statement_templates = [
            "Patients with {condition} should be treated with {treatment}.",
            "{treatment} has been shown effective for {condition} in clinical trials.",
            "Recent studies suggest {treatment} may reduce symptoms of {condition}.",
            "Clinical guidelines recommend {treatment} as first-line therapy for {condition}.",
            "Evidence shows {treatment} improves outcomes in patients with {condition}."
        ]
        
        conditions = {
            "cardiology": ["hypertension", "atrial fibrillation", "heart failure", "coronary artery disease"],
            "oncology": ["breast cancer", "lung cancer", "colon cancer", "lymphoma"],
            "neurology": ["migraine", "epilepsy", "Parkinson's disease", "multiple sclerosis"],
            "infectious_disease": ["pneumonia", "urinary tract infection", "cellulitis", "COVID-19"],
            "pharmacology": ["pain", "nausea", "inflammation", "insomnia"]
        }
        
        treatments = {
            "cardiology": ["ACE inhibitors", "beta blockers", "aspirin", "statins"],
            "oncology": ["chemotherapy", "radiation therapy", "immunotherapy", "targeted therapy"],
            "neurology": ["anticonvulsants", "dopamine agonists", "triptans", "corticosteroids"],
            "infectious_disease": ["antibiotics", "antivirals", "antifungals", "vaccines"],
            "pharmacology": ["NSAIDs", "opioids", "antihistamines", "benzodiazepines"]
        }
        
        condition = random.choice(conditions.get(domain, conditions["pharmacology"]))
        treatment = random.choice(treatments.get(domain, treatments["pharmacology"]))
        
        statement_1 = random.choice(statement_templates).format(condition=condition, treatment=treatment)
        
        # For contradicting or temporally_superseded, create a statement that disagrees
        if category in ["contradicting", "temporally_superseded"]:
            contradiction_templates = [
                "Studies have shown {treatment} is ineffective for {condition}.",
                "{treatment} is contraindicated for patients with {condition}.",
                "Recent evidence suggests {treatment} may worsen outcomes in {condition}.",
                "Clinical guidelines advise against using {treatment} for {condition}.",
                "Research indicates {treatment} has no significant benefit for {condition}."
            ]
            statement_2 = random.choice(contradiction_templates).format(condition=condition, treatment=treatment)
        else:
            # For supporting or unrelated, create an appropriate statement
            if category == "supporting":
                support_templates = [
                    "Multiple studies confirm {treatment} is effective for {condition}.",
                    "Meta-analyses support the use of {treatment} in {condition}.",
                    "Clinical experience has validated {treatment} for {condition}.",
                    "{treatment} remains a cornerstone of therapy for {condition}.",
                    "Research consistently demonstrates the benefits of {treatment} for {condition}."
                ]
                statement_2 = random.choice(support_templates).format(condition=condition, treatment=treatment)
            else:  # unrelated
                unrelated_condition = random.choice([c for c in conditions.get(domain, conditions["pharmacology"]) if c != condition])
                unrelated_treatment = random.choice([t for t in treatments.get(domain, treatments["pharmacology"]) if t != treatment])
                statement_2 = random.choice(statement_templates).format(condition=unrelated_condition, treatment=unrelated_treatment)
        
        # Create example entry
        example = {
            "statement_1": statement_1,
            "statement_2": statement_2,
            "true_category": category,
            "predicted_category": predicted_category,
            "correct": is_correct,
            "domain": domain
        }
        
        examples.append(example)
    
    # Create confusion matrix (4x4 for the 4 categories)
    confusion_matrix = [[0 for _ in range(4)] for _ in range(4)]
    
    # Map categories to indices
    category_indices = {cat: i for i, cat in enumerate(categories)}
    
    # Fill confusion matrix
    for example in examples:
        true_idx = category_indices[example["true_category"]]
        pred_idx = category_indices[example["predicted_category"]]
        confusion_matrix[true_idx][pred_idx] += 1
    
    # Create final evaluation results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "by_category": category_metrics,
        "by_domain": domain_metrics,
        "confusion_matrix": confusion_matrix,
        "examples": examples
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Generated synthetic evaluation results at {output_path}")
    print(f"Overall accuracy: {accuracy:.2f}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation results")
    parser.add_argument("--output", 
                      type=str, 
                      default="data/healthcare/evaluation/synthetic_results.json",
                      help="Path to save evaluation results")
    parser.add_argument("--accuracy", 
                      type=float, 
                      default=0.75,
                      help="Overall accuracy to simulate")
    args = parser.parse_args()
    
    generate_synthetic_evaluation(args.output, args.accuracy)
    
    return 0

if __name__ == "__main__":
    main()
