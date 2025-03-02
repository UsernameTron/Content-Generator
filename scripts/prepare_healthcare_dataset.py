#!/usr/bin/env python3
"""
Prepare healthcare contradiction and cross-reference dataset.
This script creates a specialized dataset for training the model
on healthcare-specific contradiction identification and
cross-referencing tasks.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("healthcare-dataset")

# Sample healthcare contradictions for dataset creation
SAMPLE_CONTRADICTIONS = [
    {
        "statement_1": "A meta-analysis of 12 randomized controlled trials showed that statin therapy reduces cardiovascular events by 25% in patients with elevated LDL cholesterol.",
        "statement_2": "Recent long-term observational studies suggest that statin therapy has minimal impact on cardiovascular outcomes when adjusted for lifestyle factors.",
        "relationship": "contradicting",
        "domain": "cardiology",
        "evidence_types": ["meta_analysis", "observational_study"]
    },
    {
        "statement_1": "The 2019 AHA guidelines recommend aspirin for primary prevention of cardiovascular disease in adults aged 40-70 with high cardiovascular risk.",
        "statement_2": "The 2022 AHA guidelines no longer recommend routine aspirin use for primary prevention due to increased bleeding risks.",
        "relationship": "temporally_superseded",
        "domain": "cardiology",
        "evidence_types": ["clinical_guideline", "clinical_guideline"]
    },
    {
        "statement_1": "Hydroxychloroquine showed promising results for COVID-19 treatment in early in vitro studies.",
        "statement_2": "Multiple randomized controlled trials demonstrated that hydroxychloroquine provides no benefit for COVID-19 treatment and may increase adverse events.",
        "relationship": "contradicting",
        "domain": "infectious_disease",
        "evidence_types": ["in_vitro_study", "randomized_controlled_trial"]
    }
]

# Sample evidence strength hierarchy
EVIDENCE_HIERARCHY = [
    {"type": "meta_analysis", "strength": 1, "description": "Systematic review and meta-analysis of randomized controlled trials"},
    {"type": "randomized_controlled_trial", "strength": 2, "description": "Individual randomized controlled trials"},
    {"type": "cohort_study", "strength": 3, "description": "Cohort studies and outcomes research"},
    {"type": "case_control", "strength": 4, "description": "Case-control studies"},
    {"type": "case_series", "strength": 5, "description": "Case series, case reports"},
    {"type": "expert_opinion", "strength": 6, "description": "Expert opinion without explicit critical appraisal"}
]

class HealthcareDatasetCreator:
    """Create healthcare contradiction and cross-reference dataset."""
    
    def __init__(self, output_dir="data/healthcare", config_path=None):
        """Initialize the dataset creator.
        
        Args:
            output_dir: Directory to save the dataset
            config_path: Path to configuration file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                self.config = full_config.get('healthcare_cross_reference', {})
        
        # Set default values
        self.contradiction_categories = self.config.get('contradiction_detection', {}).get(
            'categories', ["supporting", "contradicting", "unrelated", "temporally_superseded"])
        self.evidence_types = self.config.get('evidence_strength', {}).get(
            'types', ["rct", "meta_analysis", "cohort_study", "case_control", "case_series", "expert_opinion"])
        self.medical_domains = self.config.get('medical_domains', 
            ["cardiology", "oncology", "neurology", "infectious_disease", "pharmacology"])
    
    def create_contradiction_dataset(self, samples_per_category=50, evaluation_split=0.2):
        """Create a dataset of healthcare statement pairs with labeled relationships.
        
        Args:
            samples_per_category: Number of samples per contradiction category
            evaluation_split: Fraction of data to use for evaluation
        
        Returns:
            train_data, eval_data: Training and evaluation datasets
        """
        logger.info(f"Creating contradiction dataset with {samples_per_category} samples per category")
        
        # In a real implementation, we would:
        # 1. Extract statements from medical literature
        # 2. Use NLP techniques to identify potential contradiction pairs
        # 3. Have medical experts annotate the relationships
        
        # For this demonstration, we'll create synthetic data based on templates
        all_data = []
        
        # Use sample data and templates to generate the dataset
        for category in self.contradiction_categories:
            logger.info(f"Generating {samples_per_category} samples for category: {category}")
            
            # In a real implementation, this would be more sophisticated
            for i in range(samples_per_category):
                # Generate data based on templates and the sample contradictions
                sample_idx = i % len(SAMPLE_CONTRADICTIONS)
                sample = SAMPLE_CONTRADICTIONS[sample_idx].copy()
                
                # Override the relationship to match the current category
                sample["relationship"] = category
                
                # Modify statements slightly for variety
                if i > 0:
                    # Add minor modifications for variety
                    sample["statement_1"] += f" (Variation {i})"
                    sample["statement_2"] += f" (Variation {i})"
                
                all_data.append(sample)
        
        # Shuffle data
        np.random.shuffle(all_data)
        
        # Split into train and evaluation sets
        split_idx = int(len(all_data) * (1 - evaluation_split))
        train_data = all_data[:split_idx]
        eval_data = all_data[split_idx:]
        
        # Save datasets
        with open(self.output_dir / "contradiction_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(self.output_dir / "contradiction_eval.json", 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        logger.info(f"Saved {len(train_data)} training samples and {len(eval_data)} evaluation samples")
        
        return train_data, eval_data
    
    def create_evidence_ranking_dataset(self, samples_per_type=40, evaluation_split=0.2):
        """Create a dataset for evidence strength ranking tasks.
        
        Args:
            samples_per_type: Number of samples per evidence type
            evaluation_split: Fraction of data to use for evaluation
            
        Returns:
            train_data, eval_data: Training and evaluation datasets
        """
        logger.info(f"Creating evidence ranking dataset with {samples_per_type} samples per type")
        
        # Use the evidence hierarchy to create ranking tasks
        all_data = []
        
        for evidence_type in self.evidence_types:
            logger.info(f"Generating {samples_per_type} samples for evidence type: {evidence_type}")
            
            # Find the strength of this evidence type
            evidence_info = next((item for item in EVIDENCE_HIERARCHY if item["type"] == evidence_type), None)
            if not evidence_info:
                logger.warning(f"Evidence type {evidence_type} not found in hierarchy")
                continue
                
            for i in range(samples_per_type):
                # Generate a comparison task
                # Select a random different evidence type
                other_types = [e for e in EVIDENCE_HIERARCHY if e["type"] != evidence_type]
                other_evidence = np.random.choice(other_types)
                
                # Determine which is stronger
                if evidence_info["strength"] < other_evidence["strength"]:
                    stronger = evidence_info["type"]
                    weaker = other_evidence["type"]
                else:
                    stronger = other_evidence["type"]
                    weaker = evidence_info["type"]
                
                # Create a task description
                medical_domain = np.random.choice(self.medical_domains)
                condition = f"condition_{i}"
                
                sample = {
                    "task": f"Compare the strength of evidence between these two studies about {condition} in {medical_domain}:",
                    "evidence_1": {
                        "type": evidence_info["type"],
                        "description": f"A {evidence_info['description']} examining {condition} in {medical_domain} patients"
                    },
                    "evidence_2": {
                        "type": other_evidence["type"],
                        "description": f"A {other_evidence['description']} examining {condition} in {medical_domain} patients"
                    },
                    "stronger_evidence": stronger,
                    "weaker_evidence": weaker,
                    "domain": medical_domain
                }
                
                all_data.append(sample)
        
        # Shuffle data
        np.random.shuffle(all_data)
        
        # Split into train and evaluation sets
        split_idx = int(len(all_data) * (1 - evaluation_split))
        train_data = all_data[:split_idx]
        eval_data = all_data[split_idx:]
        
        # Save datasets
        with open(self.output_dir / "evidence_ranking_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(self.output_dir / "evidence_ranking_eval.json", 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        logger.info(f"Saved {len(train_data)} training samples and {len(eval_data)} evaluation samples")
        
        return train_data, eval_data
    
    def create_all_datasets(self):
        """Create all healthcare datasets."""
        contradictions_per_category = self.config.get('contradiction_detection', {}).get(
            'training_examples_per_category', 50)
        examples_per_evidence_type = self.config.get('evidence_strength', {}).get(
            'training_examples_per_type', 40)
        
        logger.info("Creating all healthcare datasets")
        self.create_contradiction_dataset(samples_per_category=contradictions_per_category)
        self.create_evidence_ranking_dataset(samples_per_type=examples_per_evidence_type)
        
        # Create a summary of the datasets
        summary = {
            "contradictions": {
                "categories": self.contradiction_categories,
                "samples_per_category": contradictions_per_category
            },
            "evidence_ranking": {
                "evidence_types": self.evidence_types,
                "samples_per_type": examples_per_evidence_type
            },
            "medical_domains": self.medical_domains,
            "dataset_path": str(self.output_dir)
        }
        
        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"All datasets created and saved to {self.output_dir}")
        return summary

def main():
    """Main function to run the dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare healthcare cross-reference dataset")
    parser.add_argument("--output", type=str, default="data/healthcare",
                       help="Output directory for the dataset")
    parser.add_argument("--config", type=str, default="healthcare_cross_reference_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        creator = HealthcareDatasetCreator(args.output, args.config)
        summary = creator.create_all_datasets()
        print(f"Dataset creation complete. Summary:\n{json.dumps(summary, indent=2)}")
    except Exception as e:
        logger.error(f"Error creating healthcare dataset: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
