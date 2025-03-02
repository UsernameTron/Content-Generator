#!/usr/bin/env python3
"""
Initialize data structure for healthcare continuous learning dashboard.
This script creates the necessary files and directory structure for the dashboard to function.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("initialize-dashboard")

def initialize_data_structure(data_dir="data/healthcare"):
    """Initialize data structure for dashboard.
    
    Args:
        data_dir: Directory to initialize
    """
    data_path = Path(data_dir)
    
    # Create directories
    logger.info(f"Creating directory structure in {data_path}")
    for subdir in ["training", "evaluation", "contradiction_dataset"]:
        (data_path / subdir).mkdir(exist_ok=True, parents=True)
    
    # Create learning history file if it doesn't exist
    history_path = data_path / "learning_history.json"
    if not history_path.exists():
        logger.info(f"Creating learning history file at {history_path}")
        initial_history = {
            "events": [],
            "metrics": {
                "starting_accuracy": 0.75,
                "target_accuracy": 0.95,
                "learning_rate": 0.05
            }
        }
        with open(history_path, 'w') as f:
            json.dump(initial_history, f, indent=2)
    
    # Create initial training data if it doesn't exist
    training_path = data_path / "training" / "healthcare_training.json"
    if not training_path.exists():
        logger.info(f"Creating initial training data at {training_path}")
        training_data = [
            {
                "id": "train_001",
                "text": "Aspirin may be used to treat fever and pain.",
                "category": "medication",
                "domain": "general",
                "contradiction": False,
                "explanation": "Aspirin is indicated for fever and pain relief."
            },
            {
                "id": "train_002",
                "text": "Antibiotics are effective against viral infections.",
                "category": "treatment",
                "domain": "infectious_disease",
                "contradiction": True,
                "explanation": "Antibiotics are only effective against bacterial infections, not viral infections."
            },
            {
                "id": "train_003",
                "text": "Type 2 diabetes is characterized by insulin resistance.",
                "category": "diagnosis",
                "domain": "endocrinology",
                "contradiction": False,
                "explanation": "Type 2 diabetes typically involves insulin resistance in body tissues."
            },
            {
                "id": "train_004",
                "text": "Hypertension has no symptoms in most cases.",
                "category": "symptoms",
                "domain": "cardiology",
                "contradiction": False,
                "explanation": "Hypertension is often called the 'silent killer' because it typically has no symptoms."
            },
            {
                "id": "train_005",
                "text": "Vaccines cause autism in children.",
                "category": "prevention",
                "domain": "pediatrics",
                "contradiction": True,
                "explanation": "Scientific evidence does not support any link between vaccines and autism."
            }
        ]
        with open(training_path, 'w') as f:
            json.dump(training_data, f, indent=2)
    
    # Create initial contradiction dataset if it doesn't exist
    contradiction_path = data_path / "contradiction_dataset" / "medical_contradictions.json"
    if not contradiction_path.exists():
        logger.info(f"Creating initial contradiction dataset at {contradiction_path}")
        contradiction_data = [
            {
                "id": "contra_001",
                "text": "Statins lower cholesterol by blocking liver enzymes.",
                "category": "medication",
                "domain": "cardiology",
                "contradiction": False,
                "explanation": "Statins inhibit HMG-CoA reductase, an enzyme in the liver that produces cholesterol."
            },
            {
                "id": "contra_002",
                "text": "Drinking alcohol while taking acetaminophen is safe.",
                "category": "medication",
                "domain": "toxicology",
                "contradiction": True,
                "explanation": "Combining alcohol and acetaminophen increases risk of liver damage."
            },
            {
                "id": "contra_003",
                "text": "Metformin is a first-line treatment for type 2 diabetes.",
                "category": "treatment",
                "domain": "endocrinology",
                "contradiction": False,
                "explanation": "Metformin is recommended as first-line therapy for type 2 diabetes in most guidelines."
            },
            {
                "id": "contra_004",
                "text": "Vaccines provide better immunity than natural infection.",
                "category": "prevention",
                "domain": "immunology",
                "contradiction": False,
                "explanation": "Vaccines can provide more predictable and safer immunity than natural infection."
            },
            {
                "id": "contra_005",
                "text": "COVID-19 only affects the respiratory system.",
                "category": "pathophysiology",
                "domain": "infectious_disease",
                "contradiction": True,
                "explanation": "COVID-19 can affect multiple organ systems, not just the respiratory system."
            },
            {
                "id": "contra_006",
                "text": "Hypertension increases risk of heart disease.",
                "category": "risk_factor",
                "domain": "cardiology",
                "contradiction": False,
                "explanation": "Hypertension is a well-established risk factor for heart disease."
            },
            {
                "id": "contra_007",
                "text": "All breast lumps are cancerous.",
                "category": "diagnosis",
                "domain": "oncology",
                "contradiction": True,
                "explanation": "Many breast lumps are benign (non-cancerous)."
            },
            {
                "id": "contra_008",
                "text": "Type 1 diabetes can be prevented with lifestyle changes.",
                "category": "prevention",
                "domain": "endocrinology",
                "contradiction": True,
                "explanation": "Type 1 diabetes is an autoimmune condition that cannot be prevented with lifestyle changes."
            },
            {
                "id": "contra_009",
                "text": "Regular physical activity reduces risk of chronic diseases.",
                "category": "prevention",
                "domain": "general",
                "contradiction": False,
                "explanation": "Regular physical activity is proven to reduce risk of multiple chronic diseases."
            },
            {
                "id": "contra_010",
                "text": "Antibiotics should be taken until symptoms resolve.",
                "category": "medication",
                "domain": "infectious_disease",
                "contradiction": True,
                "explanation": "Antibiotics should be taken for the full prescribed course, even if symptoms resolve earlier."
            }
        ]
        with open(contradiction_path, 'w') as f:
            json.dump(contradiction_data, f, indent=2)
    
    # Create initial evaluation file
    eval_path = data_path / "evaluation" / f"synthetic_eval_initial.json"
    if not list(data_path / "evaluation").filter(lambda x: x.name.startswith("synthetic_eval_")):
        logger.info(f"Creating initial evaluation file at {eval_path}")
        from scripts.generate_synthetic_evaluation import generate_synthetic_evaluation
        try:
            generate_synthetic_evaluation(str(eval_path), accuracy=0.75)
        except Exception as e:
            logger.error(f"Error generating evaluation data: {str(e)}")
            # Create a simple evaluation file if generation fails
            eval_data = {
                "accuracy": 0.75,
                "timestamp": datetime.now().isoformat(),
                "examples": 20,
                "results": [
                    {"id": "eval_001", "correct": True},
                    {"id": "eval_002", "correct": True},
                    {"id": "eval_003", "correct": False},
                    {"id": "eval_004", "correct": True},
                    {"id": "eval_005", "correct": True}
                ]
            }
            with open(eval_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
    
    logger.info("Data initialization complete!")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize data structure for healthcare learning dashboard")
    parser.add_argument("--data-dir", 
                       type=str, 
                       default="data/healthcare",
                       help="Path to healthcare data directory")
    parser.add_argument("--force", 
                       action="store_true",
                       help="Force reinitialization even if files exist")
    args = parser.parse_args()
    
    # If force is specified, delete existing files
    if args.force:
        data_path = Path(args.data_dir)
        if data_path.exists():
            import shutil
            logger.warning(f"Force option specified. Removing existing data in {data_path}")
            for item in data_path.glob("**/*"):
                if item.is_file():
                    item.unlink()
    
    # Initialize data structure
    success = initialize_data_structure(data_dir=args.data_dir)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
