#!/usr/bin/env python3
"""
Dataset importer for healthcare learning dashboard.
Provides functionality for adding new contradiction examples.
"""

import sys
import os
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

# Configure logging
logger = logging.getLogger("dataset-importer")
console = Console()

class DatasetImporter:
    """Dataset importer for adding contradiction examples."""
    
    def __init__(self, data_dir="data/healthcare"):
        """Initialize dataset importer.
        
        Args:
            data_dir: Directory containing healthcare data
        """
        self.data_dir = Path(data_dir)
        self.training_path = self.data_dir / "training" / "healthcare_training.json"
        self.contradiction_path = self.data_dir / "contradiction_dataset" / "medical_contradictions.json"
        self.history_path = self.data_dir / "learning_history.json"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        (self.data_dir / "training").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "contradiction_dataset").mkdir(exist_ok=True, parents=True)
        
        # Initialize datasets if they don't exist
        self._initialize_datasets()
        
        # Load existing data
        self.training_data = self._load_training_data()
        self.contradiction_data = self._load_contradiction_data()
        
    def _initialize_datasets(self):
        """Initialize datasets if they don't exist."""
        # Training data
        if not self.training_path.exists():
            with open(self.training_path, 'w') as f:
                json.dump([], f, indent=2)
                
        # Contradiction data
        if not self.contradiction_path.exists():
            with open(self.contradiction_path, 'w') as f:
                json.dump([], f, indent=2)
                
    def _load_training_data(self):
        """Load training data from disk."""
        try:
            if self.training_path.exists():
                with open(self.training_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return []
            
    def _load_contradiction_data(self):
        """Load contradiction data from disk."""
        try:
            if self.contradiction_path.exists():
                with open(self.contradiction_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading contradiction data: {str(e)}")
            return []
            
    def _save_training_data(self):
        """Save training data to disk."""
        try:
            with open(self.training_path, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Saved training data to {self.training_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving training data: {str(e)}")
            return False
            
    def _save_contradiction_data(self):
        """Save contradiction data to disk."""
        try:
            with open(self.contradiction_path, 'w') as f:
                json.dump(self.contradiction_data, f, indent=2)
            logger.info(f"Saved contradiction data to {self.contradiction_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving contradiction data: {str(e)}")
            return False
            
    def _update_learning_history(self, examples_added):
        """Update learning history with dataset import event.
        
        Args:
            examples_added: Number of examples added
        """
        try:
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
            else:
                history = {"events": [], "metrics": {}}
                
            # Add dataset import event
            event = {
                "type": "dataset_import",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "examples_added": examples_added
                }
            }
            
            history["events"].append(event)
            
            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"Updated learning history with dataset import event")
        except Exception as e:
            logger.error(f"Error updating learning history: {str(e)}")
            
    def get_available_categories(self):
        """Get list of unique categories in the contradiction dataset.
        
        Returns:
            list: Unique categories
        """
        categories = set()
        for item in self.contradiction_data:
            category = item.get("category")
            if category:
                categories.add(category)
        return sorted(list(categories))
        
    def get_available_domains(self):
        """Get list of unique domains in the contradiction dataset.
        
        Returns:
            list: Unique domains
        """
        domains = set()
        for item in self.contradiction_data:
            domain = item.get("domain")
            if domain:
                domains.add(domain)
        return sorted(list(domains))
        
    def add_example(self, text, category, domain, is_contradiction, explanation):
        """Add a new contradiction example.
        
        Args:
            text: Example text
            category: Category (e.g., medication, treatment)
            domain: Domain (e.g., cardiology, endocrinology)
            is_contradiction: Whether the example is a contradiction
            explanation: Explanation of the contradiction or fact
            
        Returns:
            dict: Newly added example
        """
        # Generate ID
        example_id = f"user_{str(uuid.uuid4())[:8]}"
        
        # Create example
        example = {
            "id": example_id,
            "text": text,
            "category": category,
            "domain": domain,
            "contradiction": bool(is_contradiction),
            "explanation": explanation,
            "source": "user_import",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to both datasets
        self.contradiction_data.append(example)
        self.training_data.append(example)
        
        # Save datasets
        self._save_contradiction_data()
        self._save_training_data()
        
        # Update learning history
        self._update_learning_history(1)
        
        logger.info(f"Added new example: {example_id}")
        return example
        
    def add_examples_batch(self, examples):
        """Add multiple examples at once.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            int: Number of examples added
        """
        if not examples:
            return 0
            
        # Process each example
        examples_added = 0
        for example_data in examples:
            try:
                # Generate ID
                example_id = f"user_{str(uuid.uuid4())[:8]}"
                
                # Create example
                example = {
                    "id": example_id,
                    "text": example_data.get("text", ""),
                    "category": example_data.get("category", "general"),
                    "domain": example_data.get("domain", "general"),
                    "contradiction": bool(example_data.get("contradiction", False)),
                    "explanation": example_data.get("explanation", ""),
                    "source": "batch_import",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Validate required fields
                if not example["text"] or not example["explanation"]:
                    logger.warning(f"Skipping example with missing required fields")
                    continue
                    
                # Add to both datasets
                self.contradiction_data.append(example)
                self.training_data.append(example)
                examples_added += 1
                
            except Exception as e:
                logger.error(f"Error adding example: {str(e)}")
                
        # Save datasets if examples were added
        if examples_added > 0:
            self._save_contradiction_data()
            self._save_training_data()
            
            # Update learning history
            self._update_learning_history(examples_added)
            
        logger.info(f"Added {examples_added} examples in batch")
        return examples_added
        
    def import_json_file(self, file_path):
        """Import examples from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            int: Number of examples imported
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return 0
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Handle different formats
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict) and "examples" in data:
                examples = data.get("examples", [])
            else:
                logger.error(f"Invalid data format in {file_path}")
                return 0
                
            return self.add_examples_batch(examples)
            
        except Exception as e:
            logger.error(f"Error importing examples from {file_path}: {str(e)}")
            return 0
            
    def print_examples_summary(self):
        """Print summary of contradiction examples."""
        # Get category and domain counts
        categories = {}
        domains = {}
        contradiction_count = 0
        
        for item in self.contradiction_data:
            category = item.get("category", "unknown")
            domain = item.get("domain", "unknown")
            is_contradiction = item.get("contradiction", False)
            
            categories[category] = categories.get(category, 0) + 1
            domains[domain] = domains.get(domain, 0) + 1
            
            if is_contradiction:
                contradiction_count += 1
                
        # Create summary tables
        console.print(f"\n[bold]Dataset Summary[/bold]")
        console.print(f"Total examples: {len(self.contradiction_data)}")
        console.print(f"Contradictions: {contradiction_count}")
        console.print(f"Non-contradictions: {len(self.contradiction_data) - contradiction_count}")
        
        # Categories table
        categories_table = Table(title="Categories")
        categories_table.add_column("Category", style="cyan")
        categories_table.add_column("Count", style="green")
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            categories_table.add_row(category, str(count))
            
        console.print(categories_table)
        
        # Domains table
        domains_table = Table(title="Domains")
        domains_table.add_column("Domain", style="cyan")
        domains_table.add_column("Count", style="green")
        
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            domains_table.add_row(domain, str(count))
            
        console.print(domains_table)
        
if __name__ == "__main__":
    # Simple test if run directly
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Dataset importer for healthcare learning")
    parser.add_argument("--data-dir", type=str, default="data/healthcare", help="Path to healthcare data directory")
    parser.add_argument("--import-file", type=str, help="Import from JSON file")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    
    args = parser.parse_args()
    
    importer = DatasetImporter(data_dir=args.data_dir)
    
    if args.import_file:
        count = importer.import_json_file(args.import_file)
        print(f"Imported {count} examples from {args.import_file}")
        
    if args.summary or not args.import_file:
        importer.print_examples_summary()
