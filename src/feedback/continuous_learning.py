"""
Continuous Learning Module for C. Pete Connor Model

This module converts collected feedback into training examples for model improvement.
It implements weighted sampling based on feedback severity, dataset preparation,
and integration with the model training system.
"""

import os
import sys
import json
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

# Ensure the parent directory is in the path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import feedback modules
from src.feedback.feedback_store import FeedbackStore
from src.feedback.feedback_capture import FeedbackCapture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContinuousLearningSystem:
    """System for continuous learning based on user feedback."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the continuous learning system.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the default path.
        """
        self.feedback_store = FeedbackStore(db_path)
        self.feedback_capture = FeedbackCapture(db_path)
        
        # Default paths
        project_dir = Path(__file__).resolve().parents[2]
        self.data_dir = project_dir / "data"
        self.dataset_dir = project_dir / "dataset"
        self.feedback_dataset_path = self.data_dir / "feedback_training_data.json"
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        logger.info("Continuous learning system initialized")
    
    def prepare_training_dataset(
        self,
        output_path: Optional[str] = None,
        positive_weight: float = 1.0,
        negative_weight: float = 2.0,
        include_annotations: bool = True,
        max_examples: int = 500,
        min_rating_threshold: int = 2,
        tag_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Prepare a training dataset from collected feedback.
        
        Args:
            output_path: Path to save the prepared dataset. If None, uses default path.
            positive_weight: Weight factor for positive examples (rating >= 4)
            negative_weight: Weight factor for negative examples (rating < 4)
            include_annotations: Whether to include inline annotations in the dataset
            max_examples: Maximum number of examples to include
            min_rating_threshold: Minimum rating threshold for inclusion (examples below this are weighted higher)
            tag_weights: Dictionary mapping tags to weight multipliers
            
        Returns:
            str: Path to the prepared dataset
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.dataset_dir, f"feedback_training_{timestamp}.json")
        
        # Set default tag weights if not provided
        if tag_weights is None:
            tag_weights = {
                "hallucination": 3.0,
                "voice_mismatch": 2.5,
                "content_quality": 2.0,
                "audience_mismatch": 1.8,
                "platform_mismatch": 1.8,
                "coherence": 1.5
            }
        
        # Get all feedback with detailed information
        all_feedback = self.feedback_store.query_feedback(
            limit=1000,  # Get a large sample to select from
            include_tags=True,
            include_annotations=include_annotations
        )
        
        if not all_feedback:
            logger.warning("No feedback found in the database")
            return ""
        
        # Apply weights based on rating, tags, and annotations
        weighted_feedback = []
        for entry in all_feedback:
            # Base weight by feedback type (positive or negative)
            if entry.get('is_positive', False):
                base_weight = positive_weight
            else:
                base_weight = negative_weight
            
            # Increase weight for very low ratings
            rating = entry.get('rating', 3)
            if rating < min_rating_threshold:
                base_weight *= 1.5
            
            # Add weights for tags
            tag_weight_multiplier = 1.0
            for tag in entry.get('tags', []):
                if tag in tag_weights:
                    tag_weight_multiplier *= tag_weights[tag]
            
            # Final weight calculation
            final_weight = base_weight * tag_weight_multiplier
            
            # Add to weighted list
            weighted_feedback.append((entry, final_weight))
        
        # Normalize weights to probabilities
        total_weight = sum(weight for _, weight in weighted_feedback)
        probabilities = [weight / total_weight for _, weight in weighted_feedback]
        
        # Sample examples based on weights
        num_samples = min(max_examples, len(weighted_feedback))
        selected_indices = np.random.choice(
            len(weighted_feedback),
            size=num_samples,
            replace=False,
            p=probabilities
        )
        
        # Create training dataset
        training_data = []
        for idx in selected_indices:
            entry, _ = weighted_feedback[idx]
            
            # Create the training example
            training_example = {
                "original_prompt": entry.get("original_prompt", ""),
                "generated_content": entry.get("content_text", ""),
                "rating": entry.get("rating", 0),
                "platform": entry.get("platform", ""),
                "audience": entry.get("audience", ""),
                "domain": entry.get("domain", ""),
                "feedback": entry.get("comment", ""),
                "tags": entry.get("tags", [])
            }
            
            # Include annotations if requested
            if include_annotations and 'annotations' in entry and entry['annotations']:
                training_example["annotations"] = entry['annotations']
            
            training_data.append(training_example)
        
        # Save the dataset
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Prepared training dataset with {len(training_data)} examples saved to {output_path}")
        return output_path
    
    def convert_feedback_to_training_format(
        self,
        output_path: Optional[str] = None,
        separate_files: bool = False
    ) -> Tuple[str, Dict[str, int]]:
        """
        Convert feedback into training data format with positive and negative examples.
        
        Args:
            output_path: Path to save the converted dataset. If None, uses default path.
            separate_files: Whether to create separate files for positive and negative examples
            
        Returns:
            Tuple of (path to dataset, statistics dictionary)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.dataset_dir, f"feedback_training_{timestamp}.json")
        
        # Get positive examples (good content to emulate)
        positive_examples = self.feedback_store.get_positive_examples(limit=300)
        
        # Get negative examples (content to avoid)
        negative_examples = self.feedback_store.get_negative_examples(limit=300)
        
        # Create the combined training data
        training_data = []
        
        # Process positive examples
        for entry in positive_examples:
            example = {
                "text": entry.get("content_text", ""),
                "platform": entry.get("platform", ""),
                "audience": entry.get("audience", ""),
                "rating": entry.get("rating", 0),
                "is_positive": True
            }
            training_data.append(example)
        
        # Process negative examples
        for entry in negative_examples:
            example = {
                "text": entry.get("content_text", ""),
                "platform": entry.get("platform", ""),
                "audience": entry.get("audience", ""),
                "rating": entry.get("rating", 0),
                "is_positive": False,
                "tags": entry.get("tags", []),
                "comment": entry.get("comment", "")
            }
            
            # Add annotations if available
            if "annotations" in entry and entry["annotations"]:
                example["annotations"] = entry["annotations"]
            
            training_data.append(example)
        
        # Save the combined dataset
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # If requested, create separate files for positive and negative examples
        if separate_files:
            positive_path = output_path.replace('.json', '_positive.json')
            negative_path = output_path.replace('.json', '_negative.json')
            
            with open(positive_path, 'w') as f:
                json.dump([ex for ex in training_data if ex.get("is_positive", False)], f, indent=2)
            
            with open(negative_path, 'w') as f:
                json.dump([ex for ex in training_data if not ex.get("is_positive", False)], f, indent=2)
        
        # Generate statistics
        stats = {
            "total_examples": len(training_data),
            "positive_examples": len(positive_examples),
            "negative_examples": len(negative_examples),
            "output_path": output_path
        }
        
        logger.info(f"Converted {stats['total_examples']} feedback entries to training format")
        return output_path, stats
    
    def create_fine_tuning_dataset(
        self,
        output_path: Optional[str] = None,
        incorporate_annotations: bool = True,
        min_examples: int = 20
    ) -> Tuple[str, Dict[str, int]]:
        """
        Create a fine-tuning dataset that incorporates feedback for model improvement.
        
        Args:
            output_path: Path to save the fine-tuning dataset
            incorporate_annotations: Whether to incorporate inline annotations into examples
            min_examples: Minimum number of examples required to create the dataset
            
        Returns:
            Tuple of (path to dataset, statistics dictionary)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.dataset_dir, f"finetune_feedback_{timestamp}.json")
        
        # Get positive and negative examples
        positive_examples = self.feedback_store.get_positive_examples(limit=200)
        negative_examples = self.feedback_store.get_negative_examples(limit=200)
        
        # Check if we have enough examples
        total_examples = len(positive_examples) + len(negative_examples)
        if total_examples < min_examples:
            logger.warning(f"Not enough feedback examples for fine-tuning (found {total_examples}, need {min_examples})")
            return "", {"error": "Insufficient examples", "found": total_examples, "required": min_examples}
        
        # Create fine-tuning examples
        fine_tuning_examples = []
        
        # Process positive examples - these we want the model to emulate
        for entry in positive_examples:
            # Skip entries missing necessary data
            if not entry.get("content_text"):
                continue
            
            example = {
                "input": entry.get("original_prompt", "Generate content"),
                "output": entry.get("content_text", "")
            }
            
            # Add context about platform and audience if available
            context_parts = []
            if entry.get("platform"):
                context_parts.append(f"Platform: {entry.get('platform')}")
            if entry.get("audience"):
                context_parts.append(f"Audience: {entry.get('audience')}")
            
            if context_parts:
                example["input"] = f"{' | '.join(context_parts)}\n{example['input']}"
            
            fine_tuning_examples.append(example)
        
        # Process negative examples with annotations - create corrected versions
        for entry in negative_examples:
            # Skip entries missing necessary data
            if not entry.get("content_text") or not entry.get("original_prompt"):
                continue
            
            # If we have annotations and should incorporate them
            if incorporate_annotations and "annotations" in entry and entry["annotations"]:
                # Create a corrected version based on annotations
                content = entry.get("content_text", "")
                
                # Sort annotations by start_index if available, to process from end to beginning
                annotations = entry.get("annotations", [])
                sorted_annotations = sorted(
                    [a for a in annotations if a.get("start_index") is not None and a.get("end_index") is not None],
                    key=lambda x: x.get("start_index", 0),
                    reverse=True  # Process from end to beginning to avoid index shifts
                )
                
                # Apply simple corrections based on annotations
                for annotation in sorted_annotations:
                    start = annotation.get("start_index")
                    end = annotation.get("end_index")
                    
                    if start is not None and end is not None:
                        # If there's a suggested correction in the comment, apply it
                        comment = annotation.get("comment", "")
                        if comment.startswith("Suggestion:"):
                            suggestion = comment.split("Suggestion:", 1)[1].strip()
                            content = content[:start] + suggestion + content[end:]
                
                # Create example with corrected content
                example = {
                    "input": entry.get("original_prompt", "Generate content"),
                    "output": content
                }
                
                # Add context about platform and audience if available
                context_parts = []
                if entry.get("platform"):
                    context_parts.append(f"Platform: {entry.get('platform')}")
                if entry.get("audience"):
                    context_parts.append(f"Audience: {entry.get('audience')}")
                
                if context_parts:
                    example["input"] = f"{' | '.join(context_parts)}\n{example['input']}"
                
                fine_tuning_examples.append(example)
        
        # Save the dataset
        with open(output_path, 'w') as f:
            json.dump(fine_tuning_examples, f, indent=2)
        
        # Generate statistics
        stats = {
            "total_examples": len(fine_tuning_examples),
            "from_positive": len(positive_examples),
            "from_negative": len(negative_examples),
            "output_path": output_path
        }
        
        logger.info(f"Created fine-tuning dataset with {stats['total_examples']} examples saved to {output_path}")
        return output_path, stats
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the continuous learning system.
        
        Returns:
            Dict containing statistics about feedback and learning
        """
        try:
            # Get basic feedback stats
            feedback_stats = self.feedback_store.get_feedback_stats()
            
            # Add additional learning-specific stats
            learning_stats = {
                "feedback_stats": feedback_stats,
                "total_feedback_entries": feedback_stats.get("total_count", 0),
                "positive_examples_available": feedback_stats.get("positive_count", 0),
                "negative_examples_available": feedback_stats.get("negative_count", 0),
                "ready_for_training": feedback_stats.get("total_count", 0) >= 20,
                "most_common_issues": []
            }
            
            # Get most common issues
            tag_distribution = feedback_stats.get("tag_distribution", {})
            sorted_tags = sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)
            learning_stats["most_common_issues"] = [{"tag": tag, "count": count} for tag, count in sorted_tags[:5]]
            
            return learning_stats
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Create continuous learning system
    cls = ContinuousLearningSystem()
    
    # Prepare a training dataset
    dataset_path = cls.prepare_training_dataset()
    
    # Get learning statistics
    stats = cls.get_learning_statistics()
    print(json.dumps(stats, indent=2))
