"""
Feedback Capture Interface for C. Pete Connor Model

This module provides functions to capture user feedback on generated content,
including ratings, issue tagging, and inline annotations. The feedback is stored
in the feedback database for continuous learning.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

# Ensure the parent directory is in the path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import feedback store
from src.feedback.feedback_store import FeedbackStore, FEEDBACK_TAGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FeedbackCapture:
    """Interface for capturing user feedback on generated content."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the feedback capture system.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the default path.
        """
        self.feedback_store = FeedbackStore(db_path)
        logger.info("Feedback capture interface initialized")
    
    def get_available_tags(self) -> List[str]:
        """Get the list of available feedback tags."""
        return FEEDBACK_TAGS
    
    def capture_basic_feedback(
        self,
        content_text: str,
        rating: int,
        original_prompt: Optional[str] = None,
        platform: Optional[str] = None,
        audience: Optional[str] = None,
        domain: Optional[str] = None,
        comment: Optional[str] = None
    ) -> int:
        """
        Capture basic feedback without tags or annotations.
        
        Args:
            content_text: The generated content that received feedback
            rating: Rating on a scale of 1-5
            original_prompt: The original prompt used to generate the content
            platform: Target platform (twitter, linkedin, etc.)
            audience: Target audience (executive, practitioner, general)
            domain: Content domain or topic area
            comment: Overall comment on the generated content
            
        Returns:
            int: ID of the stored feedback entry
        """
        try:
            # Validate rating
            if not 1 <= rating <= 5:
                logger.error(f"Invalid rating: {rating}. Must be between 1 and 5.")
                return -1
            
            # Store the feedback
            feedback_id = self.feedback_store.store_feedback(
                content_text=content_text,
                rating=rating,
                original_prompt=original_prompt,
                platform=platform,
                audience=audience,
                domain=domain,
                comment=comment
            )
            
            logger.info(f"Captured basic feedback with ID {feedback_id}, rating {rating}")
            return feedback_id
        
        except Exception as e:
            logger.error(f"Error capturing basic feedback: {e}")
            return -1
    
    def capture_detailed_feedback(
        self,
        content_text: str,
        rating: int,
        original_prompt: Optional[str] = None,
        platform: Optional[str] = None,
        audience: Optional[str] = None,
        domain: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Capture detailed feedback including tags and annotations.
        
        Args:
            content_text: The generated content that received feedback
            rating: Rating on a scale of 1-5
            original_prompt: The original prompt used to generate the content
            platform: Target platform (twitter, linkedin, etc.)
            audience: Target audience (executive, practitioner, general)
            domain: Content domain or topic area
            comment: Overall comment on the generated content
            tags: List of issue tags (must be from FEEDBACK_TAGS)
            annotations: List of inline annotations (dicts with text_segment, comment, start_index, end_index)
            metadata: Additional metadata as a dictionary
            
        Returns:
            int: ID of the stored feedback entry
        """
        try:
            # Validate rating
            if not 1 <= rating <= 5:
                logger.error(f"Invalid rating: {rating}. Must be between 1 and 5.")
                return -1
            
            # Validate tags
            if tags:
                for tag in tags:
                    if tag not in FEEDBACK_TAGS:
                        logger.warning(f"Unknown tag: {tag}. Tag will be stored but may not be categorized properly.")
            
            # Store the feedback
            feedback_id = self.feedback_store.store_feedback(
                content_text=content_text,
                rating=rating,
                original_prompt=original_prompt,
                platform=platform,
                audience=audience,
                domain=domain,
                comment=comment,
                tags=tags,
                annotations=annotations,
                metadata=metadata
            )
            
            logger.info(f"Captured detailed feedback with ID {feedback_id}, rating {rating}, tags {tags}")
            return feedback_id
        
        except Exception as e:
            logger.error(f"Error capturing detailed feedback: {e}")
            return -1
    
    def add_tags_to_feedback(self, feedback_id: int, tags: List[str]) -> bool:
        """
        Add tags to an existing feedback entry.
        
        Args:
            feedback_id: ID of the feedback entry
            tags: List of tags to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get existing feedback
            feedback = self.feedback_store.get_feedback(feedback_id)
            
            # Get current tags
            current_tags = feedback.get('tags', [])
            
            # Add new tags (avoiding duplicates)
            all_tags = list(set(current_tags + tags))
            
            # Store updated feedback
            # Note: Since we're not directly exposing a method to update just tags,
            # we'll need to re-store the entire feedback with updated tags
            self.feedback_store.delete_feedback(feedback_id)
            
            new_id = self.feedback_store.store_feedback(
                content_text=feedback['content_text'],
                rating=feedback['rating'],
                original_prompt=feedback.get('original_prompt'),
                platform=feedback.get('platform'),
                audience=feedback.get('audience'),
                domain=feedback.get('domain'),
                comment=feedback.get('comment'),
                tags=all_tags,
                annotations=feedback.get('annotations'),
                metadata=feedback.get('metadata')
            )
            
            logger.info(f"Added tags {tags} to feedback ID {feedback_id}, new ID is {new_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding tags to feedback {feedback_id}: {e}")
            return False
    
    def add_annotation_to_feedback(
        self,
        feedback_id: int,
        text_segment: str,
        comment: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ) -> bool:
        """
        Add an inline annotation to an existing feedback entry.
        
        Args:
            feedback_id: ID of the feedback entry
            text_segment: The text segment being annotated
            comment: The annotation comment
            start_index: Start index of the segment in the original text
            end_index: End index of the segment in the original text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get existing feedback
            feedback = self.feedback_store.get_feedback(feedback_id)
            
            # Get current annotations
            current_annotations = feedback.get('annotations', [])
            
            # Add new annotation
            new_annotation = {
                'text_segment': text_segment,
                'comment': comment,
                'start_index': start_index,
                'end_index': end_index
            }
            
            all_annotations = current_annotations + [new_annotation]
            
            # Store updated feedback
            self.feedback_store.delete_feedback(feedback_id)
            
            new_id = self.feedback_store.store_feedback(
                content_text=feedback['content_text'],
                rating=feedback['rating'],
                original_prompt=feedback.get('original_prompt'),
                platform=feedback.get('platform'),
                audience=feedback.get('audience'),
                domain=feedback.get('domain'),
                comment=feedback.get('comment'),
                tags=feedback.get('tags'),
                annotations=all_annotations,
                metadata=feedback.get('metadata')
            )
            
            logger.info(f"Added annotation to feedback ID {feedback_id}, new ID is {new_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding annotation to feedback {feedback_id}: {e}")
            return False
    
    def get_feedback_summary(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get a summary of recent feedback.
        
        Args:
            limit: Maximum number of recent feedback entries to include
            
        Returns:
            Dict containing feedback statistics and recent entries
        """
        try:
            # Get statistics
            stats = self.feedback_store.get_feedback_stats()
            
            # Get recent feedback
            recent_feedback = self.feedback_store.query_feedback(
                limit=limit,
                include_tags=True,
                include_annotations=False
            )
            
            # Create summary
            summary = {
                'statistics': stats,
                'recent_feedback': recent_feedback
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            return {'error': str(e)}
    
    def export_training_data(
        self,
        output_path: str,
        positive_only: bool = False,
        negative_only: bool = False,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        limit: int = 1000
    ) -> int:
        """
        Export feedback as training data for continuous learning.
        
        Args:
            output_path: Path to save the exported dataset
            positive_only: Whether to include only positive examples (rating >= 4)
            negative_only: Whether to include only negative examples (rating < 4)
            min_rating: Minimum rating to include
            max_rating: Maximum rating to include
            limit: Maximum number of examples to export
            
        Returns:
            int: Number of examples exported
        """
        try:
            # Check for conflicting parameters
            if positive_only and negative_only:
                logger.error("Cannot specify both positive_only and negative_only")
                return 0
            
            # Set is_positive based on parameters
            is_positive = None
            if positive_only:
                is_positive = True
            elif negative_only:
                is_positive = False
            
            # Export the data
            num_exported = self.feedback_store.export_feedback_dataset(
                output_path=output_path,
                min_rating=min_rating,
                max_rating=max_rating,
                is_positive=is_positive,
                limit=limit
            )
            
            return num_exported
        
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return 0


# Example usage
if __name__ == "__main__":
    # Create feedback capture system
    feedback_capture = FeedbackCapture()
    
    # Capture some sample feedback
    feedback_id = feedback_capture.capture_detailed_feedback(
        content_text="AI technology is transforming industries at an unprecedented rate.",
        rating=3,
        original_prompt="Write about AI impacts",
        platform="blog",
        audience="general",
        domain="technology",
        comment="Somewhat generic, needs more specific examples",
        tags=["content_quality", "verbosity"],
        annotations=[
            {
                "text_segment": "at an unprecedented rate",
                "comment": "Cliché phrase, be more specific",
                "start_index": 47,
                "end_index": 71
            }
        ]
    )
    
    # Add an additional tag
    feedback_capture.add_tags_to_feedback(feedback_id, ["voice_mismatch"])
    
    # Add an additional annotation
    feedback_capture.add_annotation_to_feedback(
        feedback_id=feedback_id,
        text_segment="AI technology",
        comment="Too vague - specify which AI technology",
        start_index=0,
        end_index=12
    )
    
    # Get summary
    summary = feedback_capture.get_feedback_summary()
    print(json.dumps(summary, indent=2))
    
    # Export for training
    num_exported = feedback_capture.export_training_data(
        output_path="data/feedback_training_data.json", 
        limit=100
    )
    print(f"Exported {num_exported} feedback entries for training")
