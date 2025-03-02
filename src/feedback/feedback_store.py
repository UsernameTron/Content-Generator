"""
Feedback Storage System for C. Pete Connor Model

This module implements a SQLite database for storing user feedback on generated content,
including ratings, comments, and improvement suggestions. The stored feedback will be
used for continuous learning and model improvement.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define feedback issue tags
FEEDBACK_TAGS = [
    "hallucination",         # Factually incorrect information
    "voice_mismatch",        # Not matching C. Pete Connor's writing style
    "content_quality",       # Poor quality content
    "formatting_issue",      # Problems with formatting
    "irrelevant",            # Content not relevant to prompt
    "audience_mismatch",     # Not appropriate for target audience
    "platform_mismatch",     # Not suitable for target platform
    "tone_issue",            # Incorrect tone or sentiment
    "jargon_level",          # Inappropriate level of technical jargon
    "verbosity",             # Too verbose or too concise
    "creativity",            # Lack of creativity or overly formulaic
    "coherence",             # Logical flow and coherence issues
    "other"                  # Other issues not covered above
]

class FeedbackStore:
    """SQLite-based storage for user feedback on generated content."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the feedback storage system.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the default path.
        """
        if db_path is None:
            # Default path in project directory
            project_dir = Path(__file__).resolve().parents[2]
            db_path = project_dir / "data" / "feedback.db"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = str(db_path)
        self._create_tables_if_not_exist()
        logger.info(f"Feedback store initialized with database at {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)
    
    def _create_tables_if_not_exist(self):
        """Create the necessary tables if they don't already exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            rating INTEGER NOT NULL,
            content_text TEXT NOT NULL,
            original_prompt TEXT,
            platform TEXT,
            audience TEXT,
            domain TEXT,
            is_positive BOOLEAN NOT NULL,
            comment TEXT,
            metadata TEXT
        )
        ''')
        
        # Create tags table for issue tagging
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
        )
        ''')
        
        # Create annotations table for inline comments
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL,
            text_segment TEXT NOT NULL,
            comment TEXT NOT NULL,
            start_index INTEGER,
            end_index INTEGER,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database tables created or verified")
    
    def store_feedback(
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
        Store feedback for generated content.
        
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
        # Validate rating
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        # Validate tags
        if tags:
            for tag in tags:
                if tag not in FEEDBACK_TAGS:
                    logger.warning(f"Unknown tag: {tag}. Will be stored but not categorized properly.")
        
        # Determine if positive example (rating >= 4)
        is_positive = (rating >= 4)
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert main feedback
        cursor.execute('''
        INSERT INTO feedback 
        (timestamp, rating, content_text, original_prompt, platform, audience, domain, is_positive, comment, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            rating,
            content_text,
            original_prompt,
            platform,
            audience,
            domain,
            is_positive,
            comment,
            metadata_json
        ))
        
        # Get the feedback ID
        feedback_id = cursor.lastrowid
        
        # Insert tags
        if tags:
            tag_values = [(feedback_id, tag) for tag in tags]
            cursor.executemany('''
            INSERT INTO feedback_tags (feedback_id, tag)
            VALUES (?, ?)
            ''', tag_values)
        
        # Insert annotations
        if annotations:
            annotation_values = []
            for ann in annotations:
                annotation_values.append((
                    feedback_id,
                    ann.get('text_segment', ''),
                    ann.get('comment', ''),
                    ann.get('start_index'),
                    ann.get('end_index')
                ))
            
            cursor.executemany('''
            INSERT INTO annotations (feedback_id, text_segment, comment, start_index, end_index)
            VALUES (?, ?, ?, ?, ?)
            ''', annotation_values)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored feedback (ID: {feedback_id}) with rating {rating}")
        return feedback_id
    
    def get_feedback(
        self,
        feedback_id: int,
        include_tags: bool = True,
        include_annotations: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve a specific feedback entry by ID.
        
        Args:
            feedback_id: ID of the feedback to retrieve
            include_tags: Whether to include associated tags
            include_annotations: Whether to include inline annotations
            
        Returns:
            Dict containing the feedback data
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get main feedback
        cursor.execute('SELECT * FROM feedback WHERE id = ?', (feedback_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise ValueError(f"No feedback found with ID {feedback_id}")
        
        # Convert to dict
        feedback = dict(row)
        
        # Parse metadata JSON
        if feedback['metadata']:
            feedback['metadata'] = json.loads(feedback['metadata'])
        
        # Get tags if requested
        if include_tags:
            cursor.execute('SELECT tag FROM feedback_tags WHERE feedback_id = ?', (feedback_id,))
            tags = [row['tag'] for row in cursor.fetchall()]
            feedback['tags'] = tags
        
        # Get annotations if requested
        if include_annotations:
            cursor.execute('SELECT * FROM annotations WHERE feedback_id = ?', (feedback_id,))
            annotations = [dict(row) for row in cursor.fetchall()]
            feedback['annotations'] = annotations
        
        conn.close()
        return feedback
    
    def query_feedback(
        self,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        is_positive: Optional[bool] = None,
        platform: Optional[str] = None,
        audience: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_tags: bool = True,
        include_annotations: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query feedback entries based on various criteria.
        
        Args:
            min_rating: Minimum rating threshold
            max_rating: Maximum rating threshold
            is_positive: Filter by positive/negative feedback
            platform: Filter by platform
            audience: Filter by audience
            domain: Filter by domain
            tags: Filter by tags (must have ALL specified tags)
            limit: Maximum number of results to return
            offset: Offset for pagination
            include_tags: Whether to include associated tags
            include_annotations: Whether to include inline annotations
            
        Returns:
            List of dicts containing the feedback entries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT * FROM feedback WHERE 1=1'
        params = []
        
        if min_rating is not None:
            query += ' AND rating >= ?'
            params.append(min_rating)
        
        if max_rating is not None:
            query += ' AND rating <= ?'
            params.append(max_rating)
        
        if is_positive is not None:
            query += ' AND is_positive = ?'
            params.append(is_positive)
        
        if platform is not None:
            query += ' AND platform = ?'
            params.append(platform)
        
        if audience is not None:
            query += ' AND audience = ?'
            params.append(audience)
        
        if domain is not None:
            query += ' AND domain = ?'
            params.append(domain)
        
        # Add order by, limit and offset
        query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        # Execute main query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        
        # Filter by tags if needed (done in Python as it requires joining)
        if tags:
            filtered_results = []
            for result in results:
                cursor.execute('SELECT tag FROM feedback_tags WHERE feedback_id = ?', (result['id'],))
                result_tags = set(row['tag'] for row in cursor.fetchall())
                if all(tag in result_tags for tag in tags):
                    filtered_results.append(result)
            results = filtered_results
        
        # Parse metadata JSON
        for result in results:
            if result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
        
        # Add tags if requested
        if include_tags:
            for result in results:
                cursor.execute('SELECT tag FROM feedback_tags WHERE feedback_id = ?', (result['id'],))
                result['tags'] = [row['tag'] for row in cursor.fetchall()]
        
        # Add annotations if requested
        if include_annotations:
            for result in results:
                cursor.execute('SELECT * FROM annotations WHERE feedback_id = ?', (result['id'],))
                result['annotations'] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_positive_examples(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get examples with positive feedback (rating >= 4)."""
        return self.query_feedback(is_positive=True, limit=limit)
    
    def get_negative_examples(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get examples with negative feedback (rating < 4)."""
        return self.query_feedback(is_positive=False, limit=limit)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored feedback.
        
        Returns:
            Dict containing various statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total feedback count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        stats['total_count'] = cursor.fetchone()[0]
        
        # Average rating
        cursor.execute('SELECT AVG(rating) FROM feedback')
        stats['average_rating'] = cursor.fetchone()[0]
        
        # Rating distribution
        cursor.execute('SELECT rating, COUNT(*) FROM feedback GROUP BY rating')
        stats['rating_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Platform distribution
        cursor.execute('SELECT platform, COUNT(*) FROM feedback WHERE platform IS NOT NULL GROUP BY platform')
        stats['platform_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Audience distribution
        cursor.execute('SELECT audience, COUNT(*) FROM feedback WHERE audience IS NOT NULL GROUP BY audience')
        stats['audience_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Positive/negative ratio
        cursor.execute('SELECT is_positive, COUNT(*) FROM feedback GROUP BY is_positive')
        positive_negative = {bool(row[0]): row[1] for row in cursor.fetchall()}
        stats['positive_count'] = positive_negative.get(True, 0)
        stats['negative_count'] = positive_negative.get(False, 0)
        
        # Tag distribution
        cursor.execute('''
        SELECT tag, COUNT(*) FROM feedback_tags 
        GROUP BY tag ORDER BY COUNT(*) DESC
        ''')
        stats['tag_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return stats
    
    def export_feedback_dataset(
        self,
        output_path: str,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        is_positive: Optional[bool] = None,
        limit: int = 1000
    ) -> int:
        """
        Export feedback as a dataset for model training.
        
        Args:
            output_path: Path to save the exported dataset
            min_rating: Minimum rating to include
            max_rating: Maximum rating to include
            is_positive: Whether to include only positive or negative examples
            limit: Maximum number of examples to export
            
        Returns:
            int: Number of examples exported
        """
        feedback_entries = self.query_feedback(
            min_rating=min_rating,
            max_rating=max_rating,
            is_positive=is_positive,
            limit=limit,
            include_tags=True,
            include_annotations=True
        )
        
        # Transform to training dataset format
        dataset = []
        for entry in feedback_entries:
            dataset_item = {
                "original_prompt": entry.get("original_prompt", ""),
                "content": entry.get("content_text", ""),
                "platform": entry.get("platform", ""),
                "audience": entry.get("audience", ""),
                "domain": entry.get("domain", ""),
                "rating": entry.get("rating", 0),
                "is_positive": entry.get("is_positive", False),
                "tags": entry.get("tags", [])
            }
            
            # Include annotations as feedback
            if "annotations" in entry and entry["annotations"]:
                dataset_item["annotations"] = entry["annotations"]
            
            dataset.append(dataset_item)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Exported {len(dataset)} feedback entries to {output_path}")
        return len(dataset)
    
    def delete_feedback(self, feedback_id: int) -> bool:
        """
        Delete a feedback entry.
        
        Args:
            feedback_id: ID of the feedback to delete
            
        Returns:
            bool: Whether deletion was successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete related tags
        cursor.execute('DELETE FROM feedback_tags WHERE feedback_id = ?', (feedback_id,))
        
        # Delete related annotations
        cursor.execute('DELETE FROM annotations WHERE feedback_id = ?', (feedback_id,))
        
        # Delete the feedback
        cursor.execute('DELETE FROM feedback WHERE id = ?', (feedback_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if success:
            logger.info(f"Deleted feedback with ID {feedback_id}")
        else:
            logger.warning(f"No feedback found with ID {feedback_id} to delete")
        
        return success


# Simple usage example
if __name__ == "__main__":
    # Create feedback store
    feedback_store = FeedbackStore()
    
    # Add sample feedback
    feedback_id = feedback_store.store_feedback(
        content_text="AI systems are revolutionizing how businesses operate, with 45% reporting increased efficiency.",
        rating=4,
        original_prompt="Write about AI impact on business",
        platform="linkedin",
        audience="executive",
        domain="ai",
        comment="Good executive summary but could use more specific metrics",
        tags=["content_quality", "audience_mismatch"],
        annotations=[
            {
                "text_segment": "AI systems are revolutionizing",
                "comment": "Too generic, needs more specificity",
                "start_index": 0,
                "end_index": 27
            }
        ],
        metadata={"generation_model": "outputs/finetune/final", "temperature": 0.8}
    )
    
    # Retrieve and print the feedback
    print(f"Added feedback with ID {feedback_id}")
    feedback = feedback_store.get_feedback(feedback_id)
    print(f"Retrieved feedback: {feedback}")
    
    # Print some stats
    stats = feedback_store.get_feedback_stats()
    print(f"Feedback stats: {stats}")
