"""
Retrieval mechanism for identifying relevant past content during generation.

This module implements functions to retrieve relevant past content during
the content generation process.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

from .vector_db import ContentVectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContentRetriever:
    """Retriever for finding relevant past content."""
    
    def __init__(self, vector_db: Optional[ContentVectorDB] = None):
        """
        Initialize the content retriever.
        
        Args:
            vector_db: Vector database instance, or None to create a new one
        """
        if vector_db is None:
            self.vector_db = ContentVectorDB()
        else:
            self.vector_db = vector_db
    
    def retrieve_relevant_content(
        self, 
        query: str, 
        k: int = 3,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve relevant content based on query.
        
        Args:
            query: The query text or prompt
            k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant content items with similarity scores
        """
        results = self.vector_db.search(query, k=k)
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if r.get("similarity", 0) >= min_similarity]
        
        return filtered_results
    
    def retrieve_with_context(
        self, 
        query: str,
        context: Dict,
        k: int = 3
    ) -> List[Dict]:
        """
        Retrieve content considering additional context.
        
        Args:
            query: The query text
            context: Additional context like audience, platform, etc.
            k: Maximum number of results to return
            
        Returns:
            List of relevant content items
        """
        # Enrich query with context
        enhanced_query = query
        
        if "audience" in context:
            enhanced_query += f" for {context['audience']} audience"
            
        if "platform" in context:
            enhanced_query += f" on {context['platform']}"
            
        if "tone" in context:
            enhanced_query += f" with {context['tone']} tone"
        
        # Retrieve relevant content with enhanced query
        results = self.vector_db.search(enhanced_query, k=k)
        
        return results
    
    def retrieve_for_generation(
        self,
        prompt: str,
        generation_params: Dict,
        k: int = 3
    ) -> Tuple[List[Dict], str]:
        """
        Retrieve content specifically for the generation process.
        
        Args:
            prompt: The generation prompt
            generation_params: Parameters for content generation
            k: Maximum number of references to include
            
        Returns:
            Tuple of (relevant_content, enhanced_prompt)
        """
        # Extract context from generation parameters
        context = {
            key: generation_params.get(key) 
            for key in ["audience", "platform", "tone", "style"] 
            if key in generation_params
        }
        
        # Retrieve relevant content
        results = self.retrieve_with_context(prompt, context, k=k)
        
        # If we have relevant results, enhance the prompt
        if results:
            references = []
            for idx, item in enumerate(results, 1):
                text = item.get("text", "")
                similarity = item.get("similarity", 0)
                if text and similarity > 0.75:  # Only use high-quality matches
                    references.append(f"Reference {idx}: {text}")
            
            if references:
                # Create enhanced prompt with references
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    f"Here are some relevant past content examples for reference:\n"
                    f"{'\n\n'.join(references)}\n\n"
                    f"Use these examples as references for your content while maintaining your unique voice."
                )
                return results, enhanced_prompt
        
        # Return original prompt if no good matches
        return results, prompt
