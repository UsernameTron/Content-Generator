"""
Reference integration functions for incorporating callbacks to previous work.

This module provides functions to seamlessly integrate cross-references and
callbacks to previous work into generated content.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

from .retrieval import ContentRetriever
from .vector_db import ContentVectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ReferenceIntegrator:
    """Integrates references to past content in generated text."""
    
    def __init__(self, retriever: Optional[ContentRetriever] = None):
        """
        Initialize the reference integrator.
        
        Args:
            retriever: Content retriever instance, or None to create a new one
        """
        if retriever is None:
            self.retriever = ContentRetriever()
        else:
            self.retriever = retriever
    
    def enhance_prompt_with_references(
        self,
        prompt: str,
        context: Dict = None,
        max_references: int = 3
    ) -> str:
        """
        Enhance a prompt with relevant references.
        
        Args:
            prompt: The original generation prompt
            context: Additional context (audience, platform, etc.)
            max_references: Maximum number of references to include
            
        Returns:
            Enhanced prompt with references
        """
        if context is None:
            context = {}
            
        # Retrieve relevant content
        results = self.retriever.retrieve_with_context(prompt, context, k=max_references)
        
        # If we have results, enhance the prompt
        if results:
            references = []
            for idx, item in enumerate(results, 1):
                text = item.get("text", "")
                similarity = item.get("similarity", 0)
                if text and similarity > 0.7:  # Only use good matches
                    references.append(f"Reference {idx}: {text}")
            
            if references:
                return (
                    f"{prompt}\n\n"
                    f"Consider these relevant examples from your past work:\n"
                    f"{'\n\n'.join(references)}\n\n"
                    f"You can reference these examples while creating new content."
                )
        
        return prompt
    
    def add_references_to_generated_content(
        self,
        content: str,
        relevant_references: List[Dict],
        explicit_citation: bool = False
    ) -> str:
        """
        Add references to generated content if appropriate.
        
        Args:
            content: The generated content
            relevant_references: List of relevant reference items
            explicit_citation: Whether to add explicit citations
            
        Returns:
            Content with added references if appropriate
        """
        if not relevant_references:
            return content
            
        # If explicit citation is requested, add them at the end
        if explicit_citation:
            citations = []
            for idx, ref in enumerate(relevant_references, 1):
                text_preview = ref.get("text", "")[:100] + "..." if len(ref.get("text", "")) > 100 else ref.get("text", "")
                citations.append(f"[{idx}] Based on previous content: \"{text_preview}\"")
            
            if citations:
                return f"{content}\n\n**References:**\n" + "\n".join(citations)
        
        return content
    
    def detect_reference_opportunities(self, content: str) -> List[Dict]:
        """
        Detect opportunities to add references in content.
        
        Args:
            content: The content to analyze
            
        Returns:
            List of potential reference points
        """
        # Split content into paragraphs
        paragraphs = content.split("\n\n")
        
        opportunities = []
        for i, para in enumerate(paragraphs):
            # Skip short paragraphs
            if len(para) < 50:
                continue
                
            # Retrieve potentially relevant content for this paragraph
            results = self.retriever.retrieve_relevant_content(para, k=2, min_similarity=0.75)
            
            if results:
                opportunities.append({
                    "paragraph_index": i,
                    "paragraph_text": para,
                    "potential_references": results
                })
        
        return opportunities
    
    def generate_with_references(
        self,
        generator_func: Any,
        prompt: str,
        params: Dict = None,
        max_references: int = 3
    ) -> Tuple[str, List[Dict]]:
        """
        Generate content with integrated references.
        
        Args:
            generator_func: Function to generate content
            prompt: The generation prompt
            params: Generation parameters
            max_references: Maximum number of references
            
        Returns:
            Tuple of (generated_content, used_references)
        """
        if params is None:
            params = {}
        
        # Get context from params
        context = {
            key: params.get(key) 
            for key in ["audience", "platform", "tone", "style"] 
            if key in params
        }
        
        # Retrieve relevant content
        references, enhanced_prompt = self.retriever.retrieve_for_generation(
            prompt, params, k=max_references
        )
        
        # Generate content with enhanced prompt
        generated_content = generator_func(enhanced_prompt, **params)
        
        return generated_content, references
