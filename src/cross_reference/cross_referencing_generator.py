"""
Extension of ModelContentGenerator with cross-referencing capabilities.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from ..models.model_content_generator import ModelContentGenerator
from .integration import ReferenceIntegrator
from .vector_db import ContentVectorDB
from .retrieval import ContentRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CrossReferencingGenerator(ModelContentGenerator):
    """Content generator with cross-referencing capabilities."""
    
    def __init__(
        self,
        model_dir: str = "outputs/finetune/final",
        device: str = "auto",
        use_wandb: bool = True,
        max_references: int = 3,
        reference_threshold: float = 0.7
    ):
        """
        Initialize the cross-referencing content generator.
        
        Args:
            model_dir: Directory containing the model
            device: Device to use (auto, cpu, cuda, mps)
            use_wandb: Whether to use Weights & Biases for logging
            max_references: Maximum number of references to include
            reference_threshold: Minimum similarity threshold for references
        """
        super().__init__(model_dir, device, use_wandb)
        
        # Initialize cross-referencing components
        self.vector_db = ContentVectorDB()
        self.retriever = ContentRetriever(self.vector_db)
        self.reference_integrator = ReferenceIntegrator(self.retriever)
        
        # Cross-referencing settings
        self.max_references = max_references
        self.reference_threshold = reference_threshold
        self.cross_referencing_enabled = True
    
    def generate_content_with_references(
        self,
        prompt: str,
        audience: str = "general",
        platform: str = "blog",
        max_length: int = 1000,
        temperature: float = 0.7,
        use_references: bool = True,
        return_references: bool = False
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """
        Generate content with cross-references to previous content.
        
        Args:
            prompt: The content generation prompt
            audience: Target audience
            platform: Content platform
            max_length: Maximum content length
            temperature: Generation temperature
            use_references: Whether to use cross-references
            return_references: Whether to return used references
            
        Returns:
            Generated content, optionally with references used
        """
        params = {
            "audience": audience,
            "platform": platform,
            "max_length": max_length,
            "temperature": temperature
        }
        
        if use_references and self.cross_referencing_enabled:
            # Generate with references
            generated_content, references = self.reference_integrator.generate_with_references(
                lambda p, **kwargs: super().generate_content(
                    p, kwargs.get("audience"), kwargs.get("platform"), 
                    kwargs.get("max_length"), kwargs.get("temperature")
                ),
                prompt,
                params,
                self.max_references
            )
            
            # Add the generated content to the vector DB for future reference
            if generated_content:
                metadata = {
                    "prompt": prompt,
                    "audience": audience,
                    "platform": platform,
                    "source": "generated"
                }
                self.vector_db.add_content(generated_content, metadata)
            
            if return_references:
                return generated_content, references
            return generated_content
        else:
            # Generate without references
            generated_content = super().generate_content(
                prompt, audience, platform, max_length, temperature
            )
            
            # Still add to vector DB for future reference
            if generated_content:
                metadata = {
                    "prompt": prompt,
                    "audience": audience,
                    "platform": platform,
                    "source": "generated"
                }
                self.vector_db.add_content(generated_content, metadata)
            
            if return_references:
                return generated_content, []
            return generated_content
    
    def index_existing_content(self, source_paths: List[str] = None) -> int:
        """
        Index existing content for cross-referencing.
        
        Args:
            source_paths: List of content source paths, or None to use defaults
            
        Returns:
            Number of content items indexed
        """
        if source_paths is None:
            source_paths = ["data/training_data.jsonl", "data/validation_data.jsonl"]
        
        total_indexed = 0
        for path in source_paths:
            count = self.vector_db.index_from_training_data(path)
            total_indexed += count
            
        logger.info(f"Indexed {total_indexed} content items from {len(source_paths)} sources")
        return total_indexed
    
    def toggle_cross_referencing(self, enabled: bool = True) -> None:
        """
        Enable or disable cross-referencing.
        
        Args:
            enabled: Whether cross-referencing should be enabled
        """
        self.cross_referencing_enabled = enabled
        logger.info(f"Cross-referencing {'enabled' if enabled else 'disabled'}")
