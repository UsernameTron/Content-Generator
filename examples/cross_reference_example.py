"""
Example usage of the cross-referencing functionality.

This script demonstrates how to use the cross-referencing functionality
to generate content with references to past work.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set environment variables for Apple Silicon optimization
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import cross-referencing functionality
from src.cross_reference.cross_referencing_generator import CrossReferencingGenerator
from src.cross_reference.vector_db import ContentVectorDB
from src.cross_reference.retrieval import ContentRetriever
from src.cross_reference.integration import ReferenceIntegrator

def main():
    """Run cross-referencing example."""
    logger.info("Starting cross-referencing example")
    
    # 1. Initialize the vector database and index content if needed
    logger.info("Initializing vector database")
    vector_db = ContentVectorDB()
    
    # 2. Initialize the cross-referencing generator
    logger.info("Initializing cross-referencing generator")
    generator = CrossReferencingGenerator(
        model_dir="outputs/finetune/final",  # Use your model directory
        max_references=3
    )
    
    # 3. Index existing content if the database is empty
    if len(vector_db.content_metadata) == 0:
        logger.info("Indexing existing content")
        generator.index_existing_content()
    
    # 4. Examples of content generation with cross-references
    
    # Example 1: Generate blog content with references
    logger.info("Example 1: Generate blog content with references")
    blog_prompt = "Write a blog post about the importance of AI ethics in modern technology development."
    
    blog_content, references = generator.generate_content_with_references(
        prompt=blog_prompt,
        audience="technical",
        platform="blog",
        max_length=1000,
        temperature=0.7,
        return_references=True
    )
    
    logger.info(f"Generated blog content with {len(references)} references")
    logger.info("Blog content preview: " + blog_content[:200] + "...")
    
    # Example 2: Generate social media content with references
    logger.info("Example 2: Generate social media content with references")
    social_prompt = "Create a LinkedIn post about career transitions into data science."
    
    social_content, references = generator.generate_content_with_references(
        prompt=social_prompt,
        audience="professional",
        platform="linkedin",
        max_length=500,
        temperature=0.7,
        return_references=True
    )
    
    logger.info(f"Generated social content with {len(references)} references")
    logger.info("Social content preview: " + social_content[:150] + "...")
    
    # Example 3: Direct use of the reference integrator
    logger.info("Example 3: Direct use of the reference integrator")
    integrator = ReferenceIntegrator()
    
    enhanced_prompt = integrator.enhance_prompt_with_references(
        prompt="Write tips for effective remote work collaboration.",
        context={"audience": "business", "platform": "newsletter"}
    )
    
    logger.info("Enhanced prompt preview: " + enhanced_prompt[:200] + "...")
    
    logger.info("Cross-referencing example completed successfully")

if __name__ == "__main__":
    main()
