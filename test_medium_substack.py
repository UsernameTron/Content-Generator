#!/usr/bin/env python3
"""
Test script for generating content for Medium and Substack platforms using Pete Connor's style.
This script demonstrates how the content generator processes inputs and generates platform-specific outputs.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("medium_substack_test")

# Import necessary modules
from src.models.model_content_generator import ModelContentGenerator
from src.models.platform_specs import get_platform_specs

def test_medium_substack_generation():
    """Test content generation for Medium and Substack platforms."""
    # Sample content for testing - using a tech-related topic
    content = """
    The latest AI models claim to be revolutionary, but a closer look reveals they're often 
    repeating patterns we've seen before. While marketing teams promote each new iteration as 
    groundbreaking, many fundamental challenges remain unsolved. Data shows that despite increasing 
    model size and compute requirements, improvements in reasoning capabilities have plateaued 
    in several benchmarks. Companies continue to hype incremental advances as transformative leaps.
    """
    
    platforms = ["medium", "substack"]
    sentiment = "thoughtful"  # Can be: informative, professional, casual, enthusiastic, thoughtful
    
    # Initialize the model content generator with Pete Connor's style
    logger.info("Initializing content generator with Pete Connor's style")
    generator = ModelContentGenerator(style="pete_connor")
    
    results = {}
    
    # Generate and display content for each platform
    for platform in platforms:
        logger.info(f"Generating content for {platform.title()} with sentiment: {sentiment}")
        
        # Get platform specs for reference
        platform_specs = get_platform_specs(platform)
        logger.info(f"{platform.title()} optimal length range: {platform_specs.optimal_length}")
        
        start_time = time.time()
        
        # Generate content
        generated = generator.generate_content(
            content=content,
            platform=platform.lower(),
            sentiment=sentiment
        )
        
        generation_time = time.time() - start_time
        
        if generated:
            results[platform] = {
                "content": generated[0],
                "character_count": len(generated[0]),
                "word_count": len(generated[0].split()),
                "generation_time": generation_time,
                "within_limits": len(generated[0]) <= platform_specs.max_length
            }
        else:
            logger.error(f"No content generated for {platform}")
            results[platform] = {
                "content": "Generation failed",
                "character_count": 0,
                "word_count": 0,
                "generation_time": generation_time,
                "within_limits": False
            }
    
    # Display results
    for platform, result in results.items():
        print(f"\n{'='*80}\n{platform.upper()} CONTENT\n{'='*80}\n")
        print(result["content"])
        
        print(f"\n{'-'*40}\nANALYSIS\n{'-'*40}")
        print(f"Character count: {result['character_count']}")
        print(f"Word count: {result['word_count']}")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Within platform limits: {'Yes' if result['within_limits'] else 'No'}")
        
        # Further analyze the content for Pete Connor style elements
        content = result["content"].lower()
        
        # Check for Pete Connor style elements
        style_elements = {
            "Data references": any(term in content for term in ["data shows", "research indicates", "survey", "study", "percent", "statistics", "%"]),
            "Contrarian view": any(term in content for term in ["contrary", "however", "despite", "but", "reality", "truth"]),
            "Satirical tone": any(term in content for term in ["hype", "buzzword", "supposedly", "allegedly", "so-called", "claims", "grift"]),
            "One-liner": "one-liner" in content or "**one-liner**" in result["content"],
            "Hashtags": "#" in result["content"],
            "Sections with headings": "# " in result["content"] or "## " in result["content"]
        }
        
        print("\nPete Connor style elements:")
        for element, present in style_elements.items():
            print(f"- {element}: {'Present' if present else 'Missing'}")
    
    # Clean up resources
    generator.close()
    logger.info("Test completed")

if __name__ == "__main__":
    test_medium_substack_generation()