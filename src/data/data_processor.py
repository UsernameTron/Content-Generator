"""
Data processing module for the multi-platform content generator.
This module prepares the training data from the JSON style guide.
"""

import json
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_style_data(json_path: str) -> Dict[str, Any]:
    """Load the writing style data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded style data from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading style data: {str(e)}")
        raise


def prepare_training_data(json_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Transform the JSON dataset into training examples with platform variations."""
    logger.info("Preparing training data with platform variations...")
    training_examples = []
    
    # Define platform-specific adaptations
    platforms = {
        "LinkedIn": {"max_length": 1300, "format": "professional", "hashtags": 2},
        "Twitter": {"max_length": 280, "format": "concise", "hashtags": 1},
        "Medium": {"max_length": 5000, "format": "detailed", "hashtags": 3},
        "Substack": {"max_length": 8000, "format": "newsletter", "hashtags": 4},
        "Instagram": {"max_length": 500, "format": "visual", "hashtags": 5},
        "Facebook": {"max_length": 2000, "format": "conversational", "hashtags": 3}
    }
    
    # Process existing examples for each platform
    for example in json_data.get("examples", []):
        base_content = example.get("content", "")
        base_description = example.get("description", "")
        
        for platform_name, platform_specs in platforms.items():
            # Create platform-specific adaptation prompt
            prompt = (
                f"Adapt the following content about {base_description} "
                f"for {platform_name} in C. Pete Connor's style. "
                f"Maximum length: {platform_specs['max_length']} characters. "
                f"Format: {platform_specs['format']}. "
                f"Use {platform_specs['hashtags']} hashtags."
            )
            
            # Add to training examples
            training_examples.append({
                "prompt": prompt,
                "completion": base_content  # Original content as starting point
            })
    
    # Create examples from topic patterns for each platform
    for topic_name, topic_info in json_data.get("topic_patterns", {}).items():
        for platform_name, platform_specs in platforms.items():
            prompt = (
                f"Create {platform_name} content about {topic_name} "
                f"focusing on {topic_info.get('angle', '')} in C. Pete Connor's style. "
                f"Use data from {', '.join(topic_info.get('data_sources', [])[:2])}. "
                f"Maximum length: {platform_specs['max_length']} characters. "
                f"Format: {platform_specs['format']}."
            )
            
            training_examples.append({
                "prompt": prompt,
                "completion": "Generate completion based on style guide"  # Placeholder
            })
    
    logger.info(f"Created {len(training_examples)} training examples with platform variations")
    return training_examples


def extract_key_topics(text: str, json_data: Optional[Dict[str, Any]] = None) -> List[str]:
    """Extract key topics from input text."""
    logger.info("Extracting key topics from input text...")
    
    # Get topic keywords from the JSON configuration if available
    all_keywords = []
    if json_data:
        for topic_pattern in json_data.get("topic_patterns", {}).values():
            all_keywords.extend(topic_pattern.get("keywords", []))
    
    # Find matching keywords in the text
    found_topics = []
    for keyword in all_keywords:
        if keyword.lower() in text.lower():
            found_topics.append(keyword)
    
    # If no topics found, extract nouns as fallback
    if not found_topics:
        # Simple fallback to default topics
        return ["technology", "industry trends", "corporate practices"]
    
    return found_topics[:5]  # Limit to top 5 topics


if __name__ == "__main__":
    # Test the module
    style_data = load_style_data("../../src/data/writing_style.json")
    examples = prepare_training_data(style_data)
    print(f"Generated {len(examples)} training examples")
