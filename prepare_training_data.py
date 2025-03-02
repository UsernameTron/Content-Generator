"""
Prepare training data for fine-tuning the C. Pete Connor style model.

This script converts the writing_style.json and any additional examples 
into properly formatted JSONL training data for model fine-tuning.
It also integrates:
- Hugging Face datasets for irony, sarcasm, and humor
- Proselint repository for post-processing stylistic improvements
- Anti-pattern examples to avoid AI-typical phrasing
"""

import os
import json
import random
import logging
import subprocess
import re
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple
import requests

try:
    from datasets import load_dataset
except ImportError:
    logging.warning("HuggingFace datasets library not installed. Run: pip install datasets")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
PROSELINT_REPO = "https://github.com/amperser/proselint.git"
PROSELINT_DIR = "proselint_repo"
HF_DATASETS = [
    ("ColumbiaNLP/MELD", "train", "text", "Emotion-based conversational dataset"),
    ("Fraser/sarcasm-detection", "train", "text", "Sarcasm detection dataset"),
    ("Abirate/english_quotes", "train", "quote", "Quotes and sayings dataset")
]
AI_PATTERNS = [
    "game changer",
    "here's the kicker",
    "cutting-edge",
    "revolutionary",
    "disruptive", 
    "innovative",
    "seamless customer journey",
    "delightful experience",
    "AI-powered",
    "frictionless",
    "hyper-personalization",
    "digital transformation",
    "paradigm shift",
    "synergy",
    "leverage",
    "holistic approach",
    "robust solution",
    "scalable architecture"
]

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_writing_style(style_file: str) -> Dict[str, Any]:
    """
    Load writing style from JSON file.
    
    Args:
        style_file: Path to writing style JSON
        
    Returns:
        Writing style dictionary
    """
    logger.info(f"Loading writing style from {style_file}")
    with open(style_file, 'r') as f:
        return json.load(f)

def expand_examples_from_style(style_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract examples from writing style data.
    
    Args:
        style_data: Writing style dictionary
        
    Returns:
        List of example dictionaries
    """
    examples = []
    
    # Extract style attributes
    logger.info("Extracting style attributes into training examples")
    if "voice" in style_data:
        examples.append({"text": f"Writing style voice: {style_data['voice']}"})
    
    if "tone" in style_data:
        examples.append({"text": f"Writing style tone: {style_data['tone']}"})
    
    if "characteristics" in style_data:
        for char in style_data["characteristics"]:
            examples.append({"text": f"Writing style characteristic: {char}"})
    
    # Extract platform-specific examples
    if "platforms" in style_data:
        logger.info("Extracting platform-specific examples")
        for platform, data in style_data["platforms"].items():
            if "examples" in data:
                for example in data["examples"]:
                    examples.append({
                        "text": f"Example {platform} content: {example}"
                    })
    
    return examples

def create_synthetic_examples(style_data: Dict[str, Any], num_examples: int = 50) -> List[Dict[str, str]]:
    """
    Create synthetic training examples based on style data.
    
    Args:
        style_data: Writing style dictionary
        num_examples: Number of synthetic examples to create
        
    Returns:
        List of synthetic example dictionaries
    """
    logger.info(f"Creating {num_examples} synthetic examples")
    synthetic_examples = []
    
    # Extract components for generation
    phrases = []
    if "phrases" in style_data:
        phrases = style_data["phrases"]
    
    characteristics = []
    if "characteristics" in style_data:
        characteristics = style_data["characteristics"]
    
    # Templates for synthetic examples
    templates = [
        "Here's a tech analysis written in C. Pete Connor's style: {characteristic}. {phrase}",
        "This is how C. Pete Connor would analyze this: {phrase} {characteristic}",
        "C. Pete Connor's take on this technology would be: {characteristic} followed by {phrase}",
        "Writing as C. Pete Connor: {phrase} This perfectly illustrates {characteristic}",
        "In the distinctive voice of C. Pete Connor: {characteristic} which is why {phrase}"
    ]
    
    # Generate synthetic examples
    for _ in range(num_examples):
        template = random.choice(templates)
        characteristic = random.choice(characteristics) if characteristics else ""
        phrase = random.choice(phrases) if phrases else ""
        
        text = template.format(characteristic=characteristic, phrase=phrase)
        synthetic_examples.append({"text": text})
    
    return synthetic_examples

def fetch_huggingface_datasets(num_examples=100):
    """Fetch sarcasm, irony, and humor datasets from Hugging Face."""
    examples = []
    try:
        for dataset_name, split, text_column, description in HF_DATASETS:
            logger.info(f"Fetching dataset: {dataset_name}")
            try:
                dataset = load_dataset(dataset_name, split=split)
                subset = dataset.shuffle(seed=42).select(range(min(len(dataset), num_examples // len(HF_DATASETS))))
                
                for item in subset:
                    if text_column in item and item[text_column]:
                        text = item[text_column]
                        if len(text.split()) > 10:  # Ensure the text is substantial
                            examples.append({
                                "prompt": f"Create content similar to this style: ",
                                "response": text,
                                "metadata": {
                                    "source": dataset_name,
                                    "description": description
                                }
                            })
                            
                logger.info(f"Added {len(subset)} examples from {dataset_name}")
            except Exception as e:
                logger.warning(f"Error fetching dataset {dataset_name}: {str(e)}")
                continue
                
        logger.info(f"Added {len(examples)} examples from Hugging Face datasets")
    except Exception as e:
        logger.error(f"Error fetching Hugging Face datasets: {str(e)}")
    
    return examples

def setup_proselint():
    """
    Set up Proselint repository for stylistic analysis.
    
    Returns:
        True if setup succeeded, False otherwise
    """
    if Path(PROSELINT_DIR).exists():
        logger.info(f"Proselint repository already exists at {PROSELINT_DIR}")
        return True
    
    try:
        logger.info(f"Cloning Proselint repository from {PROSELINT_REPO}")
        subprocess.run(["git", "clone", PROSELINT_REPO, PROSELINT_DIR], check=True)
        
        # Install proselint
        logger.info("Installing Proselint")
        subprocess.run(["pip", "install", "-e", PROSELINT_DIR], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up Proselint: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting up Proselint: {str(e)}")
        return False

def apply_proselint(text):
    """Apply Proselint to improve text quality."""
    try:
        # Ensure text is a string
        if not isinstance(text, str):
            return text
            
        # Get Proselint suggestions
        suggestions = proselint.tools.lint(text)
        
        # Apply suggestions (simple version, could be more sophisticated)
        improved_text = text
        offset = 0
        
        for suggestion in sorted(suggestions, key=lambda x: x[0]):
            # Skip suggestions that might change the meaning
            if "spelling" in suggestion[1] or "typography" in suggestion[1]:
                continue
                
            start, end = suggestion[0]
            replacement = suggestion[4] if len(suggestion) > 4 and suggestion[4] else ""
            
            # Apply the replacement
            if replacement:
                improved_text = improved_text[:start+offset] + replacement + improved_text[end+offset:]
                offset += len(replacement) - (end - start)
        
        return improved_text
    except Exception as e:
        logger.warning(f"Error applying Proselint: {str(e)}")
        return text

def create_anti_pattern_examples(style_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Create examples that demonstrate poor AI-typical writing patterns to avoid.
    These serve as negative examples for the model.
    
    Args:
        style_data: Writing style dictionary
        
    Returns:
        List of anti-pattern example dictionaries
    """
    anti_examples = []
    
    # Create examples with common AI patterns to avoid
    for pattern in AI_PATTERNS:
        topics = style_data.get("domain_expertise", ["customer experience", "AI", "machine learning"])
        
        for topic in topics:
            # Bad example with buzzwords and symmetrical structures
            bad_text = f"""
When it comes to {topic}, {pattern} is absolutely essential for businesses today.
On one hand, organizations need to innovate; on the other hand, they must maintain stability.
Here's the kicker: most companies fail to implement properly.
The solution is cutting-edge technology combined with revolutionary approaches.
This end-to-end, robust solution creates a seamless customer journey.
            """.strip()
            
            # Corresponding good example in Pete's style
            good_text = f"""
Contrary to what every consultant will tell you, {topic} isn't about {pattern}.
The data actually shows most implementations fail spectacularly while executives marvel at their digital transformation.
Ironically, companies obsessed with "innovation" typically produce the most mediocre results.
Statistically speaking, you'd be better off ignoring the buzzwords entirely and focusing on practical implementation.
            """.strip()
            
            anti_examples.append({
                "prompt": f"Rewrite this text in C. Pete Connor's style, avoiding typical AI patterns:\n\n{bad_text}",
                "response": good_text
            })
    
    logger.info(f"Created {len(anti_examples)} anti-pattern examples")
    return anti_examples

def prepare_data(
    style_file="data/writing_style.json",
    output_file="data/training_data.jsonl",
    num_synthetic=100,
    num_huggingface=100,
    use_proselint=True,
    anti_patterns=True
):
    """Prepare training data from multiple sources."""
    all_examples = []
    
    # Extract examples from writing style
    style_examples = expand_examples_from_style(load_writing_style(style_file))
    all_examples.extend(style_examples)
    logger.info(f"Extracted {len(style_examples)} examples from writing style")
    
    # Create synthetic examples
    synthetic_examples = create_synthetic_examples(load_writing_style(style_file), num_synthetic)
    all_examples.extend(synthetic_examples)
    logger.info(f"Added {len(synthetic_examples)} synthetic examples, total: {len(all_examples)}")
    
    # Fetch examples from Hugging Face
    huggingface_examples = fetch_huggingface_datasets(num_huggingface)
    all_examples.extend(huggingface_examples)
    logger.info(f"Added {len(huggingface_examples)} examples from Hugging Face, total: {len(all_examples)}")
    
    # Add anti-pattern examples if requested
    if anti_patterns:
        anti_pattern_examples = create_anti_pattern_examples(load_writing_style(style_file))
        all_examples.extend(anti_pattern_examples)
        logger.info(f"Added {len(anti_pattern_examples)} anti-pattern examples, total: {len(all_examples)}")
    
    # Install and use Proselint if requested
    if use_proselint:
        setup_proselint()
        logger.info("Applying Proselint to improve text quality")
        
        # Apply Proselint to responses (not prompts)
        for example in all_examples:
            if "response" in example and example["response"]:
                example["response"] = apply_proselint(example["response"])
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format examples into jsonl format compatible with fine-tuning
    with open(output_file, 'w') as f:
        for example in all_examples:
            # Ensure example has the required fields
            if "prompt" in example and "response" in example:
                formatted_example = {
                    "messages": [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["response"]}
                    ]
                }
                f.write(json.dumps(formatted_example) + '\n')
    
    logger.info(f"Saved {len(all_examples)} examples to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for C. Pete Connor style model")
    parser.add_argument("--style_file", type=str, default="data/writing_style.json", help="Path to writing style JSON")
    parser.add_argument("--output_file", type=str, default="data/training_data.jsonl", help="Output file for dataset")
    parser.add_argument("--num_synthetic", type=int, default=100, help="Number of synthetic examples to create")
    parser.add_argument("--num_huggingface", type=int, default=100, help="Number of Hugging Face examples to fetch")
    parser.add_argument("--no_proselint", action="store_true", help="Disable Proselint processing")
    parser.add_argument("--no_anti_patterns", action="store_true", help="Disable anti-pattern examples")
    
    args = parser.parse_args()
    
    prepare_data(
        args.style_file, 
        args.output_file, 
        args.num_synthetic,
        args.num_huggingface,
        not args.no_proselint,
        not args.no_anti_patterns
    )
