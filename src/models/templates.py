"""
Content generation templates for various platforms incorporating C. Pete Connor's style.
"""

import logging
import random
import json
import os
from typing import Dict, List, Any
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path to custom templates
CUSTOM_TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    'data', 'custom_templates.json')

def load_pete_connor_templates():
    """
    Load C. Pete Connor templates from writing_style.json
    """
    try:
        style_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'data', 'writing_style.json')
        
        if os.path.exists(style_path):
            with open(style_path, 'r') as f:
                style_data = json.load(f)
                
            # Extract content formulas and linguistic patterns for templates
            formulas = style_data.get('content_formulas', {})
            patterns = style_data.get('linguistic_patterns', {})
            
            # Create LinkedIn templates based on Pete Connor's style
            linkedin_templates = [
                "# {topic}: The Reality vs. The Hype\n\n{main_point}\n\nThe data tells a different story:\n{supporting_points}\n\n{call_to_action}\n\n{hashtags}",
                "Let's talk about {topic}...\n\nWhile vendors claim: {main_point}\n\nThe uncomfortable truth is: {supporting_points}\n\nInstead of falling for the hype: {call_to_action}\n\n{hashtags}",
                "There's a fascinating contradiction in {topic}:\n\n{main_point}\n\nThe numbers are brutally clear:\n{supporting_points}\n\nThe question is: {question}\n\n{hashtags}",
                "I've been analyzing {topic} data, and...\n\n{main_point}\n\nHere's what's actually happening:\n{supporting_points}\n\nThe key takeaway: {call_to_action}\n\n{hashtags}"
            ]
            
            # Get example posts from the JSON
            examples = style_data.get('examples', [])
            for example in examples:
                content = example.get('content', '')
                if content and 'LinkedIn' in content:
                    linkedin_templates.append(content)
            
            # Add Pete Connor templates to platforms
            pete_connor_templates = {
                "LinkedIn": linkedin_templates,
                # Create similar templates for other platforms
                "Twitter": [
                    "Hot take on {topic}: {main_point} The data: {supporting_points_brief} {hashtags}",
                    "{topic} reality check: {main_point} {hashtags}",
                    "Let's be real about {topic}: {main_point} The numbers don't lie. {hashtags}",
                    "{question} {main_point} Data says otherwise. {hashtags}"
                ],
                "Blog": [
                    "# The {topic} Emperor Has No Clothes\n\n## The Claim\n{main_point}\n\n## The Data\n{supporting_points}\n\n## The Reality\n{call_to_action}\n\n## Conclusion\n{conclusion}",
                    "# {topic}: Separating Hype from Reality\n\n{introduction}\n\n## The Industry Narrative\n{main_point}\n\n## What The Data Actually Shows\n{supporting_points}\n\n## Implications\n{implications}\n\n## The Path Forward\n{conclusion}"
                ],
                "Facebook": [
                    "I've been looking at {topic} data lately and, wow, the disconnect between marketing and reality is fascinating.\n\n{main_point}\n\n{supporting_points}\n\nWhat's your experience been? {hashtags}",
                    "Today's tech myth-busting: {topic}\n\n{main_point}\n\n{supporting_points}\n\nWho else has noticed this gap between promises and reality? {hashtags}"
                ],
                "Instagram": [
                    "📊 Reality Check: {topic} 📊\n\n{main_point}\n\n{supporting_points_brief}\n\n{hashtags}",
                    "The {topic} disconnect in one post:\n\nWhat they say: {main_point}\n\nWhat the data shows: {supporting_points_brief}\n\n{hashtags}"
                ],
                "Email Newsletter": [
                    "Subject: {title} - Reality Check\n\nGreetings truth-seekers,\n\n{introduction}\n\n# {topic} - What You're Not Being Told\n\n{main_point}\n\n{supporting_points}\n\n# The Data That Matters\n\n{key_points}\n\n{call_to_action}\n\nStay skeptical,\nC. Pete Connor"
                ]
            }
            
            logger.info("Loaded C. Pete Connor writing style templates")
            return pete_connor_templates
        else:
            logger.warning(f"Writing style file not found at {style_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading C. Pete Connor templates: {str(e)}")
        return {}

# Load custom templates if they exist
def load_custom_templates():
    """Load custom templates from JSON file if it exists"""
    try:
        if os.path.exists(CUSTOM_TEMPLATES_PATH):
            with open(CUSTOM_TEMPLATES_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading custom templates: {str(e)}")
        return {}

# Initial template definitions
TEMPLATES = {
    # Twitter templates focus on concise messages with hashtags
    "Twitter": [
        "🚨 {main_point} The data tells a different story. {hashtags}",
        "Let's be honest about {main_point}. The reality: {supporting_points_brief} {hashtags}",
        "Tech reality check: {main_point} Data shows {supporting_points_brief} {hashtags}",
        "🤔 {question} {main_point} {hashtags}",
        "📊 Unpopular truth: {main_point} {hashtags}"
    ],
    
    # Medium templates for longer, more technical content
    "Medium": [
        "# {title}\n\n## Introduction\n{introduction}\n\n## The Real Story Behind {topic}\n{main_point}\n\n{supporting_points}\n\n## Case Studies\n{case_studies}\n\n## Industry Analysis\n{industry_trends}\n\n## Conclusion\n{conclusion}\n\n{call_to_action}",
        "# {title}: Separating Reality from Vendor Fiction\n\n{introduction}\n\n## The Problem\n{problem_statement}\n\n## The Evidence\n{main_point}\n\n{supporting_points}\n\n## Case Studies\n{case_studies}\n\n## What This Means For You\n{conclusion}",
        "# The {topic} Emperor Has No Clothes\n\n{introduction}\n\n## The Industry Narrative\n{main_point}\n\n## What The Data Actually Shows\n{supporting_points}\n\n## Case Studies\n{case_studies}\n\n## Industry Trends\n{industry_trends}\n\n## The Way Forward\n{conclusion}"
    ],
    
    # Substack templates for newsletter-style content
    "Substack": [
        "Subject: {title} - The Reality Nobody's Talking About\n\nDear Fellow Skeptic,\n\n{introduction}\n\n# {topic} - What The Industry Doesn't Want You To Know\n\n{main_point}\n\n{supporting_points}\n\n# Case Studies: The Hall of Shame\n{case_studies}\n\n# The Circle of Hype\n{industry_trends}\n\n# Bottom Line\n{conclusion}\n\nStay skeptical,\nC. Pete Connor\n\nP.S. {call_to_action}",
        "Subject: The {topic} Scam Everyone's Falling For\n\nHello Truth-Seekers,\n\n{introduction}\n\nThis week, I need to talk about the absolute disaster that is {topic}.\n\n{main_point}\n\n{supporting_points}\n\n# The Corporate Hall of Shame\n{case_studies}\n\n# The Inconvenient Truth\n{conclusion}\n\nThoughts? Just hit reply. I read everything.\n\nUntil next time,\nC. Pete Connor\n\nP.S. {call_to_action}"
    ],
    
    # LinkedIn templates focus on professional content
    "LinkedIn": [
        "# 🚨 The {topic} Reality Check\n\n{main_point}\n\n{supporting_points}\n\n**One-Liner:** {call_to_action}\n\n{hashtags}",
        "I've been analyzing {topic} data, and the disconnect between marketing and reality is fascinating.\n\n{main_point}\n\n{supporting_points}\n\n**The hard truth:** {call_to_action}\n\n{hashtags}",
        "# 📊 {topic}: The Emperor Has No Clothes\n\n{main_point}\n\n{supporting_points}\n\nDon't just take my word for it - look at the data.\n\n**One-Liner:** {call_to_action}\n\n{hashtags}",
        "Let's talk about the {topic} elephant in the room...\n\n{main_point}\n\n{supporting_points}\n\n**Reality check:** {call_to_action}\n\n{hashtags}"
    ],
    
    # Facebook templates are more conversational
    "Facebook": [
        "I was just thinking about {topic} and wanted to share...\n\n{main_point}\n\n{supporting_points}\n\nAnyone else have thoughts on this? {hashtags}",
        "{question}\n\n{main_point}\n\n{supporting_points}\n\nWhat do you all think? {hashtags}",
        "So this happened today: {main_point}\n\n{supporting_points}\n\n{call_to_action} {hashtags}",
        "Can we talk about {topic} for a minute?\n\n{main_point}\n\n{supporting_points} {hashtags}"
    ],
    
    # Instagram templates focus on visual storytelling
    "Instagram": [
        "{main_point}\n\n{supporting_points}\n\n{hashtags}",
        "Today's thoughts on {topic}:\n\n{main_point}\n\n{call_to_action}\n\n{hashtags}",
        "{question}\n\n{main_point}\n\n{supporting_points}\n\n{hashtags}",
        "✨ {main_point} ✨\n\n{supporting_points}\n\n{hashtags}"
    ],
    
    # Blog templates are more detailed and structured
    "Blog": [
        "# {title}\n\n## Introduction\n{introduction}\n\n## {topic}\n{main_point}\n\n{supporting_points}\n\n## Key Takeaways\n{key_points}\n\n## Conclusion\n{conclusion}\n\n{call_to_action}",
        "# {title}\n\n{introduction}\n\n## The Problem\n{problem_statement}\n\n## Analysis\n{main_point}\n\n{supporting_points}\n\n## Solution\n{solution}\n\n## Conclusion\n{conclusion}",
        "# {title}\n\n{introduction}\n\n## Background\n{background}\n\n## Current State\n{main_point}\n\n{supporting_points}\n\n## Future Implications\n{implications}\n\n## Conclusion\n{conclusion}"
    ],
    
    # Email Newsletter templates
    "Email Newsletter": [
        "Subject: {title}\n\nDear Reader,\n\n{introduction}\n\n# {topic}\n\n{main_point}\n\n{supporting_points}\n\n# Key Insights\n\n{key_points}\n\n{call_to_action}\n\nBest regards,\n[Your Name]",
        "Subject: {title}\n\nHello,\n\n{introduction}\n\nThis week, I wanted to discuss {topic}.\n\n{main_point}\n\n{supporting_points}\n\nThoughts or questions? Just reply to this email.\n\nUntil next time,\n[Your Name]",
        "Subject: {title}\n\nGreetings!\n\n{introduction}\n\n# Spotlight on {topic}\n\n{main_point}\n\n{supporting_points}\n\n# What This Means For You\n\n{implications}\n\n{call_to_action}\n\nWarm regards,\n[Your Name]"
    ]
}

# Generic templates as fallback
GENERIC_TEMPLATES = [
    "{main_point}\n\n{supporting_points}",
    "Topic: {topic}\n\n{main_point}\n\n{supporting_points}",
    "{introduction}\n\n{main_point}\n\n{supporting_points}\n\n{conclusion}"
]

# Merge the templates with C. Pete Connor's style templates
PETE_CONNOR_TEMPLATES = load_pete_connor_templates()
CUSTOM_TEMPLATES = load_custom_templates()

# Merge all templates, prioritizing custom templates, then Pete Connor templates, then defaults
for platform, templates in PETE_CONNOR_TEMPLATES.items():
    if platform in TEMPLATES:
        TEMPLATES[platform].extend(templates)
    else:
        TEMPLATES[platform] = templates

# Add custom templates if available
for platform, templates in CUSTOM_TEMPLATES.items():
    if platform in TEMPLATES:
        TEMPLATES[platform].extend(templates)
    else:
        TEMPLATES[platform] = templates

def get_template(platform: str, sentiment: str = "neutral", style: str = None) -> str:
    """
    Get a random template for the specified platform and sentiment.
    
    Args:
        platform: Target platform name
        sentiment: Detected sentiment (positive, negative, neutral)
        style: Writing style preference (e.g., "pete_connor", "default", "custom")
        
    Returns:
        str: Content template
    """
    # Normalize platform name
    normalized_platform = platform.strip().title()
    
    # Get platform-specific templates or fall back to generic
    if normalized_platform in TEMPLATES:
        logger.debug(f"Using templates for platform: {normalized_platform}")
        templates = TEMPLATES[normalized_platform]
        
        # If pete_connor style is requested, filter for Pete Connor templates
        if style == "pete_connor" and normalized_platform in PETE_CONNOR_TEMPLATES:
            templates = PETE_CONNOR_TEMPLATES[normalized_platform]
    else:
        logger.warning(f"No templates found for platform: {platform}. Using generic templates.")
        templates = GENERIC_TEMPLATES
    
    # Select a random template
    selected_template = random.choice(templates)
    logger.debug(f"Selected template for {platform} with {sentiment} sentiment")
    
    return selected_template

def format_hashtags(keywords: List[str], count: int = 3) -> str:
    """
    Format keywords as hashtags.
    
    Args:
        keywords: List of keywords
        count: Maximum number of hashtags to include
        
    Returns:
        str: Formatted hashtags
    """
    # Limit the number of hashtags
    selected_keywords = keywords[:min(count, len(keywords))]
    
    # Format as hashtags
    hashtags = " ".join([f"#{keyword.replace(' ', '')}" for keyword in selected_keywords])
    
    return hashtags
