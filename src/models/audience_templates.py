"""
Audience-specific templates for adapting content to different audience types:
- Executive: Senior decision-makers, strategic focus, metrics-oriented
- Practitioner: Technical implementers, detailed focus, methodology-oriented
- General Public: Non-technical consumers, simplified concepts, analogy-oriented
"""

import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Audience conditioning tokens
AUDIENCE_TOKENS = {
    "executive": "[EXEC]",
    "practitioner": "[PRAC]",
    "general": "[GEN]"
}

# Default audience if none specified
DEFAULT_AUDIENCE = "practitioner"

# Audience-specific templates by platform and content type
AUDIENCE_TEMPLATES = {
    "executive": {
        "LinkedIn": [
            "{audience_token} Key finding on {topic}: {main_point}\n\nBusiness impact:\n{supporting_points}\n\nStrategic implication: {call_to_action}\n\n{hashtags}",
            "{audience_token} Executive summary: {topic}\n\nMarket position: {main_point}\n\nMetrics that matter:\n{supporting_points}\n\nRecommended action: {call_to_action}\n\n{hashtags}",
            "{audience_token} {topic} performance metrics:\n\n{main_point}\n\nComparative analysis:\n{supporting_points}\n\nROI opportunity: {call_to_action}\n\n{hashtags}"
        ],
        "Twitter": [
            "{audience_token} {topic} insight: {main_point} Key metric: {supporting_points_brief} {hashtags}",
            "{audience_token} Strategic take on {topic}: {main_point} Business impact: {supporting_points_brief} {hashtags}"
        ],
        "Blog": [
            "{audience_token} # Executive Briefing: {topic}\n\n## Strategic Overview\n{main_point}\n\n## Performance Metrics\n{supporting_points}\n\n## Business Implications\n{implications}\n\n## Action Plan\n{conclusion}",
            "{audience_token} # {topic}: The Executive Perspective\n\n{introduction}\n\n## Market Position\n{main_point}\n\n## Performance Indicators\n{supporting_points}\n\n## Strategic Options\n{implications}\n\n## Recommended Approach\n{conclusion}"
        ],
        "Email Newsletter": [
            "{audience_token} Subject: {title} - Executive Summary\n\nDear Leadership Team,\n\n{introduction}\n\n# {topic} - Strategic Analysis\n\n{main_point}\n\n# Key Performance Indicators\n\n{supporting_points}\n\n# ROI Considerations\n\n{key_points}\n\n{call_to_action}\n\nRegards,\nC. Pete Connor"
        ]
    },
    "practitioner": {
        "LinkedIn": [
            "{audience_token} Technical deep-dive: {topic}\n\n{main_point}\n\nImplementation considerations:\n{supporting_points}\n\nBest practices: {call_to_action}\n\n{hashtags}",
            "{audience_token} {topic} implementation analysis:\n\n{main_point}\n\nTechnical specifications:\n{supporting_points}\n\nMethodology recommendation: {call_to_action}\n\n{hashtags}",
            "{audience_token} Hands-on with {topic}:\n\n{main_point}\n\nTechnical findings:\n{supporting_points}\n\nPractical approach: {call_to_action}\n\n{hashtags}"
        ],
        "Twitter": [
            "{audience_token} {topic} implementation tip: {main_point} Technical detail: {supporting_points_brief} {hashtags}",
            "{audience_token} For practitioners working with {topic}: {main_point} Method: {supporting_points_brief} {hashtags}"
        ],
        "Blog": [
            "{audience_token} # Technical Guide: Implementing {topic}\n\n## Core Methodology\n{main_point}\n\n## Technical Specifications\n{supporting_points}\n\n## Implementation Steps\n{implications}\n\n## Validation Approach\n{conclusion}",
            "{audience_token} # {topic}: A Practitioner's Handbook\n\n{introduction}\n\n## Technical Foundation\n{main_point}\n\n## Implementation Details\n{supporting_points}\n\n## Problem-Solving Approaches\n{implications}\n\n## Deployment Considerations\n{conclusion}"
        ],
        "Email Newsletter": [
            "{audience_token} Subject: {title} - Technical Implementation Guide\n\nFellow Practitioners,\n\n{introduction}\n\n# {topic} - Technical Foundation\n\n{main_point}\n\n# Implementation Specifics\n\n{supporting_points}\n\n# Common Challenges & Solutions\n\n{key_points}\n\n{call_to_action}\n\nHappy building,\nC. Pete Connor"
        ]
    },
    "general": {
        "LinkedIn": [
            "{audience_token} Let's talk about {topic} in simple terms:\n\n{main_point}\n\nHere's what this means for you:\n{supporting_points}\n\nTake-home message: {call_to_action}\n\n{hashtags}",
            "{audience_token} Breaking down {topic} into everyday language:\n\n{main_point}\n\nWhat you need to know:\n{supporting_points}\n\nSimple next step: {call_to_action}\n\n{hashtags}",
            "{audience_token} {topic} simplified:\n\n{main_point}\n\nReal-world examples:\n{supporting_points}\n\nEveryday application: {call_to_action}\n\n{hashtags}"
        ],
        "Twitter": [
            "{audience_token} {topic} explained simply: {main_point} In everyday terms: {supporting_points_brief} {hashtags}",
            "{audience_token} What everyone should know about {topic}: {main_point} {hashtags}"
        ],
        "Blog": [
            "{audience_token} # {topic} Explained: No Technical Background Required\n\n## The Simple Version\n{main_point}\n\n## Real-World Examples\n{supporting_points}\n\n## Why This Matters To You\n{implications}\n\n## The Bottom Line\n{conclusion}",
            "{audience_token} # Understanding {topic} in Everyday Terms\n\n{introduction}\n\n## The Basics\n{main_point}\n\n## How It Works (Simplified)\n{supporting_points}\n\n## What This Means For You\n{implications}\n\n## Taking Action\n{conclusion}"
        ],
        "Email Newsletter": [
            "{audience_token} Subject: {title} - Explained Simply\n\nHi there,\n\n{introduction}\n\n# {topic} - The Basics\n\n{main_point}\n\n# Everyday Examples\n\n{supporting_points}\n\n# Why This Matters\n\n{key_points}\n\n{call_to_action}\n\nSimply put,\nC. Pete Connor"
        ]
    }
}

def get_audience_template(platform: str, audience: Optional[str] = None) -> str:
    """
    Get a template appropriate for the specified audience and platform.
    
    Args:
        platform: Target platform (LinkedIn, Twitter, etc.)
        audience: Target audience (executive, practitioner, general)
        
    Returns:
        str: Audience-appropriate content template
    """
    # Normalize platform name
    normalized_platform = platform.strip().title()
    
    # Use default audience if none specified
    if not audience:
        audience = DEFAULT_AUDIENCE
    
    # Normalize audience name
    audience = audience.lower()
    if audience not in AUDIENCE_TOKENS:
        logger.warning(f"Unknown audience: {audience}. Using default: {DEFAULT_AUDIENCE}")
        audience = DEFAULT_AUDIENCE
    
    # Get audience token
    audience_token = AUDIENCE_TOKENS[audience]
    
    # Get templates for the specified audience and platform
    if audience in AUDIENCE_TEMPLATES and normalized_platform in AUDIENCE_TEMPLATES[audience]:
        templates = AUDIENCE_TEMPLATES[audience][normalized_platform]
        if templates:
            import random
            template = random.choice(templates)
            # Add audience token if not already in the template
            if "{audience_token}" in template:
                return template
            else:
                return f"{audience_token} {template}"
    
    # Fall back to regular templates with audience token prepended
    from .templates import get_template
    generic_template = get_template(platform)
    return f"{audience_token} {generic_template}"

def get_audience_description(audience: str) -> str:
    """
    Get a description of the specified audience for guidance.
    
    Args:
        audience: Target audience (executive, practitioner, general)
        
    Returns:
        str: Description of the audience characteristics
    """
    descriptions = {
        "executive": (
            "Content for senior decision-makers focusing on strategic implications, "
            "business metrics, ROI, and high-level insights. Emphasize bottom-line impact, "
            "competitive advantage, and market positioning. Use concise language and "
            "highlight quantitative results."
        ),
        "practitioner": (
            "Content for technical implementers focusing on methodologies, technical "
            "specifications, and hands-on guidance. Include specific details on implementation "
            "approaches, technical challenges, and best practices. Use domain-specific "
            "terminology appropriate for subject matter experts."
        ),
        "general": (
            "Content for non-technical audiences focusing on simplification, real-world "
            "examples, and practical applications. Avoid jargon, use analogies, and connect "
            "concepts to everyday experiences. Emphasize why the topic matters and what "
            "actions individuals can take."
        )
    }
    
    audience = audience.lower()
    if audience in descriptions:
        return descriptions[audience]
    else:
        return descriptions[DEFAULT_AUDIENCE]

def get_all_audience_types() -> List[str]:
    """
    Get all available audience types.
    
    Returns:
        List[str]: List of available audience types
    """
    return list(AUDIENCE_TOKENS.keys())
