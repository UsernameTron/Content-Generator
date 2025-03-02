"""
Content transformer module that implements the CANDOR method
"""

import logging
import re
import random

# Configure logging
logger = logging.getLogger("CANDOR.content_transformer")

# Satirical transition phrases
SATIRICAL_TRANSITIONS = [
    "In a shocking development that surprised absolutely no one,",
    "Breaking news from the department of the painfully obvious:",
    "Prepare to have your mind marginally nudged, if not completely blown:",
    "In what may be the least surprising revelation since water was found to be wet,",
    "According to experts who clearly needed a PhD to figure this out,",
    "Brace yourself for this groundbreaking revelation:",
    "In a twist that shocked precisely zero people,",
    "Channeling my inner captain obvious for a moment,",
    "Scientists have confirmed what your grandmother already knew:",
    "After extensive research that could have been replaced by common sense,"
]

# Corporate jargon replacements (original -> satirical)
JARGON_REPLACEMENTS = {
    "synergy": "magical business alchemy",
    "leverage": "squeeze until it hurts",
    "optimize": "fiddle with until deadline",
    "paradigm shift": "slightly different idea with a fancy name",
    "disrupt": "mildly inconvenience",
    "innovative solution": "thing that already existed but with a new logo",
    "deliverable": "thing we promised but might not deliver",
    "actionable": "vague enough to avoid accountability",
    "streamline": "fire people",
    "cutting-edge": "released by our competitors two years ago",
    "best practices": "things other companies do that we copy",
    "thought leader": "person with a Medium account",
    "value-add": "feature nobody asked for",
    "core competency": "the one thing we're not terrible at",
    "move the needle": "cause a barely perceptible change",
    "drill down": "ask increasingly uncomfortable questions",
    "circle back": "postpone indefinitely",
    "low-hanging fruit": "easy stuff we should have done already",
    "holistic approach": "making it up as we go along",
    "alignment": "forcing everyone to agree with the boss",
    "stakeholder": "person who will complain if not consulted",
    "ideate": "have a thought like a normal human being",
    "agile": "no documentation or planning",
    "pivot": "abandon failing strategy while pretending it was intentional",
    "mission-critical": "thing the CEO mentioned once in passing",
    "strategic": "expensive",
    "engagement": "clicks, hopefully",
    "transformation": "new logo and website",
    "utilize": "use",
    "incentivize": "bribe",
    "onboard": "trap in orientation meetings"
}

def transform_content(content, sentiment_data):
    """
    Transform content using the CANDOR method:
    C - Contextualize the content for satire
    A - Amplify corporate jargon to highlight absurdity
    N - Neutralize PR-speak with candid alternatives
    D - Dramatize statistics and claims
    O - Overstate importance of trivial details
    R - Reframe from an irreverent perspective
    
    Args:
        content (str): The original content to transform
        sentiment_data (dict): Sentiment analysis results
        
    Returns:
        dict: Transformed content with various versions:
            - base_satire: Base satirical version
            - exaggerated: More extreme satirical version
            - subtle: More subtle satirical version
    """
    # 1. BASE TRANSFORMATION
    # Start with a satirical introduction
    intro = random.choice(SATIRICAL_TRANSITIONS)
    
    # Replace corporate jargon with satirical alternatives
    transformed = content
    for jargon, replacement in JARGON_REPLACEMENTS.items():
        # Case-insensitive replacement that preserves capitalization
        pattern = re.compile(re.escape(jargon), re.IGNORECASE)
        transformed = pattern.sub(lambda m: _match_case(m.group(0), replacement), transformed)
    
    # Add satirical commentary based on sentiment
    if sentiment_data['tone'] == 'positive':
        commentary = "\n\nIn other words, everything is absolutely perfect and there are definitely no underlying issues whatsoever. None. Zero. Don't even think about it."
    elif sentiment_data['tone'] == 'negative':
        commentary = "\n\nSurprisingly, complaining about problems doesn't automatically solve them. Who knew?"
    else:
        commentary = "\n\nSomehow managing to say a lot while communicating almost nothing of substance. Impressive."
    
    # Add random satirical hashtags
    satirical_hashtags = [
        "#CorporateNonsense",
        "#BusinessFluff",
        "#JargonBingo",
        "#PretendingToWork",
        "#MeetingsThatCouldBeEmails",
        "#DisruptivelyObvious",
        "#ThoughtLeadershipThoughts",
        "#ProfessionallyVague"
    ]
    random.shuffle(satirical_hashtags)
    hashtag_selection = " ".join(satirical_hashtags[:3] + sentiment_data['hashtags'][:2])
    
    # Assemble base satire
    base_satire = f"{intro}\n\n{transformed}{commentary}\n\n{hashtag_selection}"
    
    # 2. CREATE EXAGGERATED VERSION
    # More extreme version with additional satirical elements
    exaggerated_intro = f"BREAKING: In a development that has literally changed the course of human history (or at least that's what the press release claims),\n\n"
    
    # Exaggerate statistics and claims
    exaggerated = transformed
    exaggerated = re.sub(r'(\d+)%', lambda m: f"a WHOPPING {m.group(1)}% (give or take 50%)", exaggerated)
    exaggerated = re.sub(r'significant', "EARTH-SHATTERING", exaggerated, flags=re.IGNORECASE)
    exaggerated = re.sub(r'important', "REVOLUTIONARY", exaggerated, flags=re.IGNORECASE)
    
    # Overstate importance
    exaggerated_commentary = "\n\nThis information is so incredibly important that you should probably stop whatever you're doing, even if it's brain surgery or defusing a bomb, to fully absorb these GAME-CHANGING insights."
    
    # Assemble exaggerated version
    exaggerated_version = f"{exaggerated_intro}{exaggerated}{exaggerated_commentary}\n\n{hashtag_selection} #LifeChanging #MindBlown"
    
    # 3. CREATE SUBTLE VERSION
    # More subtle version with less obvious satire
    subtle_intro = "Interesting development in the world of corporate communication:"
    
    # Subtle jargon replacement (only replace some instances)
    subtle = content
    for jargon, replacement in list(JARGON_REPLACEMENTS.items())[:10]:  # Use fewer replacements
        # Only replace some instances (50% chance for each match)
        pattern = re.compile(re.escape(jargon), re.IGNORECASE)
        subtle = pattern.sub(lambda m: _match_case(m.group(0), replacement) if random.random() > 0.5 else m.group(0), subtle)
    
    # Subtle commentary as a "note"
    subtle_commentary = "\n\nNote: One can't help but wonder if there might be a slightly more straightforward way to express these ideas."
    
    # Assemble subtle version
    subtle_version = f"{subtle_intro}\n\n{subtle}{subtle_commentary}\n\n{' '.join(sentiment_data['hashtags'][:3])}"
    
    logger.info("Content transformed successfully using CANDOR method")
    
    return {
        'base_satire': base_satire,
        'exaggerated': exaggerated_version,
        'subtle': subtle_version
    }

def _match_case(original, replacement):
    """Match the case pattern of the original word in the replacement"""
    # If original is all uppercase, convert replacement to uppercase
    if original.isupper():
        return replacement.upper()
    # If original is title case, convert replacement to title case
    elif original[0].isupper():
        return replacement.capitalize()
    # Otherwise, use lowercase
    else:
        return replacement.lower()
