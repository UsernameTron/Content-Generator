"""
Content transformer module that implements the CANDOR method
"""

import logging
import re
import random
import sys
import os

# Try to import the model-based content generator
try:
    from src.models.model_content_generator import ModelContentGenerator
    USE_MODEL = True
    # Initialize the model once
    model_generator = ModelContentGenerator(style="pete_connor")
except (ImportError, Exception) as e:
    print(f"Warning: Could not load model-based generator: {e}")
    print("Falling back to template-based generation")
    USE_MODEL = False
    model_generator = None

# Configure logging
logger = logging.getLogger("CANDOR.content_transformer")

# Satirical transition phrases - Pete Connor style
SATIRICAL_TRANSITIONS = [
    "In a shocking development that surprised absolutely no one with a functioning brain,",
    "Breaking news from the department of the painfully obvious that corporate America still doesn't get:",
    "Hold onto your ergonomic office chairs as I reveal what anyone with half a brain cell already knows:",
    "In what may be the least surprising revelation since we discovered executives actually don't deserve their salaries,",
    "According to experts who clearly needed a PhD to figure out what the janitor already knew:",
    "Prepare for the mind-shattering, earth-shaking revelation that somehow required a 6-figure consulting fee:",
    "In a twist that shocked precisely zero people who've ever worked in an office for more than 3 days,",
    "Let me channel my inner corporate bullshit translator for a moment:",
    "After countless hours spent in meaningless meetings, we've brilliantly concluded what was obvious from the start:",
    "Fresh from the land of overpriced consultants and PowerPoint abusers comes this astounding insight:",
    "Brace yourself for some weapons-grade corporate wisdom that somehow required seventeen meetings to produce:",
    "Stop the presses! After burning through their entire Q2 budget, management discovered what the interns knew day one:",
    "Alert the media! The C-suite geniuses earning 300x your salary just realized something any functioning adult already knew:",
    "After an exhaustive six-month study that could have been done in six minutes by anyone with common sense:"
]

# Corporate jargon replacements (original -> satirical) - Pete Connor style
JARGON_REPLACEMENTS = {
    "synergy": "bullshit corporate magic that doesn't actually exist",
    "leverage": "exploit mercilessly until completely drained of value",
    "optimize": "rearrange deck chairs on the Titanic until the deadline hits",
    "paradigm shift": "marginally different idea with an unnecessarily pretentious name",
    "disrupt": "piss off established players with a half-baked app",
    "innovative solution": "thing that already existed but we're claiming we invented it",
    "deliverable": "half-finished crap we'll convince you is worth paying for",
    "actionable": "vague enough to deflect all responsibility when it inevitably fails",
    "streamline": "fire enough people to hit quarterly numbers and secure executive bonuses",
    "cutting-edge": "barely functional technology our competitors mastered two years ago",
    "best practices": "obvious things we're pretending required expertise to figure out",
    "thought leader": "LinkedIn addict with a pathological need for validation",
    "value-add": "pointless feature nobody wanted that we'll use to justify a price increase",
    "core competency": "the one thing we suck at slightly less than everything else",
    "move the needle": "create an illusion of progress while fundamentally changing nothing",
    "drill down": "interrogate subordinates until finding someone to blame",
    "circle back": "postpone indefinitely in hopes you'll forget you ever asked",
    "low-hanging fruit": "blindingly obvious tasks we're embarrassingly late on addressing",
    "holistic approach": "making shit up as we go along but with a spiritual veneer",
    "alignment": "brutal process of crushing dissent until everyone parrots the boss's opinion",
    "stakeholder": "person with just enough power to derail everything but not enough to help",
    "ideate": "have a normal human thought but charge consulting rates for it",
    "agile": "chaotic development with enough jargon to sound intentional",
    "pivot": "abandon failing strategy while desperately pretending it was the plan all along",
    "mission-critical": "whatever random thing the CEO became obsessed with this week",
    "strategic": "ruinously expensive and impossible to measure the value of",
    "engagement": "tricking users into clicking things that make us money",
    "transformation": "expensive rebranding exercise that changes nothing fundamental",
    "utilize": "use (but said by someone who thinks they're too important to say 'use')",
    "incentivize": "manipulate with the smallest possible reward we can get away with",
    "onboard": "subject to soul-crushing orientation process that crushes all enthusiasm",
    "deep dive": "superficial glance at a spreadsheet with dramatic commentary",
    "think outside the box": "come up with ideas I can take credit for",
    "bandwidth": "capacity to tolerate additional abuse from management",
    "touch base": "interrupt your actual work for a pointless status update",
    "ecosystem": "collection of incompatible products we're forcing customers to use together",
    "customer-centric": "making decisions based entirely on profit while claiming it's for users",
    "scalable": "works for exactly three months until we need another complete rewrite",
    "takeaway": "obvious conclusion that didn't require a meeting",
    "visibility": "suffocating transparency allowing micromanagement from every direction",
    "proactive": "doing someone else's job before they fail at it themselves"
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
    # Try to use the model-based generator if available
    if USE_MODEL and model_generator is not None:
        try:
            # Generate using Pete Connor model for each version
            logger.info("Attempting to use model-based generation")
            base_model_content = model_generator.generate_content(content, "medium", sentiment=sentiment_data.get('tone', 'neutral'))[0]
            exaggerated_model_content = model_generator.generate_content(content, "medium", sentiment="negative")[0]
            subtle_model_content = model_generator.generate_content(content, "linkedin", sentiment="neutral")[0]
            
            if base_model_content and len(base_model_content) > 100:
                logger.info("Successfully used model-based generation")
                # Use model-generated content if successful
                return {
                    'base_satire': base_model_content,
                    'exaggerated': exaggerated_model_content,
                    'subtle': subtle_model_content
                }
            else:
                logger.warning("Model returned empty/short content, falling back to template")
        except Exception as e:
            logger.error(f"Model-based generation failed: {e}")
            logger.info("Falling back to template-based generation")
    
    # Template-based generation (fallback)
    # 1. BASE TRANSFORMATION
    # Start with a satirical introduction
    intro = random.choice(SATIRICAL_TRANSITIONS)
    
    # Replace corporate jargon with satirical alternatives
    transformed = content
    # Make sure we're actually replacing some jargon for humorous effect
    jargon_replacement_count = 0
    for jargon, replacement in JARGON_REPLACEMENTS.items():
        # Case-insensitive replacement that preserves capitalization
        pattern = re.compile(re.escape(jargon), re.IGNORECASE)
        original_text = transformed
        transformed = pattern.sub(lambda m: _match_case(m.group(0), replacement), transformed)
        # Count successful replacements
        if original_text != transformed:
            jargon_replacement_count += 1
    
    # If no jargon was found to replace, insert some funny corporate speak just to make sure we get some humor
    # Pete Connor style - more aggressive funny insertions and always insert at least one even if jargon was found
    funny_insertions = [
        " (which is just corporate-speak for 'doing the fucking obvious') ",
        " - a groundbreaking concept that absolutely no one with half a brain cell has thought of before - ",
        " (please pause while I feign excitement for this utterly mundane concept) ",
        " - and I use these buzzwords with all the soul-crushing corporate sincerity my dead-inside self can muster - ",
        " which is just a fancy way of saying we're doing the same damn thing as everyone else but spending 10x more on consultants ",
        " (pause for awkward executive chuckle that signals you should laugh too if you want that promotion) ",
        " — let's all pretend this is revolutionary and not something a halfway competent intern could have suggested — ",
        " *insert mandatory corporate enthusiasm here while died-inside employees nod along* ",
        " which suspiciously resembles what we mocked our competitors for doing last quarter ",
        " (and yes, they're seriously expecting us to be impressed by this) ",
        " — please contain your overwhelming excitement at this utterly pedestrian concept — ",
        " because apparently stating the obvious deserves a seven-figure bonus these days "
    ]
    
    # Insert at least one funny comment regardless of jargon count
    sentences = transformed.split('. ')
    if len(sentences) > 2:
        # Insert at least 1-2 funny comments depending on content length
        num_insertions = min(2, len(sentences) // 3)  # Approximately one comment per 3 sentences
        num_insertions = max(1, num_insertions)  # But at least one
        
        # Select random positions for insertions
        potential_positions = list(range(1, len(sentences) - 1))  # Avoid first and last sentences
        if potential_positions:
            insertion_positions = random.sample(potential_positions, min(num_insertions, len(potential_positions)))
            
            # Insert comments
            for pos in sorted(insertion_positions, reverse=True):  # Work backwards to maintain indices
                sentences[pos] = sentences[pos] + random.choice(funny_insertions)
                
        transformed = '. '.join(sentences)
    
    # Add satirical commentary based on sentiment - Pete Connor style
    if sentiment_data['tone'] == 'positive':
        commentary_options = [
            "\n\nIn other words, everything is absolutely perfect and we've achieved corporate nirvana. Just ignore the smell of burning money and the faint sound of employees updating their resumes.",
            "\n\nWow! It's amazing how perfect everything sounds when your bonus depends on pretending problems don't exist. I'm sure those 'minor issues' will just fix themselves!",
            "\n\nThere you have it, folks! A glowing success story unsullied by reality, metrics, or any form of objective measurement. Why let facts interfere with a good narrative?"
        ]
        commentary = random.choice(commentary_options)
    elif sentiment_data['tone'] == 'negative':
        commentary_options = [
            "\n\nSurprisingly, identifying problems and then doing absolutely nothing about them doesn't fix anything. Who could have possibly predicted this revolutionary insight?",
            "\n\nSo in summary: everything is broken, morale is in the toilet, but we're 'cautiously optimistic' about next quarter because admitting the truth would require actual leadership.",
            "\n\nThe good news is that we've successfully identified all the problems. The bad news is that fixing them would require admitting someone made a mistake, so we'll just rebrand them as 'growth opportunities' instead."
        ]
        commentary = random.choice(commentary_options)
    else:
        commentary_options = [
            "\n\nThis masterpiece of corporate communication has achieved the impossible: using hundreds of words to say absolutely nothing of substance. The pinnacle of modern business communication.",
            "\n\nI've seen more meaningful content generated by random word generators. At least those occasionally stumble into an accidental insight between the buzzwords.",
            "\n\nIf vague, non-committal corporate-speak were an Olympic sport, this would take gold, silver, AND bronze. A true masterclass in saying nothing with maximum verbosity."
        ]
        commentary = random.choice(commentary_options)
    
    # Add random satirical hashtags - Pete Connor style
    satirical_hashtags = [
        "#CorporateBullshitTranslator",
        "#BuzzwordBingoChampion",
        "#ThisMeetingCouldHaveBeenAnEmail",
        "#PretendingToWorkSince9AM",
        "#BullshitAsAService",
        "#ThoughtLeadershipIsMadeUp",
        "#PowerPointPurgatory",
        "#ExecutiveTimeWasting",
        "#CorpSpeak",
        "#JargonDetox",
        "#BrainDeadManagement",
        "#ConsultantScam",
        "#DeathByPowerPoint",
        "#LinkedInLunacy",
        "#QuarterlyResultsDesperation",
        "#MiddleManagementHell",
        "#MissionStatementMadLibs",
        "#KPIKillingProductivity",
        "#VisionaryBullshit",
        "#StakeholderStockholmSyndrome",
        "#DisruptionMyAss",
        "#InnovationTheaterProductions"
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
