"""
Platform Adapter Module

This module transforms satirical content to meet the formatting requirements,
character limits, and user expectations of specific social media platforms. 
Each adapter creates platform-optimized content that maintains the satirical 
elements while conforming to platform-specific best practices.
"""

import logging
import re
import random
from datetime import datetime
from src.models.platform_specs import PLATFORM_SPECS

# Configure logging
logger = logging.getLogger("CANDOR.platform_adapter")

def adapt_for_platforms(transformed_content, sentiment_data, platforms):
    """
    Adapt transformed content for specific platforms
    
    This function routes content to the appropriate platform-specific adapter based
    on the requested platforms. Each adapter applies formatting, character limits,
    and structural changes that optimize the content for that platform.
    
    Args:
        transformed_content (dict): Content transformed by the CANDOR method
                                   (contains 'base_satire', 'exaggerated', 'subtle')
        sentiment_data (dict): Comprehensive sentiment analysis results
        platforms (list): List of target platforms
        
    Returns:
        dict: Platform-specific content versions mapped by platform name
    """
    results = {}
    
    # Extract content metadata for adapters
    content_metadata = _extract_content_metadata(transformed_content, sentiment_data)
    
    for platform in platforms:
        try:
            # Check if adapter exists for this platform
            if platform in PLATFORM_ADAPTERS:
                # Apply platform-specific adaptation with enhanced metadata
                results[platform] = PLATFORM_ADAPTERS[platform](
                    transformed_content, 
                    sentiment_data,
                    content_metadata
                )
                logger.info(f"Successfully adapted content for {platform}")
            else:
                logger.warning(f"No adapter available for platform: {platform}")
                results[platform] = f"Error: No adapter available for {platform}"
                
        except Exception as e:
            # Handle adaptation errors gracefully
            logger.error(f"Error adapting content for {platform}: {str(e)}")
            results[platform] = f"Error generating content for {platform}: {str(e)}"
    
    return results

def _extract_content_metadata(transformed_content, sentiment_data):
    """
    Extract useful metadata from the content and sentiment analysis
    for use by platform adapters
    
    Args:
        transformed_content (dict): Transformed content versions
        sentiment_data (dict): Sentiment analysis results
        
    Returns:
        dict: Content metadata for platform adaptation
    """
    # Get base content for analysis
    base_content = transformed_content.get('base_satire', '')
    
    # Extract paragraphs for structure analysis
    paragraphs = [p for p in base_content.split('\n\n') if p.strip()]
    
    # Get main topics and entities
    topics = sentiment_data.get('topics', [])
    if not topics and 'keywords' in sentiment_data:
        # Fall back to keywords if topics not available
        topics = sentiment_data.get('keywords', [])[:3]
        
    # Get entities if available
    entities = sentiment_data.get('entities', {})
    
    # Determine industry/domain
    industry = sentiment_data.get('industry', 'general')
    
    # Determine content length category
    if len(base_content) < 500:
        length_category = "short"
    elif len(base_content) < 1500:
        length_category = "medium"
    else:
        length_category = "long"
        
    # Generate dynamic title options based on topics and sentiment
    title_options = _generate_title_options(topics, sentiment_data)
    
    # Generate dynamic section headings
    section_headings = _generate_section_headings(topics, industry)
    
    return {
        'topics': topics,
        'entities': entities,
        'industry': industry,
        'length_category': length_category,
        'paragraph_count': len(paragraphs),
        'paragraphs': paragraphs,
        'title_options': title_options,
        'section_headings': section_headings
    }

def _generate_title_options(topics, sentiment_data):
    """Generate content-specific title options based on topics and sentiment"""
    title_options = []
    
    # Extract key information for title generation
    main_topic = topics[0].capitalize() if topics else "Corporate Communication"
    tone = sentiment_data.get('tone', 'neutral')
    
    # Basic title patterns
    title_options.append(f"The Absurdity of {main_topic}: A Satirical Analysis")
    
    # Add tone-specific titles
    if tone == 'positive':
        title_options.append(f"Too Good to Be True: The {main_topic} Hype Cycle")
        title_options.append(f"The Suspiciously Upbeat World of {main_topic}")
    elif tone == 'negative':
        title_options.append(f"The Problem with {main_topic} Nobody Talks About")
        title_options.append(f"When {main_topic} Goes Wrong: A Satirical Intervention")
    else:
        title_options.append(f"Decoding the {main_topic} Buzzword Bingo")
        title_options.append(f"{main_topic}: What They Say vs. What They Mean")
    
    # Add secondary topic if available
    if len(topics) > 1:
        second_topic = topics[1].capitalize()
        title_options.append(f"Where {main_topic} Meets {second_topic}: A Comedy of Errors")
    
    # Add industry-specific title if available
    industry = sentiment_data.get('industry', '')
    if industry and industry != 'general':
        title_options.append(f"The {industry.capitalize()} Industry's Obsession with {main_topic}")
    
    return title_options

def _generate_section_headings(topics, industry):
    """Generate content-specific section headings based on topics and industry"""
    headings = []
    
    # Basic section headings
    headings.append("The Art of Saying Nothing with Many Words")
    headings.append("Translating Corporate-Speak into Human Language")
    headings.append("What This Actually Means")
    headings.append("The Bottom Line")
    
    # Topic-based headings
    for topic in topics[:3]:
        topic_cap = topic.capitalize()
        headings.append(f"The {topic_cap} Paradox")
        headings.append(f"Unpacking the {topic_cap} Mythology")
        headings.append(f"When {topic_cap} Is Just a Fancy Label")
    
    # Industry-specific headings
    industry_headings = {
        'technology': [
            "Tech Jargon Decoded",
            "Silicon Valley Translation Service",
            "When Engineers Try Marketing"
        ],
        'business': [
            "MBA-Speak for the Rest of Us",
            "The Executive Summary of Nothing",
            "Boardroom Bingo Winners"
        ],
        'marketing': [
            "Marketing Fluff Detector",
            "Brand Nonsense Translated",
            "The Engagement Illusion"
        ],
        'healthcare': [
            "Medical Jargon Demystified",
            "Healthcare Promises vs. Reality",
            "The Wellness Industrial Complex"
        ],
        'finance': [
            "Financial Doublespeak Dictionary",
            "Money Talk Translator",
            "Banking Buzzwords Decoded"
        ]
    }
    
    # Add industry-specific headings if available
    if industry in industry_headings:
        headings.extend(industry_headings[industry])
    
    # Shuffle for variety but keep some structure
    random.shuffle(headings)
    
    return headings

def adapt_for_youtube(transformed_content, sentiment_data, content_metadata):
    """
    Create YouTube-optimized script and description
    
    Transforms content into a YouTube script format with clear sections,
    timestamps, call-to-action, and SEO-optimized description.
    
    Args:
        transformed_content (dict): Transformed content versions
        sentiment_data (dict): Sentiment analysis results
        content_metadata (dict): Additional content metadata
        
    Returns:
        str: Formatted YouTube script and description
    """
    # Use base satire for YouTube content
    content_base = transformed_content['base_satire']
    
    # Extract metadata for personalization
    topics = content_metadata['topics']
    industry = content_metadata['industry']
    paragraphs = content_metadata['paragraphs']
    
    # Generate title based on content
    title_options = content_metadata['title_options']
    video_title = title_options[0] if title_options else "Corporate Speak: A Satirical Analysis"
    
    # Create personalized video introduction
    intro_templates = [
        f"Hey everyone, welcome back to the channel! Today we're taking a satirical look at {topics[0] if topics else 'corporate communication'}.",
        f"What's up, content rebels! In this video, we're dissecting the world of {topics[0] if topics else 'business jargon'} with some much-needed humor.",
        f"Hello truth-seekers! Ready to decode the absurdity of {topics[0] if topics else 'corporate speak'}? Let's dive in!"
    ]
    intro = random.choice(intro_templates)
    
    # Create script sections
    script_parts = []
    
    # Add title and intro
    script_parts.append(f"TITLE: {video_title}")
    script_parts.append("\nINTRO:")
    script_parts.append(intro)
    
    # Add context based on topics
    if topics:
        context = f"\nCONTEXT:\nToday we're looking at {', '.join(topics[:2])} - topics that have generated endless buzzwords and corporate nonsense."
        script_parts.append(context)
    
    # Add main content sections
    script_parts.append("\nMAIN CONTENT:")
    
    # Use section headings for structure
    section_headings = content_metadata['section_headings']
    
    # Determine how many paragraphs to use based on content length
    if len(paragraphs) <= 3:
        # Use all paragraphs for short content
        content_paragraphs = paragraphs
    else:
        # Select key paragraphs for longer content
        content_paragraphs = [paragraphs[0]]  # Always include intro
        
        # Add middle paragraphs
        middle_idx = len(paragraphs) // 2
        content_paragraphs.append(paragraphs[middle_idx])
        
        # Add conclusion if more than 3 paragraphs
        if len(paragraphs) > 3:
            content_paragraphs.append(paragraphs[-1])
    
    # Create content sections with headings
    for i, para in enumerate(content_paragraphs):
        heading = section_headings[i % len(section_headings)]
        script_parts.append(f"\nSECTION {i+1}: {heading}")
        script_parts.append(para)
    
    # Add content-specific call to action
    cta_templates = [
        f"If you enjoyed this satirical take on {topics[0] if topics else 'corporate jargon'}, hit that like button and subscribe for more content that cuts through the nonsense!",
        f"Found this analysis helpful? Subscribe for weekly breakdowns of {industry} buzzwords and corporate absurdity!",
        "Want more corporate satire? Hit subscribe and let me know in the comments what other ridiculous business trends you want me to dissect next!"
    ]
    
    script_parts.append("\nCALL TO ACTION:")
    script_parts.append(random.choice(cta_templates))
    
    # Create full script
    script = "\n".join(script_parts)
    
    # Create SEO-optimized video description
    # Get metadata for description
    keywords = ", ".join(sentiment_data.get('keywords', [])[:5])
    hashtags = " ".join(sentiment_data.get('hashtags', [])[:5])
    
    # Calculate video structure with timestamps
    # Assuming 5 minutes total video length with appropriate divisions
    timestamps = [
        "0:00 - Introduction",
        "0:30 - Context and Background",
        "1:15 - " + (section_headings[0] if section_headings else "Main Analysis"),
        "3:00 - " + (section_headings[1] if len(section_headings) > 1 else "Key Takeaways"),
        "4:30 - Conclusion and Final Thoughts"
    ]
    
    # Create description with timestamps and SEO elements
    description = f"""
🎯 {video_title}

{content_paragraphs[0][:150]}... [Watch to see the full breakdown!]

TIMESTAMPS:
{timestamps[0]}
{timestamps[1]}
{timestamps[2]}
{timestamps[3]}
{timestamps[4]}

Keywords: {keywords}

Follow me on other platforms:
Twitter: @CPeteConnor
LinkedIn: /in/cpeteconnor
Medium: @cpeteconnor

{hashtags}

#Satire #{industry.capitalize()} #CorporateHumor #BusinessJargon
"""
    
    # Apply platform constraints (YouTube description maximum is ~5000 chars)
    if len(description) > 5000:
        description = description[:4997] + "..."
    
    # Combine script and description in a format ready for publishing
    youtube_content = f"VIDEO TITLE:\n{video_title}\n\nVIDEO SCRIPT:\n{'-'*80}\n{script}\n\n{'='*80}\n\nVIDEO DESCRIPTION:\n{'-'*80}\n{description}"
    
    return youtube_content

def adapt_for_medium(transformed_content, sentiment_data, content_metadata):
    """
    Create Medium-optimized blog post format
    
    Transforms content into a well-structured Medium article with proper
    headings, formatting, quotes, and SEO elements.
    
    Args:
        transformed_content (dict): Transformed content versions
        sentiment_data (dict): Sentiment analysis results
        content_metadata (dict): Additional content metadata
        
    Returns:
        str: Formatted Medium article
    """
    # Choose content version based on sentiment and length
    tone = sentiment_data.get('tone', 'neutral')
    length_category = content_metadata.get('length_category', 'medium')
    
    # Select content version based on tone and length
    if tone == 'positive' and length_category != 'short':
        # Use exaggerated version for positive, longer content
        content_base = transformed_content['exaggerated']
    elif length_category == 'short':
        # Use subtle version for short content
        content_base = transformed_content['subtle']
    else:
        # Use base satire for most content
        content_base = transformed_content['base_satire']
    
    # Extract metadata for personalization
    topics = content_metadata['topics']
    industry = content_metadata['industry']
    paragraphs = content_metadata['paragraphs']
    title_options = content_metadata['title_options']
    section_headings = content_metadata['section_headings']
    
    # Select a title that works well for Medium
    title = title_options[0] if title_options else "The Art of Corporate Nonsense: A Satirical Analysis"
    
    # Create subtitle using topics or industry
    subtitle_templates = [
        "Decoding corporate jargon one buzzword at a time",
        f"A satirical look at {industry} communication",
        f"When {topics[0] if topics else 'business-speak'} goes too far"
    ]
    subtitle = random.choice(subtitle_templates)
    
    # Create content sections based on length
    article_sections = []
    
    # Introduction always included
    introduction = paragraphs[0] if paragraphs else ""
    article_sections.append(("", introduction))  # No heading for intro
    
    # For medium articles, use proper section structure based on content length
    if length_category == 'short':
        # For short content, use minimal sections
        if len(paragraphs) > 1:
            article_sections.append((section_headings[0], paragraphs[1]))
        
        if len(paragraphs) > 2:
            article_sections.append(("The Bottom Line", paragraphs[2]))
            
    elif length_category == 'medium':
        # For medium content, use 2-3 sections
        sections_to_use = min(3, len(paragraphs) - 1)
        
        for i in range(sections_to_use):
            section_idx = i + 1  # Skip intro paragraph
            if section_idx < len(paragraphs):
                heading = section_headings[i % len(section_headings)]
                article_sections.append((heading, paragraphs[section_idx]))
        
    else:
        # For long content, use more sections with subheadings
        # Group paragraphs into logical sections
        remaining_paragraphs = paragraphs[1:]  # Skip intro
        sections_to_create = min(5, len(remaining_paragraphs))
        
        paragraphs_per_section = max(1, len(remaining_paragraphs) // sections_to_create)
        
        for i in range(sections_to_create):
            start_idx = i * paragraphs_per_section
            end_idx = start_idx + paragraphs_per_section
            
            if start_idx < len(remaining_paragraphs):
                heading = section_headings[i % len(section_headings)]
                
                # Combine paragraphs for this section
                section_content = "\n\n".join(remaining_paragraphs[start_idx:end_idx])
                article_sections.append((heading, section_content))
    
    # Create conclusion
    conclusion_templates = [
        "In conclusion, perhaps the most revolutionary business strategy would be simply communicating clearly. But where's the fun in that?",
        f"At the end of the day, {topics[0] if topics else 'corporate communication'} would benefit from more honesty and less jargon. But then what would satirists like me write about?",
        f"The real innovation in {industry} won't come from buzzwords, but from clear communication that actually means something. Revolutionary concept, I know."
    ]
    conclusion = random.choice(conclusion_templates)
    
    # Add conclusion as final section
    article_sections.append(("The Bottom Line", conclusion))
    
    # Add pull quote for visual interest (Medium best practice)
    if len(paragraphs) > 2:
        # Find a good sentence for pull quote
        potential_quotes = []
        for para in paragraphs[1:3]:  # Look in 2nd and 3rd paragraphs
            sentences = para.split('. ')
            potential_quotes.extend([s + '.' for s in sentences if 15 < len(s) < 120])
        
        pull_quote = random.choice(potential_quotes) if potential_quotes else ""
    else:
        pull_quote = ""
    
    # Add author bio with correct Medium formatting
    author_bio = """
---

*Written by C. Pete Connor, a satirist who spent too many years in corporate meeting rooms before escaping to write about their absurdity. Subscribe for weekly doses of corporate humor that might be too real.*
"""
    
    # Assemble full article with proper Medium formatting
    medium_article = f"""# {title}

### {subtitle}

{introduction}

"""
    
    # Add pull quote if available
    if pull_quote:
        medium_article += f"> {pull_quote}\n\n"
    
    # Add all sections with proper headings
    for heading, content in article_sections[1:]:  # Skip intro which is already added
        if heading:
            medium_article += f"## {heading}\n\n"
        medium_article += f"{content}\n\n"
    
    # Add bio and tags
    medium_article += f"{author_bio}\n\n"
    
    # Add SEO-friendly tags (Medium allows up to 5 tags)
    hashtags = sentiment_data.get('hashtags', [])
    tags = [tag.replace('#', '') for tag in hashtags[:5]]
    medium_article += f"Tags: {' '.join(tags)}"
    
    return medium_article

def adapt_for_linkedin(transformed_content, sentiment_data, content_metadata):
    """
    Create LinkedIn-optimized post format
    
    Transforms content into a professional LinkedIn post with appropriate
    length, formatting, and engagement elements.
    
    Args:
        transformed_content (dict): Transformed content versions
        sentiment_data (dict): Sentiment analysis results
        content_metadata (dict): Additional content metadata
        
    Returns:
        str: Formatted LinkedIn post
    """
    # LinkedIn posts work best with the subtle version
    content = transformed_content['subtle']
    
    # Extract metadata for personalization
    topics = content_metadata['topics']
    industry = content_metadata['industry']
    
    # LinkedIn has strict character limits (around 3000 for posts)
    # Aim for 1300-2000 chars for optimal LinkedIn engagement
    MAX_LINKEDIN_CHARS = 2000
    
    # If content is too long, intelligently trim it
    if len(content) > MAX_LINKEDIN_CHARS:
        # Try to find a good breakpoint (end of paragraph)
        breakpoint = content.rfind('\n\n', 0, MAX_LINKEDIN_CHARS - 100)
        
        if breakpoint > 0:
            content = content[:breakpoint] + "\n\n..."
        else:
            # If no good breakpoint, just trim at character limit
            content = content[:MAX_LINKEDIN_CHARS - 3] + "..."
    
    # Create professional framing based on topics/industry
    intro_templates = [
        f"Some thoughts on {topics[0] if topics else 'corporate communication'} trends:",
        f"I've been reflecting on {topics[0] if topics else 'business language'} recently:",
        f"An observation about {industry} communication that might resonate:",
        "Interesting perspective on professional communication patterns:"
    ]
    intro = random.choice(intro_templates)
    
    # Add topic-specific engagement question
    engagement_templates = [
        f"What's your experience with {topics[0] if topics else 'corporate communication'}? Have you noticed similar patterns?",
        f"Do you think we could communicate more effectively in {industry} contexts?",
        f"What's your approach to keeping {topics[0] if topics else 'business language'} clear and meaningful?",
        "How do you maintain authenticity in professional communications?",
        f"Have you found ways to simplify {industry} language in your organization?"
    ]
    question = random.choice(engagement_templates)
    
    # Create industry-specific hashtags for LinkedIn visibility
    linkedin_hashtags = {
        'technology': ['#TechLeadership', '#Innovation', '#DigitalTransformation'],
        'business': ['#BusinessStrategy', '#Leadership', '#ProfessionalDevelopment'],
        'marketing': ['#MarketingStrategy', '#ContentMarketing', '#BrandCommunication'],
        'healthcare': ['#HealthcareLeadership', '#MedicalCommunication', '#HealthTech'],
        'finance': ['#FinancialServices', '#BusinessInsights', '#FinancialStrategy'],
        'education': ['#EducationLeadership', '#ProfessionalLearning', '#EdTech']
    }
    
    # Get industry-specific hashtags or use general ones
    industry_tags = linkedin_hashtags.get(industry, ['#ProfessionalDevelopment', '#BusinessCommunication'])
    
    # Get content hashtags
    content_tags = sentiment_data.get('hashtags', [])[:2]
    
    # Combine and limit hashtags (LinkedIn best practice: 3-5 hashtags)
    hashtags = ' '.join(industry_tags + content_tags)
    
    # Format with line breaks and proper LinkedIn structure
    # LinkedIn posts perform better with short paragraphs and line breaks
    content_paragraphs = content.split('\n\n')
    
    # Format content with appropriate line breaks for LinkedIn readability
    formatted_content = intro + "\n\n"
    
    for para in content_paragraphs:
        # Keep paragraphs short for LinkedIn
        if len(para) > 150:
            # Split long paragraphs
            sentences = para.split('. ')
            current_para = ""
            
            for sentence in sentences:
                if not sentence.endswith('.'):
                    sentence += '.'
                
                if len(current_para) + len(sentence) < 150:
                    current_para += " " + sentence if current_para else sentence
                else:
                    formatted_content += current_para.strip() + "\n\n"
                    current_para = sentence
            
            if current_para:
                formatted_content += current_para.strip() + "\n\n"
        else:
            formatted_content += para + "\n\n"
    
    # Add engagement question and hashtags
    formatted_content += question + "\n\n" + hashtags
    
    return formatted_content

def adapt_for_substack(transformed_content, sentiment_data, content_metadata):
    """
    Create Substack-optimized newsletter format
    
    Transforms content into an email newsletter format with greeting,
    sections, and subscriber-focused elements.
    
    Args:
        transformed_content (dict): Transformed content versions
        sentiment_data (dict): Sentiment analysis results
        content_metadata (dict): Additional content metadata
        
    Returns:
        str: Formatted Substack newsletter
    """
    # Use base satire for newsletter content
    content = transformed_content['base_satire']
    
    # Extract metadata for personalization
    topics = content_metadata['topics']
    industry = content_metadata['industry']
    paragraphs = content_metadata['paragraphs']
    title_options = content_metadata['title_options']
    
    # Create engaging newsletter title
    title_templates = [
        f"This Week in {topics[0].capitalize() if topics else industry.capitalize()}: A Satirical Take",
        f"The {industry.capitalize()} Nonsense Translator: {datetime.now().strftime('%B %Y')} Edition",
        f"Decoding {topics[0].capitalize() if topics else 'Corporate'} Speak: This Week's Analysis"
    ]
    
    # Use content title or template
    title = title_options[0] if title_options else random.choice(title_templates)
    
    # Create personalized greeting
    greeting_templates = [
        "Hello Fellow Corporate Survivors,",
        f"Greetings, {industry.capitalize()} Truth-Seekers,",
        "Dear Readers Who Appreciate Straight Talk,"
    ]
    greeting = random.choice(greeting_templates)
    
    # Create newsletter intro based on topics
    intro_templates = [
        f"Welcome to this week's edition of 'Corporate Nonsense Translated.' Today, we're examining {topics[0] if topics else 'business communication'} that deserves our satirical attention.",
        f"Thanks for joining me for another round of {industry} satire. This week's specimen comes from the world of {topics[0] if topics else 'corporate jargon'}.",
        f"I hope your week has been jargon-free, but if not, you're in the right place. Today we're dissecting {topics[0] if topics else 'corporate language'} in all its absurd glory."
    ]
    intro = random.choice(intro_templates)
    
    # Structure newsletter content based on length
    newsletter_sections = []
    
    # This Week's Specimen section (main content)
    specimen_heading = "This Week's Specimen"
    
    # Select appropriate amount of content
    if len(paragraphs) <= 2:
        specimen_content = "\n\n".join(paragraphs)
    else:
        specimen_content = "\n\n".join(paragraphs[:2])
    
    newsletter_sections.append((specimen_heading, specimen_content))
    
    # What This Actually Means section
    meaning_templates = [
        f"When we strip away the jargon and the carefully constructed {industry}-speak, what we're really seeing is an attempt to make the ordinary sound extraordinary, the mundane sound revolutionary, and the questionable sound definitive.",
        f"Translated to human language, this is really just saying: '{_generate_plain_english_version(topics)}' But that wouldn't sound impressive enough in a {industry} meeting, would it?",
        f"The real message here, without all the fancy words, is about as groundbreaking as discovering that coffee is hot. But in the world of {topics[0] if topics else industry}, stating the obvious with complex terminology is considered visionary."
    ]
    
    meaning_heading = "What This Actually Means"
    meaning_content = random.choice(meaning_templates)
    newsletter_sections.append((meaning_heading, meaning_content))
    
    # Add This Week's Jargon Awards section (a Substack favorite)
    jargon_awards = _generate_jargon_awards(topics, industry)
    newsletter_sections.append(("This Week's Jargon Awards 🏆", jargon_awards))
    
    # Add subscriber exclusive if content is long enough
    if len(paragraphs) > 3:
        exclusive_heading = "For Paid Subscribers Only: The Director's Cut"
        exclusive_teaser = f"For those who've upgraded to paid subscriptions, I've included an extended analysis of {topics[0] if topics else 'this communication trend'}, with additional examples and a downloadable 'Corporate Bingo' card customized for {industry} meetings. [Preview below, full access for subscribers]"
        newsletter_sections.append((exclusive_heading, exclusive_teaser))
    
    # Create sign-off based on persona
    signoff_templates = [
        "Until next week,\n\nC. Pete Connor\nChief Jargon Deconstruction Officer",
        "Stay clear, stay direct,\n\nC. Pete Connor\nCorporate Translator",
        f"Fighting {industry} nonsense one newsletter at a time,\n\nC. Pete Connor"
    ]
    signoff = random.choice(signoff_templates)
    
    # Add witty unsubscribe notice (Substack allows custom unsubscribe messages)
    unsubscribe_templates = [
        "_To unsubscribe from this newsletter, simply respond with 'I prefer my corporate nonsense untranslated, thank you.'_",
        "_If this newsletter isn't adding value to your strategic paradigm, click unsubscribe. No hard feelings, we'll still synergize at the ideation phase._",
        f"_Unsubscribe if you must, but who else will translate {industry} jargon for you every week?_"
    ]
    unsubscribe = random.choice(unsubscribe_templates)
    
    # Assemble newsletter with proper Substack formatting
    newsletter = f"""# {title}

{greeting}

{intro}

"""
    
    # Add all sections
    for heading, content in newsletter_sections:
        newsletter += f"## {heading}\n\n{content}\n\n"
    
    # Add sign-off and unsubscribe
    newsletter += f"{signoff}\n\n{unsubscribe}"
    
    return newsletter

def _generate_plain_english_version(topics):
    """Generate a 'plain English' version of what corporate speak actually means"""
    if not topics:
        return "we're doing the same things everyone else is doing, but with a fancier name"
    
    templates = [
        f"we need to focus on {topics[0]} more effectively",
        f"our approach to {topics[0]} needs improvement",
        f"we should pay attention to {topics[0]} like our competitors are doing",
        f"{topics[0]} is important and we're not handling it well enough"
    ]
    
    return random.choice(templates)

def _generate_jargon_awards(topics, industry):
    """Generate satirical 'jargon awards' with content-specific examples"""
    # Industry-specific jargon examples
    industry_jargon = {
        'technology': ["digital transformation", "AI-powered", "blockchain-enabled", "cloud-native", "tech stack"],
        'business': ["synergy", "paradigm shift", "strategic alignment", "circle back", "deep dive"],
        'marketing': ["brand activation", "content strategy", "customer journey", "engagement metrics", "omnichannel"],
        'healthcare': ["patient-centered", "value-based care", "clinical workflows", "holistic approach", "care coordination"],
        'finance': ["financial wellness", "value proposition", "market disruption", "ROI optimization", "fiscal alignment"],
        'education': ["learning outcomes", "student engagement", "pedagogical innovation", "curricular alignment", "assessment metrics"]
    }
    
    # Get industry jargon or use general business jargon
    jargon_list = industry_jargon.get(industry, industry_jargon['business'])
    
    # Award templates with placeholders
    award_templates = [
        "**The 'Most Unnecessary Buzzword' Award**: Goes to \"{jargon}\" - because apparently \"{translation}\" wasn't fancy enough.",
        "**The 'Most Confusing Metaphor' Award**: \"{jargon} to {jargon2} on this {topic} initiative.\"",
        "**The 'Email That Could Have Been a Text' Award**: The 500-word message that ultimately just asked \"{simple_question}\"",
        "**The 'Stating the Obvious' Award**: For the revolutionary insight that \"{obvious_statement}\"",
        "**The 'Buzzword Bingo Champion' Award**: For fitting \"{jargon}\", \"{jargon2}\", and \"{jargon3}\" into a single sentence."
    ]
    
    # Generate 3 random awards
    awards = []
    templates_to_use = random.sample(award_templates, 3)
    
    for template in templates_to_use:
        # Select random jargon terms
        jargon = random.choice(jargon_list)
        jargon2 = random.choice([j for j in jargon_list if j != jargon])
        jargon3 = random.choice([j for j in jargon_list if j != jargon and j != jargon2])
        
        # Simple translations for jargon
        translations = {
            "digital transformation": "using computers",
            "AI-powered": "has if-statements",
            "blockchain-enabled": "has a database",
            "cloud-native": "uses the internet",
            "tech stack": "software we use",
            "synergy": "working together",
            "paradigm shift": "doing something different",
            "strategic alignment": "agreeing on stuff",
            "circle back": "talk later",
            "deep dive": "look closely",
            "brand activation": "marketing campaign",
            "content strategy": "what to post",
            "customer journey": "how people buy stuff",
            "engagement metrics": "likes and comments",
            "omnichannel": "being everywhere",
            "patient-centered": "caring about patients",
            "value-based care": "good healthcare",
            "clinical workflows": "how doctors work",
            "holistic approach": "looking at everything",
            "care coordination": "talking to each other",
            "financial wellness": "having enough money",
            "value proposition": "why people should care",
            "market disruption": "doing something different",
            "ROI optimization": "making more money",
            "fiscal alignment": "budget planning",
            "learning outcomes": "what students learn",
            "student engagement": "keeping students interested",
            "pedagogical innovation": "new teaching methods",
            "curricular alignment": "making classes work together",
            "assessment metrics": "test scores"
        }
        
        translation = translations.get(jargon, "using normal words")
        
        # Topic reference
        topic = topics[0] if topics else "core business"
        
        # Simple questions that get wrapped in corporate speak
        simple_questions = [
            "Are we still meeting today?",
            f"Have you finished the {topic} report?",
            "What's the deadline?",
            "Can you send me that file?",
            "Do you have time to talk this week?"
        ]
        simple_question = random.choice(simple_questions)
        
        # Obvious statements that get treated as insights
        obvious_statements = [
            f"customers prefer products that work correctly",
            f"employees are more productive when they're not constantly interrupted",
            f"projects finish faster when they have clear requirements",
            f"people don't read long emails carefully",
            f"meetings without agendas waste time"
        ]
        obvious_statement = random.choice(obvious_statements)
        
        # Format the award with all placeholders filled
        award = template.format(
            jargon=jargon,
            jargon2=jargon2,
            jargon3=jargon3,
            translation=translation,
            topic=topic,
            simple_question=simple_question,
            obvious_statement=obvious_statement
        )
        
        awards.append(award)
    
    # Join awards with line breaks and bullet points
    return "* " + "\n* ".join(awards)

# Register platform adapters
PLATFORM_ADAPTERS = {
    'YouTube': adapt_for_youtube,
    'Medium': adapt_for_medium,
    'LinkedIn': adapt_for_linkedin,
    'Substack': adapt_for_substack
}
