"""
Text Processor Module

This module handles direct text input processing for the CANDOR system, preparing
raw text for sentiment analysis and content transformation. It implements intelligent
text cleaning, structure preservation, and basic statistical analysis to ensure
optimal input quality for downstream processing.

The module focuses on maintaining the original meaning and intent of the text while
removing artifacts that could interfere with analysis, such as inconsistent formatting
or extraneous whitespace.
"""

import logging
import re
import string
from typing import Dict, Any, List, Tuple
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CANDOR.text_processor")

# Constants for text processing
MAX_LINE_LENGTH = 100  # Characters before suggesting line breaks for readability
JARGON_TERMS = [
    "synergy", "leverage", "optimize", "paradigm shift", "disrupt", "innovative",
    "solution", "deliverable", "actionable", "bandwidth", "ecosystem", "scalable",
    "robust", "streamline", "cutting-edge", "best practices", "thought leader",
    "value-add", "core competency", "move the needle", "drill down", "circle back",
    "low-hanging fruit", "holistic", "alignment", "stakeholder", "ideate", "agile",
    "pivot", "mission-critical"
]

def process_text(text: str) -> Dict[str, Any]:
    """
    Process raw text input to prepare it for analysis and transformation.
    
    This function cleans, normalizes, and analyzes text input to ensure
    high-quality content processing. It preserves meaningful structure
    while removing artifacts that could interfere with downstream analysis.
    
    Args:
        text (str): Raw text input from user
        
    Returns:
        Dict[str, Any]: Processed text with metadata including:
            - processed_text: Cleaned and normalized text
            - char_count: Character count
            - word_count: Word count
            - sentence_count: Sentence count
            - paragraph_count: Paragraph count
            - avg_sentence_length: Average sentence length (words)
            - readability_score: Simple readability metric
            - jargon_density: Percentage of corporate jargon terms
            - processing_notes: List of processing actions taken
    """
    if not text:
        raise ValueError("Empty text input provided. Please enter some text to analyze.")
    
    # Initialize result dictionary and processing notes
    result = {}
    processing_notes = []
    
    # Step 1: Basic cleaning
    original_length = len(text)
    processed_text = text.strip()
    processing_notes.append("Removed leading/trailing whitespace")
    
    # Step 2: Fix inconsistent line endings (normalize to \n)
    processed_text = processed_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Step 3: Preserve paragraph structure while removing excessive newlines
    processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
    if '\n\n\n' in text:
        processing_notes.append("Normalized paragraph breaks (3+ newlines → 2)")
    
    # Step 4: Remove excessive whitespace within lines
    processed_text = re.sub(r' {2,}', ' ', processed_text)
    if '  ' in text:
        processing_notes.append("Removed excessive spaces within text")
    
    # Step 5: Fix spacing after punctuation
    processed_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', processed_text)
    
    # Step 6: Analyze text structure
    stats = _analyze_text_structure(processed_text)
    result.update(stats)
    
    # Step 7: Check for readability issues
    readability_issues = _check_readability(processed_text)
    if readability_issues:
        processing_notes.extend(readability_issues)
    
    # Step 8: Detect jargon density
    jargon_info = _detect_jargon(processed_text)
    result.update(jargon_info)
    
    # Step 9: Suggest structural improvements if needed
    structure_suggestions = _suggest_structural_improvements(processed_text, stats)
    if structure_suggestions:
        processing_notes.extend(structure_suggestions)
    
    # Add processed text and notes to result
    result['processed_text'] = processed_text
    result['processing_notes'] = processing_notes
    
    # Calculate efficiency metrics
    char_reduction = original_length - len(processed_text)
    if char_reduction > 0:
        efficiency = (char_reduction / original_length) * 100
        processing_notes.append(f"Increased content efficiency by {efficiency:.1f}% ({char_reduction} chars removed)")
    
    # Log processing summary
    logger.info(f"Processed text input: {original_length} chars → {len(processed_text)} chars, "
               f"{stats['word_count']} words, {stats['paragraph_count']} paragraphs")
    
    return result

def _analyze_text_structure(text: str) -> Dict[str, Any]:
    """
    Analyze the structure of text to provide meaningful statistics.
    
    Args:
        text: Processed text to analyze
        
    Returns:
        Dict containing text structure statistics
    """
    # Split into paragraphs, sentences, and words
    paragraphs = text.split('\n\n')
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out empty elements
    paragraphs = [p for p in paragraphs if p.strip()]
    sentences = [s for s in sentences if s.strip()]
    
    # Calculate statistics
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    
    # Avoid division by zero
    avg_sentence_length = word_count / max(1, sentence_count)
    avg_paragraph_length = sentence_count / max(1, paragraph_count)
    
    # Calculate a simple readability score (based on avg sentence length)
    # Lower is easier to read
    readability_score = min(10, avg_sentence_length / 5 * 10)
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_paragraph_length': round(avg_paragraph_length, 1),
        'readability_score': round(readability_score, 1)
    }

def _check_readability(text: str) -> List[str]:
    """
    Check for common readability issues in text.
    
    Args:
        text: Processed text to check
        
    Returns:
        List of readability issues found
    """
    issues = []
    
    # Check for very long sentences (>40 words)
    sentences = re.split(r'[.!?]+', text)
    long_sentences = [s for s in sentences if len(s.split()) > 40 and s.strip()]
    if long_sentences:
        issues.append(f"Found {len(long_sentences)} very long sentences (>40 words)")
    
    # Check for very long paragraphs (>8 sentences)
    paragraphs = text.split('\n\n')
    for i, paragraph in enumerate(paragraphs):
        sentence_count = len(re.split(r'[.!?]+', paragraph))
        if sentence_count > 8:
            issues.append(f"Paragraph {i+1} contains {sentence_count} sentences (consider breaking up)")
    
    # Check for very long lines without breaks
    lines = text.split('\n')
    long_lines = [l for l in lines if len(l) > MAX_LINE_LENGTH]
    if long_lines:
        issues.append(f"Found {len(long_lines)} lines exceeding {MAX_LINE_LENGTH} characters")
    
    return issues

def _detect_jargon(text: str) -> Dict[str, Any]:
    """
    Detect and analyze corporate jargon in text.
    
    Args:
        text: Processed text to analyze
        
    Returns:
        Dict with jargon analysis
    """
    words = re.findall(r'\b\w+\b', text.lower())
    jargon_count = sum(1 for word in words if any(jargon.lower() in text.lower() for jargon in JARGON_TERMS))
    jargon_density = 0 if not words else (jargon_count / len(words)) * 100
    
    # Find specific jargon instances
    found_jargon = []
    for term in JARGON_TERMS:
        if term.lower() in text.lower():
            found_jargon.append(term)
    
    return {
        'jargon_count': jargon_count,
        'jargon_density': round(jargon_density, 1),
        'jargon_terms': found_jargon[:5]  # Limit to top 5 for brevity
    }

def _suggest_structural_improvements(text: str, stats: Dict[str, Any]) -> List[str]:
    """
    Suggest structural improvements based on text analysis.
    
    Args:
        text: Processed text
        stats: Text structure statistics
        
    Returns:
        List of improvement suggestions
    """
    suggestions = []
    
    # Suggest paragraph breaks for wall-of-text
    if stats['paragraph_count'] == 1 and stats['sentence_count'] > 5:
        suggestions.append("Consider adding paragraph breaks to improve readability")
    
    # Suggest subheadings for longer content
    if stats['word_count'] > 500 and '#' not in text and stats['paragraph_count'] > 3:
        suggestions.append("Consider adding subheadings to organize longer content")
    
    # Suggest bullet points for lists
    if re.search(r'\b(first|second|third|fourth|finally)\b', text.lower()):
        if not re.search(r'(\n\s*[-*]\s+)', text):
            suggestions.append("Consider using bullet points for list-like content")
    
    return suggestions

def process_text_for_platform(text: str, platform: str) -> Dict[str, Any]:
    """
    Process text with platform-specific optimizations.
    
    This function extends the basic text processing with additional
    platform-specific optimizations to better prepare content for
    a specific target platform.
    
    Args:
        text: Raw text input
        platform: Target platform name
        
    Returns:
        Dict containing processed text and platform-specific metadata
    """
    # First apply standard processing
    result = process_text(text)
    processed_text = result['processed_text']
    platform_notes = []
    
    # Apply platform-specific processing
    platform = platform.lower()
    
    if platform == 'linkedin':
        # LinkedIn benefits from professional language and clear structure
        if result['jargon_density'] > 5:
            platform_notes.append("Consider reducing jargon density for better LinkedIn engagement")
        
        # Check if text is too long for optimal LinkedIn viewing
        if result['char_count'] > 1300:
            platform_notes.append("Text exceeds optimal LinkedIn length (1300 chars); consider shortening")
            
        # Suggest adding a call-to-action if missing
        if not re.search(r'\b(agree|thoughts|comment|share|follow)\b', processed_text.lower()):
            platform_notes.append("Consider adding an engagement question/CTA for LinkedIn")
    
    elif platform == 'twitter':
        # Twitter has strict character limits
        if result['char_count'] > 280:
            platform_notes.append(f"Text exceeds Twitter's 280 character limit by {result['char_count'] - 280} chars")
            
        # Check hashtag usage
        hashtag_count = processed_text.count('#')
        if hashtag_count == 0:
            platform_notes.append("Consider adding 1-2 relevant hashtags for Twitter visibility")
        elif hashtag_count > 3:
            platform_notes.append("Too many hashtags for Twitter; consider reducing to 2-3 maximum")
    
    elif platform == 'medium':
        # Medium favors well-structured, substantial content
        if result['word_count'] < 300:
            platform_notes.append("Content may be too brief for Medium; consider expanding")
            
        # Check for subheadings in longer content
        if result['word_count'] > 700 and '#' not in processed_text:
            platform_notes.append("Consider adding headings for better structure on Medium")
            
        # Check for proper intro and conclusion
        paragraphs = processed_text.split('\n\n')
        if len(paragraphs) >= 3:
            if len(paragraphs[0].split()) < 40:
                platform_notes.append("Consider expanding introduction for Medium readers")
            if len(paragraphs[-1].split()) < 40:
                platform_notes.append("Consider adding a stronger conclusion for Medium")
    
    elif platform == 'substack':
        # Substack newsletters benefit from personal connection
        if not re.search(r'\b(hi|hello|greetings|dear)\b', processed_text.lower()):
            platform_notes.append("Consider adding a greeting for Substack newsletter format")
            
        # Check for sign-off
        if not re.search(r'\b(cheers|thanks|sincerely|regards|until next time)\b', processed_text.lower()):
            platform_notes.append("Consider adding a sign-off for Substack newsletter format")
    
    # Add platform-specific notes to result
    result['platform_specific_notes'] = platform_notes
    
    # Generate timestamp for processing record
    result['processing_timestamp'] = datetime.datetime.now().isoformat()
    result['target_platform'] = platform
    
    logger.info(f"Applied {platform}-specific optimizations to text input")
    return result

def identify_content_type(text: str) -> str:
    """
    Identify the likely content type based on text analysis.
    
    Args:
        text: Processed text to analyze
        
    Returns:
        str: Likely content type
    """
    # Count signal words/phrases for different content types
    signals = {
        'industry_insight': len(re.findall(r'\b(industry|trend|market|data shows|research|according to|study|report)\b', text.lower())),
        'opinion': len(re.findall(r'\b(i think|i believe|in my view|from my perspective|i disagree|i agree)\b', text.lower())),
        'how_to': len(re.findall(r'\b(how to|step|guide|tutorial|process|method|approach|strategy|tips|tricks)\b', text.lower())),
        'narrative': len(re.findall(r'\b(story|experience|journey|happened|when i|narrative|account)\b', text.lower())),
        'satire': len(re.findall(r'\b(ridiculous|absurd|ironic|satirical|humor|amusing|supposedly|allegedly)\b', text.lower())),
    }
    
    # Add structure-based signals
    paragraphs = text.split('\n\n')
    
    # How-to content often has numbered steps or bullet points
    if re.search(r'(\n\s*\d+\.|\n\s*[-*]\s+)', text):
        signals['how_to'] += 5
    
    # Narrative content often has temporal markers
    if re.search(r'\b(yesterday|today|last week|recently|ago|then|after that)\b', text.lower()):
        signals['narrative'] += 3
    
    # Industry insights often have statistics or citations
    if re.search(r'\b(\d+%|according to|cited|referenced)\b', text.lower()):
        signals['industry_insight'] += 3
    
    # Satire often has exaggeration or sarcasm markers
    if re.search(r'(!{2,}|\?{2,}|air quotes|supposedly|allegedly|revolutionary|game-changing|synergy)', text.lower()):
        signals['satire'] += 3
    
    # Return the content type with the highest signal count
    return max(signals.items(), key=lambda x: x[1])[0]

def extract_key_topics(text: str, max_topics: int = 3) -> List[str]:
    """
    Extract key topics from text using simplified NLP approach.
    
    Args:
        text: Processed text to analyze
        max_topics: Maximum number of topics to extract
        
    Returns:
        List of extracted topic strings
    """
    # In a production system, this would use proper NLP techniques
    # This is a simplified implementation
    
    # Remove common stopwords
    stopwords = {'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'what',
                'when', 'where', 'how', 'for', 'with', 'is', 'are', 'was', 'were', 
                'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'to', 'will',
                'would', 'should', 'can', 'could', 'of', 'at', 'by', 'about', 'like'}
    
    # Extract all words, lowercase
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(filtered_words)
    
    # Get the most common words
    common_words = word_counts.most_common(max_topics * 2)
    
    # Look for adjacent word pairs (bigrams) containing common words
    topics = []
    for word, _ in common_words[:max_topics]:
        # Look for word in context (simple bigram extraction)
        word_pattern = r'\b(' + word + r'\s+\w+|\w+\s+' + word + r')\b'
        bigram_matches = re.findall(word_pattern, text.lower())
        
        if bigram_matches:
            # Use the most common bigram
            bigram_counter = Counter(bigram_matches)
            most_common_bigram = bigram_counter.most_common(1)[0][0]
            topics.append(most_common_bigram)
        else:
            # Fall back to single word
            topics.append(word)
    
    # Ensure we have max_topics unique topics
    unique_topics = list(dict.fromkeys(topics))[:max_topics]
    
    return unique_topics