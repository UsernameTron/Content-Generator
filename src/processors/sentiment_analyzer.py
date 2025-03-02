"""
Sentiment Analyzer Module

This module provides comprehensive sentiment and content analysis using natural language
processing techniques. It determines emotional tone, extracts key topics, identifies
entities, and generates relevant hashtags based on actual content meaning.
"""

import logging
import re
import random
from collections import Counter

# Configure logging
logger = logging.getLogger("CANDOR.sentiment_analyzer")

# Initialize NLP components lazily to improve startup time
nltk_initialized = False
spacy_initialized = False
textblob_initialized = False

# Industry and domain dictionaries for detection
INDUSTRY_KEYWORDS = {
    'technology': [
        'software', 'hardware', 'app', 'algorithm', 'data', 'digital', 'tech', 'code',
        'programming', 'developer', 'startup', 'innovation', 'cloud', 'mobile',
        'interface', 'platform', 'network', 'device', 'computer', 'online'
    ],
    'business': [
        'business', 'company', 'corporate', 'management', 'executive', 'strategy',
        'leadership', 'market', 'finance', 'investment', 'revenue', 'profit', 'growth',
        'stakeholder', 'organization', 'enterprise', 'industry', 'commercial', 'client', 'customer'
    ],
    'marketing': [
        'marketing', 'brand', 'advertising', 'content', 'social media', 'campaign',
        'engagement', 'audience', 'consumer', 'conversion', 'customer', 'promotion',
        'target', 'communication', 'message', 'segment', 'demographic', 'outreach'
    ],
    'healthcare': [
        'health', 'medical', 'patient', 'doctor', 'hospital', 'clinical', 'care',
        'treatment', 'diagnosis', 'wellness', 'therapy', 'pharmaceutical', 'medicine',
        'symptom', 'disease', 'healthcare', 'provider', 'procedure', 'prescription'
    ],
    'education': [
        'education', 'student', 'learning', 'school', 'teacher', 'classroom', 'course',
        'curriculum', 'academic', 'university', 'college', 'teaching', 'knowledge',
        'training', 'skill', 'lesson', 'professor', 'faculty', 'degree', 'educational'
    ],
    'finance': [
        'finance', 'banking', 'investment', 'financial', 'stock', 'market', 'fund',
        'trading', 'investor', 'portfolio', 'asset', 'wealth', 'capital', 'currency',
        'retirement', 'loan', 'credit', 'debt', 'transaction', 'payment'
    ]
}

def _ensure_nltk_initialized():
    """Initialize NLTK resources on first use"""
    global nltk_initialized
    if not nltk_initialized:
        try:
            import nltk
            
            # Download necessary NLTK resources if not already available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
            try:
                nltk.data.find('chunkers/maxent_ne_chunker')
            except LookupError:
                nltk.download('maxent_ne_chunker', quiet=True)
                
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words', quiet=True)
                
            nltk_initialized = True
            logger.info("NLTK resources initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLTK: {e}. Using fallback analysis methods.")

def _ensure_textblob_initialized():
    """Initialize TextBlob on first use"""
    global textblob_initialized
    if not textblob_initialized:
        try:
            import textblob
            textblob_initialized = True
            logger.info("TextBlob initialized successfully")
        except ImportError:
            logger.error("TextBlob not available. Using fallback sentiment analysis.")

def _ensure_spacy_initialized():
    """Initialize spaCy on first use"""
    global spacy_initialized, nlp
    if not spacy_initialized:
        try:
            import spacy
            try:
                # Try to load the model, downloading if necessary
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                # If model not found, download it
                logger.info("Downloading spaCy model...")
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                              check=True)
                nlp = spacy.load("en_core_web_sm")
            
            spacy_initialized = True
            logger.info("spaCy initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {e}. Using fallback entity recognition.")

def analyze_sentiment(text):
    """
    Perform comprehensive sentiment and content analysis on text.
    
    This function analyzes text to determine emotional tone, extract key topics,
    identify entities, and generate relevant hashtags. It uses multiple NLP
    techniques to provide a rich analysis that captures the actual meaning and
    context of the content.
    
    Args:
        text (str): The text content to analyze
        
    Returns:
        dict: Analysis results containing:
            - sentiment_score: float from -1.0 (negative) to 1.0 (positive)
            - tone: str label (positive, neutral, negative)
            - topics: list of main topics in the content
            - entities: dict of entities by type (people, organizations, etc.)
            - keywords: list of important keywords
            - hashtags: list of relevant hashtags
            - industry: likely industry/domain of the content
            - corporate_jargon_level: measure of corporate jargon usage
    """
    if not text or not text.strip():
        raise ValueError("Empty text provided for analysis")
    
    # Track timing for performance monitoring
    import time
    start_time = time.time()
    
    # Initialize result dictionary
    analysis = {
        'sentiment_score': 0.0,
        'tone': 'neutral',
        'topics': [],
        'entities': {},
        'keywords': [],
        'hashtags': [],
        'industry': 'general',
        'corporate_jargon_level': 0.0
    }
    
    # 1. Sentiment Analysis
    sentiment_result = _analyze_sentiment_score(text)
    analysis['sentiment_score'] = sentiment_result['score']
    analysis['tone'] = sentiment_result['tone']
    
    # 2. Topic and Keyword Extraction
    topic_result = _extract_topics_and_keywords(text)
    analysis['topics'] = topic_result['topics']
    analysis['keywords'] = topic_result['keywords']
    
    # 3. Entity Recognition
    analysis['entities'] = _extract_entities(text)
    
    # 4. Industry Detection
    analysis['industry'] = _detect_industry(text, topic_result['keywords'])
    
    # 5. Jargon Analysis
    analysis['corporate_jargon_level'] = _measure_corporate_jargon(text)
    
    # 6. Generate Hashtags
    analysis['hashtags'] = _generate_hashtags(
        analysis['topics'], 
        analysis['keywords'],
        analysis['entities'],
        analysis['industry']
    )
    
    # Log performance
    elapsed_time = time.time() - start_time
    logger.info(f"Sentiment analysis completed in {elapsed_time:.2f}s: score={analysis['sentiment_score']:.2f}, tone={analysis['tone']}")
    
    return analysis

def _analyze_sentiment_score(text):
    """Determine sentiment score and tone using TextBlob or fallback method"""
    # Try using TextBlob for sentiment analysis
    try:
        _ensure_textblob_initialized()
        from textblob import TextBlob
        
        # TextBlob analysis
        blob = TextBlob(text)
        
        # Get polarity score (-1.0 to 1.0)
        score = blob.sentiment.polarity
        
        # Determine tone based on polarity
        if score > 0.15:
            tone = "positive"
        elif score < -0.15:
            tone = "negative"
        else:
            tone = "neutral"
            
        return {
            'score': score,
            'tone': tone,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    except Exception as e:
        logger.warning(f"TextBlob sentiment analysis failed: {e}. Using fallback method.")
        
        # Fallback to simple word counting
        return _fallback_sentiment_analysis(text)

def _fallback_sentiment_analysis(text):
    """Simple word-based sentiment analysis as fallback"""
    # Lists of sentiment words
    POSITIVE_WORDS = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'terrific', 'outstanding', 'superior', 'positive', 'nice', 'brilliant',
        'exceptional', 'perfect', 'incredible', 'love', 'happy', 'joy', 'excited',
        'inspiring', 'success', 'successful', 'innovative', 'progress', 'benefit',
        'improve', 'advantage', 'efficient', 'effective', 'solution', 'opportunity'
    ]

    NEGATIVE_WORDS = [
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative', 'disappointing',
        'failure', 'failed', 'mistake', 'problem', 'issue', 'trouble', 'difficult',
        'challenging', 'worst', 'inferior', 'mediocre', 'inadequate', 'subpar',
        'unacceptable', 'unhappy', 'angry', 'frustrated', 'sad', 'hate', 'dislike',
        'expensive', 'costly', 'waste', 'useless', 'broken', 'damage', 'harm'
    ]
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Handle negations by marking words that follow negations
    negations = ['not', 'no', "n't", 'never', 'neither', 'nor', 'hardly', 'barely']
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count positive and negative words, accounting for negations
    positive_count = 0
    negative_count = 0
    negation_active = False
    
    for i, word in enumerate(words):
        # Check if this word is a negation
        if word in negations:
            negation_active = True
            continue
            
        # Reset negation after 3 words
        if negation_active and i > 0 and i % 3 == 0:
            negation_active = False
            
        # Count sentiment based on word and negation status
        if word in POSITIVE_WORDS:
            if negation_active:
                negative_count += 1
            else:
                positive_count += 1
                
        elif word in NEGATIVE_WORDS:
            if negation_active:
                positive_count += 1
            else:
                negative_count += 1
    
    # Calculate sentiment score
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words == 0:
        score = 0.0
    else:
        score = (positive_count - negative_count) / total_sentiment_words
    
    # Determine tone
    if score > 0.15:
        tone = "positive"
    elif score < -0.15:
        tone = "negative"
    else:
        tone = "neutral"
        
    return {
        'score': score,
        'tone': tone,
        'subjectivity': 0.5  # Default subjectivity when using fallback
    }

def _extract_topics_and_keywords(text):
    """Extract main topics and important keywords from text"""
    try:
        _ensure_nltk_initialized()
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.tag import pos_tag
        from nltk.util import ngrams
        from string import punctuation
        
        # Tokenize text
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text.lower())
        
        # Get stop words and customize for our needs
        stop_words = set(stopwords.words('english'))
        stop_words.update(['would', 'could', 'should', 'will', 'may', 'many', 
                          'much', 'also', 'however', 'though', 'although'])
        
        # Filter out stop words, punctuation, and short words
        filtered_tokens = [word for word in tokens 
                          if word not in stop_words 
                          and word not in punctuation
                          and len(word) > 2]
        
        # Get POS tags for filtered tokens
        pos_tokens = pos_tag(filtered_tokens)
        
        # Extract nouns and noun phrases as potential topics/keywords
        important_tokens = [word for word, pos in pos_tokens 
                           if pos.startswith('NN') or pos.startswith('JJ')]
        
        # Count frequencies
        word_counter = Counter(important_tokens)
        
        # Extract bigrams (two-word phrases)
        bi_grams = list(ngrams(filtered_tokens, 2))
        bigram_counter = Counter(bi_grams)
        
        # Generate potential topics from most common nouns and bigrams
        topic_candidates = []
        
        # Add top nouns
        topic_candidates.extend([word for word, count in word_counter.most_common(10)
                               if len(word) > 3])
        
        # Add top bigrams as phrases
        topic_candidates.extend([' '.join(gram) for gram, count in bigram_counter.most_common(5)
                                if count > 1 and gram[0] not in stop_words and gram[1] not in stop_words])
        
        # Remove duplicates while preserving order
        seen = set()
        topics = [x for x in topic_candidates 
                 if not (x in seen or seen.add(x)) and len(x) > 3][:7]
        
        # Keywords are a combination of important words + POS tags
        keywords = []
        
        # Add important nouns
        for word, pos in pos_tokens:
            if pos.startswith('NN') and word not in keywords and len(word) > 3:
                keywords.append(word)
        
        # Add important adjectives
        for word, pos in pos_tokens:
            if pos.startswith('JJ') and word not in keywords and len(word) > 3:
                keywords.append(word)
        
        # Add important verbs
        for word, pos in pos_tokens:
            if pos.startswith('VB') and word not in keywords and len(word) > 3:
                keywords.append(word)
        
        # Limit keywords to top 15
        keywords = keywords[:15]
        
        return {
            'topics': topics,
            'keywords': keywords
        }
        
    except Exception as e:
        logger.warning(f"Advanced topic extraction failed: {e}. Using fallback method.")
        
        # Fallback to simple word frequency
        return _fallback_topic_extraction(text)

def _fallback_topic_extraction(text):
    """Simple frequency-based topic and keyword extraction as fallback"""
    # Common stop words to filter out
    common_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'it', 'for', 
                        'as', 'with', 'on', 'by', 'this', 'be', 'or', 'at', 'an', 'from',
                        'was', 'were', 'are', 'have', 'has', 'had', 'not', 'but', 'what',
                        'all', 'when', 'if', 'their', 'one', 'can', 'so', 'you', 'there'])
    
    # Convert to lowercase and tokenize
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Filter out common words and short words
    filtered_words = [word for word in words 
                     if word not in common_words and len(word) > 3]
    
    # Count frequencies
    word_freq = Counter(filtered_words)
    
    # Extract keywords (most common words)
    keywords = [word for word, count in word_freq.most_common(15)]
    
    # Use top keywords as topics
    topics = keywords[:7]
    
    return {
        'topics': topics,
        'keywords': keywords
    }

def _extract_entities(text):
    """Extract named entities (people, organizations, locations, etc.)"""
    # Try using spaCy for entity recognition
    try:
        _ensure_spacy_initialized()
        import spacy
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Organize entities by type
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'products': [],
            'other': []
        }
        
        # Map spaCy entity types to our categories
        type_mapping = {
            'PERSON': 'people',
            'ORG': 'organizations',
            'GPE': 'locations',
            'LOC': 'locations',
            'PRODUCT': 'products',
            'WORK_OF_ART': 'products'
        }
        
        # Extract and categorize entities
        for ent in doc.ents:
            category = type_mapping.get(ent.label_, 'other')
            # Avoid duplicates
            if ent.text not in entities[category]:
                entities[category].append(ent.text)
        
        return entities
        
    except Exception as e:
        logger.warning(f"SpaCy entity extraction failed: {e}. Using NLTK fallback.")
        
        # Try NLTK named entity recognition as fallback
        try:
            _ensure_nltk_initialized()
            import nltk
            from nltk import word_tokenize, pos_tag, ne_chunk
            from nltk.chunk import tree2conlltags
            
            # Tokenize, POS tag, and extract named entities
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            ne_tree = ne_chunk(pos_tags)
            
            # Convert tree to IOB tags
            iob_tags = tree2conlltags(ne_tree)
            
            # Organize entities by type
            entities = {
                'people': [],
                'organizations': [],
                'locations': [],
                'products': [],
                'other': []
            }
            
            # Extract entities based on IOB tags
            current_entity = []
            current_type = None
            
            for word, pos, tag in iob_tags:
                if tag.startswith('B-'):
                    # Start of a new entity
                    if current_entity:
                        entity_text = ' '.join(current_entity)
                        if current_type == 'PERSON':
                            entities['people'].append(entity_text)
                        elif current_type == 'ORGANIZATION':
                            entities['organizations'].append(entity_text)
                        elif current_type == 'GPE' or current_type == 'LOCATION':
                            entities['locations'].append(entity_text)
                        else:
                            entities['other'].append(entity_text)
                    
                    current_entity = [word]
                    current_type = tag[2:]  # Remove B- prefix
                    
                elif tag.startswith('I-') and current_entity:
                    # Continuation of current entity
                    current_entity.append(word)
                    
                elif tag == 'O':
                    # Outside any entity
                    if current_entity:
                        entity_text = ' '.join(current_entity)
                        if current_type == 'PERSON':
                            entities['people'].append(entity_text)
                        elif current_type == 'ORGANIZATION':
                            entities['organizations'].append(entity_text)
                        elif current_type == 'GPE' or current_type == 'LOCATION':
                            entities['locations'].append(entity_text)
                        else:
                            entities['other'].append(entity_text)
                        
                        current_entity = []
                        current_type = None
            
            # Handle any remaining entity
            if current_entity:
                entity_text = ' '.join(current_entity)
                if current_type == 'PERSON':
                    entities['people'].append(entity_text)
                elif current_type == 'ORGANIZATION':
                    entities['organizations'].append(entity_text)
                elif current_type == 'GPE' or current_type == 'LOCATION':
                    entities['locations'].append(entity_text)
                else:
                    entities['other'].append(entity_text)
            
            return entities
            
        except Exception as e2:
            logger.warning(f"NLTK entity extraction failed: {e2}. Using minimal entity extraction.")
            
            # Minimal entity extraction as last resort
            return {
                'people': [],
                'organizations': [],
                'locations': [],
                'products': [],
                'other': []
            }

def _detect_industry(text, keywords):
    """Detect likely industry or domain of the content"""
    text_lower = text.lower()
    
    # Count industry keywords in text
    industry_scores = {}
    
    for industry, terms in INDUSTRY_KEYWORDS.items():
        # Count keyword matches
        keyword_matches = sum(1 for term in terms if term in text_lower)
        
        # Count partial matches in phrases
        phrase_matches = sum(1 for term in terms for phrase in text_lower.split('.') if term in phrase)
        
        # Calculate industry score
        industry_scores[industry] = keyword_matches + (phrase_matches * 0.5)
    
    # Check keywords for additional signals
    for keyword in keywords:
        for industry, terms in INDUSTRY_KEYWORDS.items():
            if keyword in terms:
                industry_scores[industry] = industry_scores.get(industry, 0) + 1.5
    
    # If no clear industry detected, return general
    if not industry_scores or max(industry_scores.values()) < 2:
        return 'general'
    
    # Return industry with highest score
    return max(industry_scores.items(), key=lambda x: x[1])[0]

def _measure_corporate_jargon(text):
    """Measure the level of corporate jargon in the text"""
    # List of corporate jargon and buzzwords
    CORPORATE_JARGON = [
        'synergy', 'leverage', 'optimize', 'paradigm', 'disrupt', 'innovative',
        'solution', 'deliverable', 'actionable', 'bandwidth', 'ecosystem', 'scalable',
        'robust', 'streamline', 'cutting-edge', 'best practices', 'thought leader',
        'value-add', 'core competency', 'move the needle', 'drill down', 'circle back',
        'low-hanging fruit', 'holistic', 'alignment', 'stakeholder', 'ideate', 'agile',
        'pivot', 'mission-critical', 'strategy', 'strategic', 'engagement', 'transform',
        'empower', 'incentivize', 'monetize', 'disruptive', 'bleeding edge', 'synergize',
        'verticalization', 'touch base', 'out of the box', 'push the envelope',
        'game changer', 'change agent', 'deep dive', 'ecosystem', 'bandwidth',
        'value proposition', 'proactive', 'thought leadership', 'customer centric'
    ]
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count words
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    # Count jargon terms
    jargon_instances = 0
    for jargon_term in CORPORATE_JARGON:
        if ' ' in jargon_term:
            # Multi-word jargon
            jargon_instances += text_lower.count(jargon_term)
        else:
            # Single-word jargon
            jargon_instances += sum(1 for word in words if word == jargon_term)
    
    # Calculate jargon level (normalized to 0.0-1.0)
    jargon_level = min(1.0, jargon_instances / (total_words * 0.1))  # 10% max
    
    return jargon_level

def _generate_hashtags(topics, keywords, entities, industry):
    """Generate relevant hashtags based on content analysis"""
    hashtags = []
    
    # Add industry-specific hashtags
    INDUSTRY_HASHTAGS = {
        'technology': ['#TechTrends', '#Innovation', '#DigitalTransformation', '#TechLife'],
        'business': ['#BusinessStrategy', '#Leadership', '#Management', '#GrowthMindset'],
        'marketing': ['#MarketingTips', '#ContentMarketing', '#DigitalMarketing', '#BrandStrategy'],
        'healthcare': ['#HealthTech', '#MedicalInnovation', '#Healthcare', '#WellnessTrends'],
        'education': ['#EdTech', '#LearningJourney', '#Education', '#TeachingInnovation'],
        'finance': ['#FinTech', '#InvestmentStrategy', '#FinancialFreedom', '#MoneyMatters'],
        'general': ['#Insights', '#Perspective', '#ThoughtLeadership', '#TrendWatch']
    }
    
    # Add 1-2 industry hashtags
    industry_tags = INDUSTRY_HASHTAGS.get(industry, INDUSTRY_HASHTAGS['general'])
    hashtags.extend(random.sample(industry_tags, min(2, len(industry_tags))))
    
    # Add hashtags from top topics
    for topic in topics[:3]:
        if ' ' in topic:
            # Multi-word topic, camel case
            words = topic.split()
            hashtag = '#' + ''.join(word.capitalize() for word in words)
            hashtags.append(hashtag)
        else:
            # Single word topic, capitalize
            hashtags.append('#' + topic.capitalize())
    
    # Add entity hashtags (organizations, products)
    for org in entities.get('organizations', [])[:2]:
        words = org.split()
        if len(words) <= 3:  # Only use shorter organization names
            hashtag = '#' + ''.join(word.capitalize() for word in words)
            hashtags.append(hashtag)
    
    for product in entities.get('products', [])[:1]:
        words = product.split()
        if len(words) <= 2:  # Only use shorter product names
            hashtag = '#' + ''.join(word.capitalize() for word in words)
            hashtags.append(hashtag)
    
    # Add keyword-based hashtags
    remaining_slots = max(0, 8 - len(hashtags))
    if remaining_slots > 0 and keywords:
        for keyword in keywords[:remaining_slots]:
            if keyword not in [tag[1:].lower() for tag in hashtags]:  # Avoid duplicates
                hashtags.append('#' + keyword.capitalize())
    
    # Add satirical hashtags for corporate content
    SATIRICAL_HASHTAGS = [
        "#CorporateNonsense",
        "#JargonAlert",
        "#BusinessBuzzwords",
        "#PretendingToWork",
        "#MeetingsThatCouldBeEmails",
        "#ThoughtLeadershipThoughts",
        "#DisruptivelyObvious",
        "#ProfessionallyVague"
    ]
    
    # Add 1-2 satirical hashtags
    hashtags.extend(random.sample(SATIRICAL_HASHTAGS, min(2, len(SATIRICAL_HASHTAGS))))
    
    # Limit to 10 hashtags maximum to avoid hashtag spam
    return hashtags[:10]