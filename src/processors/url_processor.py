"""
URL Processor Module

This module extracts and analyzes content from web pages for the CANDOR system.
It implements intelligent web scraping, content extraction, and preprocessing
to transform online content into a format suitable for satirical transformation.

The module handles various types of web content differently based on domain
patterns, content structure, and semantic analysis, ensuring appropriate 
extraction of the core message while filtering out noise like navigation,
footers, and advertisements.
"""

import logging
import re
import urllib.parse
import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CANDOR.url_processor")

# Constants for content extraction
MAX_TITLE_LENGTH = 100
MAX_CONTENT_LENGTH = 10000
ARTICLE_INDICATORS = ["article", "post", "story", "news", "blog", "release"]
COMMERCE_INDICATORS = ["product", "shop", "store", "buy", "purchase", "order"]
CORPORATE_INDICATORS = ["about", "company", "team", "careers", "leadership"]

# Corporate jargon patterns to detect in web content
JARGON_PATTERNS = [
    r'\b(synerg(y|ies|istic))\b',
    r'\b(leverage(d|s)?)\b',
    r'\b(optimiz(e|ing|ed|ation))\b',
    r'\b(paradigm shift(s|ing)?)\b',
    r'\b(disrupt(ive|ion|ing|ed)?)\b',
    r'\b(innovat(e|ion|ing|ive|ively|or|ors)?)\b',
    r'\b(solution(s)?)\b',
    r'\b(deliverable(s)?)\b',
    r'\b(actionable)\b',
    r'\b(stakeholder(s)?)\b',
    r'\b(best practi(ce|ces))\b',
    r'\b(thought leader(ship)?)\b',
    r'\b(value[- ]add(ed)?)\b',
    r'\b(core competenc(y|ies))\b',
    r'\b(move the needle)\b',
    r'\b(circle back)\b',
    r'\b(low[- ]hanging fruit)\b',
    r'\b(holistic approach)\b',
    r'\b(robust)\b',
    r'\b(scalable)\b',
    r'\b(agile)\b',
    r'\b(customer[- ]centric)\b',
    r'\b(streamline(d|s)?)\b'
]

class ContentExtractionError(Exception):
    """Exception raised for content extraction errors."""
    pass

class InvalidURLError(Exception):
    """Exception raised for invalid URL format or structure."""
    pass

def process_url(url: str) -> Dict[str, Any]:
    """
    Extract and analyze content from a URL.
    
    This function validates, processes, and extracts content from a web URL,
    preparing it for downstream analysis and transformation. It handles
    different types of web content appropriately based on the URL structure
    and domain type.
    
    Args:
        url (str): URL to extract content from
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - title: Extracted title
            - content: Extracted main content
            - domain: Source domain
            - content_type: Detected content type
            - word_count: Count of words in extracted content
            - jargon_count: Count of corporate jargon terms
            - jargon_density: Percentage of text that is corporate jargon
            - extraction_timestamp: When the content was extracted
            - author: Author if detected, otherwise None
            - publish_date: Publication date if detected, otherwise None
            - metadata: Additional metadata about the content
            
    Raises:
        InvalidURLError: When the URL is empty or malformed
        ContentExtractionError: When content cannot be extracted
    """
    if not url:
        raise InvalidURLError("Empty URL provided. Please enter a valid URL to process.")
    
    # Step 1: Validate and normalize URL
    normalized_url = _normalize_url(url)
    logger.info(f"Processing URL: {normalized_url}")
    
    try:
        # Step 2: Parse the URL to extract components
        parsed_url = urllib.parse.urlparse(normalized_url)
        if not parsed_url.netloc:
            raise InvalidURLError(f"Invalid URL structure: {normalized_url}")
        
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        
        # Step 3: Analyze URL structure to determine content type
        content_type = _determine_content_type(domain, path, query)
        logger.info(f"Detected content type: {content_type} for {domain}")
        
        # Step 4: In a real implementation, fetch the web content
        # For this prototype, generate simulated content
        title, raw_content = _extract_content_simulation(domain, path, content_type)
        
        # Step 5: Analyze extracted content
        content_analysis = _analyze_content(raw_content, domain, content_type)
        
        # Step 6: Prepare final result
        result = {
            'url': normalized_url,
            'title': title,
            'content': raw_content,
            'domain': domain,
            'path': path,
            'content_type': content_type,
            'word_count': content_analysis['word_count'],
            'jargon_count': content_analysis['jargon_count'],
            'jargon_density': content_analysis['jargon_density'],
            'extraction_timestamp': datetime.datetime.now().isoformat(),
            'author': None,  # Would be extracted from actual content
            'publish_date': None,  # Would be extracted from actual content
            'metadata': {
                'domain_category': _categorize_domain(domain),
                'estimated_read_time': _calculate_read_time(content_analysis['word_count']),
                'promotional_score': content_analysis['promotional_score'],
                'corporate_speak_level': content_analysis['corporate_speak_level'],
                'detected_topics': content_analysis['detected_topics']
            }
        }
        
        logger.info(f"Successfully extracted content from {domain}: {content_analysis['word_count']} words, "
                   f"{content_analysis['jargon_count']} jargon terms")
        
        return result
        
    except InvalidURLError as e:
        logger.error(f"Invalid URL error: {str(e)}")
        raise
    
    except ContentExtractionError as e:
        logger.error(f"Content extraction error: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error processing URL {normalized_url}: {str(e)}")
        raise ContentExtractionError(f"Failed to process URL: {str(e)}")

def _normalize_url(url: str) -> str:
    """
    Normalize URL format by ensuring protocol and handling common issues.
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL string
        
    Raises:
        InvalidURLError: When URL format cannot be normalized
    """
    # Remove leading/trailing whitespace
    url = url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Handle common URL format issues
    if '//' in url[8:]:  # Double slash after protocol
        url = url.replace('https://', 'https:/').replace('http://', 'http:/')
        url = url.replace('https:/', 'https://').replace('http:/', 'http://')
        
    # Handle spaces in URLs (replace with %20)
    if ' ' in url:
        url = url.replace(' ', '%20')
    
    # Validate basic URL structure
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc or '.' not in parsed.netloc:
            raise InvalidURLError(f"URL lacks valid domain structure: {url}")
        return url
    except Exception as e:
        raise InvalidURLError(f"Could not parse URL {url}: {str(e)}")

def _determine_content_type(domain: str, path: str, query: str) -> str:
    """
    Determine the type of content based on URL structure.
    
    Args:
        domain: Website domain
        path: URL path
        query: URL query parameters
        
    Returns:
        String indicating content type
    """
    domain_lower = domain.lower()
    path_lower = path.lower()
    
    # Check for common news/article sites
    if any(news_site in domain_lower for news_site in [
        'news', 'times', 'post', 'herald', 'tribune', 'journal', 
        'gazette', 'reuters', 'bloomberg', 'bbc', 'cnn', 'nyt'
    ]):
        return 'news_article'
    
    # Check for common blogging platforms
    if any(blog_site in domain_lower for blog_site in [
        'medium.com', 'wordpress', 'blogger', 'tumblr', 'blogspot',
        'substack', 'ghost', 'hashnode'
    ]):
        return 'blog_post'
    
    # Check for professional/business sites
    if any(prof_site in domain_lower for prof_site in [
        'linkedin', 'company', 'corp', 'inc', 'ltd', 'enterprise',
        'group', 'solutions', 'global', 'international', 'consulting'
    ]):
        return 'corporate_content'
    
    # Check for e-commerce
    if any(shop_site in domain_lower for shop_site in [
        'shop', 'store', 'buy', 'product', 'amazon', 'ebay', 'etsy',
        'walmart', 'purchase', 'deal', 'price'
    ]):
        return 'product_page'
    
    # Check path for content indicators
    if any(indicator in path_lower for indicator in ARTICLE_INDICATORS):
        return 'article'
    
    if any(indicator in path_lower for indicator in COMMERCE_INDICATORS):
        return 'product_page'
        
    if any(indicator in path_lower for indicator in CORPORATE_INDICATORS):
        return 'corporate_content'
    
    # Check for query parameters indicating content type
    if query:
        query_params = query.lower()
        if any(param in query_params for param in ['article', 'post', 'blog']):
            return 'article'
        if any(param in query_params for param in ['product', 'item', 'shop']):
            return 'product_page'
    
    # Default to generic webpage if we can't determine specific type
    return 'generic_webpage'

def _categorize_domain(domain: str) -> str:
    """
    Categorize the domain type based on domain name patterns.
    
    Args:
        domain: Website domain
        
    Returns:
        Domain category string
    """
    domain_lower = domain.lower()
    
    categories = {
        'news': ['news', 'times', 'post', 'journal', 'tribune', 'herald', 'reuters', 'cnn', 'bbc', 'nyt'],
        'tech': ['tech', 'software', 'app', 'digital', 'cloud', 'cyber', 'ai', 'data', 'compute', 'dev'],
        'business': ['business', 'finance', 'invest', 'capital', 'fund', 'corp', 'inc', 'ltd', 'group'],
        'shopping': ['shop', 'store', 'buy', 'mall', 'market', 'amazon', 'ebay', 'etsy', 'walmart'],
        'social': ['facebook', 'twitter', 'instagram', 'linkedin', 'pinterest', 'tiktok', 'social', 'community'],
        'education': ['edu', 'university', 'college', 'school', 'academy', 'learn', 'course', 'training'],
        'government': ['gov', 'state', 'federal', 'agency', 'department', 'administration', 'official']
    }
    
    # Check domain against category patterns
    for category, patterns in categories.items():
        if any(pattern in domain_lower for pattern in patterns):
            return category
            
    # Default category
    return 'general'

def _extract_content_simulation(domain: str, path: str, content_type: str) -> Tuple[str, str]:
    """
    Simulate content extraction based on URL components.
    
    In a production system, this would use a proper web scraping implementation.
    This function generates realistic simulated content for demonstration purposes.
    
    Args:
        domain: Website domain
        path: URL path
        content_type: Detected content type
        
    Returns:
        Tuple of (title, content)
    """
    # Extract domain name without TLD
    domain_parts = domain.split('.')
    if len(domain_parts) > 1:
        site_name = domain_parts[-2].capitalize()
    else:
        site_name = domain.capitalize()
    
    # Extract last path component as topic
    path_parts = [p for p in path.split('/') if p]
    topic = "General Topic"
    if path_parts:
        topic = path_parts[-1].replace('-', ' ').replace('_', ' ').capitalize()
    
    # Generate different content templates based on content type
    if content_type == 'news_article':
        title = f"{topic}: What You Need to Know About This Developing Story"
        content = _generate_news_article(site_name, topic)
        
    elif content_type == 'blog_post':
        title = f"{topic}: The Ultimate Guide You Never Knew You Needed"
        content = _generate_blog_post(site_name, topic)
        
    elif content_type == 'corporate_content':
        title = f"How {site_name} Is Revolutionizing {topic} Through Innovation"
        content = _generate_corporate_content(site_name, topic)
        
    elif content_type == 'product_page':
        title = f"Introducing the New {site_name} {topic} - Game-Changing Innovation"
        content = _generate_product_page(site_name, topic)
        
    else:  # generic_webpage
        title = f"{site_name} - Your Destination for {topic}"
        content = _generate_generic_webpage(site_name, topic)
    
    return title, content

def _generate_news_article(site_name: str, topic: str) -> str:
    """Generate simulated news article content."""
    date = datetime.datetime.now().strftime("%B %d, %Y")
    author = random.choice(["Sarah Johnson", "Michael Chen", "Aisha Patel", "Robert Williams", "Emma Garcia"])
    
    return f"""
By {author} | {date} | {site_name} News

{topic.upper()} - In a dramatic development that has caught industry observers by surprise, the world of {topic} has been fundamentally altered by recent events, according to sources familiar with the matter.

Experts are calling this a potential watershed moment for {topic}, with wide-ranging implications that could reshape the landscape for years to come.

"We're seeing unprecedented changes in how {topic} is being approached," said Dr. Jennifer Reynolds, a leading authority in the field. "The data suggests this isn't just a temporary shift but potentially a new paradigm."

The announcement comes after months of speculation about the future direction of {topic}, with stakeholders expressing both optimism and concern about what these developments might mean.

KEY POINTS:

- Market analysts predict a 37% increase in {topic}-related activities over the next quarter
- Industry leaders are scrambling to adapt their strategies in response
- Regulatory bodies have announced plans to review existing frameworks
- Consumer advocacy groups have raised questions about potential impacts

"What makes this particularly significant is the timing," explained industry analyst Mark Thompson. "Coming on the heels of last month's developments, this creates a perfect storm scenario that few had anticipated."

Critics, however, point out that similar claims about {topic} have been made before, with minimal long-term impact. "We've seen this cycle play out repeatedly," noted skeptic and commentator Lisa Rodriguez. "The initial excitement rarely translates into meaningful change."

Nevertheless, the implications for everyday consumers remain unclear, as experts continue to debate the real-world applications of these developments.

{site_name} has learned that several major companies are already investing heavily in {topic}-adjacent technologies, suggesting a broader industry consensus about its importance.

"The race is definitely on," said tech entrepreneur David Singh. "Whoever cracks the code on {topic} first will have a significant advantage."

As this story develops, {site_name} will continue to provide updates on the evolving situation.
"""

def _generate_blog_post(site_name: str, topic: str) -> str:
    """Generate simulated blog post content."""
    return f"""
{topic}: The Ultimate Guide You Never Knew You Needed

Published on {datetime.datetime.now().strftime("%B %d, %Y")}

Are you maximizing your {topic} potential? If you're like most people, you're probably making these 5 common mistakes without even realizing it.

In my 10+ years of {topic} experience, I've discovered that the key to success is not what most "experts" tell you. In fact, the conventional wisdom about {topic} is completely wrong.

THE HARSH TRUTH ABOUT {topic.upper()}

Let's be honest: most advice about {topic} comes from people trying to sell you something. They're not interested in your actual results—they just want your credit card number.

Here's what the data actually shows:

- 78% of people approach {topic} completely backwards
- Only 12% achieve their desired outcomes
- The average person wastes 4.3 hours per week on ineffective {topic} strategies
- The most common advice is precisely what's holding you back

Why is nobody talking about this? Because the truth doesn't sell fancy courses or generate affiliate commissions.

MY JOURNEY WITH {topic}

I was stuck in the same cycle until I accidentally discovered what I now call the "Reverse {topic} Method." It happened when I was working with a client who was doing everything "right" but getting terrible results.

Out of desperation, we decided to try the exact opposite approach. The results were immediate and shocking.

Let me walk you through my proven 3-step process that has helped thousands transform their approach to {topic}:

1. REFRAME YOUR {topic.upper()} MINDSET

First, you need to completely abandon the conventional thinking around {topic}. This is the hardest step for most people because we've been conditioned to approach it from exactly the wrong angle.

The key insight: {topic} isn't about [common misconception]. It's actually about [counterintuitive truth].

2. IMPLEMENT THE COUNTERINTUITIVE {topic.upper()} STRATEGY

Once your mindset is properly aligned, you'll need to implement these core tactics:

- Start with the end state and work backwards
- Focus on [unexpected approach] instead of [common approach]
- Measure what matters: [specific metric] is the only number you should care about
- Build in immediate feedback loops to accelerate your progress

3. SCALE YOUR {topic.upper()} ECOSYSTEM

With the foundation in place, now you can expand your approach across different contexts:

- Adapt the core principles to your specific situation
- Build systems that make implementation effortless
- Create accountability mechanisms that actually work
- Leverage [specific tool] to automate the repetitive aspects

The results speak for themselves. After applying these principles, my clients have seen a 300% increase in {topic} engagement and a 10x ROI on their {topic} investments.

WHAT NEXT?

If you're serious about transforming your approach to {topic}, download my free worksheet: "The {topic} Transformation Blueprint" and join 50,000+ others who have already seen dramatic results.

Let me know in the comments: What's your biggest struggle with {topic}?
"""

def _generate_corporate_content(site_name: str, topic: str) -> str:
    """Generate simulated corporate webpage content."""
    return f"""
{site_name} » About » {topic}

HOW {site_name.upper()} IS REVOLUTIONIZING {topic.upper()} THROUGH INNOVATION

At {site_name}, we're proud to be at the forefront of the {topic} revolution. Our cutting-edge approach combines deep industry expertise with next-generation technology to deliver unparalleled value to our stakeholders.

OUR {topic.upper()} VISION

{site_name} is committed to leveraging synergistic partnerships to create a paradigm shift in how {topic} is approached in today's rapidly evolving landscape. By harnessing the power of disruptive innovation, we're not just adapting to change—we're driving it.

Through our proprietary {topic.capitalize()} Solutions Framework™, we've helped organizations across 17 industries:

- Optimize their {topic} infrastructure for maximum efficiency
- Streamline {topic} processes to eliminate redundancies
- Scale {topic} initiatives across enterprise ecosystems
- Drive actionable insights through data-powered {topic} analytics

"Our partnership with {site_name} transformed our approach to {topic}," says John Williams, CIO of a Fortune 500 company. "Their innovative solutions helped us achieve a 43% increase in operational efficiency while reducing overhead costs by 27%."

THE {site_name.upper()} DIFFERENCE

What sets our {topic} approach apart is our holistic methodology:

1. ASSESS
First, our team of seasoned experts conducts a comprehensive assessment of your current {topic} landscape, identifying opportunities for optimization and growth.

2. STRATEGIZE
Next, we develop a customized roadmap aligned with your business objectives, ensuring that our {topic} solution drives meaningful outcomes that impact your bottom line.

3. IMPLEMENT
Our agile implementation process ensures minimal disruption to your operations while maximizing the speed to value.

4. MEASURE
Using our proprietary metrics framework, we continuously monitor performance to quantify ROI and identify opportunities for further enhancement.

5. OPTIMIZE
The journey doesn't end at implementation. We partner with you to continuously refine and evolve your {topic} strategy as market conditions change.

INDUSTRY RECOGNITION

{site_name}'s innovative approach to {topic} has been recognized by leading analysts and industry bodies:

- Named "Leader" in {topic} by Gartner Magic Quadrant (2023)
- Awarded "{topic} Innovator of the Year" by Industry Association
- Featured in Forbes' "Top 10 Companies Revolutionizing {topic}"
- Recipient of the prestigious {topic} Excellence Award for three consecutive years

READY TO TRANSFORM YOUR {topic.upper()} APPROACH?

Contact us today to schedule a complimentary {topic} assessment and discover how {site_name} can help you unlock unprecedented value.
"""

def _generate_product_page(site_name: str, topic: str) -> str:
    """Generate simulated product page content."""
    price = random.randint(49, 499)
    rating = round(random.uniform(4.2, 4.9), 1)
    reviews = random.randint(120, 3500)
    
    return f"""
{site_name} » Products » {topic}

INTRODUCING THE {site_name.upper()} {topic.upper()} PRO

${price}.99 | ★★★★★ {rating} ({reviews} reviews) | ✓ In Stock

THE FUTURE OF {topic.upper()} HAS ARRIVED

Meet the groundbreaking {site_name} {topic} Pro—engineered to revolutionize how you experience {topic} with unparalleled performance, sleek design, and intuitive functionality.

WHY THE {topic.upper()} PRO STANDS APART

- 2x Faster: Powered by our next-generation technology
- Smart Integration: Seamlessly connects with your existing ecosystem
- Enhanced Efficiency: Reduces {topic} time by up to 43%
- Advanced Protection: Industry-leading security features
- Eco-Friendly: Made with 70% recycled materials

"The {site_name} {topic} Pro has completely transformed my daily routine. It's not just an upgrade—it's a revolution." — Featured Customer Review

TECHNICAL SPECIFICATIONS

- Dimensions: 10.4" x 7.2" x 0.8"
- Weight: Just 1.6 lbs
- Battery Life: Up to 18 hours
- Connectivity: WiFi 6, Bluetooth 5.2, USB-C
- Storage: 256GB (expandable to 1TB)
- Display: Ultra HD resolution with TrueTone technology
- Warranty: 2-year limited warranty included

SPECIAL OFFER: Purchase today and receive our exclusive {topic} Essentials Kit (a $79 value) absolutely FREE!

CUSTOMER FAVORITES

WHAT OUR CUSTOMERS ARE SAYING

★★★★★ "Game-changer! The {topic} Pro exceeded all my expectations. Worth every penny." — Jason M.

★★★★★ "As someone who uses {topic} tools daily, I can confidently say this is the best on the market." — Sarah L.

★★★★ "Great product with intuitive controls. Took off one star for the app learning curve." — Michael T.

★★★★★ "The {topic} Pro's efficiency has saved me countless hours. Highly recommend!" — Emma R.

FREQUENTLY ASKED QUESTIONS

Q: How does the {topic} Pro compare to previous models?
A: The {topic} Pro offers 2x faster performance, 40% longer battery life, and 3 new innovative features not available in previous generations.

Q: Is the {topic} Pro compatible with other {site_name} products?
A: Yes! The {topic} Pro is designed to integrate seamlessly with the entire {site_name} ecosystem.

Q: What's included in the box?
A: Your {topic} Pro comes with a quick-start guide, charging cable, protective case, and access to our premium support service.

ADD TO CART | BUY NOW | ADD TO WISHLIST
"""

def _generate_generic_webpage(site_name: str, topic: str) -> str:
    """Generate simulated generic webpage content."""
    return f"""
{site_name} - Your destination for all things {topic}!

WELCOME TO {site_name.upper()}

Learn how our cutting-edge {topic} solutions can transform your experience.
With industry-leading expertise and proprietary technology, we deliver
unparalleled {topic} results.

WHAT WE OFFER

Our {topic} platform features:
- Advanced {topic} algorithms
- Seamless {topic} integration
- Enterprise-grade {topic} security
- Real-time {topic} analytics

WHY CHOOSE {site_name.upper()}

✓ Trusted by over 10,000 customers worldwide
✓ Industry-leading expertise in {topic}
✓ Award-winning customer support
✓ Continuous innovation and improvement

Join thousands of satisfied customers who have revolutionized their {topic}
approach with {site_name}.

OUR SERVICES

{topic} CONSULTATION
Our experts will analyze your current {topic} approach and identify opportunities for optimization.

{topic} IMPLEMENTATION
We'll help you implement best practices and cutting-edge solutions for maximum {topic} efficiency.

{topic} MANAGEMENT
Ongoing support and management to ensure your {topic} strategy continues to deliver results.

{topic} TRAINING
Comprehensive training programs to empower your team with {topic} expertise.

GET STARTED TODAY

Contact us to schedule your free {topic} consultation!

Phone: (555) 123-4567
Email: info@{domain.lower()}
"""

def _analyze_content(content: str, domain: str, content_type: str) -> Dict[str, Any]:
    """
    Analyze extracted content to extract metadata and insights.
    
    Args:
        content: Extracted content text
        domain: Website domain
        content_type: Detected content type
        
    Returns:
        Dict containing content analysis results
    """
    # Count words
    words = re.findall(r'\b\w+\b', content.lower())
    word_count = len(words)
    
    # Detect jargon
    jargon_count = 0
    jargon_instances = []
    
    for pattern in JARGON_PATTERNS:
        matches = re.findall(pattern, content.lower())
        jargon_count += len(matches)
        if matches:
            # Get the actual matched text
            for match in matches:
                if isinstance(match, tuple):  # Some regex groups return tuples
                    jargon_instances.append(match[0])
                else:
                    jargon_instances.append(match)
    
    # Calculate jargon density
    jargon_density = 0 if word_count == 0 else (jargon_count / word_count) * 100
    
    # Calculate promotional score (0-10)
    promotional_indicators = [
        '!', 'best', 'revolutionary', 'innovative', 'groundbreaking',
        'exclusive', 'limited', 'special offer', 'discount', 'free',
        'unprecedented', 'remarkable', 'extraordinary', 'exceptional'
    ]
    
    promotional_score = 0
    for indicator in promotional_indicators:
        promotional_score += content.lower().count(indicator.lower())
    
    # Normalize promotional score to 0-10 range
    promotional_score = min(10, promotional_score / 5)
    
    # Calculate corporate speak level (0-10)
    corporate_speak_level = min(10, jargon_count)
    
    # Detect main topics (simplified approach)
    detected_topics = _extract_topics(content, content_type)
    
    return {
        'word_count': word_count,
        'jargon_count': jargon_count,
        'jargon_instances': jargon_instances[:5],  # Top 5 for brevity
        'jargon_density': round(jargon_density, 1),
        'promotional_score': round(promotional_score, 1),
        'corporate_speak_level': round(corporate_speak_level, 1),
        'detected_topics': detected_topics
    }

def _extract_topics(content: str, content_type: str) -> List[str]:
    """
    Extract main topics from content text.
    
    Args:
        content: Content text
        content_type: Type of content
        
    Returns:
        List of detected topics
    """
    # In a real implementation, this would use NLP techniques
    # This is a simplified approach
    
    # Look for capitalized phrases as potential topics
    topic_candidates = re.findall(r'\b[A-Z][A-Za-z]+(?: [A-Za-z]+){0,3}\b', content)
    
    # Look for repeated terms
    words = re.findall(r'\b[a-zA-Z]{5,}\b', content.lower())
    word_counts = {}
    for word in words:
        if word not in ['about', 'these', 'their', 'there', 'would', 'should', 'could']:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top repeated terms
    repeated_terms = [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    # Combine sources of topics
    all_topics = topic_candidates + repeated_terms
    
    # Remove duplicates while preserving order
    unique_topics = []
    seen = set()
    for topic in all_topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            unique_topics.append(topic)
    
    return unique_topics[:5]  # Return top 5 topics

def _calculate_read_time(word_count: int) -> int:
    """
    Calculate estimated reading time in minutes.
    
    Args:
        word_count: Number of words in content
        
    Returns:
        Reading time in minutes
    """
    # Average reading speed is about 200-250 words per minute
    reading_speed = 225
    read_time_minutes = max(1, round(word_count / reading_speed))
    return read_time_minutes

def extract_url_metadata(url: str) -> Dict[str, Any]:
    """
    Extract metadata from a URL without fetching content.
    
    This function analyzes URL structure to extract useful metadata
    about the likely content without making an HTTP request.
    
    Args:
        url: URL to analyze
        
    Returns:
        Dict containing URL metadata
    """
    try:
        # Normalize URL
        normalized_url = _normalize_url(url)
        
        # Parse URL
        parsed_url = urllib.parse.urlparse(normalized_url)
        
        # Extract components
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        
        # Determine content type without fetching content
        content_type = _determine_content_type(domain, path, query)
        
        # Categorize domain
        domain_category = _categorize_domain(domain)
        
        # Extract topic from path
        path_parts = [p for p in path.split('/') if p]
        topic = None
        if path_parts:
            topic = path_parts[-1].replace('-', ' ').replace('_', ' ').capitalize()
        
        # Analyze query parameters
        query_params = {}
        if query:
            for param in query.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    query_params[key] = value
        
        # Extract likely content format indicators
        format_indicators = []
        format_extensions = {
            '.html': 'HTML webpage',
            '.php': 'Dynamic webpage',
            '.aspx': 'ASP.NET webpage',
            '.jsp': 'Java Server Page',
            '.pdf': 'PDF document',
            '.doc': 'Word document',
            '.docx': 'Word document',
            '.ppt': 'PowerPoint presentation',
            '.pptx': 'PowerPoint presentation',
            '.xls': 'Excel spreadsheet',
            '.xlsx': 'Excel spreadsheet'
        }
        
        for ext, format_name in format_extensions.items():
            if path.endswith(ext):
                format_indicators.append(format_name)
                break
        
        # Build metadata response
        metadata = {
            'url': normalized_url,
            'domain': domain,
            'path': path,
            'likely_content_type': content_type,
            'domain_category': domain_category,
            'extracted_topic': topic,
            'query_parameters': query_params,
            'format_indicators': format_indicators,
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        return metadata
        
    except InvalidURLError as e:
        logger.error(f"Error extracting URL metadata: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error extracting URL metadata: {str(e)}")
        raise InvalidURLError(f"Could not analyze URL: {str(e)}")

def detect_url_sentiment(url: str) -> Dict[str, Any]:
    """
    Analyze URL to estimate likely content sentiment without fetching content.
    
    This function examines URL structure and components to make educated
    guesses about the likely sentiment and tone of the content.
    
    Args:
        url: URL to analyze
        
    Returns:
        Dict with sentiment analysis
    """
    try:
        # Normalize and parse URL
        normalized_url = _normalize_url(url)
        parsed_url = urllib.parse.urlparse(normalized_url)
        
        domain = parsed_url.netloc
        path = parsed_url.path.lower()
        query = parsed_url.query.lower()
        
        # Initialize sentiment indicators
        positive_indicators = 0
        negative_indicators = 0
        neutral_indicators = 0
        
        # Check domain for sentiment indicators
        positive_domains = ['positive', 'good', 'happy', 'best', 'top', 'success']
        negative_domains = ['negative', 'bad', 'problem', 'issue', 'critic', 'fail']
        
        for term in positive_domains:
            if term in domain.lower():
                positive_indicators += 1
                
        for term in negative_domains:
            if term in domain.lower():
                negative_indicators += 1
        
        # Check path for sentiment indicators
        positive_path_terms = ['success', 'solution', 'guide', 'help', 'benefit', 'improve', 'best', 'top', 'great']
        negative_path_terms = ['problem', 'issue', 'error', 'warning', 'danger', 'risk', 'worst', 'bad', 'fail']
        
        for term in positive_path_terms:
            if term in path:
                positive_indicators += 1
                
        for term in negative_path_terms:
            if term in path:
                negative_indicators += 1
        
        # Check query parameters for sentiment indicators
        if 'success' in query or 'solved' in query:
            positive_indicators += 1
        if 'error' in query or 'issue' in query:
            negative_indicators += 1
        
        # Check for question indicators (neutral/inquisitive)
        if 'how' in path or 'what' in path or 'why' in path or 'when' in path:
            neutral_indicators += 2
        
        # Check for news indicators (can be positive or negative)
        if 'news' in domain.lower() or 'news' in path:
            neutral_indicators += 1
            
        # Determine content type for context
        content_type = _determine_content_type(domain, path, query)
        
        # Product pages tend to be positive
        if content_type == 'product_page':
            positive_indicators += 2
            
        # Corporate content tends to be positive
        if content_type == 'corporate_content':
            positive_indicators += 2
            
        # News can be either positive or negative
        if content_type == 'news_article':
            neutral_indicators += 1
        
        # Determine overall sentiment
        total_indicators = positive_indicators + negative_indicators + neutral_indicators
        
        if total_indicators == 0:
            sentiment = "neutral"
            confidence = 0.5
        else:
            # Calculate sentiment scores
            positive_score = positive_indicators / total_indicators
            negative_score = negative_indicators / total_indicators
            neutral_score = neutral_indicators / total_indicators
            
            # Determine dominant sentiment
            if positive_score > negative_score and positive_score > neutral_score:
                sentiment = "positive"
                confidence = positive_score
            elif negative_score > positive_score and negative_score > neutral_score:
                sentiment = "negative"
                confidence = negative_score
            else:
                sentiment = "neutral"
                confidence = neutral_score
        
        return {
            'url': normalized_url,
            'estimated_sentiment': sentiment,
            'confidence': round(confidence, 2),
            'positive_indicators': positive_indicators,
            'negative_indicators': negative_indicators,
            'neutral_indicators': neutral_indicators,
            'domain_category': _categorize_domain(domain),
            'content_type': content_type,
            'analysis_method': "URL structure analysis (no content fetch)",
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing URL sentiment: {str(e)}")
        raise InvalidURLError(f"Could not analyze URL sentiment: {str(e)}")

def process_url_for_platform(url: str, platform: str) -> Dict[str, Any]:
    """
    Process URL content specifically for a target platform.
    
    This function extracts URL content and applies platform-specific
    preprocessing to prepare it for transformation into satirical content
    optimized for a specific platform.
    
    Args:
        url: URL to process
        platform: Target platform (e.g., 'LinkedIn', 'Twitter')
        
    Returns:
        Dict containing processed content with platform-specific metadata
    """
    # Extract content using standard process
    content_data = process_url(url)
    
    # Add platform-specific preprocessing
    platform = platform.lower()
    platform_notes = []
    
    if platform == 'linkedin':
        # Check for corporate language that would be good satire fodder
        if content_data['jargon_density'] > 5:
            platform_notes.append("High corporate jargon density - excellent for LinkedIn satire")
        
        # Check for excessive self-promotion
        if content_data['metadata']['promotional_score'] > 7:
            platform_notes.append("Heavy self-promotion detected - prime target for satirical transformation")
            
        # Check if content is long enough for LinkedIn
        if content_data['word_count'] < 100:
            platform_notes.append("Content may be too brief for effective LinkedIn satire - consider expanding")
            
    elif platform == 'twitter':
        # Twitter has strict character limits
        if content_data['word_count'] > 50:
            platform_notes.append("Content will need significant condensing for Twitter format")
            
        # Check if content has clear talking points for tweets
        if len(content_data['metadata']['detected_topics']) < 2:
            platform_notes.append("Limited clear topics detected - may need to sharpen focus for Twitter")
            
    elif platform == 'medium':
        # Medium works best with substantive content
        if content_data['word_count'] < 300:
            platform_notes.append("Content may be too brief for substantive Medium satire")
            
        # Check for structure that works well in Medium articles
        if content_data['content_type'] not in ['blog_post', 'news_article', 'corporate_content']:
            platform_notes.append("Content format may need restructuring for effective Medium article")
            
    elif platform == 'substack':
        # Substack newsletters need personality and audience connection
        platform_notes.append("Consider adding a personal angle to the satire for Substack format")
        
        # Check for clear sections that can be expanded
        if content_data['content_type'] == 'generic_webpage':
            platform_notes.append("Content lacks clear sections - consider restructuring for newsletter format")
    
    # Add platform-specific notes to result
    content_data['platform_specific_notes'] = platform_notes
    content_data['target_platform'] = platform
    content_data['processing_timestamp'] = datetime.datetime.now().isoformat()
    
    return content_data

def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract valid URLs from a text string.
    
    Args:
        text: Text to scan for URLs
        
    Returns:
        List of extracted valid URLs
    """
    # Basic URL pattern matching
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Find all URL-like strings
    potential_urls = url_pattern.findall(text)
    
    # Validate each URL
    valid_urls = []
    for url in potential_urls:
        try:
            normalized_url = _normalize_url(url)
            valid_urls.append(normalized_url)
        except InvalidURLError:
            # Skip invalid URLs
            continue
    
    # Also look for "domain.com" style URLs without protocol
    domain_pattern = re.compile(
        r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
    )
    
    domain_matches = domain_pattern.findall(text)
    for domain in domain_matches:
        # Skip domains that are already part of extracted URLs
        if not any(domain in url for url in valid_urls):
            try:
                normalized_url = _normalize_url(domain)
                valid_urls.append(normalized_url)
            except InvalidURLError:
                continue
    
    return valid_urls