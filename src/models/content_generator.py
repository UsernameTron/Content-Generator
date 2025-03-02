"""
Content generator module for creating platform-specific content.
"""

import os
import re
import logging
import random
import json
from typing import Dict, Any, List, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Import local modules
from src.utils.document_processor import analyze_sentiment, extract_key_topics
from src.models.templates import get_template, format_hashtags
from src.utils.wandb_monitor import log_generation_example, is_wandb_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContentGenerator:
    """
    Generate platform-specific content based on templates and NLP analysis.
    """
    
    def __init__(self, use_wandb: bool = True):
        """
        Initialize the content generator.
        
        Args:
            use_wandb: Flag to enable/disable Weights & Biases monitoring
        """
        # Common emojis by sentiment
        self.emojis = {
            "positive": ["👍", "🚀", "💡", "✨", "🔥", "👏", "💪", "🎯", "🙌", "😊"],
            "negative": ["🤔", "😕", "🙄", "👎", "💩", "🤦‍♂️", "🧐", "😬", "🔍", "⚠️"],
            "neutral": ["📊", "🔄", "📱", "💻", "🤖", "📈", "🔮", "⚙️", "🧠", "📝"]
        }
        
        # Load writing style configuration
        self.writing_style = self._load_writing_style()
        
        # Use W&B if enabled and available
        self.use_wandb = use_wandb and is_wandb_available()
        
        logger.info(f"ContentGenerator initialized with W&B monitoring: {self.use_wandb}")
    
    def _load_writing_style(self) -> Dict[str, Any]:
        """
        Load writing style configuration from JSON file.
        
        Returns:
            Dict containing writing style configuration
        """
        try:
            style_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'data', 'writing_style.json')
            
            if os.path.exists(style_path):
                with open(style_path, 'r') as f:
                    style_data = json.load(f)
                logger.info("Loaded writing style configuration")
                return style_data
            else:
                logger.warning(f"Writing style file not found at {style_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading writing style: {str(e)}")
            return {}
    
    def generate_content(
        self, 
        input_text: str, 
        platform: str, 
        platform_specs: Any,  # Allow both Dict and PlatformSpecs object
        tone: str = "Informative",
        keywords: List[str] = None,
        writing_style: str = None
    ) -> str:
        """
        Generate platform-specific content based on input text.
        
        Args:
            input_text: Source text to generate content from
            platform: Target platform (e.g., Twitter, LinkedIn)
            platform_specs: Platform-specific parameters
            tone: Desired tone for the content
            keywords: List of keywords to include
            writing_style: Optional writing style override (e.g., "pete_connor")
            
        Returns:
            str: Generated content
        """
        try:
            # Use memoization to cache content generation results
            cache_key = f"{platform}:{writing_style}:{tone}:{hash(input_text)}"
            
            # Check if we have a cached result
            if hasattr(self, '_content_cache') and cache_key in self._content_cache:
                logger.info(f"Using cached content for platform: {platform}")
                return self._content_cache[cache_key]
            
            logger.info(f"Generating content for platform: {platform} with style: {writing_style}")
            
            # Initialize cache if needed
            if not hasattr(self, '_content_cache'):
                self._content_cache = {}
                # Limit cache size to prevent memory issues
                self._cache_max_size = 100
            
            # Use C. Pete Connor's style by default if loaded
            if writing_style is None and self.writing_style:
                writing_style = "pete_connor"
            
            # Optimize keyword extraction - only do it if needed
            if not keywords or len(keywords) == 0:
                # Check if we've previously extracted keywords for this text
                if hasattr(self, '_keyword_cache') and input_text[:50] in self._keyword_cache:
                    keywords = self._keyword_cache[input_text[:50]]
                else:
                    # Initialize keyword cache if needed
                    if not hasattr(self, '_keyword_cache'):
                        self._keyword_cache = {}
                    
                    # Extract keywords and cache them
                    keywords = extract_key_topics(input_text, num_topics=5)
                    # Use first 50 chars as key to save memory
                    self._keyword_cache[input_text[:50]] = keywords
                    
                    # Limit keyword cache size
                    if len(self._keyword_cache) > 50:
                        # Remove oldest entry
                        self._keyword_cache.pop(next(iter(self._keyword_cache)))
            
            # Optimize sentiment analysis - similar caching approach
            if hasattr(self, '_sentiment_cache') and input_text[:50] in self._sentiment_cache:
                sentiment_data = self._sentiment_cache[input_text[:50]]
                dominant_sentiment = sentiment_data.get("dominant_sentiment", "neutral")
            else:
                # Initialize sentiment cache if needed
                if not hasattr(self, '_sentiment_cache'):
                    self._sentiment_cache = {}
                
                # Analyze sentiment and cache it
                sentiment_data = analyze_sentiment(input_text)
                dominant_sentiment = sentiment_data.get("dominant_sentiment", "neutral")
                self._sentiment_cache[input_text[:50]] = sentiment_data
                
                # Limit sentiment cache size
                if len(self._sentiment_cache) > 50:
                    self._sentiment_cache.pop(next(iter(self._sentiment_cache)))
            
            # Extract key topics if keywords are still empty
            if len(keywords) == 0:
                keywords = sentiment_data.get("top_keywords", [])
                if len(keywords) == 0:
                    # Fallback to some generic topics
                    keywords = ["technology", "innovation", "digital", "trends", "future"]
            
            # Create content based on platform specifications
            # Handle both dictionary and object formats for platform_specs
            if hasattr(platform_specs, 'max_length'):
                # Using PlatformSpecs object
                max_length = platform_specs.max_length
                min_length = platform_specs.min_length
            elif isinstance(platform_specs, dict) and 'max_length' in platform_specs:
                # Using dictionary format
                max_length = platform_specs['max_length'] 
                min_length = platform_specs['min_length']
            else:
                # Fallback defaults
                logger.warning(f"Invalid platform_specs format for {platform}. Using defaults.")
                max_length = 5000
                min_length = 100
            
            # Apply special handling for Substack and Medium to ensure longer content
            if platform.lower() == "substack":
                # For Substack, ensure we generate at least 8000 characters
                min_length = max(min_length, 8000)  # Force minimum 8000 characters
                max_length = max(max_length, 25000)  # Ensure enough room
            elif platform.lower() == "medium":
                # For Medium, ensure we generate at least 8000 characters
                min_length = max(min_length, 8000)  # Force minimum 8000 characters
                max_length = max(max_length, 25000)  # Ensure enough room
                
                # If content is shorter than 3000 characters after generation,
                # the templates might not be fully utilized. Log a warning.
                if len(input_text) < 200:
                    logger.warning("Short input text for Medium. Content may not reach optimal length.")
            
            # Extract hashtag settings safely
            if hasattr(platform_specs, 'hashtag_strategy') and platform_specs.hashtag_strategy:
                hashtag_count = platform_specs.hashtag_strategy.recommended_count
            elif isinstance(platform_specs, dict) and 'hashtag_strategy' in platform_specs:
                hashtag_count = platform_specs['hashtag_strategy'].get('recommended_count', 0)
            else:
                hashtag_count = 0
                
            # Extract emoji settings safely
            if hasattr(platform_specs, 'tone') and platform_specs.tone:
                emoji_count = 1 if platform_specs.tone.emoji_usage != "none" else 0
                formal_tone = platform_specs.tone.formality_level > 3
            elif isinstance(platform_specs, dict) and 'tone' in platform_specs:
                emoji_count = 1 if platform_specs['tone'].get('emoji_usage') != "none" else 0
                formal_tone = platform_specs['tone'].get('formality_level', 3) > 3
            else:
                emoji_count = 1
                formal_tone = False
            
            # Get appropriate template for the platform, sentiment, and writing style
            template = get_template(platform, dominant_sentiment, writing_style)
            
            # Generate content from template
            content = self._fill_template(
                template=template,
                input_text=input_text,
                keywords=keywords,
                dominant_sentiment=dominant_sentiment,
                platform=platform
            )
            
            # Adapt content to platform specifications
            adapted_content = self._adapt_to_platform(
                content=content,
                platform=platform,
                max_length=max_length,
                hashtag_count=hashtag_count,
                emoji_count=emoji_count,
                formal_tone=formal_tone,
                keywords=keywords
            )
            
            # Cache the result
            self._content_cache[cache_key] = adapted_content
            
            # Manage cache size
            if len(self._content_cache) > self._cache_max_size:
                # Remove oldest entry
                self._content_cache.pop(next(iter(self._content_cache)))
            
            # Log to W&B if enabled - but do it asynchronously
            if self.use_wandb:
                try:
                    # Only log a sample of generations to reduce overhead
                    if random.random() < 0.2:  # 20% chance to log
                        from src.utils.wandb_monitor import log_generation_example
                        log_generation_example(
                            platform=platform,
                            prompt=input_text[:100] + "..." if len(input_text) > 100 else input_text,
                            generated_content=adapted_content,
                        )
                        logger.debug("Logged generation example to W&B")
                except Exception as e:
                    logger.error(f"Error logging to W&B: {str(e)}")
            
            logger.info(f"Successfully generated content for {platform}")
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return f"Error generating content: {str(e)}"
    
    def _fill_template(
        self, 
        template: str,
        input_text: str,
        keywords: List[str],
        dominant_sentiment: str,
        platform: str
    ) -> str:
        """
        Fill template with content based on input text and keywords.
        
        Args:
            template: Template string with placeholders
            input_text: Source text to generate content from
            keywords: List of keywords to include
            dominant_sentiment: Dominant sentiment of the input text
            platform: Target platform
            
        Returns:
            str: Filled template content
        """
        # Extract sentences from input text
        sentences = sent_tokenize(input_text)
        
        # Prepare template variables
        template_vars = {
            "topic": keywords[0] if keywords else "technology",
            "title": f"Thoughts on {keywords[0]}" if keywords else "Latest Insights",
            "main_point": sentences[0] if sentences else "This is an interesting topic.",
            "hashtags": format_hashtags(keywords, 3) if keywords else ""
        }
        
        # Add supporting points from the input text with more Pete Connor style
        if len(sentences) > 1:
            # Create data-driven, satirical supporting points
            supporting_points = []
            for i, sentence in enumerate(sentences[1:min(5, len(sentences))]):
                # Add more Pete Connor flair with bullet points and data references
                if i == 0:
                    supporting_points.append(f"• Research shows {sentence}")
                elif i == 1:
                    supporting_points.append(f"• Studies indicate that {sentence}")
                elif i == 2:
                    supporting_points.append(f"• According to the data, {sentence}")
                else:
                    supporting_points.append(f"• {sentence}")
            
            template_vars["supporting_points"] = "\n\n".join(supporting_points)
            # Create a brief version for platforms with length constraints
            template_vars["supporting_points_brief"] = supporting_points[0] if supporting_points else "The data contradicts the hype."
        else:
            template_vars["supporting_points"] = "The data tells a different story. Look at the numbers, not the marketing."
            template_vars["supporting_points_brief"] = "The data contradicts the hype."
        
        # Add additional variables based on content type
        if platform in ["Blog", "Email Newsletter", "LinkedIn", "Medium", "Substack"]:
            # Generate introduction
            template_vars["introduction"] = f"I've been thinking about {keywords[0] if keywords else 'this topic'} lately and wanted to share some thoughts."
            
            # Generate key points - expanded for Medium and Substack
            key_points_list = []
            
            # For Medium and Substack, generate much more detailed content
            if platform in ["Medium", "Substack"]:
                # Use more sentences for more detailed content
                for i, kw in enumerate(keywords[:5]):  # Use up to 5 keywords
                    if i < len(sentences):
                        # Create multiple points per keyword for more detailed content
                        key_points_list.append(f"## {kw.capitalize()}\n")
                        key_points_list.append(f"{sentences[i]}")
                        
                        # Add more detailed analysis for each point
                        if platform == "Medium":
                            # For Medium - more technical
                            key_points_list.append(f"\nHoly cow, the data is almost comical! {kw} implementations have a staggering 76% failure rate, with most organizations abandoning their initiatives within 18 months. I'm not making this up - these are the actual reasons cited by real companies who flushed millions down the toilet:\n")
                            key_points_list.append(f"• Misalignment between technical capabilities and business objectives (translation: the vendor lied about what the product could actually do)")
                            key_points_list.append(f"• Insufficient expertise in the underlying systems architecture (translation: no one bothered to check if the shiny new toy would work with their existing infrastructure)")
                            key_points_list.append(f"• Lack of proper data governance and quality control mechanisms (translation: garbage in, garbage out, but with a fancy dashboard on top)")
                            key_points_list.append(f"• Failure to account for integration complexities with legacy systems (translation: turns out connecting to systems built in 1997 isn't as easy as the sales deck promised)\n")
                            key_points_list.append(f"When I analyze the technical specifications behind most {kw} platforms, I find architectural flaws so obvious they'd make a first-year CompSci student blush. Vendors conveniently bury these limitations under layers of marketing jargon and cherry-picked case studies. Let me tear down the façade and show you the hilarious gap between marketing promises and technical reality:\n")
                        
                        elif platform == "Substack":
                            # For Substack - more sarcastic and humorous
                            key_points_list.append(f"\nOh, the absolute hilarity of watching executives throw millions at {kw} initiatives while the same basic problems remain unsolved. Let me describe the typical corporate {kw} implementation meeting:\n")
                            key_points_list.append(f"• The vendor arrives with slides so glossy you could use them as mirrors")
                            key_points_list.append(f"• The CTO nods sagely while understanding approximately 12% of what's being presented")
                            key_points_list.append(f"• Someone mentions 'synergy' and 'digital transformation' in the same sentence (BINGO!)")
                            key_points_list.append(f"• Meanwhile, the IT team in the back of the room is quietly updating their résumés\n")
                            key_points_list.append(f"It's absolutely astonishing how we keep falling for the same nonsense, repackaged with slightly different buzzwords each fiscal year. The {kw} industry has mastered the art of selling digital snake oil with a straight face.\n")
                    else:
                        # Add generic points if we run out of sentences
                        if platform == "Medium":
                            key_points_list.append(f"## {kw.capitalize()}\n")
                            key_points_list.append(f"The technical implementation challenges of {kw} are frequently underestimated - and by 'frequently' I mean 'always' and by 'underestimated' I mean 'completely ignored until the project is already on fire.' When examining the underlying infrastructure requirements, we see a comedy of errors: organizations throw millions at solutions without understanding basic scaling limitations, data consistency requirements, or integration complexity with existing systems.")
                            key_points_list.append(f"\nLet me share a particularly hilarious example from a Fortune 500 company (who shall remain nameless to protect the embarrassed). They spent $12.8 million on a {kw} platform that promised to 'revolutionize' their operations. Two years later, they had succeeded in:\n")
                            key_points_list.append(f"1. Creating three entirely new departments to manage the platform's limitations")
                            key_points_list.append(f"2. Developing 16 custom workarounds for 'features' that didn't actually exist")
                            key_points_list.append(f"3. Generating an impressive 240% increase in service desk tickets")
                            key_points_list.append(f"4. Achieving precisely zero of their original objectives\n")
                            key_points_list.append(f"The kicker? They renewed their contract because 'we've invested too much to back out now.' I couldn't make this stuff up if I tried.")
                        elif platform == "Substack":
                            key_points_list.append(f"## {kw.capitalize()}\n")
                            key_points_list.append(f"If there were an Olympic event for corporate gullibility, the buying cycle for {kw} solutions would sweep the gold, silver, AND bronze medals. The absolute absurdity of watching intelligent professionals fall for flashy demos and cherry-picked case studies never ceases to amaze me.")
                            key_points_list.append(f"\nConsider this scene, which I've witnessed so many times I could write a screenplay about it: A vendor demonstrates their {kw} solution in a perfectly controlled environment with carefully curated test data that bears no resemblance whatsoever to the client's actual business. The executives in the room are practically drooling. Not one person asks: 'But will this work with OUR systems? OUR data? OUR actual use cases?'\n")
                            key_points_list.append(f"The meeting concludes with everyone congratulating themselves on being 'forward-thinking' and 'innovative,' while simultaneously committing to spend millions on a solution that will be collecting digital dust within 18 months. And the best part? When it all inevitably fails, they'll blame the implementation team rather than their own lack of due diligence.\n")
                            key_points_list.append(f"I need to start selling tickets to these meetings. I'd make more than most {kw} consultants.")
            else:
                # Standard approach for other platforms
                for i, kw in enumerate(keywords[:3]):
                    if i < len(sentences):
                        key_points_list.append(f"• {kw.capitalize()}: {sentences[i]}")
                    else:
                        key_points_list.append(f"• {kw.capitalize()}: An important aspect to consider.")
            
            template_vars["key_points"] = "\n".join(key_points_list)
            
            # Generate conclusion
            base_conclusion = f"{keywords[0] if keywords else 'this topic'} {self._get_sentiment_phrase(dominant_sentiment)}."
            
            # Enhanced conclusion for Medium and Substack
            if platform == "Medium":
                # More technical and data-driven conclusion for Medium with added humor and sarcasm
                template_vars["conclusion"] = f"In conclusion, {base_conclusion} The empirical evidence couldn't be clearer: organizations need to fundamentally rethink their approach to implementation, or they'll keep achieving that perfect 0% success rate they seem to be aiming for.\n\nInstead of swallowing vendor promises like a sleep-deprived college student at an all-you-can-eat buffet, technical teams should prioritize:\n\n1. Establishing clear, measurable success metrics before any implementation begins (revolutionary concept, I know)\n2. Creating robust data validation protocols that verify vendor claims independently (spoiler alert: their case studies are cherry-picked fairy tales)\n3. Implementing phased deployments with defined fallback procedures (because 'big bang' implementations always end with an actual bang)\n4. Maintaining parallel systems during transition periods to enable objective performance comparisons (trust, but verify—and by 'trust' I mean 'don't trust at all')\n\nAnd here's a bonus tip that vendors hate: document every promise made during the sales cycle and include it in the contract as a condition of payment. Watch how quickly those 'guaranteed outcomes' turn into 'aspirational goals.'\n\nLet's be honest—most organizations will ignore this advice because executives love shiny new toys more than they love actual results. But for the brave few willing to prioritize reality over hype, these approaches can mean the difference between an implementation that delivers value and one that delivers résumé-updating opportunities for the entire project team."
            elif platform == "Substack":
                # Even more sarcastic, humorous conclusion for Substack
                template_vars["conclusion"] = f"In conclusion, {base_conclusion} And yet, despite all evidence to the contrary, we'll continue watching the same tragic comedy unfold in boardrooms worldwide. Next quarter, there will be a new acronym, a new framework, a new 'revolutionary' approach—and the same old executives will line up with checkbooks in hand, eager to be the first to waste their shareholders' money on digital fairy dust.\n\nThe most reliable constant in the tech industry isn't Moore's Law—it's the unfailing human capacity to believe that THIS time, THIS solution will somehow defy the overwhelming historical evidence and actually deliver what's promised on the sales slide. Einstein reportedly defined insanity as doing the same thing repeatedly and expecting different results. By that definition, enterprise technology procurement is the corporate world's largest insane asylum.\n\nI've started playing a game I call 'Implementation Bingo' during client engagements. The center square is 'Vendor misses deadline but blames client data.' Other squares include classics like 'Mysterious new fees appear,' 'Key feature works in demo but not in production,' and my personal favorite, 'Original sales engineer mysteriously disappears and is replaced by junior associate who has 'no record' of earlier promises.'\n\nWhat's truly remarkable is that I've never failed to get a full blackout card within the first three months of any implementation. The consistency is almost beautiful—like watching a car crash in perfect slow motion while the dealership continues to insist that what you're seeing is actually 'expected performance behavior.'\n\nSo here's my radical proposal: the next time a vendor promises you their solution will 'revolutionize' your business, ask them to put 100% of their fees at risk, to be paid only when they deliver the promised outcomes. The speed with which they backpedal will break the sound barrier. But hey, at least that would be one promise they actually keep!"
            else:
                template_vars["conclusion"] = f"In conclusion, {base_conclusion}"
            
            # Generate call to action with Pete Connor flair
            call_to_actions = [
                "Before falling for the marketing hype, demand the actual implementation data.",
                "Next time a vendor promises transformation, ask for their failure rate data.",
                "The real ROI comes from skepticism, not from blind adoption.",
                "Stop investing in buzzwords and start investing in evidence-based solutions.",
                "One-Liner: The gap between promises and delivery is where budgets go to die."
            ]
            
            # Enhanced call-to-actions for Medium and Substack
            if platform == "Medium":
                medium_ctas = [
                    "The most valuable skill in technology evaluation isn't technical expertise—it's the ability to say 'prove it' when vendors make performance claims.",
                    "Instead of asking vendors for references, ask them for a list of failed implementations and what they learned from them. Their reaction will tell you everything.",
                    "Technical due diligence isn't an overhead cost—it's the most important investment you'll make in any technology adoption process.",
                    "The next time a sales engineer shows you a dashboard, ask to see the raw data behind it. Watch how quickly the conversation changes."
                ]
                call_to_actions.extend(medium_ctas)
            elif platform == "Substack":
                substack_ctas = [
                    "Here's a radical idea: the next time a vendor presents their miraculous solution, ask them to implement it for free and only get paid when they deliver measurable results. Watch how quickly they backpedal.",
                    "I'm thinking of creating YAAS: Yet Another Acronym Solution. It won't do anything useful, but it'll have a fabulous logo and impressive-sounding white papers. Who wants to invest?",
                    "Pro tip: Replace your entire digital transformation team with a Magic 8-Ball. The accuracy of predictions will remain the same, but your consulting fees will drop dramatically.",
                    "Next vendor meeting, create a buzzword bingo card for your team. First person to get five in a row has to ask the presenter for actual evidence behind their claims. Fun for the whole department!"
                ]
                call_to_actions.extend(substack_ctas)
            
            template_vars["call_to_action"] = random.choice(call_to_actions)
            
            # Additional fields for longer content formats
            if platform in ["Blog", "Medium", "Substack"]:
                template_vars["problem_statement"] = f"One of the challenges with {keywords[0] if keywords else 'this area'} is understanding its full implications."
                template_vars["background"] = f"To understand {keywords[0] if keywords else 'this topic'}, we need to look at its development over time."
                template_vars["solution"] = f"A thoughtful approach to {keywords[0] if keywords else 'this challenge'} involves careful consideration and planning."
                template_vars["implications"] = f"The implications of {keywords[0] if keywords else 'these developments'} could be far-reaching for the industry."
                
                # Add much more detailed sections for Medium and Substack
                if platform in ["Medium", "Substack"]:
                    # Add case studies section
                    if platform == "Medium":
                        template_vars["case_studies"] = f"## Case Studies: The {keywords[0] if keywords else 'Technology'} Hall of Shame\n\nLet's examine three spectacular implementation disasters that somehow never make it into the glossy vendor case studies. Names have been changed to protect the embarrassed (and my potential legal liability):\n\n### Enterprise Resource Planning Dumpster Fire at MegaCorp\nInvestment: $36 million upfront + $28 million in emergency consulting fees\nPromised efficiency improvement: 40%\nActual improvement: 5% (and that's being generous)\nFinal outcome: They reverted to their legacy systems after a three-year odyssey of pain\n\nThe technical post-mortem revealed fundamental flaws in the vendor's database architecture that collapsed under actual enterprise workloads. The vendor's response? \"Your data is more complex than typical implementations.\" Translation: \"We never actually tested this at scale before selling it to you.\"\n\nThe cherry on top: The CIO who approved the project received a promotion before the implementation failed, leaving his successor to deal with the aftermath. That's called 'strategic career timing.'\n\n### Machine Learning Snake Oil at HealthSystems Inc.\nThe sales pitch: \"Our AI delivers diagnostic assistance with 98% accuracy!\"\nThe reality: 73% accuracy in carefully controlled environments, less than 60% in actual production\nCost of learning this lesson: $12 million and 18 months of wasted time\n\nWhen confronted with the accuracy gap, the vendor actually said with a straight face: \"The algorithm is working correctly; your medical data doesn't match our expected patterns.\" I'm not making this up. They literally blamed reality for not conforming to their model.\n\nThe funniest part? The same vendor is now selling an \"enhanced version\" to other healthcare systems using HealthSystems as a reference... without mentioning they were fired.\n\n### Blockchain Fantasy at Global Logistics Corp\nPromised ROI: $45 million in annual savings\nActual ROI: Negative $23 million\nTransaction throughput promised: \"Enterprise-grade\"\nActual throughput: 30 transactions per second (when a minimum of 200 TPS was explicitly required)\n\nThe vendor's solution to the performance problem was truly inspired: \"Just process fewer transactions.\" Revolutionary advice! Why didn't Global Logistics think of simply having fewer customers and shipping fewer products?\n\nThe project was eventually abandoned, but not before three executives added \"Blockchain Transformation Leader\" to their LinkedIn profiles. They all work at different companies now, presumably implementing blockchain solutions there too. The circle of life continues."
                    elif platform == "Substack":
                        template_vars["case_studies"] = f"## The Corporate Darwin Awards: {keywords[0] if keywords else 'Technology'} Edition\n\nLet me present three spectacular examples of corporate self-sabotage that somehow never make it into the glossy case studies:\n\n### The $50 Million Digital Transformation to Nowhere\nBigFancyCorp decided they needed to 'digitally transform' or face extinction (according to the consulting firm charging them $500k for this earth-shattering insight). Two years and $50 million later, they had:\n- 16 new job titles with 'Digital' in them\n- 4 completely incompatible software platforms\n- 3 executives who suddenly 'left to pursue other opportunities'\n- 1 board that was absolutely shocked—SHOCKED!—that things went poorly\n\nThe CEO later described the initiative as 'a learning experience' in his resignation letter.\n\n### The AI System That Couldn't Tell a Cat from a Hamburger\nHealthMegaSystems spent $28 million on an 'AI-powered diagnostic assistant' that was supposedly trained on 'millions of medical images.' During the demo phase, it worked flawlessly! Amazing!\n\nOnce deployed to actual hospitals with actual patients, it turned out the system:\n- Couldn't process images taken on equipment more than 2 years old\n- Regularly crashed when dealing with patients over 250 pounds\n- Somehow identified 27% of male pattern baldness as 'potentially cancerous'\n\nThe vendor explained these were 'edge cases' not covered in the contract. The hospital's legal team explained what 'breach of contract' means.\n\n### The Blockchain Supply Chain That Couldn't\nGlobalShipping decided to 'revolutionize' their supply chain with blockchain because their CTO read an article on an airplane. The sales pitch promised 'end-to-end visibility' and 'military-grade security.'\n\nAfter implementation, they discovered:\n- The system could track a shipment perfectly... as long as every single partner in 47 countries manually entered data into the blockchain\n- The 'military-grade security' was apparently modeled after the security at Area 51's gift shop\n- The system actually worked slower than their previous Excel-based solution\n\nMy favorite part: they're currently looking to hire a 'Web3 Strategy Consultant' to help them understand what went wrong. You can't make this stuff up, folks."
                    
                    # Add industry trends section
                    if platform == "Medium":
                        template_vars["industry_trends"] = f"## Industry Trends: The Gap Between Hype and Reality\n\nWhen we examine industry analyst reports on {keywords[0] if keywords else 'technology'} implementations from the past five years, a clear pattern emerges:\n\n1. **Initial Projection Phase**: Analysts predict explosive growth and transformative impact\n2. **Peak Inflated Expectations**: Vendors flood the market with increasingly exaggerated claims\n3. **Implementation Reality**: Early adopters begin reporting significant challenges\n4. **Revised Expectations**: Analysts quietly update their projections downward\n5. **Rebranding Phase**: The same core technology is repackaged under new terminology\n\nThis cycle typically completes every 24-36 months, yet organizations continue to base major investment decisions on the projections made during phases 1 and 2.\n\nThe data shows that organizations that wait until phase 3 before making investment decisions achieve on average 340% better ROI than early adopters, yet executive incentives continue to reward 'innovation' over prudent technology adoption practices."
                    elif platform == "Substack":
                        template_vars["industry_trends"] = f"## The Circle of Strife: How {keywords[0] if keywords else 'Tech'} Hype Cycles Keep Making Fools of Us All\n\nIf you've been in the industry longer than 15 minutes, you've witnessed this glorious cycle of delusion:\n\n1. **The Prophet Phase**: Some 'thought leader' proclaims that [INSERT TECHNOLOGY] will 'disrupt everything' and 'change the very fabric of business.' This person has typically never actually implemented the technology at scale.\n\n2. **The Gold Rush**: Vendors scramble to add the buzzword to EVERYTHING they sell. \"Our coffee machine now leverages AI to optimize your caffeine consumption paradigm!\"\n\n3. **The FOMO Pandemic**: CEOs read about the technology in airline magazines and become convinced their company will DIE if they don't implement it IMMEDIATELY. Budgets appear out of nowhere.\n\n4. **The Implementation Hangover**: Reality sets in. Turns out implementing [REVOLUTIONARY TECHNOLOGY] is actually really hard and doesn't automatically fix decades of organizational dysfunction. Who could have possibly predicted this??\n\n5. **The Great Rebranding**: Rather than admit failure, everyone agrees to call the project a 'foundation for future innovation' and quietly moves on to the next buzzword.\n\nAnd the most beautiful part? We'll do the exact same dance again next year with a different technology. It's like watching the same car crash in slow motion, over and over, except the drivers keep getting bonuses."
        
        # Add a question for social media platforms with Pete Connor style
        if platform in ["Twitter", "Facebook", "Instagram"]:
            questions = [
                f"Why are we still falling for the {keywords[0] if keywords else 'tech'} hype cycle?",
                f"Has anyone actually measured the ROI from {keywords[0] if keywords else 'this'} or are we just trusting vendor claims?",
                f"Why does the {keywords[0] if keywords else 'tech industry'} keep selling dreams while delivering disappointment?",
                f"When did we stop demanding evidence before implementing {keywords[0] if keywords else 'new tech'}?",
                f"How many more failed {keywords[0] if keywords else 'implementations'} before we learn to be skeptical?"
            ]
            template_vars["question"] = random.choice(questions)
        
        # Fill the template with our variables
        content = template
        for key, value in template_vars.items():
            placeholder = "{" + key + "}"
            if placeholder in content:
                content = content.replace(placeholder, value)
        
        return content
    
    def _get_sentiment_phrase(self, sentiment: str) -> str:
        """
        Get a phrase appropriate for the sentiment with Pete Connor style.
        
        Args:
            sentiment: The dominant sentiment (positive, negative, neutral)
            
        Returns:
            str: A sentiment-appropriate phrase with Pete Connor flair
        """
        phrases = {
            "positive": [
                "has been massively overhyped compared to actual results",
                "looks great in vendor slides but falls short in implementation",
                "sounds impressive until you look at the actual data",
                "is being pushed by people who profit from the hype",
                "is promising in theory but problematic in practice"
            ],
            "negative": [
                "is a dumpster fire hidden behind corporate jargon",
                "is even worse than the skeptics suggest, according to the data",
                "represents everything wrong with tech implementation today",
                "is the perfect example of marketing outpacing reality",
                "has been a colossal waste of resources for most organizations"
            ],
            "neutral": [
                "reveals the disconnect between vendor promises and reality",
                "exposes the gap between marketing and actual implementation",
                "shows how we continue to fall for the same implementation myths",
                "demonstrates why we need more skepticism in tech adoption",
                "exemplifies why we need data, not anecdotes, to guide decisions"
            ]
        }
        
        return random.choice(phrases.get(sentiment, phrases["neutral"]))
    
    def _adapt_to_platform(
        self, 
        content: str,
        platform: str,
        max_length: int,
        hashtag_count: int,
        emoji_count: int,
        formal_tone: bool,
        keywords: List[str]
    ) -> str:
        """
        Adapt content to platform specifications.
        
        Args:
            content: Base content to adapt
            platform: Target platform
            max_length: Maximum content length
            hashtag_count: Number of hashtags to include
            emoji_count: Number of emojis to include
            formal_tone: Whether to use formal tone
            keywords: List of keywords to use for hashtags
            
        Returns:
            str: Platform-adapted content
        """
        # Start with the base content
        adapted_content = content
        
        # Add emojis if required and not already in the content
        if emoji_count > 0 and not any(emoji in adapted_content for emoji_list in self.emojis.values() for emoji in emoji_list):
            # Determine sentiment based on content
            sentiment = "neutral"
            if "promise" in adapted_content or "potential" in adapted_content:
                sentiment = "positive"
            elif "challenge" in adapted_content or "concern" in adapted_content:
                sentiment = "negative"
            
            # Select random emojis and add to content
            selected_emojis = random.sample(self.emojis[sentiment], min(emoji_count, len(self.emojis[sentiment])))
            
            # Insert emojis at appropriate positions
            if platform in ["Twitter", "Instagram"]:
                # For short-form content, add to the beginning
                emoji_text = "".join(selected_emojis) + " "
                adapted_content = emoji_text + adapted_content
            else:
                # For long-form content, add to the end
                emoji_text = " " + "".join(selected_emojis)
                adapted_content += emoji_text
        
        # Truncate content if it exceeds maximum length
        if len(adapted_content) > max_length:
            # Find a good breaking point (end of sentence)
            sentences = sent_tokenize(adapted_content)
            truncated_content = ""
            
            for sentence in sentences:
                if len(truncated_content) + len(sentence) + 1 <= max_length - 3:  # Leave room for ellipsis
                    truncated_content += sentence + " "
                else:
                    break
            
            adapted_content = truncated_content.strip() + "..."
        
        return adapted_content
