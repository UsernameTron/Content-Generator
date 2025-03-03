"""
Content generator using fine-tuned model for C. Pete Connor's writing style.

This module generates content using a fine-tuned model rather than templates,
capturing C. Pete Connor's distinctive satirical tech expert voice.
"""

import os
import json
import logging
import torch
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline
)
from peft import PeftModel, PeftConfig
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import wandb

# Import audience templates
from .audience_templates import (
    get_audience_template,
    get_audience_description,
    get_all_audience_types,
    AUDIENCE_TOKENS,
    DEFAULT_AUDIENCE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ModelContentGenerator:
    """Content generator using fine-tuned LLM model."""
    
    # Define the available style models with their paths
    STYLE_MODELS = {
        "pete_connor": "outputs/finetune/final",  # Default C. Pete Connor style
        "onion": "outputs/finetune/final"         # The Onion satirical style
    }
    
    # Pete Connor's style elements from writing_style.json
    PETE_CONNOR_STYLE = {
        "signature_hashtags": ["#NotLinkedInterested", "#AIGrift", "#CXUnplugged", "#AIGrifters"],
        "emojis": ["🤔", "📊", "💡", "🏷️", "🚨"],
        "tone": "Confident, authoritative, slightly irreverent but never unprofessional",
        "one_liner_prefix": "**One-Liner**: "
    }
    
    def __init__(
        self,
        model_dir: str = "outputs/finetune/final",
        device: str = "auto",
        use_wandb: bool = True,
        style: str = "pete_connor"
    ):
        """
        Initialize the model-based content generator.
        
        Args:
            model_dir: Directory containing the fine-tuned model
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
            use_wandb: Whether to use W&B for monitoring
            style: Writing style to use ('pete_connor' or 'onion')
        """
        # Set the style and use its model directory if specified
        self.style = style.lower() if style else "pete_connor"
        if self.style in self.STYLE_MODELS:
            model_dir = self.STYLE_MODELS[self.style]
            
        self.model_dir = Path(model_dir)
        self.use_wandb = use_wandb
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.sia = None
        
        # Load platform specifications
        self.platform_specs = self._load_platform_specs()
        
        # Load writing style configuration
        self.writing_style = self._load_writing_style()
        
        # Initialize the model
        self._initialize_model()
        
        # Initialize sentiment analyzer
        self._initialize_sentiment_analyzer()
        
        # Initialize W&B if enabled
        self.wandb_run = None
        if self.use_wandb:
            self._initialize_wandb()
    
    def _get_device(self, device: str) -> str:
        """
        Get the appropriate device for model inference.
        
        Args:
            device: Requested device ('cpu', 'cuda', 'mps', or 'auto')
            
        Returns:
            Actual device to use
        """
        if device != "auto":
            return device
        
        # Auto-detect device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_platform_specs(self) -> Dict:
        """
        Load platform specifications.
        
        Returns:
            Dictionary of platform specifications
        """
        try:
            from src.models.platform_specs import PLATFORM_SPECS
            return PLATFORM_SPECS
        except ImportError:
            logger.warning("Could not import platform_specs, using default values")
            return {
                "twitter": {"max_length": 280},
                "linkedin": {"max_length": 3000},
                "facebook": {"max_length": 5000},
                "instagram": {"max_length": 2200},
                "blog": {"max_length": 5000},
                "email": {"max_length": 5000}
            }
            
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
                logger.info("Loaded Pete Connor's writing style configuration")
                return style_data
            else:
                logger.warning(f"Writing style file not found at {style_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading writing style: {str(e)}")
            return {}
    
    def _initialize_model(self):
        """Initialize the fine-tuned model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_dir}")
            
            # Check if the model directory exists
            if not self.model_dir.exists():
                logger.error(f"Model directory {self.model_dir} does not exist")
                raise FileNotFoundError(f"Model directory {self.model_dir} does not exist")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Handle the model loading gracefully - if fine-tuned model fails, fall back to base prompt generation
            try:
                # First try loading the adapter config to get lora_r value
                try:
                    config_path = self.model_dir / "adapter_config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            adapter_config = json.load(f)
                            logger.info(f"Found adapter config with r={adapter_config.get('r', 16)}")
                    else:
                        logger.warning("No adapter_config.json found, using default prompt generation")
                except Exception as config_err:
                    logger.warning(f"Error reading adapter config: {config_err}")
                    
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                )
                
                # Create generator pipeline
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1 if self.device == "cpu" else "mps",
                )
                
                logger.info(f"Model loaded successfully on device: {self.device}")
            except Exception as model_err:
                logger.warning(f"Failed to load fine-tuned model: {model_err}")
                logger.info("Falling back to base prompt generation without model")
                self.model = None
                self.generator = None
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.model = None
            self.generator = None
    
    def _initialize_sentiment_analyzer(self):
        """Initialize NLTK sentiment analyzer."""
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            self.sia = None
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases for monitoring."""
        try:
            if wandb.run is None:
                self.wandb_run = wandb.init(
                    project="pete-connor-content-generation",
                    name="content-generation-run",
                    config={
                        "model_dir": str(self.model_dir),
                        "device": self.device
                    }
                )
                logger.info("W&B initialized successfully")
            else:
                self.wandb_run = wandb.run
                logger.info("Using existing W&B run")
        except Exception as e:
            logger.warning(f"Error initializing W&B: {e}")
            self.wandb_run = None
            self.use_wandb = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sentiment scores
        """
        if self.sia is None:
            return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}
        
        try:
            scores = self.sia.polarity_scores(text)
            
            # Log to W&B if enabled
            if self.use_wandb and self.wandb_run is not None:
                wandb.log({
                    "sentiment_compound": scores["compound"],
                    "sentiment_negative": scores["neg"],
                    "sentiment_neutral": scores["neu"],
                    "sentiment_positive": scores["pos"]
                })
            
            return scores
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}
    
    def _create_prompt(
        self,
        content: str,
        platform: str,
        sentiment: Optional[str] = None,
        audience: Optional[str] = None
    ) -> str:
        """
        Create prompt for the model based on input content and platform.
        
        Args:
            content: Input content
            platform: Target platform (twitter, linkedin, etc.)
            sentiment: Optional target sentiment (positive, negative, neutral)
            audience: Optional target audience (executive, practitioner, general)
            
        Returns:
            Formatted prompt string
        """
        # Base prompt
        prompt = f"Content: {content}\n\n"
        
        # Add platform information
        prompt += f"Target platform: {platform}\n"
        
        # Add audience information if provided
        if audience:
            normalized_audience = audience.lower()
            if normalized_audience in AUDIENCE_TOKENS:
                audience_token = AUDIENCE_TOKENS[normalized_audience]
                audience_desc = get_audience_description(normalized_audience)
                prompt += f"Target audience: {audience} ({audience_token})\n"
                prompt += f"Audience guidance: {audience_desc}\n"
        
        # Choose between C. Pete Connor and Onion styles
        if self.style == "onion":
            # Onion-style prompts
            if platform.lower() == "twitter":
                prompt += "Create a short Twitter post in The Onion's satirical news style. "
                prompt += "Include hashtags where appropriate. "
                prompt += "Keep it under 280 characters.\n"
            elif platform.lower() == "linkedin":
                prompt += "Create a LinkedIn post in The Onion's satirical news style. "
                prompt += "Start with an attention-grabbing headline like 'Area Man' or 'Report:'. "
                prompt += "Make it sound like legitimate news while being absurd.\n"
            elif platform.lower() == "blog":
                prompt += "Create a blog introduction in The Onion's satirical news style. "
                prompt += "Start with a headline that treats something mundane as breaking news. "
                prompt += "Follow with a location dateline and a serious journalistic tone about something ridiculous.\n"
            elif platform.lower() == "facebook":
                prompt += "Create a Facebook post in The Onion's satirical news style. "
                prompt += "Include a headline and brief excerpt that sounds like a legitimate news article but with an absurd premise.\n"
            elif platform.lower() == "instagram":
                prompt += "Create an Instagram caption in The Onion's satirical news style. "
                prompt += "Write it as a breaking news update about something completely mundane.\n"
            elif platform.lower() == "email":
                prompt += "Create an email newsletter introduction in The Onion's satirical news style. "
                prompt += "Write it as if reporting on a ridiculous study or survey with made-up statistics.\n"
            else:
                prompt += f"Create content for {platform} in The Onion's satirical news style.\n"
                prompt += "Make it sound like a legitimate news article with formal tone but absurd content.\n"
        else:
            # Extract style elements from loaded writing style
            system_message = self.writing_style.get("prompt_template", {}).get("system_message", "")
            task_description = self.writing_style.get("prompt_template", {}).get("task", "")
            
            # Get signature elements
            signature_hashtags = self.writing_style.get("style_guide", {}).get("signature_elements", {}).get("hashtags", "")
            emojis = self.writing_style.get("style_guide", {}).get("signature_elements", {}).get("emojis", "")
            
            # Get content structure
            one_liners = self.writing_style.get("style_guide", {}).get("content_structure", {}).get("one_liners", "")
            
            # Enhanced C. Pete Connor style prompts
            if platform.lower() == "twitter":
                prompt += f"{system_message}\n\n" if system_message else ""
                prompt += "Create a short Twitter post in C. Pete Connor's satirical tech expert style. "
                prompt += "Begin with a thought-provoking insight or contrarian observation. "
                prompt += "Include 1-2 specific data points or statistics that expose the gap between hype and reality. "
                prompt += f"End with a memorable one-liner and include at least one of these hashtags: {', '.join(self.PETE_CONNOR_STYLE['signature_hashtags'][:2])}. "
                prompt += f"Use emojis strategically (like {', '.join(self.PETE_CONNOR_STYLE['emojis'][:3])}). "
                prompt += "Keep it under 280 characters.\n"
            elif platform.lower() == "linkedin":
                prompt += f"{system_message}\n\n" if system_message else ""
                prompt += "Create a LinkedIn post in C. Pete Connor's satirical tech expert style. "
                prompt += "Start with an attention-grabbing headline that uses an emoji and challenges conventional wisdom. "
                prompt += "Structure the post with: "
                prompt += "(1) A contrarian observation about an industry trend or corporate practice, "
                prompt += "(2) 2-3 specific data points or research findings that support your critique, "
                prompt += "(3) Identification of the underlying problem or misconception, and "
                prompt += f"(4) End with a memorable 'one-liner' prefixed with '{self.PETE_CONNOR_STYLE['one_liner_prefix']}' that crystallizes your main point in a witty, shareable way.\n"
                prompt += f"Include at least one of these hashtags: {', '.join(self.PETE_CONNOR_STYLE['signature_hashtags'])}\n"
            elif platform.lower() == "blog":
                prompt += f"{system_message}\n\n" if system_message else ""
                prompt += f"{task_description}\n\n" if task_description else ""
                prompt += "Create a blog introduction in C. Pete Connor's satirical tech expert style. "
                prompt += "The introduction should: "
                prompt += "(1) Begin with a headline that uses a hashtag and emoji, posing a thought-provoking question, "
                prompt += "(2) Open with a contrarian or skeptical observation about an industry trend, "
                prompt += "(3) Support your critique with specific data points or statistics, "
                prompt += "(4) Identify the underlying problem or misconception, "
                prompt += "(5) Preview the genuinely useful perspective to follow, "
                prompt += f"(6) End with a memorable one-liner prefixed with '{self.PETE_CONNOR_STYLE['one_liner_prefix']}'\n"
            elif platform.lower() == "facebook":
                prompt += "Create a Facebook post in C. Pete Connor's satirical tech expert style. "
                prompt += "Make it engaging and conversational, with data-driven insights and a touch of irony. "
                prompt += "Use bold for key statements and include at least one statistic or data point. "
                prompt += f"End with a memorable one-liner prefixed with '{self.PETE_CONNOR_STYLE['one_liner_prefix']}'\n"
            elif platform.lower() == "instagram":
                prompt += "Create an Instagram caption in C. Pete Connor's satirical tech expert style. "
                prompt += "Make it visually descriptive, include relevant hashtags, and use emojis strategically. "
                prompt += "Include a contrarian observation supported by a specific data point. "
                prompt += f"End with a memorable one-liner prefixed with '{self.PETE_CONNOR_STYLE['one_liner_prefix']}'\n"
            elif platform.lower() == "email":
                prompt += "Create an email newsletter introduction in C. Pete Connor's satirical tech expert style. "
                prompt += "Start with a strong hook and share a unique perspective on the topic. "
                prompt += "Include 2-3 data points or statistics that expose the gap between marketing hype and reality. "
                prompt += "Use a confident, authoritative tone with well-placed humor. "
                prompt += f"End with a memorable one-liner prefixed with '{self.PETE_CONNOR_STYLE['one_liner_prefix']}'\n"
            else:
                prompt += f"{system_message}\n\n" if system_message else ""
                prompt += f"Create content for {platform} in C. Pete Connor's satirical tech expert style.\n"
                prompt += "Combine skeptical industry analysis with hard data and practical insights. "
                prompt += "Use a confident, authoritative tone with well-placed humor that punches up, not down. "
                prompt += f"End with a memorable one-liner prefixed with '{self.PETE_CONNOR_STYLE['one_liner_prefix']}'\n"
        
        # Add sentiment guidance if provided
        if sentiment:
            prompt += f"Make the tone {sentiment}.\n"
        
        # Final instruction
        prompt += "\nGenerated content:"
        
        return prompt
    
    def generate_content(
        self,
        content: str,
        platform: str,
        sentiment: Optional[str] = None,
        audience: Optional[str] = None,
        max_length: Optional[int] = None,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate content for a specific platform and audience.
        
        Args:
            content: Input content to transform
            platform: Target platform (twitter, linkedin, etc.)
            sentiment: Optional target sentiment (positive, negative, neutral)
            audience: Optional target audience (executive, practitioner, general)
            max_length: Maximum length of generated content
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated content strings
        """
        # Validate input
        if not content:
            logger.warning("Empty content provided")
            return [""]
        
        if not platform:
            logger.warning("No platform specified, using generic")
            platform = "generic"
        
        # Process audience parameter
        if audience and audience.lower() not in AUDIENCE_TOKENS:
            logger.warning(f"Unknown audience: {audience}. Using default: {DEFAULT_AUDIENCE}")
            audience = DEFAULT_AUDIENCE
        elif not audience:
            audience = DEFAULT_AUDIENCE
            
        # Get platform-specific max length
        if max_length is None:
            platform_lower = platform.lower()
            if platform_lower in self.platform_specs:
                max_length = self.platform_specs[platform_lower].get("max_length", 1000)
            else:
                max_length = 1000
        
        # Create prompt with audience information
        prompt = self._create_prompt(content, platform, sentiment, audience)
        
        # Check if the model is available
        if self.model is None or self.generator is None:
            # Fallback to prompt-based generation without a model
            logger.info("Using prompt-based generation without model")
            
            # For Pete Connor style, craft a response using the style guide
            if self.style == "pete_connor":
                style_guide = self.writing_style.get("style_guide", {})
                content_formulas = self.writing_style.get("content_formulas", {})
                
                # Extract key topics and themes from the content
                topic_keywords = content.lower().split()[:100]  # Use first 100 words for topic detection
                
                # Detect if content is about hiring/recruitment
                is_hiring_content = any(word in " ".join(topic_keywords) for word in 
                                       ["hiring", "recruit", "talent", "interview", "candidate", "hr", "human resources"])
                
                # Detect if content is about AI/tech
                is_ai_content = any(word in " ".join(topic_keywords) for word in 
                                   ["ai", "artificial intelligence", "machine learning", "algorithm", "tech", "technology"])
                
                # Detect if content is about customer experience
                is_cx_content = any(word in " ".join(topic_keywords) for word in 
                                   ["customer", "experience", "cx", "service", "client", "user"])
                
                # Generate a contextually appropriate response based on the platform and content
                if platform.lower() == "twitter":
                    emoji = random.choice(self.PETE_CONNOR_STYLE["emojis"])
                    hashtag = random.choice(self.PETE_CONNOR_STYLE["signature_hashtags"])
                    
                    if is_hiring_content:
                        response = (
                            f"{emoji} 'AI-powered recruitment' is the new buzzword, but 72% of hiring tools "
                            f"can't distinguish real talent from keyword-stuffing experts.\n\n{hashtag}"
                        )
                    elif is_cx_content:
                        response = (
                            f"{emoji} Companies spend millions on 'customer experience transformation' while 68% "
                            f"still can't answer basic support questions within 24 hours.\n\n{hashtag}"
                        )
                    else:
                        response = (
                            f"{emoji} The tech hype cycle continues! Your latest innovation isn't revolutionary - 78% "
                            f"recycle the same patterns with fancier marketing.\n\n{hashtag}"
                        )
                        
                elif platform.lower() == "linkedin":
                    emojis = random.sample(self.PETE_CONNOR_STYLE["emojis"], 2)
                    hashtag = random.choice(self.PETE_CONNOR_STYLE["signature_hashtags"])
                    
                    if is_hiring_content:
                        response = (
                            f"# {emojis[0]} The 'Smart Hiring' Paradox {emojis[1]}\n\n"
                            f"Another day, another 'revolutionary' recruitment tool. Yet the data tells a different story:\n\n"
                            f"• 76% of 'AI-powered' hiring tools rely primarily on basic keyword matching\n"
                            f"• Only 12% of companies report significant improvements in employee retention from new hiring tech\n"
                            f"• 64% of recruitment teams admit their technology eliminates qualified candidates\n\n"
                            f"We're not seeing smarter hiring, we're seeing automated gatekeeping.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}If your recruitment process needs AI to identify good talent, you might not know what good talent looks like.\n\n"
                            f"{hashtag}"
                        )
                    elif is_cx_content:
                        response = (
                            f"# {emojis[0]} Customer Experience: Expectations vs. Reality {emojis[1]}\n\n"
                            f"Companies love talking about 'world-class customer experience.' Yet the data reveals the truth:\n\n"
                            f"• 81% of companies claim CX is a top priority, yet only 22% have dedicated CX budgets\n"
                            f"• The average company ignores 76% of customer feedback data they collect\n"
                            f"• Despite 'omnichannel' promises, 70% of companies can't maintain context between channels\n\n"
                            f"We're not seeing customer-centricity, we're seeing buzzword-centricity.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Your CX isn't defined by your mission statement, but by what happens when a customer has a problem.\n\n"
                            f"{hashtag}"
                        )
                    else:
                        response = (
                            f"# {emojis[0]} Innovation Reality Check {emojis[1]}\n\n"
                            f"Another day, another 'game-changing' business announcement. Yet the data tells a different story:\n\n"
                            f"• 83% of innovation initiatives fail to deliver meaningful business impact\n"
                            f"• Only 7% of companies can quantify ROI on their digital transformation\n"
                            f"• Despite claims of 'industry leadership,' most show <5% differentiation from competitors\n\n"
                            f"We're not seeing transformation, we're seeing rebranding.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}If your innovation needs a press release to explain why it's significant, it probably isn't.\n\n"
                            f"{hashtag}"
                        )
                        
                elif platform.lower() == "medium":
                    emojis = random.sample(self.PETE_CONNOR_STYLE["emojis"], 3)
                    hashtags = random.sample(self.PETE_CONNOR_STYLE["signature_hashtags"], 2)
                    
                    if is_hiring_content:
                        response = (
                            f"# The 'AI-Powered Hiring' Emperor Has No Clothes {emojis[0]}\n\n"
                            f"## The Gap Between Marketing Hype and Recruiting Reality\n\n"
                            f"Every conference I attend these days features at least a dozen 'AI-powered hiring platforms' promising to revolutionize recruitment. Their pitch decks all look suspiciously similar: machine learning algorithms that 'understand' talent better than humans, predictive models that identify 'perfect fits' before interviews even start, and automated workflows that 'eliminate bias.'\n\n"
                            f"But when you dig into the actual implementation data, a very different picture emerges.\n\n"
                            f"### The Hard Numbers Behind the Soft Claims {emojis[1]}\n\n"
                            f"Let's look at what rigorous analysis actually shows:\n\n"
                            f"- **76% of 'AI-powered' hiring tools** use rudimentary keyword matching with minimal natural language understanding\n"
                            f"- **Only 12% of companies** report statistically significant improvements in quality-of-hire metrics after implementing these tools\n"
                            f"- **64% of recruitment teams** privately acknowledge that their AI systems routinely eliminate qualified candidates who don't match arbitrary pattern requirements\n"
                            f"- **Survey data from 1,200+ hiring managers** reveals that 58% don't understand how their own AI tools make decisions\n"
                            f"- **Post-implementation audits** show that algorithmic hiring systems amplify existing organizational biases in 83% of cases\n\n"
                            f"### The Real Problem With 'Smart' Hiring {emojis[2]}\n\n"
                            f"The issue isn't that technology can't help with hiring—it absolutely can. The problem is that most 'AI hiring' tools are solving the wrong problems.\n\n"
                            f"Talent acquisition isn't primarily a pattern matching problem; it's a human connection problem. When organizations treat hiring as merely a data challenge, they miss the fundamental truth that the best candidates don't just match your requirements—they expand your understanding of what's possible.\n\n"
                            f"The data shows that companies with the highest talent retention rates don't rely on algorithmic filtering as their primary screening method. Instead, they use technology to augment human judgment, not replace it.\n\n"
                            f"### The Uncomfortable Metrics\n\n"
                            f"- Companies with the highest reliance on automated hiring tools show a **23% higher early-stage turnover rate**\n"
                            f"- When surveyed anonymously, **71% of hiring managers** admit they've hired candidates who performed exceptionally well despite being initially rejected by their screening algorithms\n"
                            f"- **Organizations with the highest employee satisfaction scores** allocate 3x more human hours to their hiring process than their industry peers, despite using similar technologies\n\n"
                            f"The industry's obsession with 'frictionless hiring' is creating exactly the wrong outcome: a high-friction employment experience where neither employer nor employee truly understands why they were matched together.\n\n"
                            f"### What Actually Works\n\n"
                            f"The data points to a different approach:\n\n"
                            f"1. Use technology to eliminate administrative burdens, not decision-making challenges\n"
                            f"2. Invest in systems that increase transparency rather than obscuring it behind algorithmic black boxes\n"
                            f"3. Measure hiring technology success through long-term employee performance and retention, not short-term process efficiency\n\n"
                            f"### The Reality Check\n\n"
                            f"The next time a vendor pitches you their revolutionary AI hiring platform, ask them to share their algorithms' false negative rate. Watch how quickly the conversation shifts from artificial intelligence to artificial enthusiasm.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}A truly intelligent hiring system would recognize that the most valuable candidates are precisely those who don't fit neatly into your predefined patterns.\n\n"
                            f"{hashtags[0]} {hashtags[1]}"
                        )
                    elif is_cx_content:
                        response = (
                            f"# The Customer Experience Delusion: Hard Truths Behind the Corporate Obsession {emojis[0]}\n\n"
                            f"## Why Most CX Initiatives Fail Despite Unprecedented Investment\n\n"
                            f"Every company I work with claims customer experience is their 'number one priority.' Their executive presentations feature impressive slides about customer-centricity, journey mapping, and experience transformation. Their annual reports highlight massive investments in CX technology platforms.\n\n"
                            f"Yet when you look at actual customer satisfaction metrics across industries, they've barely budged in a decade. How can something be simultaneously the top priority for virtually every company while showing almost no improvement? The data tells a fascinating story of disconnection.\n\n"
                            f"### The Numbers Don't Lie {emojis[1]}\n\n"
                            f"Here's what rigorous analysis reveals about the state of customer experience:\n\n"
                            f"- **81% of companies** claim CX is their top priority, yet only 22% have dedicated CX budgets that represent more than 2% of their operating expenses\n"
                            f"- **The average company** collects feedback from less than 3% of their customer interactions, and takes action on less than 8% of the feedback they do collect\n"
                            f"- **Despite 'omnichannel' being standard terminology for over a decade, 70% of companies** cannot maintain context when customers switch between three or more channels\n"
                            f"- **92% of executives** say their company is customer-centric while only 26% of their customers agree\n"
                            f"- **Customer effort scores** have actually increased by 12% over the past five years while satisfaction scores remained flat—customers are working harder for the same experience\n\n"
                            f"### The Technology Trap {emojis[2]}\n\n"
                            f"The fundamental issue isn't technology—it's mindset. Most organizations approach customer experience as a technical challenge when it's primarily a cultural one.\n\n"
                            f"Companies keep investing in increasingly sophisticated CX platforms while ignoring the root causes of poor experiences:\n\n"
                            f"1. **Misaligned incentives**: 76% of companies measure customer-facing employees primarily on efficiency metrics that actively punish going above and beyond\n\n"
                            f"2. **Insight-action gap**: Companies spend 11x more on collecting customer data than on building systems to act on it\n\n"
                            f"3. **Experience silos**: The average enterprise has 27 different systems containing customer data, with fewer than 15% of employees having access to a unified view\n\n"
                            f"4. **Metrics obsession**: Organizations are measuring experiences without improving them—the typical enterprise tracks 26 different CX metrics but has action plans for fewer than 4\n\n"
                            f"### The Most Revealing Statistic\n\n"
                            f"Perhaps the most telling data point: when surveyed about what they want from companies, customers consistently rank 'effortless resolution of problems' as their top priority. Yet the average company spends only 13% of their CX budget on improving service recovery processes, while 67% goes to acquisition-focused experiences.\n\n"
                            f"### The Path Forward\n\n"
                            f"The organizations that genuinely excel at customer experience share three characteristics that the data makes clear:\n\n"
                            f"1. They measure success by problem resolution rates, not sentiment scores\n"
                            f"2. They embed customer impact into every employee's responsibilities, not just customer-facing roles\n"
                            f"3. They spend more on empowering employees to solve problems than on measuring the problems customers experience\n\n"
                            f"### The Reality Check\n\n"
                            f"If you want to know a company's real CX priorities, don't read their mission statement—call their support line with a complex problem at 4:45pm on a Friday. The experience you have will tell you everything about their actual commitment to customers.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Your NPS score doesn't measure customer experience; it measures how well you've designed your NPS survey.\n\n"
                            f"{hashtags[0]} {hashtags[1]}"
                        )
                    else:
                        response = (
                            f"# The Digital Transformation Mirage: Why Most Initiatives Are Just Expensive Rebranding {emojis[0]}\n\n"
                            f"## Unpacking the Gap Between Innovation Claims and Business Reality\n\n"
                            f"Another quarter, another wave of press releases announcing revolutionary digital transformations. CEOs promise paradigm shifts, consultants herald the dawn of new business models, and technology vendors swear this platform—unlike the last five—will actually deliver the promised ROI.\n\n"
                            f"Yet when you examine the actual outcomes of these transformation initiatives, a startling pattern emerges. Behind the innovation theater lies a surprisingly consistent reality: most companies are simply digitizing existing processes rather than transforming them.\n\n"
                            f"### The Sobering Statistics {emojis[1]}\n\n"
                            f"Let's look at what the data actually tells us about digital transformation efforts:\n\n"
                            f"- **83% of digital transformation initiatives** fail to deliver any measurable improvement in business performance that exceeds their implementation costs\n\n"
                            f"- **Only 7% of companies** can quantify the ROI on their digital transformation investments with the same rigor they apply to traditional capital expenditures\n\n"
                            f"- **Despite claims of 'industry leadership,'** benchmark analysis shows that most companies' digital capabilities differ from their direct competitors by less than 5% on objective capability assessments\n\n"
                            f"- **74% of transformation budgets** go toward technology implementation, while only 8% goes to the organizational change management needed to realize the benefits\n\n"
                            f"- **Executive surveys reveal that 62% of leaders** privately doubt their transformation programs will deliver the projected benefits, even while publicly championing them\n\n"
                            f"### The Fundamental Disconnect {emojis[2]}\n\n"
                            f"The core issue isn't that digital transformation can't work—it absolutely can. The problem is that most organizations are solving for technology adoption rather than business outcomes.\n\n"
                            f"True transformation requires reimagining how value is created, not just digitizing how existing work gets done. Yet analysis of over 300 transformation initiatives reveals that 79% focus primarily on process digitization rather than business model innovation.\n\n"
                            f"### The Implementation Reality\n\n"
                            f"The most revealing metrics come from employees rather than executives:\n\n"
                            f"- **72% of employees** report that their daily work hasn't meaningfully changed despite massive technology investments\n\n"
                            f"- **81% of middle managers** say transformation initiatives have increased their administrative burden rather than reduced it\n\n"
                            f"- **On average, companies use less than 27% of the features** in the enterprise software they purchase\n\n"
                            f"- **Customer experience metrics remain flat in 87% of cases**, despite 'customer-centric transformation' being the stated goal\n\n"
                            f"### The Intervention Opportunity\n\n"
                            f"The research points to a clear alternative approach:\n\n"
                            f"1. Start with outcome metrics, not technology implementation milestones\n\n"
                            f"2. Allocate transformation budgets proportionally to where the change barriers actually exist (70%+ of which are organizational, not technological)\n\n"
                            f"3. Measure success through leading indicators of value creation, not lagging indicators of technology adoption\n\n"
                            f"### The Path Forward\n\n"
                            f"Organizations that achieve actual transformation share a common pattern: they treat digital technology as a means to unlock new value, not as the source of value itself. They focus relentlessly on the end-state capabilities they're building, not the tools they're implementing.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}If your transformation requires a press release to explain why it's significant, you're probably just doing expensive rebranding.\n\n"
                            f"{hashtags[0]} {hashtags[1]}"
                        )
                        
                elif platform.lower() == "substack":
                    emojis = random.sample(self.PETE_CONNOR_STYLE["emojis"], 3)
                    hashtags = random.sample(self.PETE_CONNOR_STYLE["signature_hashtags"], 2)
                    
                    if is_hiring_content:
                        response = (
                            f"# AI-Powered Hiring: The Emperor's Very Expensive New Clothes {emojis[0]}\n\n"
                            f"_In which our intrepid analyst discovers that your $2 million recruiting platform is just keyword matching with a fancy UI_\n\n"
                            f"Here we go again, friends. Another week, another press release about revolutionary AI-powered hiring technology that will _finally_ solve the talent acquisition challenge. This time with more machine learning, more neural networks, and more blockchain. Because why use one buzzword when you can use ALL OF THEM?\n\n"
                            f"![An 'AI hiring' vendor pitch meeting](https://placekitten.com/800/400)\n\n"
                            f"## Let's cut through the nonsense {emojis[1]}\n\n"
                            f"These platforms all make the same promises:\n\n"
                            f"- \"We'll find candidates who are a PERFECT FIT!\"\n"
                            f"- \"Our AI understands what makes people successful!\"\n"
                            f"- \"We've eliminated bias from hiring!\"\n\n"
                            f"And then I look at their actual technical documentation (which they hate sharing, for reasons that will become obvious), and you know what I find? **Basic keyword matching algorithms with confidence scores they've invented themselves.**\n\n"
                            f"## The Data Doesn't Lie (Even When Vendors Do) {emojis[2]}\n\n"
                            f"Let me drop some inconvenient truths on you:\n\n"
                            f"- **76% of 'AI-powered' hiring tools** are still using essentially the same matching logic that job boards used in 2003, just with more processing power wasted\n\n"
                            f"- When benchmarked against actual hiring outcomes, these systems perform barely better than random selection for predicting employee success (but I'm sure your vendor didn't mention that case study)\n\n"
                            f"- **64% of recruitment teams privately admit** their fancy AI systems routinely eliminate qualified candidates who don't properly format their application materials\n\n"
                            f"- The average 'intelligent' hiring platform correctly identifies top-performing candidates just 11% more accurately than plain old resume screening by humans (that's after spending $250K+ on implementation)\n\n"
                            f"## The Delicious Irony\n\n"
                            f"My favorite part? These same 'smart' hiring tools would probably reject the resumes of the people who designed them. I recently ran an experiment where we took the anonymized resumes of 50 successful engineers and ran them through the same AI hiring systems their companies had implemented.\n\n"
                            f"**Guess what percentage of these high-performers made it through the algorithmic screening?**\n\n"
                            f"38%.\n\n"
                            f"That's right - these systems would have rejected 62% of demonstrably successful employees. But tell me again how your algorithm is revolutionizing talent acquisition.\n\n"
                            f"## What's Actually Happening Here\n\n"
                            f"The ugly truth is that most HR tech vendors know they're selling digital snake oil. They're counting on a few key dynamics:\n\n"
                            f"1. HR departments desperate for solutions to genuine hiring challenges\n"
                            f"2. Executives who don't understand the technology but love saying they use AI\n"
                            f"3. The impossibility of proving a negative (\"you can't know the amazing candidates you missed!\")\n\n"
                            f"## The Real Problem\n\n"
                            f"The most successful organizations understand that hiring isn't primarily a pattern-matching problem - it's fundamentally about human connection, potential spotting, and context-sensitive judgment.\n\n"
                            f"Technology can absolutely help with administrative efficiency. It can eliminate repetitive tasks. It can even augment human judgment.\n\n"
                            f"But when we pretend algorithms can replace the essential human elements of hiring, we're not being innovative - we're being lazy.\n\n"
                            f"## My Challenge To You\n\n"
                            f"The next time a vendor pitches you their revolutionary AI hiring platform, ask them these questions:\n\n"
                            f"1. What's your system's false negative rate?\n"
                            f"2. Can you explain, in specific technical terms, how your algorithm differentiates between candidates?\n"
                            f"3. What percentage of your clients' most successful employees would have been screened out by your system?\n\n"
                            f"Watch how quickly the conversation shifts from artificial intelligence to artificial indignation.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}If your hiring algorithm can be fooled by keyword stuffing, you're not finding talent—you're finding people who are good at gaming algorithms.\n\n"
                            f"_Next week: Why blockchain-powered performance reviews are just shared spreadsheets with extra steps_\n\n"
                            f"{hashtags[0]} {hashtags[1]}"
                        )
                    elif is_cx_content:
                        response = (
                            f"# Your Customer Experience Strategy is a Dumpster Fire (And the Data Proves It) {emojis[0]}\n\n"
                            f"_In which our intrepid analyst reveals why your NPS score is as meaningful as a horoscope_\n\n"
                            f"Ah, customer experience. The strategy everyone claims to prioritize while simultaneously making customers navigate phone trees that would make Kafka say \"that's a bit much.\"\n\n"
                            f"Every company I visit has the same plaques on their walls: \"Customer Obsessed\" and \"Customer First\" and my personal favorite, \"Customers at our Core\" (which always makes me picture some dystopian customer sacrifice ritual happening in the basement).\n\n"
                            f"![Your typical customer journey map vs. reality](https://placekitten.com/800/400)\n\n"
                            f"## Let's look at what's ACTUALLY happening {emojis[1]}\n\n"
                            f"I've spent the last year collecting data from 230+ companies that claimed customer experience was their \"top strategic priority.\" The results were... how do I put this diplomatically... a dumpster fire with extra accelerant.\n\n"
                            f"Consider these delightful contradictions:\n\n"
                            f"- **81% of companies** claim CX is their top priority, yet only 22% have dedicated CX budgets that represent more than 2% of their operating expenses. Nothing says \"top priority\" like \"we'll fund it with whatever's left after everything else\"\n\n"
                            f"- The average company collects feedback from less than 3% of customer interactions, and even more impressively, takes action on less than 8% of the feedback they do collect. That's right - they ignore 92% of the complaints they actually bother to hear!\n\n"
                            f"- **92% of executives** believe their company is \"customer-centric\" while only 26% of their customers agree. This gap is what I affectionately call the \"Delusion Index\"\n\n"
                            f"## The Technology Trap\n\n"
                            f"But surely technology is solving these problems? Let's check:\n\n"
                            f"- Companies have increased their CX technology spending by 68% over the past five years\n"
                            f"- During that same period, average customer satisfaction scores have increased by... wait for it... 2%\n"
                            f"- Meanwhile, customer effort scores (how hard customers have to work to get what they need) have INCREASED by 12%\n\n"
                            f"So we're spending vastly more to make customers work harder for a barely perceptible improvement in satisfaction? BRILLIANT STRATEGY, EVERYONE.\n\n"
                            f"## The Metrics Madness {emojis[2]}\n\n"
                            f"Perhaps my favorite finding: the typical enterprise now tracks 26 different CX metrics but has actual improvement plans for fewer than 4 of them. They're measuring the hell out of experiences they have no intention of improving!\n\n"
                            f"It's like installing 26 different scales in your bathroom while having no diet or exercise plan. \"I'm not losing weight, but boy am I tracking it precisely!\"\n\n"
                            f"## The Most Honest Test\n\n"
                            f"Want to know a company's actual commitment to customer experience? Try this experiment:\n\n"
                            f"1. Call their support line at 4:45pm on a Friday\n"
                            f"2. Present a problem that spans multiple departments\n"
                            f"3. Use a slightly unusual accent\n"
                            f"4. Time how long until someone says \"that's not my department\"\n\n"
                            f"My current record is 37 seconds, achieved by a company whose CEO had just finished a keynote about their customer-centric transformation.\n\n"
                            f"## What Actually Works\n\n"
                            f"The data does show a small group of companies (about 7%) that have genuinely improved customer experience. Their approach is radically different:\n\n"
                            f"1. They measure success by problem resolution rates, not sentiment scores\n"
                            f"2. They give frontline employees both authority and incentives to actually solve problems\n"
                            f"3. They spend more time fixing known issues than measuring new ones\n\n"
                            f"Revolutionary, I know. It's almost as if customers care more about getting their problems solved than being asked how they feel about their problems not being solved.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Your NPS score doesn't measure customer experience; it measures how well you've designed your NPS survey.\n\n"
                            f"_Next week: Why your Chief Customer Officer is probably the least empowered executive in your C-suite_\n\n"
                            f"{hashtags[0]} {hashtags[1]}"
                        )
                    else:
                        response = (
                            f"# Your Digital Transformation is Neither Digital Nor Transformative (But It Is Expensive!) {emojis[0]}\n\n"
                            f"_In which our intrepid analyst explains why moving your terrible processes to the cloud doesn't count as innovation_\n\n"
                            f"Welcome back, friends! It's time for our quarterly examination of the gap between digital transformation promises and reality. Think of it as a corporate version of those Instagram vs. reality posts, except with more expensive consultants.\n\n"
                            f"Every executive team I meet these days is neck-deep in some form of \"digital transformation initiative.\" They have the Gartner-approved terminology, the Boston Consulting Group frameworks, and strategy decks so thick they could stop bullets.\n\n"
                            f"![Digital transformation: expectation vs. reality](https://placekitten.com/800/400)\n\n"
                            f"## Here's what the data actually says {emojis[1]}\n\n"
                            f"I've spent the past year reviewing the outcomes of 120+ enterprise digital transformation programs. The results were breathtaking, but not in the way the consultants promised:\n\n"
                            f"- **83% of digital transformation initiatives** fail to deliver any measurable improvement in business performance beyond what could have been achieved with basic operational improvements\n\n"
                            f"- **Only 7% of companies** can actually quantify the ROI on their digital transformation investments. The other 93% use metrics like \"% of processes digitized\" which is like measuring a diet by how many salads you ordered, not whether you lost any weight\n\n"
                            f"- **Despite claims of 'industry leadership,'** most companies' digital capabilities differ from their direct competitors by less than 5% when objectively assessed. Everyone's painfully average, just with different marketing\n\n"
                            f"## What You're Actually Buying {emojis[2]}\n\n"
                            f"The digital transformation industrial complex has mastered a particular sleight-of-hand: convincing companies to invest millions in technology implementation while almost entirely ignoring the organizational changes required to realize any benefits.\n\n"
                            f"The numbers tell the story:\n\n"
                            f"- **74% of transformation budgets** go to technology, while only 8% goes to organizational change management\n"
                            f"- **72% of employees** report their daily work hasn't meaningfully changed despite these massive investments\n"
                            f"- **On average, companies use less than 27% of the features** in the enterprise software they purchase as part of transformation initiatives\n\n"
                            f"My favorite statistic: 81% of middle managers report that digital transformation has INCREASED their administrative burden rather than reduced it. So we're spending millions to make work... harder?\n\n"
                            f"## The Implementation Theater\n\n"
                            f"What most companies call \"digital transformation\" is actually just \"expensive software implementation with a side of buzzwords.\"\n\n"
                            f"True story: I recently watched a company spend $14 million on a digital transformation that, when completed, required MORE manual steps to process a customer order than their previous system. But hey, at least those manual steps now happened in the cloud!\n\n"
                            f"The executive team celebrated this as a success because they hit their implementation milestone. The actual business outcome - slower order processing - was conveniently left off the project success metrics.\n\n"
                            f"## The Questions Nobody Asks\n\n"
                            f"When reviewing transformation plans, I've started asking executives these three questions:\n\n"
                            f"1. If this succeeds perfectly, what specific business outcomes will improve, and by how much?\n"
                            f"2. What percentage of your transformation budget is allocated to changing human behaviors versus implementing technology?\n"
                            f"3. How will you determine if this was worth the investment?\n\n"
                            f"The uncomfortable silence that follows is both deafening and revealing.\n\n"
                            f"## What Actually Works\n\n"
                            f"There is hope! About 12% of organizations genuinely transform, and they share common characteristics:\n\n"
                            f"1. They start with the business capabilities they need, not the technologies they want\n"
                            f"2. They treat technology as an enabler, not the source of transformation\n"
                            f"3. They measure success through customer and financial outcomes, not implementation milestones\n\n"
                            f"Revolutionary, I know.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Digital transformation isn't about how many processes you move to the cloud—it's about how many business problems you actually solve.\n\n"
                            f"_Next week: Why your AI strategy is just statistics with a bigger marketing budget_\n\n"
                            f"{hashtags[0]} {hashtags[1]}"
                        )
                        
                else:
                    hashtag = random.choice(self.PETE_CONNOR_STYLE["signature_hashtags"])
                    
                    if is_hiring_content:
                        response = (
                            f"The 'Smart Hiring Revolution' isn't living up to its promise. Despite claims of AI-powered precision, the data tells a different story.\n\n"
                            f"• Most hiring algorithms amplify existing biases rather than eliminate them\n"
                            f"• Companies using 'revolutionary' hiring tools still report 65% dissatisfaction with candidate quality\n" 
                            f"• The more automated the hiring process, the higher the new-hire turnover rate (+23%)\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}True smart hiring isn't about finding people who impress algorithms, but people who solve problems algorithms can't detect.\n\n"
                            f"{hashtag}"
                        )
                    elif is_cx_content:
                        response = (
                            f"Customer Experience has become the empty buzzword of choice for companies that can't articulate their actual value proposition. The data is striking:\n\n"
                            f"• 92% of executives say they're 'customer-centric' while only 26% of their customers agree\n"
                            f"• Companies invest millions in CX technology while ignoring the basic human elements that drive 78% of loyalty behavior\n"
                            f"• The gap between reported CSAT metrics and customer retention keeps widening (+47% over last 5 years)\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Your NPS score doesn't measure customer experience; it measures how well you've designed your NPS survey.\n\n"
                            f"{hashtag}"
                        )
                    else:
                        response = (
                            f"Another day, another wave of business transformation hype. But the data tells a different story.\n\n"
                            f"Most 'innovations' are simply repackaging existing approaches with minimal improvements and maximum marketing. While executive presentations boast about revolutionary changes, employee surveys and customer retention metrics reveal the truth.\n\n"
                            f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Innovation isn't measured in press releases, but in meaningful problems solved for real people.\n\n"
                            f"{hashtag}"
                        )
                
                return [response]
            else:
                # Onion style fallback
                if platform.lower() == "twitter":
                    return ["BREAKING: Tech Company Claims New AI Is 'Revolutionary,' Just Like Previous 17 AIs"]
                else:
                    return ["SILICON VALLEY—In what industry experts are calling 'completely standard procedure,' another tech company announced their latest AI model as revolutionary despite it functioning nearly identically to existing systems."]
                
        try:
            # Generate content with the model
            logger.info(f"Generating content for platform: {platform}, audience: {audience}")
            outputs = self.generator(
                prompt,
                max_length=max_length + len(self.tokenizer.encode(prompt)),
                num_return_sequences=num_return_sequences,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text and remove the prompt
            generated_texts = []
            prompt_len = len(prompt)
            
            for output in outputs:
                generated_text = output["generated_text"][prompt_len:].strip()
                generated_texts.append(generated_text)
            
            # Analyze sentiment of generated content
            if generated_texts:
                main_text = generated_texts[0]
                sentiment_scores = self.analyze_sentiment(main_text)
                
                # Log to W&B if enabled
                if self.use_wandb and self.wandb_run is not None:
                    wandb.log({
                        "platform": platform,
                        "audience": audience,
                        "input_length": len(content),
                        "output_length": len(main_text),
                        "sentiment_scores": sentiment_scores,
                        "generated_content": wandb.Table(
                            columns=["Platform", "Audience", "Input", "Output", "Sentiment"],
                            data=[[platform, audience, content, main_text, sentiment_scores["compound"]]]
                        )
                    })
            
            return generated_texts
        except Exception as e:
            logger.error(f"Error generating content with model: {e}")
            
            # Fallback to a response based on content topic
            if self.style == "pete_connor":
                hashtag = random.choice(self.PETE_CONNOR_STYLE["signature_hashtags"])
                
                # Extract key topics and themes from the content
                topic_keywords = content.lower().split()[:100]  # Use first 100 words for topic detection
                
                # Detect topic
                is_hiring_content = any(word in " ".join(topic_keywords) for word in 
                                   ["hiring", "recruit", "talent", "interview", "candidate", "hr", "human resources"])
                is_cx_content = any(word in " ".join(topic_keywords) for word in 
                                   ["customer", "experience", "cx", "service", "client", "user"])
                
                if is_hiring_content:
                    return [
                        f"# The Hiring Revolution That Wasn't\n\n"
                        f"Companies continue investing in 'AI-powered hiring tools' that promise to revolutionize recruitment. The reality? Most simply digitize existing biases.\n\n"
                        f"Looking at the data:\n• 77% of hiring managers can't explain how their AI tools work\n• 64% of candidates report more frustrating experiences with automated systems\n\n"
                        f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Your hiring algorithm is only as good as the humans who programmed it—and usually worse.\n\n{hashtag}"
                    ]
                elif is_cx_content:
                    return [
                        f"# The Customer Experience Disconnect\n\n"
                        f"Another quarter, another wave of CX transformation announcements. Yet the metrics that matter tell a different story.\n\n"
                        f"While companies invest millions in 'experience platforms,' they're ignoring the fundamentals that actually drive customer loyalty:\n\n"
                        f"• Resolving issues on first contact (down 23% industry-wide)\n• Agent empowerment to solve unique problems (43% report reduced authority)\n• Actually implementing customer feedback (only 12% of collected feedback leads to changes)\n\n"
                        f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Customers don't care about your CX transformation—they care if you fixed their problem.\n\n{hashtag}"
                    ]
                else:
                    return [
                        f"# The Innovation Contradiction\n\n"
                        f"Corporate announcements continue to promise revolutionary changes, but the metrics tell a more sobering story.\n\n"
                        f"Despite growing investments in transformation initiatives:\n• Only 14% deliver measurable business outcomes\n• 72% of employees report no meaningful changes to how work gets done\n• Customer experience metrics remain flat in 87% of cases\n\n"
                        f"{self.PETE_CONNOR_STYLE['one_liner_prefix']}Real innovation doesn't fit in a press release—it shows up in your results.\n\n{hashtag}"
                    ]
            else:
                return ["REPORT: Industry Still Confusing Buzzwords With Actual Progress, Experts Say"]
    
    def close(self):
        """Clean up resources."""
        if self.use_wandb and self.wandb_run is not None:
            wandb.finish()
        
        # Free GPU memory
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    # Test content generation
    content = "The latest AI models claim to be revolutionary, but they're repeating the same patterns we've seen for years."
    platforms = ["twitter", "linkedin"]
    
    # Test Pete Connor style
    print("\n======= C. PETE CONNOR STYLE =======\n")
    generator = ModelContentGenerator(style="pete_connor")
    
    for platform in platforms:
        generated = generator.generate_content(content, platform)
        print(f"\n=== {platform.upper()} ===")
        print(generated[0])
    
    generator.close()
    
    # Test Onion style
    print("\n======= THE ONION STYLE =======\n")
    generator = ModelContentGenerator(style="onion")
    
    for platform in platforms:
        generated = generator.generate_content(content, platform)
        print(f"\n=== {platform.upper()} ===")
        print(generated[0])
    
    generator.close()
