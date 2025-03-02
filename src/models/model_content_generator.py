"""
Content generator using fine-tuned model for C. Pete Connor's writing style.

This module generates content using a fine-tuned model rather than templates,
capturing C. Pete Connor's distinctive satirical tech expert voice.
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

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
    
    def __init__(
        self,
        model_dir: str = "outputs/finetune/final",
        device: str = "auto",
        use_wandb: bool = True
    ):
        """
        Initialize the model-based content generator.
        
        Args:
            model_dir: Directory containing the fine-tuned model
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
            use_wandb: Whether to use W&B for monitoring
        """
        self.model_dir = Path(model_dir)
        self.use_wandb = use_wandb
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.sia = None
        
        # Load platform specifications
        self.platform_specs = self._load_platform_specs()
        
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
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
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
        
        # Add platform-specific instructions
        if platform.lower() == "twitter":
            prompt += "Create a short Twitter post in C. Pete Connor's satirical tech expert style. "
            prompt += "Include hashtags and emojis where appropriate. "
            prompt += "Keep it under 280 characters.\n"
        elif platform.lower() == "linkedin":
            prompt += "Create a LinkedIn post in C. Pete Connor's satirical tech expert style. "
            prompt += "Start with an attention-grabbing headline. "
            prompt += "Include some data-driven insights and end with a memorable conclusion.\n"
        elif platform.lower() == "blog":
            prompt += "Create a blog introduction in C. Pete Connor's satirical tech expert style. "
            prompt += "The introduction should hook the reader with a contrarian observation, "
            prompt += "mention some data points, and set up the rest of the article.\n"
        elif platform.lower() == "facebook":
            prompt += "Create a Facebook post in C. Pete Connor's satirical tech expert style. "
            prompt += "Make it engaging and conversational, with a touch of irony.\n"
        elif platform.lower() == "instagram":
            prompt += "Create an Instagram caption in C. Pete Connor's satirical tech expert style. "
            prompt += "Make it visually descriptive and include relevant hashtags.\n"
        elif platform.lower() == "email":
            prompt += "Create an email newsletter introduction in C. Pete Connor's satirical tech expert style. "
            prompt += "Start with a strong hook and share a unique perspective on the topic.\n"
        else:
            prompt += f"Create content for {platform} in C. Pete Connor's satirical tech expert style.\n"
        
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
        
        try:
            # Generate content
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
            logger.error(f"Error generating content: {e}")
            return ["Error generating content. Please try again."]
    
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
    generator = ModelContentGenerator()
    content = "The latest AI models claim to be revolutionary, but they're repeating the same patterns we've seen for years."
    
    platforms = ["twitter", "linkedin", "blog", "facebook", "instagram", "email"]
    
    for platform in platforms:
        generated = generator.generate_content(content, platform)
        print(f"\n=== {platform.upper()} ===")
        print(generated[0])
    
    generator.close()
