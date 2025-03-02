"""
Platform-Specific Parameters for Satirical Content Generation.

This module implements detailed specifications for adapting satirical content
across multiple digital platforms. Each platform has unique requirements,
audience expectations, and optimal formatting guidelines that significantly 
impact content performance and audience engagement.

The specifications enable the CANDOR system to transform raw satirical content
into platform-optimized formats that maintain the core satirical elements while
respecting each platform's constraints and best practices.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContentFormat(Enum):
    """Content format types supported across platforms."""
    SHORT_TEXT = "short_text"
    PROFESSIONAL_POST = "professional_post"
    VIDEO_SCRIPT = "video_script"
    ARTICLE = "article"
    NEWSLETTER = "newsletter"
    CONVERSATIONAL = "conversational"
    VISUAL_CAPTION = "visual_caption"


class ContentType(Enum):
    """Content topic/purpose types supported across platforms."""
    CONCISE_OPINION = "concise_opinion"
    INDUSTRY_INSIGHT = "industry_insight"
    EDUCATIONAL_ENTERTAINMENT = "educational_entertainment" 
    IN_DEPTH_ANALYSIS = "in_depth_analysis"
    CURATED_INSIGHT = "curated_insight"
    OPINION_SHARE = "opinion_share"
    ENGAGING_STORY = "engaging_story"


@dataclass
class FormatCapabilities:
    """Formatting capabilities available on a specific platform."""
    supports_bold: bool = False
    supports_italic: bool = False
    supports_headings: bool = False
    supports_lists: bool = False
    supports_blockquotes: bool = False
    supports_links: bool = True
    supports_images: bool = False
    supports_videos: bool = False
    supports_code_blocks: bool = False
    heading_levels: List[int] = field(default_factory=list)
    
    def get_capabilities_summary(self) -> str:
        """Returns a summary of available formatting capabilities."""
        capabilities = []
        if self.supports_bold: capabilities.append("bold")
        if self.supports_italic: capabilities.append("italic")
        if self.supports_headings: capabilities.append(f"headings (levels {self.heading_levels})")
        if self.supports_lists: capabilities.append("lists")
        if self.supports_blockquotes: capabilities.append("blockquotes")
        if self.supports_links: capabilities.append("links")
        if self.supports_images: capabilities.append("images")
        if self.supports_videos: capabilities.append("videos")
        if self.supports_code_blocks: capabilities.append("code blocks")
        
        return ", ".join(capabilities)


@dataclass
class ContentStructureTemplate:
    """Template for structuring content based on platform best practices."""
    sections: List[str]
    section_order: List[str]
    recommended_section_lengths: Dict[str, Dict[str, int]] = field(default_factory=dict)
    optional_sections: List[str] = field(default_factory=list)
    
    def get_section_guidelines(self, content_length: str = "medium") -> Dict[str, int]:
        """
        Returns the recommended character count for each section.
        
        Args:
            content_length: Size category ("short", "medium", "long")
            
        Returns:
            Dict mapping section names to recommended character counts
        """
        if content_length in self.recommended_section_lengths:
            return self.recommended_section_lengths[content_length]
        else:
            # Default to medium if specified length not found
            return self.recommended_section_lengths.get("medium", {})


@dataclass
class EngagementGuidelines:
    """Guidelines for optimizing content engagement on each platform."""
    recommended_post_times: List[str]
    optimal_post_frequency: str
    engagement_prompts: List[str]
    call_to_action_styles: List[str]
    best_performing_content_types: List[str]
    
    def get_random_engagement_prompt(self) -> str:
        """Returns a random engagement prompt appropriate for the platform."""
        import random
        return random.choice(self.engagement_prompts)
    
    def get_random_cta(self) -> str:
        """Returns a random call-to-action template appropriate for the platform."""
        import random
        return random.choice(self.call_to_action_styles)


@dataclass
class HashtagStrategy:
    """Platform-specific hashtag usage guidelines."""
    recommended_count: int
    max_count: int
    placement: str  # "start", "end", "integrated", "comments"
    popular_hashtags: List[str]
    satire_specific_hashtags: List[str]
    
    def get_recommended_hashtags(self, industry: str, topics: List[str]) -> List[str]:
        """
        Returns a strategically selected set of hashtags based on content.
        
        Args:
            industry: Content's industry category
            topics: Key topics covered in the content
            
        Returns:
            List of recommended hashtags for this platform
        """
        import random
        
        # Industry-specific hashtags
        industry_hashtags = {
            "technology": ["#TechTrends", "#Innovation", "#DigitalTransformation", "#TechLife"],
            "business": ["#BusinessStrategy", "#Leadership", "#Management", "#GrowthMindset"],
            "marketing": ["#MarketingTips", "#ContentMarketing", "#DigitalMarketing", "#BrandStrategy"],
            "healthcare": ["#HealthTech", "#MedicalInnovation", "#Healthcare", "#WellnessTrends"],
            "education": ["#EdTech", "#LearningJourney", "#Education", "#TeachingInnovation"],
            "finance": ["#FinTech", "#InvestmentStrategy", "#FinancialFreedom", "#MoneyMatters"],
        }
        
        # Get industry hashtags or use general business if industry not found
        industry_tags = industry_hashtags.get(industry.lower(), industry_hashtags["business"])
        
        # Convert topics to hashtags (camelCase)
        topic_hashtags = []
        for topic in topics[:2]:  # Limit to 2 topic hashtags
            if not topic:
                continue
            # Convert multi-word topics to camelCase hashtags
            if " " in topic:
                words = topic.split()
                hashtag = "#" + words[0].lower() + "".join(word.capitalize() for word in words[1:])
            else:
                hashtag = "#" + topic.lower()
            topic_hashtags.append(hashtag)
        
        # Select satire hashtags
        satire_tags = random.sample(self.satire_specific_hashtags, 
                                  min(2, len(self.satire_specific_hashtags)))
        
        # Combine and limit based on recommended count
        all_hashtags = (
            random.sample(industry_tags, min(2, len(industry_tags))) + 
            topic_hashtags + 
            satire_tags
        )
        
        # Ensure we don't exceed max_count
        return all_hashtags[:min(self.recommended_count, self.max_count)]


@dataclass
class ToneGuidelines:
    """Platform-specific tone and voice guidelines."""
    formality_level: int  # 1-5 scale (1=casual, 5=formal)
    humor_level: int  # 1-5 scale (1=subtle, 5=overt)
    technical_depth: int  # 1-5 scale (1=beginner, 5=expert)
    satire_intensity: int  # 1-5 scale (1=gentle, 5=biting)
    emoji_usage: str  # "none", "minimal", "moderate", "liberal"
    audience_expectations: str
    taboo_topics: List[str] = field(default_factory=list)
    
    def get_tone_description(self) -> str:
        """Returns a human-readable description of the tone guidelines."""
        formality_desc = ["very casual", "casual", "balanced", "professional", "very formal"][self.formality_level-1]
        humor_desc = ["subtle", "light", "moderate", "strong", "very overt"][self.humor_level-1]
        technical_desc = ["beginner-friendly", "simplified", "balanced", "detailed", "expert-level"][self.technical_depth-1]
        satire_desc = ["gentle", "mild", "moderate", "sharp", "biting"][self.satire_intensity-1]
        
        return (f"Content should maintain a {formality_desc} tone with {humor_desc} humor. "
                f"Technical depth should be {technical_desc} with {satire_desc} satire. "
                f"Emoji usage should be {self.emoji_usage}.")


@dataclass
class VisualGuidelines:
    """Guidelines for visual elements on each platform."""
    header_image_dimensions: Optional[str] = None
    inline_image_dimensions: Optional[str] = None
    thumbnail_dimensions: Optional[str] = None
    video_dimensions: Optional[str] = None
    recommended_image_ratio: Optional[str] = None
    max_images: int = 0
    supports_carousels: bool = False
    supports_embeds: bool = False
    
    def get_image_requirements(self) -> str:
        """Returns a summary of image requirements for this platform."""
        requirements = []
        if self.header_image_dimensions:
            requirements.append(f"Header images: {self.header_image_dimensions}")
        if self.inline_image_dimensions:
            requirements.append(f"Inline images: {self.inline_image_dimensions}")
        if self.thumbnail_dimensions:
            requirements.append(f"Thumbnails: {self.thumbnail_dimensions}")
        if self.recommended_image_ratio:
            requirements.append(f"Recommended ratio: {self.recommended_image_ratio}")
        if self.max_images > 0:
            requirements.append(f"Maximum images: {self.max_images}")
        
        return "; ".join(requirements) if requirements else "No image support"


@dataclass
class PlatformFeatures:
    """Special features unique to each platform."""
    special_features: Dict[str, Any] = field(default_factory=dict)
    content_restrictions: List[str] = field(default_factory=list)
    monetization_options: List[str] = field(default_factory=list)
    audience_targeting: bool = False
    
    def has_feature(self, feature_name: str) -> bool:
        """Checks if platform supports a specific feature."""
        return feature_name in self.special_features


@dataclass
class PlatformSpecs:
    """Comprehensive specifications for a content platform."""
    name: str
    min_length: int
    max_length: int
    optimal_length: Dict[str, int]  # Maps content type to optimal character count
    
    # Core specifications
    format_capabilities: FormatCapabilities
    content_structure: ContentStructureTemplate
    engagement: EngagementGuidelines
    hashtag_strategy: HashtagStrategy
    tone: ToneGuidelines
    visual_guidelines: VisualGuidelines
    special_features: PlatformFeatures
    
    # Content adaptation preferences
    prefers_format: ContentFormat
    prefers_content_type: ContentType
    
    def get_length_requirements(self, content_type: Optional[str] = None) -> Dict[str, int]:
        """
        Gets length requirements for this platform.
        
        Args:
            content_type: Optional content type to get specific optimal length
            
        Returns:
            Dict with min, max, and optimal length
        """
        result = {
            "min_length": self.min_length,
            "max_length": self.max_length
        }
        
        if content_type and content_type in self.optimal_length:
            result["optimal_length"] = self.optimal_length[content_type]
        else:
            # Use first optimal length as default
            default_type = next(iter(self.optimal_length))
            result["optimal_length"] = self.optimal_length[default_type]
            
        return result
    
    def validate_content_length(self, content: str) -> bool:
        """
        Validates if content meets platform length requirements.
        
        Args:
            content: Content to validate
            
        Returns:
            Boolean indicating if content length is valid
        """
        content_length = len(content)
        return self.min_length <= content_length <= self.max_length
    
    def suggest_content_adjustments(self, content: str) -> List[str]:
        """
        Suggests adjustments if content doesn't match platform requirements.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of suggested adjustments
        """
        suggestions = []
        content_length = len(content)
        
        if content_length < self.min_length:
            suggestions.append(f"Content is too short ({content_length} chars). Add at least {self.min_length - content_length} characters.")
        
        if content_length > self.max_length:
            suggestions.append(f"Content is too long ({content_length} chars). Remove at least {content_length - self.max_length} characters.")
        
        # Check for hashtag usage
        hashtag_count = content.count('#')
        if hashtag_count > self.hashtag_strategy.max_count:
            suggestions.append(f"Too many hashtags ({hashtag_count}). Maximum recommended: {self.hashtag_strategy.max_count}")
        
        # Check for emoji usage
        import re
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251" 
                                   "]+", flags=re.UNICODE)
        
        emoji_count = len(emoji_pattern.findall(content))
        
        if self.tone.emoji_usage == "none" and emoji_count > 0:
            suggestions.append(f"Remove all emojis. This platform prefers no emoji usage.")
        elif self.tone.emoji_usage == "minimal" and emoji_count > 2:
            suggestions.append(f"Too many emojis ({emoji_count}). Use maximum 1-2 emojis for this platform.")
            
        return suggestions


# Define platform-specific specifications
LINKEDIN_SPECS = PlatformSpecs(
    name="LinkedIn",
    min_length=100,
    max_length=3000,
    optimal_length={
        "industry_insight": 1300,
        "educational_content": 1500,
        "company_update": 800
    },
    format_capabilities=FormatCapabilities(
        supports_bold=True,
        supports_italic=True,
        supports_headings=False,
        supports_lists=True,
        supports_blockquotes=False,
        supports_links=True,
        supports_images=True,
        supports_videos=True,
        supports_code_blocks=False,
        heading_levels=[]
    ),
    content_structure=ContentStructureTemplate(
        sections=["hook", "main_point", "supporting_evidence", "insight", "call_to_action"],
        section_order=["hook", "main_point", "supporting_evidence", "insight", "call_to_action"],
        recommended_section_lengths={
            "short": {
                "hook": 100,
                "main_point": 200,
                "supporting_evidence": 300,
                "insight": 150,
                "call_to_action": 50
            },
            "medium": {
                "hook": 150,
                "main_point": 300,
                "supporting_evidence": 500,
                "insight": 250,
                "call_to_action": 100
            }
        },
        optional_sections=["supporting_evidence", "call_to_action"]
    ),
    engagement=EngagementGuidelines(
        recommended_post_times=["Tuesday 8-10am", "Wednesday 10am-12pm", "Thursday 1-3pm"],
        optimal_post_frequency="2-3 times per week",
        engagement_prompts=[
            "What's your experience with {topic}?",
            "Have you encountered this issue in your organization?",
            "What strategies have worked for you in handling {topic}?",
            "Do you agree this is a challenge in {industry}?",
            "Would love to hear your thoughts on this approach."
        ],
        call_to_action_styles=[
            "I'd be interested to hear your thoughts on this.",
            "Has your experience been similar or different?",
            "If you found this valuable, I'd appreciate you sharing it with your network.",
            "What would you add to this analysis?"
        ],
        best_performing_content_types=["industry insights", "data-backed opinions", "professional development tips"]
    ),
    hashtag_strategy=HashtagStrategy(
        recommended_count=3,
        max_count=5,
        placement="end",
        popular_hashtags=["#Innovation", "#Leadership", "#ProfessionalDevelopment", "#FutureOfWork"],
        satire_specific_hashtags=["#CorporateNonsense", "#JargonAlert", "#BusinessBuzzwords", "#ThoughtLeadershipThoughts"]
    ),
    tone=ToneGuidelines(
        formality_level=4,  # Professional
        humor_level=3,  # Moderate
        technical_depth=4,  # Detailed
        satire_intensity=3,  # Moderate
        emoji_usage="minimal",
        audience_expectations="Professional audience expecting insightful, value-adding content with a balance of expertise and personality",
        taboo_topics=["overly political content", "highly controversial social issues", "profanity", "extreme negativity"]
    ),
    visual_guidelines=VisualGuidelines(
        header_image_dimensions="1200x627 pixels",
        inline_image_dimensions="1200x1200 pixels max",
        thumbnail_dimensions="400x400 pixels",
        recommended_image_ratio="1.91:1",
        max_images=9,
        supports_carousels=True,
        supports_embeds=True
    ),
    special_features=PlatformFeatures(
        special_features={
            "document_sharing": "PDF/PPTX documents up to 100MB",
            "polls": "Up to 4 options, 2-week maximum duration",
            "carousel_posts": "Up to 10 slides per post",
            "native_video": "Up to 10 minutes"
        },
        content_restrictions=[
            "No contests or promotions",
            "No excessive tagging",
            "No recruitment content unless using Jobs features"
        ],
        monetization_options=["Premium articles", "Creator mode", "Sponsored content"],
        audience_targeting=True
    ),
    prefers_format=ContentFormat.PROFESSIONAL_POST,
    prefers_content_type=ContentType.INDUSTRY_INSIGHT
)

YOUTUBE_SPECS = PlatformSpecs(
    name="YouTube",
    min_length=500,
    max_length=5000,
    optimal_length={
        "script": 1800,
        "description": 2000,
    },
    format_capabilities=FormatCapabilities(
        supports_bold=False,
        supports_italic=False,
        supports_headings=False,
        supports_lists=False,
        supports_blockquotes=False,
        supports_links=True,
        supports_images=False,
        supports_videos=True,
        supports_code_blocks=False,
        heading_levels=[]
    ),
    content_structure=ContentStructureTemplate(
        sections=["intro", "hook", "content_overview", "main_segments", "call_to_action", "outro"],
        section_order=["intro", "hook", "content_overview", "main_segments", "call_to_action", "outro"],
        recommended_section_lengths={
            "short": {
                "intro": 150,
                "hook": 100,
                "content_overview": 150,
                "main_segments": 1000,
                "call_to_action": 100,
                "outro": 100
            },
            "medium": {
                "intro": 200,
                "hook": 150,
                "content_overview": 200,
                "main_segments": 2000,
                "call_to_action": 150,
                "outro": 150
            }
        },
        optional_sections=["content_overview"]
    ),
    engagement=EngagementGuidelines(
        recommended_post_times=["Thursday 3-5pm", "Saturday 10am-12pm", "Sunday 9-11am"],
        optimal_post_frequency="1-2 times per week",
        engagement_prompts=[
            "Let me know in the comments if you've experienced this too.",
            "What other topics would you like me to cover?",
            "Have I missed anything important about {topic}?",
            "What's your take on this issue?",
            "Do you agree with my analysis? Let me know below."
        ],
        call_to_action_styles=[
            "If you found this valuable, smash that like button and subscribe for more content.",
            "Don't forget to subscribe and hit the notification bell to stay updated.",
            "Leave your thoughts in the comments below.",
            "Check the description for links to resources mentioned in this video."
        ],
        best_performing_content_types=["how-to guides", "satirical takes", "myth-busting", "industry analysis"]
    ),
    hashtag_strategy=HashtagStrategy(
        recommended_count=5,
        max_count=15,
        placement="description",
        popular_hashtags=["#TechTalk", "#BusinessTips", "#CareerAdvice", "#IndustryInsights"],
        satire_specific_hashtags=["#CorporateSatire", "#TechHumor", "#BusinessJokes", "#IndustryComedy"]
    ),
    tone=ToneGuidelines(
        formality_level=2,  # Casual
        humor_level=4,  # Strong
        technical_depth=3,  # Balanced
        satire_intensity=4,  # Sharp
        emoji_usage="moderate",
        audience_expectations="Visual audience expecting entertaining and educational content with personality and clear explanations",
        taboo_topics=["excessive profanity", "graphic content", "misinformation"]
    ),
    visual_guidelines=VisualGuidelines(
        thumbnail_dimensions="1280x720 pixels",
        video_dimensions="1920x1080 pixels (16:9)",
        recommended_image_ratio="16:9",
        supports_embeds=True
    ),
    special_features=PlatformFeatures(
        special_features={
            "timestamps": "Add timestamps to make content skimmable",
            "cards": "Add clickable cards at specific timestamps",
            "end_screens": "20-second end screens with calls to action",
            "chapters": "Break video into navigable sections"
        },
        content_restrictions=[
            "No copyrighted material",
            "No harmful or dangerous content",
            "No misleading metadata"
        ],
        monetization_options=["Ad revenue", "Channel memberships", "Super Chat", "Merchandise shelf"],
        audience_targeting=False
    ),
    prefers_format=ContentFormat.VIDEO_SCRIPT,
    prefers_content_type=ContentType.EDUCATIONAL_ENTERTAINMENT
)

MEDIUM_SPECS = PlatformSpecs(
    name="Medium",
    min_length=4000,
    max_length=25000,
    optimal_length={
        "article": 9000,
        "data_analysis": 10000,
        "satire": 8000
    },
    format_capabilities=FormatCapabilities(
        supports_bold=True,
        supports_italic=True,
        supports_headings=True,
        supports_lists=True,
        supports_blockquotes=True,
        supports_links=True,
        supports_images=True,
        supports_videos=False,
        supports_code_blocks=True,
        heading_levels=[1, 2, 3, 4]
    ),
    content_structure=ContentStructureTemplate(
        sections=["title", "subtitle", "hook", "introduction", "main_sections", "conclusion", "call_to_action", "bio"],
        section_order=["title", "subtitle", "hook", "introduction", "main_sections", "conclusion", "call_to_action", "bio"],
        recommended_section_lengths={
            "short": {
                "title": 60,
                "subtitle": 120,
                "hook": 150,
                "introduction": 300,
                "main_sections": 1000,
                "conclusion": 200,
                "call_to_action": 100,
                "bio": 100
            },
            "medium": {
                "title": 80,
                "subtitle": 150,
                "hook": 250,
                "introduction": 700,
                "main_sections": 9000,
                "conclusion": 800,
                "call_to_action": 200,
                "bio": 150
            }
        },
        optional_sections=["subtitle", "call_to_action"]
    ),
    engagement=EngagementGuidelines(
        recommended_post_times=["Tuesday 10am-12pm", "Thursday 8-10pm", "Sunday 3-5pm"],
        optimal_post_frequency="1-3 times per week",
        engagement_prompts=[
            "What has your experience been with {topic}?",
            "I'd love to hear your perspective on this issue.",
            "Have you found other approaches that work better?",
            "What other factors should be considered in this analysis?",
            "How has {topic} affected your industry specifically?"
        ],
        call_to_action_styles=[
            "If you enjoyed this article, consider following me for more insights on {topic}.",
            "I write about {industry} topics weekly. Follow to stay updated.",
            "What other {industry} topics would you like me to cover? Respond in the comments.",
            "Check out my other articles on {related_topic} if you found this valuable."
        ],
        best_performing_content_types=["in-depth analyses", "data storytelling", "personal experiences", "satirical takes"]
    ),
    hashtag_strategy=HashtagStrategy(
        recommended_count=5,
        max_count=5,
        placement="end",
        popular_hashtags=["#Technology", "#Data", "#Productivity", "#Leadership", "#Innovation"],
        satire_specific_hashtags=["#TechSatire", "#CorporateHumor", "#StartupLife", "#IndustryIrony"]
    ),
    tone=ToneGuidelines(
        formality_level=3,  # Balanced (less formal, more conversational)
        humor_level=4,  # Strong humor
        technical_depth=5,  # Maximum technical depth
        satire_intensity=5,  # Maximum biting satire 
        emoji_usage="minimal",
        audience_expectations="Readers seeking substantial, technically detailed content with depth and nuance, technical insights, evidence-based arguments, and data-driven analysis delivered with Pete Connor's distinctive sarcastic and humorous style",
        taboo_topics=["clickbait", "low-quality hot takes", "purely promotional content", "oversimplified explanations"]
    ),
    visual_guidelines=VisualGuidelines(
        header_image_dimensions="1500x750 pixels",
        inline_image_dimensions="1500x any height",
        recommended_image_ratio="2:1 for header, any for inline",
        max_images=20,
        supports_embeds=True
    ),
    special_features=PlatformFeatures(
        special_features={
            "publications": "Submit to publications for wider reach",
            "series": "Group related articles into series",
            "member_only_content": "Restrict content to paying Medium members",
            "stats": "Detailed analytics on article performance"
        },
        content_restrictions=[
            "No purely promotional content",
            "No content farming or low-quality articles",
            "No plagiarism"
        ],
        monetization_options=["Medium Partner Program", "Referred memberships"],
        audience_targeting=False
    ),
    prefers_format=ContentFormat.ARTICLE,
    prefers_content_type=ContentType.IN_DEPTH_ANALYSIS
)

SUBSTACK_SPECS = PlatformSpecs(
    name="Substack",
    min_length=4000,
    max_length=35000,
    optimal_length={
        "newsletter": 7000,
        "deep_dive": 10000,
        "satire": 8000,
    },
    format_capabilities=FormatCapabilities(
        supports_bold=True,
        supports_italic=True,
        supports_headings=True,
        supports_lists=True,
        supports_blockquotes=True,
        supports_links=True,
        supports_images=True,
        supports_videos=False,
        supports_code_blocks=True,
        heading_levels=[1, 2, 3]
    ),
    content_structure=ContentStructureTemplate(
        sections=["subject_line", "greeting", "introduction", "main_content", "premium_teaser", "closing", "p.s"],
        section_order=["subject_line", "greeting", "introduction", "main_content", "premium_teaser", "closing", "p.s"],
        recommended_section_lengths={
            "short": {
                "subject_line": 50,
                "greeting": 50,
                "introduction": 200,
                "main_content": 800,
                "premium_teaser": 150,
                "closing": 100,
                "p.s": 100
            },
            "medium": {
                "subject_line": 60,
                "greeting": 100,
                "introduction": 600,
                "main_content": 6000,
                "premium_teaser": 600,
                "closing": 300,
                "p.s": 300
            }
        },
        optional_sections=["premium_teaser", "p.s"]
    ),
    engagement=EngagementGuidelines(
        recommended_post_times=["Sunday 10am", "Tuesday 6am", "Friday 5pm"],
        optimal_post_frequency="1 time per week",
        engagement_prompts=[
            "Reply directly to this email with your thoughts.",
            "I'm curious about your experience with {topic}.",
            "Paid subscribers: What would you like me to cover next?",
            "Which aspect of this analysis resonated most with you?",
            "Did I miss an important angle on this topic?"
        ],
        call_to_action_styles=[
            "If you found this valuable, consider becoming a paid subscriber for exclusive content.",
            "Share this newsletter with colleagues who would appreciate this perspective.",
            "Join the community discussion by commenting below.",
            "Take the subscriber poll to help shape future content."
        ],
        best_performing_content_types=["insider analysis", "premium deep dives", "exclusive perspectives", "curation with commentary"]
    ),
    hashtag_strategy=HashtagStrategy(
        recommended_count=0,
        max_count=3,
        placement="none",
        popular_hashtags=[],
        satire_specific_hashtags=[]
    ),
    tone=ToneGuidelines(
        formality_level=2,  # More casual
        humor_level=5,  # Very strong humor
        technical_depth=4,  # Detailed
        satire_intensity=5,  # Maximum biting satire
        emoji_usage="moderate",
        audience_expectations="Direct, conversational connection with subscribers who expect aggressive sarcasm, scathing satire, and no-holds-barred critique delivered with humor",
        taboo_topics=["purely promotional content", "clickbait", "non-cohesive topics outside your established focus"]
    ),
    visual_guidelines=VisualGuidelines(
        header_image_dimensions="1600x900 pixels recommended",
        inline_image_dimensions="Any dimensions work well",
        recommended_image_ratio="16:9 for header recommended",
        max_images=20,
        supports_embeds=True
    ),
    special_features=PlatformFeatures(
        special_features={
            "email_subject_lines": "Critical for open rates (40-60 characters optimal)",
            "paid_subscriber_sections": "Content exclusive to paying subscribers",
            "discussion_threads": "Enable/disable comments per post",
            "custom_welcome_email": "Personalized for new subscribers"
        },
        content_restrictions=[
            "No spam or misleading content",
            "No excessive self-promotion",
            "No harassment or harmful content"
        ],
        monetization_options=["Paid subscriptions", "Founding member tiers", "Group subscriptions"],
        audience_targeting=True
    ),
    prefers_format=ContentFormat.NEWSLETTER,
    prefers_content_type=ContentType.CURATED_INSIGHT
)

TWITTER_SPECS = PlatformSpecs(
    name="Twitter",
    min_length=1,
    max_length=280,
    optimal_length={
        "tweet": 140,
        "thread_start": 230,
        "thread_continuation": 260
    },
    format_capabilities=FormatCapabilities(
        supports_bold=False,
        supports_italic=False,
        supports_headings=False,
        supports_lists=False,
        supports_blockquotes=False,
        supports_links=True,
        supports_images=True,
        supports_videos=True,
        supports_code_blocks=False,
        heading_levels=[]
    ),
    content_structure=ContentStructureTemplate(
        sections=["hook", "point", "link_or_cta"],
        section_order=["hook", "point", "link_or_cta"],
        recommended_section_lengths={
            "short": {
                "hook": 40,
                "point": 80,
                "link_or_cta": 20
            },
            "medium": {
                "hook": 60,
                "point": 160,
                "link_or_cta": 40
            }
        },
        optional_sections=["link_or_cta"]
    ),
    engagement=EngagementGuidelines(
        recommended_post_times=["Monday 12pm", "Wednesday 9am", "Friday 3pm"],
        optimal_post_frequency="3-5 times per day",
        engagement_prompts=[
            "What's your take?",
            "Agree or disagree?",
            "Has this been your experience?",
            "What would you add?",
            "Hot take or cold fact?"
        ],
        call_to_action_styles=[
            "RT if you agree",
            "Share this if you've experienced this too",
            "Follow for more {topic} insights",
            "Join the conversation below"
        ],
        best_performing_content_types=["hot takes", "contrarian opinions", "data insights", "thread deep-dives"]
    ),
    hashtag_strategy=HashtagStrategy(
        recommended_count=2,
        max_count=3,
        placement="integrated",
        popular_hashtags=["#TechTwitter", "#DataScience", "#Leadership", "#StartupLife"],
        satire_specific_hashtags=["#CorporateBS", "#TechHumor", "#StartupSatire", "#AIGrifters"]
    ),
    tone=ToneGuidelines(
        formality_level=2,  # Casual
        humor_level=4,  # Strong
        technical_depth=3,  # Balanced
        satire_intensity=4,  # Sharp
        emoji_usage="moderate",
        audience_expectations="Rapid, punchy insights with personality; controversy and strong opinions welcome",
        taboo_topics=["excessive self-promotion", "spam", "inflammatory political content"]
    ),
    visual_guidelines=VisualGuidelines(
        inline_image_dimensions="1200x675 pixels",
        thumbnail_dimensions="440x220 pixels",
        recommended_image_ratio="16:9",
        max_images=4,
        supports_carousels=True,
        supports_embeds=True
    ),
    special_features=PlatformFeatures(
        special_features={
            "threads": "Connected tweets for longer narratives",
            "polls": "Quick audience surveys",
            "spaces": "Live audio discussions",
            "bookmarks": "Save content for later reference"
        },
        content_restrictions=[
            "No excessive tagging",
            "No spam or misleading content",
            "No abusive behavior"
        ],
        monetization_options=["Super Follows", "Ticketed Spaces", "Twitter Blue"],
        audience_targeting=False
    ),
    prefers_format=ContentFormat.SHORT_TEXT,
    prefers_content_type=ContentType.CONCISE_OPINION
)

# Map of all available platform specifications
PLATFORM_SPECS_MAP = {
    "LinkedIn": LINKEDIN_SPECS,
    "YouTube": YOUTUBE_SPECS,
    "Medium": MEDIUM_SPECS,
    "Substack": SUBSTACK_SPECS,
    "Twitter": TWITTER_SPECS,
    # Add more platforms as they're defined
}

# Default specifications if a platform is not found
DEFAULT_SPECS = PlatformSpecs(
    name="Generic",
    min_length=100,
    max_length=2000,
    optimal_length={
        "generic": 1000
    },
    format_capabilities=FormatCapabilities(
        supports_bold=True,
        supports_italic=True,
        supports_headings=False,
        supports_lists=True,
        supports_blockquotes=False,
        supports_links=True,
        supports_images=True,
        supports_videos=False,
        heading_levels=[]
    ),
    content_structure=ContentStructureTemplate(
        sections=["introduction", "main_content", "conclusion"],
        section_order=["introduction", "main_content", "conclusion"],
        recommended_section_lengths={
            "short": {
                "introduction": 100,
                "main_content": 300,
                "conclusion": 100
            },
            "medium": {
                "introduction": 200,
                "main_content": 600,
                "conclusion": 200
            }
        },
        optional_sections=[]
    ),
    engagement=EngagementGuidelines(
        recommended_post_times=["Weekdays 9am-5pm"],
        optimal_post_frequency="1-3 times per week",
        engagement_prompts=[
            "What do you think about this perspective?",
            "I'd love to hear your thoughts on this topic.",
            "What's been your experience with this issue?",
            "Did I miss anything important?",
            "What would you add to this analysis?"
        ],
        call_to_action_styles=[
            "Let me know your thoughts in the comments.",
            "Share if you found this valuable.",
            "Follow for more content like this.",
            "What would you like me to cover next?"
        ],
        best_performing_content_types=["educational content", "opinion pieces", "analysis", "how-to guides"]
    ),
    hashtag_strategy=HashtagStrategy(
        recommended_count=3,
        max_count=5,
        placement="end",
        popular_hashtags=["#Content", "#Insights", "#Perspective", "#Analysis"],
        satire_specific_hashtags=["#Satire", "#Humor", "#PerspectiveShift", "#TakeWithGrainOfSalt"]
    ),
    tone=ToneGuidelines(
        formality_level=3,  # Balanced
        humor_level=3,  # Moderate
        technical_depth=3,  # Balanced
        satire_intensity=3,  # Moderate
        emoji_usage="minimal",
        audience_expectations="General audience expecting clear, valuable content with appropriate personality",
        taboo_topics=["offensive content", "spam", "misinformation"]
    ),
    visual_guidelines=VisualGuidelines(
        header_image_dimensions="1200x630 pixels",
        inline_image_dimensions="800x600 pixels",
        recommended_image_ratio="16:9",
        max_images=5,
        supports_carousels=False,
        supports_embeds=True
    ),
    special_features=PlatformFeatures(
        special_features={},
        content_restrictions=[
            "No spam",
            "No offensive content",
            "No misinformation"
        ],
        monetization_options=[],
        audience_targeting=False
    ),
    prefers_format=ContentFormat.PROFESSIONAL_POST,
    prefers_content_type=ContentType.INDUSTRY_INSIGHT
)


def get_platform_specs(platform: str) -> PlatformSpecs:
    """
    Get comprehensive specifications for a specific platform.
    
    This function retrieves the detailed platform specifications needed
    for optimizing content. It handles normalization of platform names
    and provides sensible defaults if a platform isn't found.
    
    Args:
        platform: Target platform name
        
    Returns:
        PlatformSpecs: Comprehensive platform specifications
    """
    # Normalize platform name (case-insensitive lookup)
    normalized_platform = platform.strip().title()
    
    # Lookup platform specs
    if normalized_platform in PLATFORM_SPECS_MAP:
        logger.debug(f"Found specifications for platform: {normalized_platform}")
        return PLATFORM_SPECS_MAP[normalized_platform]
    else:
        logger.warning(f"No specifications found for platform: {platform}. Using defaults.")
        return DEFAULT_SPECS


def get_platform_names() -> List[str]:
    """
    Get a list of all supported platform names.
    
    Returns:
        List[str]: Names of all supported platforms
    """
    return list(PLATFORM_SPECS_MAP.keys())


def get_optimal_platforms_for_content(content: str, content_type: str = None) -> List[Dict[str, Any]]:
    """
    Determine which platforms are best suited for a specific piece of content.
    
    Analyzes content length, structure, and type to suggest optimal platforms
    for distribution with minimal adaptation required.
    
    Args:
        content: The content to analyze
        content_type: Optional type specification (e.g., "industry_insight")
        
    Returns:
        List[Dict]: Platforms sorted by suitability, with reasons
    """
    content_length = len(content)
    results = []
    
    # Check all platforms for suitability
    for platform_name, specs in PLATFORM_SPECS_MAP.items():
        # Skip platforms where content is definitely too long
        if content_length > specs.max_length:
            continue
            
        # Calculate match score based on length and other factors
        length_match = 1.0
        if content_length < specs.min_length:
            length_match = content_length / specs.min_length
        elif content_length > specs.max_length:
            length_match = 0.0
        else:
            # Find the closest optimal length
            optimal_lengths = list(specs.optimal_length.values())
            closest_optimal = min(optimal_lengths, key=lambda x: abs(x - content_length))
            
            # Higher score the closer to optimal length
            deviation = abs(content_length - closest_optimal) / closest_optimal
            length_match = max(0.0, 1.0 - deviation)
        
        # Analyze content structure (simplified analysis)
        has_sections = '\n\n' in content
        has_lists = '- ' in content or '* ' in content
        
        structure_match = 0.7  # Default reasonable match
        if has_sections and specs.format_capabilities.supports_headings:
            structure_match += 0.15
        if has_lists and specs.format_capabilities.supports_lists:
            structure_match += 0.15
        
        # Calculate total suitability score
        suitability = (length_match * 0.6) + (structure_match * 0.4)
        
        # Add reasons for this score
        reasons = []
        if length_match < 0.5:
            reasons.append("Content length doesn't match platform requirements")
        else:
            reasons.append("Content length works well for this platform")
            
        if not has_sections and specs.format_capabilities.supports_headings:
            reasons.append("Could benefit from adding section headings")
            
        if has_lists and not specs.format_capabilities.supports_lists:
            reasons.append("Lists will need to be reformatted")
            
        results.append({
            "platform": platform_name,
            "suitability_score": suitability,
            "reasons": reasons,
            "optimal_length": closest_optimal if 'closest_optimal' in locals() else None,
            "format_capabilities": specs.format_capabilities.get_capabilities_summary()
        })
    
    # Sort by suitability score
    results.sort(key=lambda x: x["suitability_score"], reverse=True)
    return results


def suggest_adaptation_strategies(content: str, source_platform: str, target_platform: str) -> Dict[str, Any]:
    """
    Suggest specific strategies to adapt content from one platform to another.
    
    Analyzes the differences between platforms and provides actionable
    suggestions for effective content adaptation.
    
    Args:
        content: The content to adapt
        source_platform: Original platform
        target_platform: Destination platform
        
    Returns:
        Dict: Adaptation strategies and specific recommendations
    """
    # Get platform specifications
    source_specs = get_platform_specs(source_platform)
    target_specs = get_platform_specs(target_platform)
    
    # Initial validation
    content_length = len(content)
    
    strategies = {
        "length_adjustment": None,
        "format_changes": [],
        "tone_adjustments": [],
        "structure_changes": [],
        "hashtag_recommendations": [],
        "visual_recommendations": []
    }
    
    # Check length requirements
    if content_length > target_specs.max_length:
        strategies["length_adjustment"] = {
            "action": "shorten",
            "current_length": content_length,
            "target_length": target_specs.max_length,
            "reduction_needed": content_length - target_specs.max_length,
            "reduction_percentage": round((content_length - target_specs.max_length) / content_length * 100, 1)
        }
    elif content_length < target_specs.min_length:
        strategies["length_adjustment"] = {
            "action": "lengthen",
            "current_length": content_length,
            "target_length": target_specs.min_length,
            "addition_needed": target_specs.min_length - content_length,
            "addition_percentage": round((target_specs.min_length - content_length) / target_specs.min_length * 100, 1)
        }
    else:
        optimal_length = list(target_specs.optimal_length.values())[0]  # Use first available optimal length
        if abs(content_length - optimal_length) > optimal_length * 0.2:  # More than 20% off optimal
            action = "shorten" if content_length > optimal_length else "lengthen"
            strategies["length_adjustment"] = {
                "action": action,
                "current_length": content_length,
                "target_length": optimal_length,
                "adjustment_needed": abs(content_length - optimal_length),
                "adjustment_percentage": round(abs(content_length - optimal_length) / optimal_length * 100, 1)
            }
    
    # Check formatting capabilities
    for capability, source_has in source_specs.format_capabilities.__dict__.items():
        if isinstance(source_has, bool):
            target_has = getattr(target_specs.format_capabilities, capability)
            if source_has and not target_has:
                strategies["format_changes"].append({
                    "feature": capability.replace("supports_", ""),
                    "action": "remove",
                    "note": f"{target_platform} doesn't support {capability.replace('supports_', '')}"
                })
    
    # Check tone adjustments needed
    if source_specs.tone.formality_level != target_specs.tone.formality_level:
        action = "more formal" if target_specs.tone.formality_level > source_specs.tone.formality_level else "less formal"
        strategies["tone_adjustments"].append({
            "aspect": "formality",
            "action": action,
            "note": f"{target_platform} expects {action} tone than {source_platform}"
        })
        
    if source_specs.tone.humor_level != target_specs.tone.humor_level:
        action = "more humor" if target_specs.tone.humor_level > source_specs.tone.humor_level else "less humor"
        strategies["tone_adjustments"].append({
            "aspect": "humor",
            "action": action,
            "note": f"{target_platform} expects {action} than {source_platform}"
        })
        
    if source_specs.tone.satire_intensity != target_specs.tone.satire_intensity:
        action = "stronger satire" if target_specs.tone.satire_intensity > source_specs.tone.satire_intensity else "milder satire"
        strategies["tone_adjustments"].append({
            "aspect": "satire",
            "action": action,
            "note": f"{target_platform} works better with {action} than {source_platform}"
        })
    
    # Check structure changes
    required_sections = [s for s in target_specs.content_structure.sections 
                       if s not in target_specs.content_structure.optional_sections]
    
    # Simple heuristic to guess what sections are missing
    # (In a real implementation, this would use more sophisticated content analysis)
    for section in required_sections:
        # Simplified check - in reality would need more sophisticated analysis
        if section not in str(content).lower():
            strategies["structure_changes"].append({
                "action": "add",
                "section": section,
                "note": f"Add a {section.replace('_', ' ')} section for {target_platform}"
            })
    
    # Hashtag recommendations
    if target_specs.hashtag_strategy.recommended_count > 0:
        # Extract topics from content (simplified)
        topics = []
        for line in content.split('\n'):
            words = line.split()
            if 3 < len(words) < 10:  # Look for short sentences that might be topics
                topics.append(' '.join(words[:2]))  # Take first two words as potential topic
        
        if len(topics) > 0:
            # Generate hashtag recommendations
            recommended_hashtags = target_specs.hashtag_strategy.get_recommended_hashtags(
                "technology",  # Default industry
                topics[:2]  # Use first two detected topics
            )
            
            strategies["hashtag_recommendations"] = {
                "count": target_specs.hashtag_strategy.recommended_count,
                "placement": target_specs.hashtag_strategy.placement,
                "suggested_hashtags": recommended_hashtags
            }
    
    # Visual recommendations
    if target_specs.visual_guidelines.header_image_dimensions:
        strategies["visual_recommendations"].append({
            "type": "header_image",
            "dimensions": target_specs.visual_guidelines.header_image_dimensions,
            "note": f"Include a header image for better engagement on {target_platform}"
        })
        
    # Add platform-specific features to leverage
    if hasattr(target_specs.special_features, "special_features"):
        strategies["platform_specific_features"] = list(target_specs.special_features.special_features.keys())
    
    return strategies


def format_for_platform(content: str, platform: str, content_type: Optional[str] = None) -> str:
    """
    Format content according to platform specifications.
    
    Applies basic formatting transformations to adapt content to a 
    specific platform's requirements. For advanced adaptation,
    use the dedicated platform adapters.
    
    Args:
        content: Content to format
        platform: Target platform
        content_type: Optional content type for platform-specific processing
        
    Returns:
        str: Formatted content
    """
    platform_specs = get_platform_specs(platform)
    
    # Check if content exceeds maximum length
    if len(content) > platform_specs.max_length:
        # Truncate content with indicator
        truncated_content = content[:platform_specs.max_length - 3] + "..."
        logger.warning(f"Content truncated from {len(content)} to {len(truncated_content)} characters")
        content = truncated_content
    
    # Apply platform-specific formatting
    if platform == "Twitter":
        # For Twitter, ensure hashtags at the end and format links
        # Count existing hashtags
        hashtag_count = content.count('#')
        
        # If under the recommended count, add relevant hashtags
        if hashtag_count < platform_specs.hashtag_strategy.recommended_count:
            # In a real implementation, would analyze content for relevant hashtags
            # For this example, just add some generic ones
            additional_needed = platform_specs.hashtag_strategy.recommended_count - hashtag_count
            if additional_needed > 0:
                hashtags_to_add = " #TechInsights #DataScience"[:additional_needed * 12]  # Approx length per hashtag
                content = content.rstrip() + "\n\n" + hashtags_to_add
    
    elif platform == "LinkedIn":
        # For LinkedIn, ensure post has clear structure and appropriate professional tone
        
        # Add line breaks for readability if content is longer than 500 chars
        if len(content) > 500 and '\n\n' not in content:
            # Add paragraph breaks every ~200 characters at sentence boundaries
            sentences = content.split('. ')
            formatted_content = ""
            current_paragraph = ""
            
            for sentence in sentences:
                if not sentence.endswith('.'):
                    sentence += '.'
                    
                # Check if adding this sentence would make paragraph too long
                if len(current_paragraph) + len(sentence) > 200:
                    formatted_content += current_paragraph.strip() + "\n\n"
                    current_paragraph = sentence + " "
                else:
                    current_paragraph += sentence + " "
            
            # Add the last paragraph
            if current_paragraph:
                formatted_content += current_paragraph.strip()
                
            content = formatted_content
    
    elif platform == "Medium":
        # For Medium, ensure proper formatting with headings
        
        # If content doesn't have headings and is long enough, add some
        if len(content) > 1000 and '# ' not in content and '\n## ' not in content:
            # Split into sections (simplistic approach)
            paragraphs = content.split('\n\n')
            if len(paragraphs) >= 3:
                # Add a title and section headings
                title = paragraphs[0]
                introduction = paragraphs[1]
                main_content = paragraphs[2:-1]
                conclusion = paragraphs[-1] if len(paragraphs) > 2 else ""
                
                # Format with headings
                formatted_content = f"# {title}\n\n{introduction}\n\n"
                
                # Add section headings for main content
                for i, para in enumerate(main_content):
                    section_name = f"Section {i+1}"
                    formatted_content += f"## {section_name}\n\n{para}\n\n"
                
                # Add conclusion
                if conclusion:
                    formatted_content += f"## Conclusion\n\n{conclusion}"
                    
                content = formatted_content
    
    elif platform == "YouTube":
        # For YouTube, format as a script with timestamps if it's not already
        if "SCRIPT:" not in content and "TIMESTAMP:" not in content:
            # Basic script structure
            title = content.split('\n')[0] if '\n' in content else "Video Title"
            intro = "Hello everyone, welcome to this video on " + title
            
            # Split content into sections
            paragraphs = content.split('\n\n')
            
            # Format as script with timestamps
            script = f"TITLE: {title}\n\n"
            script += f"INTRO (0:00):\n{intro}\n\n"
            
            # Add main content sections with timestamps
            for i, para in enumerate(paragraphs[1:], 1):
                timestamp = f"{i}:00"  # Simplified timestamps
                script += f"SECTION {i} ({timestamp}):\n{para}\n\n"
            
            # Add outro
            script += "OUTRO:\nThanks for watching! Don't forget to like and subscribe for more content like this."
            
            content = script
    
    elif platform == "Substack":
        # For Substack, format as a newsletter with greeting and sign-off
        if "Dear" not in content and "Hello" not in content:
            # Add newsletter elements
            title = content.split('\n')[0] if '\n' in content else "Newsletter Title"
            greeting = "Hello Fellow Readers,"
            intro = content.split('\n\n')[0] if '\n\n' in content else content
            main_content = '\n\n'.join(content.split('\n\n')[1:]) if '\n\n' in content else ""
            
            # Format as newsletter
            newsletter = f"# {title}\n\n"
            newsletter += f"{greeting}\n\n"
            newsletter += f"{intro}\n\n"
            newsletter += f"{main_content}\n\n"
            newsletter += "Until next time,\n\nC. Pete Connor"
            
            content = newsletter
    
    return content