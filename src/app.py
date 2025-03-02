"""
Main Streamlit application for the Multi-Format Content Generator.
"""

import os
import time
import logging
import json
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import nltk

# Import local modules
from models.content_generator import ContentGenerator
from models.platform_specs import get_platform_specs
from utils.health_monitor import check_system_health
from utils.document_processor import extract_text_from_document
from utils.wandb_monitor import is_wandb_available, setup_wandb_monitoring

# Configure logging with Rich
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Create a Streamlit page config
st.set_page_config(
    page_title="Multi-Format Content Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the content generator with W&B integration
@st.cache_resource
def initialize_content_generator():
    """Initialize and cache the content generator."""
    try:
        # Load writing style
        style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'data', 'writing_style.json')
        
        json_data = {}
        if os.path.exists(style_path):
            with open(style_path, 'r') as f:
                json_data = json.load(f)
                logger.info("Loaded writing style configuration")
        
        # Initialize W&B if available
        if is_wandb_available():
            setup_wandb_monitoring(json_data)
            logger.info("Initialized W&B monitoring")
        else:
            logger.info("W&B monitoring not available (API key not set)")
        
        # Initialize content generator
        return ContentGenerator(use_wandb=is_wandb_available())
    except Exception as e:
        logger.error(f"Error initializing content generator: {str(e)}")
        return ContentGenerator(use_wandb=False)

# Initialize content generator
content_generator = initialize_content_generator()

@st.cache_data
def load_writing_styles():
    """Load available writing styles."""
    styles = [
        {"id": "default", "name": "Default Style", "description": "Standard professional tone"},
        {"id": "pete_connor", "name": "C. Pete Connor", "description": "Satirical tech expert with data-driven insights"}
    ]
    
    # Check if custom templates exist
    custom_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'data', 'custom_templates.json')
    if os.path.exists(custom_path):
        styles.append({
            "id": "custom", 
            "name": "Custom Templates", 
            "description": "User-defined templates"
        })
    
    return styles

@st.cache_data
def load_example_topics():
    """Load example topics for demonstration."""
    return [
        "AI Ethics in the Workplace",
        "The Future of Remote Work",
        "Data Privacy Concerns",
        "Blockchain Applications in Finance",
        "5G Technology Implications",
        "Cloud Computing Trends",
        "Cybersecurity Best Practices",
        "Digital Transformation Strategies"
    ]

@st.monitor_operation
def generate_content(input_text: str, platform: str, tone: str, keywords: List[str], writing_style: str = None) -> Dict[str, Any]:
    """
    Generate content for the specified platform with the given parameters.
    
    Args:
        input_text: The source text to generate content from
        platform: Target platform (e.g., Twitter, LinkedIn)
        tone: Desired tone for the content
        keywords: List of keywords to include
        writing_style: Optional writing style to use
        
    Returns:
        dict: Generated content with metadata
    """
    try:
        start_time = time.time()
        
        # Get platform specifications
        platform_specs = get_platform_specs(platform)
        
        # Generate the content
        generated_content = content_generator.generate_content(
            input_text=input_text,
            platform=platform,
            platform_specs=platform_specs,
            tone=tone,
            keywords=keywords,
            writing_style=writing_style
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Get system health info
        health_status = check_system_health()
        
        # Analyze content
        sentiment_analysis = None
        try:
            from utils.document_processor import analyze_sentiment
            sentiment_analysis = analyze_sentiment(generated_content)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
        
        # Return results
        return {
            "content": generated_content,
            "platform": platform,
            "tone": tone,
            "writing_style": writing_style,
            "generation_time_seconds": generation_time,
            "system_health": health_status["status"],
            "character_count": len(generated_content),
            "word_count": len(generated_content.split()),
            "sentiment_analysis": sentiment_analysis
        }
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return {
            "content": f"Error generating content: {str(e)}",
            "platform": platform,
            "tone": tone,
            "writing_style": writing_style,
            "generation_time_seconds": 0,
            "system_health": "error",
            "character_count": 0,
            "word_count": 0,
            "sentiment_analysis": None
        }

def display_system_health():
    """Display system health information in the sidebar."""
    # Get system health
    health_data = check_system_health()
    
    # Display health metrics
    st.sidebar.header("System Health")
    
    # Overall status
    st.sidebar.metric(
        "Status", 
        health_data["status"],
        delta="Normal" if health_data["status"] == "Healthy" else "Warning"
    )
    
    # CPU usage
    st.sidebar.metric(
        "CPU Usage", 
        f"{health_data['metrics']['cpu_percent']}%"
    )
    
    # Memory usage
    memory_percent = health_data['metrics']['memory_percent']
    st.sidebar.metric(
        "Memory Usage", 
        f"{memory_percent}%",
        delta=f"-{100-memory_percent}% available"
    )
    
    # Disk usage
    disk_percent = health_data['metrics']['disk_percent']
    st.sidebar.metric(
        "Disk Usage", 
        f"{disk_percent}%",
        delta=f"-{100-disk_percent}% available"
    )
    
    # Add refresh button
    if st.sidebar.button("Refresh Health Data"):
        st.sidebar.success("Health data refreshed!")
        st.experimental_rerun()

def main():
    """Main application function."""
    # Application header
    st.title("📝 Multi-Format Content Generator")
    st.subheader("Generate optimized content for multiple platforms")
    
    # Display system health in sidebar
    display_system_health()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Content Parameters")
        
        # Platform selection
        platform = st.selectbox(
            "Select Target Platform",
            options=["Twitter", "LinkedIn", "Facebook", "Instagram", "Blog", "Email Newsletter"],
            index=0
        )
        
        # Writing style selection
        styles = load_writing_styles()
        style_options = [style["name"] for style in styles]
        style_descriptions = {style["name"]: style["description"] for style in styles}
        style_ids = {style["name"]: style["id"] for style in styles}
        
        selected_style_name = st.selectbox(
            "Select Writing Style",
            options=style_options,
            index=1 if "C. Pete Connor" in style_options else 0,
            help="Choose the writing style for your content"
        )
        
        # Show description of selected style
        st.info(style_descriptions.get(selected_style_name, ""))
        
        # Get the style ID
        selected_style = style_ids.get(selected_style_name)
        
        # Tone selection
        tone = st.selectbox(
            "Select Content Tone",
            options=["Informative", "Professional", "Casual", "Enthusiastic", "Thoughtful"],
            index=0
        )
        
        # Keywords input
        keywords_input = st.text_area("Keywords (one per line)")
        keywords = [kw.strip() for kw in keywords_input.split("\n") if kw.strip()]
        
        # Example topics
        st.markdown("### Example Topics")
        example_topics = load_example_topics()
        selected_topic = st.selectbox(
            "Try an example topic",
            options=["Select a topic..."] + example_topics
        )
        
        if selected_topic != "Select a topic..." and st.button("Use Example Topic"):
            st.session_state.example_topic = selected_topic
        
        # Generate button
        generate_button = st.button("Generate Content", use_container_width=True)
    
    # Main content area - input methods
    tab1, tab2, tab3 = st.tabs(["Text Input", "Document Upload", "URL Input"])
    
    with tab1:
        # Pre-fill with example topic if selected
        default_text = st.session_state.get('example_topic', '')
        text_input = st.text_area(
            "Enter your text",
            height=200,
            value=default_text,
            placeholder="Enter the source text here to generate content from..."
        )
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"]
        )
        
        if uploaded_file is not None:
            # Extract text from document
            file_extension = uploaded_file.name.split(".")[-1].lower()
            text_input = extract_text_from_document(uploaded_file)
            
            # Show preview
            with st.expander("Document Text Preview", expanded=False):
                st.text(text_input[:500] + "..." if len(text_input) > 500 else text_input)
    
    with tab3:
        url_input = st.text_input("Enter URL", placeholder="https://example.com/article")
        fetch_button = st.button("Fetch Content")
        
        if fetch_button and url_input:
            with st.spinner("Fetching content from URL..."):
                try:
                    from utils.document_processor import extract_text_from_url
                    text_input = extract_text_from_url(url_input)
                    
                    # Show preview
                    with st.expander("URL Content Preview", expanded=False):
                        st.text(text_input[:500] + "..." if len(text_input) > 500 else text_input)
                        
                except Exception as e:
                    st.error(f"Error fetching content: {str(e)}")
    
    # Generate content when button is clicked
    if generate_button and text_input:
        with st.spinner("Generating content..."):
            result = generate_content(text_input, platform, tone, keywords, selected_style)
            
            # Display results
            st.subheader(f"Generated Content for {platform}")
            
            # Platform info
            platform_specs = get_platform_specs(platform)
            st.info(f"Character limit: {platform_specs.get('max_length', 'N/A')} | Current length: {result['character_count']}")
            
            # Warning if content exceeds platform limit
            if result['character_count'] > platform_specs.get('max_length', float('inf')):
                st.warning(f"⚠️ Content exceeds the {platform} character limit by {result['character_count'] - platform_specs.get('max_length')} characters")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Generation Time", f"{result['generation_time_seconds']:.2f}s")
            with col2:
                st.metric("Character Count", result["character_count"])
            with col3:
                st.metric("Word Count", result["word_count"])
            with col4:
                # Show sentiment if available
                if result.get("sentiment_analysis"):
                    sentiment = result["sentiment_analysis"].get("dominant_sentiment", "neutral")
                    st.metric("Sentiment", sentiment.capitalize())
            
            # Content display
            st.markdown("### Content:")
            st.markdown(result["content"])
            
            # Copy button
            st.button(
                "Copy to clipboard", 
                on_click=lambda: st.write("Content copied to clipboard!")
            )
            
            # Download button
            st.download_button(
                "Download as text file",
                result["content"],
                file_name=f"{platform.lower()}_content.txt",
                mime="text/plain"
            )
            
            # Sentiment analysis section
            if result.get("sentiment_analysis"):
                with st.expander("Sentiment Analysis", expanded=True):
                    sentiment_data = result["sentiment_analysis"]
                    
                    # Create columns for sentiment scores
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        st.metric("Positive", f"{sentiment_data.get('positive', 0):.2f}")
                    with score_cols[1]:
                        st.metric("Neutral", f"{sentiment_data.get('neutral', 0):.2f}")
                    with score_cols[2]:
                        st.metric("Negative", f"{sentiment_data.get('negative', 0):.2f}")
                    
                    # Display top keywords if available
                    if "top_keywords" in sentiment_data and sentiment_data["top_keywords"]:
                        st.markdown("#### Top Keywords:")
                        keyword_list = ", ".join(sentiment_data["top_keywords"])
                        st.markdown(f"*{keyword_list}*")
    
    elif generate_button:
        st.warning("Please enter some text or upload a document first.")
    
    # Add W&B status indicator
    st.sidebar.markdown("---")
    st.sidebar.write("#### Model Monitoring")
    if is_wandb_available():
        st.sidebar.success("✅ W&B Monitoring: Active")
    else:
        st.sidebar.error("❌ W&B Monitoring: Inactive (API Key not set)")
        with st.sidebar.expander("Setup Instructions"):
            st.markdown("""
            To enable W&B monitoring:
            
            1. Sign up at [wandb.ai](https://wandb.ai)
            2. Get your API key from Settings
            3. Set the environment variable:
            ```
            export WANDB_API_KEY=your_api_key_here
            ```
            4. Restart the application
            """)

if __name__ == "__main__":
    main()
