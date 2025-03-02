"""
Model-based Streamlit application for the Multi-Format Content Generator.

This version uses a fine-tuned model to generate content in C. Pete Connor's
distinctive satirical tech expert style, focusing on platform selection
rather than writing style selection.
"""

import os
import time
import logging
import json
from typing import Dict, Any, List
from pathlib import Path

import streamlit as st
import pandas as pd
import nltk
import wandb

# Import local modules
from models.model_content_generator import ModelContentGenerator
from models.platform_specs import get_platform_specs
from utils.health_monitor import check_system_health
from utils.document_processor import extract_text_from_document

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
    page_title="C. Pete Connor Content Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .title-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .title-container h1 {
        color: #0e1117;
        margin-bottom: 0.5rem;
    }
    .title-container p {
        color: #4a4a4a;
        font-size: 1.1rem;
    }
    .platform-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1a73e8;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the model-based content generator
@st.cache_resource
def initialize_content_generator():
    """Initialize and cache the model-based content generator."""
    try:
        # Determine model directory
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'outputs', 'finetune', 'final')
        
        # Check if model exists, if not, use template-based fallback
        if not Path(model_dir).exists():
            logger.warning(f"Model directory not found at {model_dir}, using fallback generator")
            model_dir = None
        
        # Initialize content generator
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        generator = ModelContentGenerator(
            model_dir=model_dir,
            device="auto",
            use_wandb=use_wandb
        )
        logger.info(f"Content generator initialized (model: {'available' if model_dir else 'unavailable'})")
        return generator
    except Exception as e:
        logger.error(f"Error initializing content generator: {e}")
        st.error(f"Failed to initialize content generator: {e}")
        return None

# Initialize content generator
content_generator = initialize_content_generator()

def load_example_topics() -> List[str]:
    """Load example topics for demonstration."""
    return [
        "The latest AI model claims to be a game-changer but has the same limitations as its predecessors.",
        "Silicon Valley just released another 'revolutionary' app that solves a problem no one has.",
        "This startup raised $50 million for an idea that already failed three times before.",
        "Tech companies are adding 'AI' to their product names without changing anything about the product.",
        "The newest smartphone has 8 cameras but still can't take good photos in low light."
    ]

def generate_content(input_text: str, platform: str, sentiment: str = None) -> Dict[str, Any]:
    """
    Generate content using the model-based generator.
    
    Args:
        input_text: The source text to generate content from
        platform: Target platform (e.g., Twitter, LinkedIn)
        sentiment: Optional sentiment direction (positive, negative, neutral)
        
    Returns:
        dict: Generated content with metadata
    """
    if not input_text or not platform:
        return {
            "content": "",
            "sentiment_scores": {"compound": 0, "neg": 0, "neu": 0, "pos": 0},
            "platform": platform,
            "processing_time": 0
        }
    
    try:
        start_time = time.time()
        
        # Log to W&B if available
        if wandb.run is not None:
            wandb.log({
                "input_platform": platform,
                "input_length": len(input_text),
                "input_content": input_text
            })
        
        # Generate content
        generated_texts = content_generator.generate_content(
            content=input_text,
            platform=platform.lower(),
            sentiment=sentiment,
            max_length=None,  # Use platform default
            num_return_sequences=1
        )
        
        # Get the first generated text
        generated_content = generated_texts[0] if generated_texts else ""
        
        # Analyze sentiment
        sentiment_scores = content_generator.analyze_sentiment(generated_content)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log generation metrics
        if wandb.run is not None:
            wandb.log({
                "output_length": len(generated_content),
                "processing_time": processing_time
            })
        
        return {
            "content": generated_content,
            "sentiment_scores": sentiment_scores,
            "platform": platform,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        st.error(f"Error generating content: {e}")
        return {
            "content": f"Error generating content: {str(e)}",
            "sentiment_scores": {"compound": 0, "neg": 0, "neu": 0, "pos": 0},
            "platform": platform,
            "processing_time": 0
        }

def display_system_health():
    """Display system health information in the sidebar."""
    with st.sidebar:
        st.markdown("### System Health")
        
        # Get health data
        health_data = check_system_health()
        
        # Display health metrics
        col1, col2, col3 = st.columns(3)
        
        # CPU usage
        cpu_usage = health_data.get("cpu_percent", 0)
        cpu_color = "green" if cpu_usage < 70 else "orange" if cpu_usage < 85 else "red"
        col1.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta=None,
            delta_color=cpu_color
        )
        
        # Memory usage
        memory_usage = health_data.get("memory_percent", 0)
        memory_color = "green" if memory_usage < 70 else "orange" if memory_usage < 85 else "red"
        col2.metric(
            "Memory",
            f"{memory_usage:.1f}%",
            delta=None,
            delta_color=memory_color
        )
        
        # Disk usage
        disk_usage = health_data.get("disk_percent", 0)
        disk_color = "green" if disk_usage < 70 else "orange" if disk_usage < 85 else "red"
        col3.metric(
            "Disk",
            f"{disk_usage:.1f}%",
            delta=None,
            delta_color=disk_color
        )
        
        # W&B Status
        wandb_status = "Connected" if wandb.run is not None else "Not Connected"
        wandb_color = "green" if wandb.run is not None else "red"
        st.markdown(f"**W&B Status:** <span style='color:{wandb_color}'>{wandb_status}</span>", unsafe_allow_html=True)
        
        # Model Status
        model_status = "Available" if content_generator.model is not None else "Not Available"
        model_color = "green" if content_generator.model is not None else "red"
        st.markdown(f"**Model Status:** <span style='color:{model_color}'>{model_status}</span>", unsafe_allow_html=True)
        
        # Hardware Acceleration
        device = content_generator.device
        device_icon = "🚀" if device in ["cuda", "mps"] else "🖥️"
        st.markdown(f"**Hardware:** {device_icon} {device.upper()}")
        
        # Add refresh button
        if st.button("Refresh Health Data"):
            st.success("Health data refreshed!")
            st.experimental_rerun()

def display_sentiment_analysis(sentiment_scores):
    """Display sentiment analysis results with visual indicators."""
    compound = sentiment_scores.get("compound", 0)
    
    # Determine sentiment category
    if compound >= 0.05:
        sentiment_category = "Positive"
        sentiment_class = "sentiment-positive"
        sentiment_emoji = "😊"
    elif compound <= -0.05:
        sentiment_category = "Negative"
        sentiment_class = "sentiment-negative"
        sentiment_emoji = "😟"
    else:
        sentiment_category = "Neutral"
        sentiment_class = "sentiment-neutral"
        sentiment_emoji = "😐"
    
    # Display sentiment summary
    st.markdown(f"### Sentiment Analysis {sentiment_emoji}")
    st.markdown(f"**Overall Sentiment:** <span class='{sentiment_class}'>{sentiment_category}</span> (Score: {compound:.2f})", unsafe_allow_html=True)
    
    # Create sentiment bar chart
    sentiment_data = {
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Score": [
            sentiment_scores.get("pos", 0),
            sentiment_scores.get("neu", 0),
            sentiment_scores.get("neg", 0)
        ]
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    st.bar_chart(sentiment_df.set_index("Sentiment"))

def main():
    """Main application function."""
    # Application header with custom styling
    st.markdown("""
    <div class="title-container">
        <h1>🎭 C. Pete Connor Content Generator</h1>
        <p>Generate content across multiple platforms in C. Pete Connor's satirical tech expert style</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Style notice - C. Pete Connor style is fixed
        st.markdown("""
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; border-left:5px solid #1890ff;">
            <strong>Writing Style:</strong> C. Pete Connor (Satirical Tech Expert)
        </div>
        """, unsafe_allow_html=True)
        
        # Sentiment direction
        sentiment = st.selectbox(
            "Sentiment Direction (Optional)",
            options=["Automatic", "More Positive", "More Negative", "More Neutral"],
            index=0
        )
        
        # Map sentiment selection to model parameter
        sentiment_param = None
        if sentiment == "More Positive":
            sentiment_param = "positive"
        elif sentiment == "More Negative":
            sentiment_param = "negative"
        elif sentiment == "More Neutral":
            sentiment_param = "neutral"
        
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
            try:
                extracted_text = extract_text_from_document(uploaded_file)
                st.markdown("### Extracted Text")
                st.text_area("Preview", value=extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""), height=200)
                
                # Store extracted text for generation
                text_input = extracted_text
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                text_input = ""
    
    with tab3:
        url_input = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article"
        )
        
        if url_input and st.button("Extract Text from URL"):
            st.info("URL text extraction not implemented in this example.")
            # In a real implementation, you would extract text from the URL here
            text_input = ""
    
    # Process input and generate content
    if generate_button and (text_input or (uploaded_file is not None)):
        with st.spinner(f"Generating {platform} content in C. Pete Connor's style..."):
            # Generate content
            result = generate_content(
                input_text=text_input,
                platform=platform,
                sentiment=sentiment_param
            )
            
            # Display generated content
            st.markdown(f"### Generated {platform} Content")
            st.markdown(f"""
            <div class="platform-box">
                {result['content']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display metadata
            with st.expander("Content Metadata"):
                st.markdown(f"**Platform:** {result['platform']}")
                st.markdown(f"**Processing Time:** {result['processing_time']:.2f} seconds")
            
            # Display sentiment analysis
            display_sentiment_analysis(result['sentiment_scores'])
            
            # Copy button
            if st.button("📋 Copy to Clipboard"):
                st.code(result['content'])
                st.success("Content copied to clipboard!")

if __name__ == "__main__":
    main()
