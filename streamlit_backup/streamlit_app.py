#!/usr/bin/env python3
"""
CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric
A satirical content generator with cross-platform adaptations - Streamlit Version
"""

import os
import sys
import logging
import tempfile
import time
import psutil
import streamlit as st
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CANDOR")

# Import processors
from src.processors.text_processor import process_text
from src.processors.document_processor import process_document
from src.processors.url_processor import process_url
from src.processors.sentiment_analyzer import analyze_sentiment
from src.processors.content_transformer import transform_content
from src.adapters.platform_adapter import adapt_for_platforms

class HealthMonitor:
    """Monitor system health and resource usage"""
    
    # Warning thresholds (70%)
    CPU_WARNING = 70.0
    MEMORY_WARNING = 70.0
    DISK_WARNING = 70.0
    
    @staticmethod
    def get_system_health():
        """Get current system health metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        return {
            "cpu": {
                "usage": cpu_percent,
                "warning": cpu_percent > HealthMonitor.CPU_WARNING
            },
            "memory": {
                "usage": memory_percent,
                "warning": memory_percent > HealthMonitor.MEMORY_WARNING,
                "total": memory.total,
                "available": memory.available
            },
            "disk": {
                "usage": disk_percent,
                "warning": disk_percent > HealthMonitor.DISK_WARNING,
                "total": disk.total,
                "free": disk.free
            }
        }


def process_content(input_text, input_type, platforms):
    """Process content through the CANDOR pipeline"""
    try:
        with st.spinner("Processing input..."):
            # Step 1: Process input based on type
            if input_type == "text":
                raw_content = process_text(input_text)
            elif input_type == "document":
                raw_content = process_document(input_text)
            elif input_type == "url":
                raw_content = process_url(input_text)
            else:
                raise ValueError(f"Unknown input type: {input_type}")
        
        with st.spinner("Analyzing sentiment..."):
            # Step 2: Analyze sentiment
            sentiment_data = analyze_sentiment(raw_content)
            
        with st.spinner("Transforming content..."):
            # Step 3: Transform content using CANDOR method
            transformed_content = transform_content(raw_content, sentiment_data)
            
        with st.spinner("Adapting for platforms..."):
            # Step 4: Adapt for selected platforms
            results = adapt_for_platforms(transformed_content, sentiment_data, platforms)
            
        return results, sentiment_data
    
    except Exception as e:
        logger.error(f"Error in content processing: {str(e)}")
        st.error(f"Error: {str(e)}")
        return None, None


def display_health_metrics():
    """Display system health metrics"""
    health = HealthMonitor.get_system_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_color = "🔴" if health["cpu"]["warning"] else "🟢"
        st.metric(
            label=f"{cpu_color} CPU Usage",
            value=f"{health['cpu']['usage']:.1f}%"
        )
        
    with col2:
        mem_color = "🔴" if health["memory"]["warning"] else "🟢"
        st.metric(
            label=f"{mem_color} Memory Usage",
            value=f"{health['memory']['usage']:.1f}%",
            delta=f"{health['memory']['available'] / (1024**3):.1f} GB Free"
        )
        
    with col3:
        disk_color = "🔴" if health["disk"]["warning"] else "🟢"
        st.metric(
            label=f"{disk_color} Disk Usage",
            value=f"{health['disk']['usage']:.1f}%",
            delta=f"{health['disk']['free'] / (1024**3):.1f} GB Free"
        )
    
    if any([health["cpu"]["warning"], health["memory"]["warning"], health["disk"]["warning"]]):
        st.warning("⚠️ System resources are running high. Performance may be affected.")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="CANDOR - Satirical Content Generator",
        page_icon="😏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric")
    st.subheader("A satirical content generator with cross-platform adaptations")
    
    # Sidebar for input and controls
    with st.sidebar:
        st.header("Input Options")
        
        input_type = st.radio(
            "Select Input Type",
            ["Text", "Document", "URL"],
            index=0
        )
        
        # Platform selection
        st.header("Target Platforms")
        platforms = ["YouTube", "Medium", "LinkedIn", "Substack"]
        selected_platforms = []
        
        cols = st.columns(2)
        for i, platform in enumerate(platforms):
            col_idx = i % 2
            with cols[col_idx]:
                if st.checkbox(platform, value=True):
                    selected_platforms.append(platform)
        
        # Action button
        generate_btn = st.button("Generate Content", use_container_width=True)
        
        # System health
        st.header("System Health")
        display_health_metrics()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Text input
        if input_type == "Text":
            user_input = st.text_area(
                "Enter your content here",
                height=300,
                placeholder="Paste your corporate content here for satirical transformation..."
            )
            input_content = user_input
            input_method = "text"
            
        # Document upload
        elif input_type == "Document":
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=["txt", "pdf", "docx", "md"],
                help="Supported formats: TXT, PDF, DOCX, MD"
            )
            
            if uploaded_file:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    input_content = tmp_file.name
                    input_method = "document"
                    st.success(f"Uploaded: {uploaded_file.name}")
            else:
                input_content = None
                input_method = None
                
        # URL input
        elif input_type == "URL":
            input_content = st.text_input(
                "Enter URL",
                placeholder="https://example.com/corporate-announcement"
            )
            input_method = "url"
    
    # Process content when button is clicked
    if generate_btn and input_content and selected_platforms:
        results, sentiment_data = process_content(input_content, input_method, selected_platforms)
        
        if results:
            # Display sentiment analysis
            with col1:
                st.header("Sentiment Analysis")
                sentiment_score = sentiment_data["score"]
                sentiment_color = "green" if sentiment_score > 0.2 else "red" if sentiment_score < -0.2 else "orange"
                
                st.metric(
                    label="Sentiment Score",
                    value=f"{sentiment_score:.2f}",
                    delta=f"{sentiment_data['tone'].capitalize()} Tone"
                )
                
                st.subheader("Keywords")
                st.write(", ".join(sentiment_data["keywords"][:10]))
                
                st.subheader("Generated Hashtags")
                st.write(" ".join(sentiment_data["hashtags"][:8]))
                
                # Corporate jargon detection
                st.subheader("Corporate Jargon")
                for jargon in sentiment_data["corporate_jargon"][:5]:
                    st.info(jargon)
            
            # Show generated content in tabs
            with col2:
                st.header("Generated Content")
                
                # Create tabs for each platform
                tabs = st.tabs(selected_platforms)
                
                for i, platform in enumerate(selected_platforms):
                    with tabs[i]:
                        if platform in results:
                            st.text_area(
                                f"{platform} Content",
                                value=results[platform],
                                height=400
                            )
                            st.download_button(
                                f"Download {platform} Content",
                                results[platform],
                                file_name=f"CANDOR_{platform}_content.txt",
                                mime="text/plain"
                            )
                            st.button(
                                f"Copy {platform} Content",
                                key=f"copy_{platform}",
                                help="Copy content to clipboard"
                            )
                        else:
                            st.error(f"No content generated for {platform}")
    
    # Documentation and help
    st.markdown("---")
    with st.expander("About CANDOR"):
        st.markdown("""
        ## CANDOR Method
        
        CANDOR transforms corporate content using our specialized method:
        
        - **C**ontextualize content for satire
        - **A**mplify corporate jargon
        - **N**eutralize PR-speak
        - **D**ramatize statistics
        - **O**verstate trivial details
        - **R**eframe with irreverent perspective
        
        ## Supported Platforms
        
        - **YouTube**: Video scripts and descriptions
        - **Medium**: Blog posts and articles
        - **LinkedIn**: Professional posts with subtle satire
        - **Substack**: Newsletter format with satirical commentary
        
        ## How to Use
        
        1. Select your input type (text, document, or URL)
        2. Provide your content
        3. Choose your target platforms
        4. Click "Generate Content"
        5. Review and download the results
        """)


if __name__ == "__main__":
    main()
