"""
Utilities for processing documents and extracting text from various sources.
"""

import io
import re
import logging
import requests
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_text_from_document(uploaded_file) -> str:
    """
    Extract text from various document formats.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text or error message
    """
    try:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        # Process based on file extension
        if file_extension == 'txt':
            # Text file
            return uploaded_file.getvalue().decode('utf-8')
            
        elif file_extension == 'pdf':
            # PDF file
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
            
        elif file_extension == 'docx':
            # Word document
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        else:
            return f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        logger.error(f"Error extracting text from document: {str(e)}")
        return f"Error processing document: {str(e)}"

def extract_text_from_url(url: str) -> str:
    """
    Extract text content from a URL.
    
    Args:
        url: URL to extract content from
        
    Returns:
        str: Extracted text or error message
    """
    try:
        # Add http:// if not present
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
        
        # Request the URL
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }, timeout=10)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
        
        # Find the main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|article|post')) or soup.body
        
        if main_content:
            # Extract text from main content
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # If no main content is identified, use the body
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
        
        return text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return f"Error processing URL: {str(e)}"

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of the input text using NLTK.
    
    Args:
        text: Input text to analyze
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        # Tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Get word count
        word_count = len(filtered_words)
        
        # Create a simple sentiment dictionary
        # This is a very basic implementation - in a real-world scenario, 
        # you'd want a more sophisticated approach
        positive_words = [
            'good', 'great', 'excellent', 'best', 'awesome', 'wonderful', 'amazing',
            'love', 'happy', 'joy', 'positive', 'success', 'successful', 'beneficial',
            'innovative', 'improvement', 'improve', 'advantage', 'impressive'
        ]
        
        negative_words = [
            'bad', 'worst', 'terrible', 'horrible', 'awful', 'poor', 'negative',
            'hate', 'sad', 'anger', 'angry', 'problem', 'failure', 'fail', 'issue',
            'disadvantage', 'difficult', 'challenge', 'error', 'mistake'
        ]
        
        # Count sentiment words
        positive_count = sum(1 for word in filtered_words if word in positive_words)
        negative_count = sum(1 for word in filtered_words if word in negative_words)
        neutral_count = word_count - positive_count - negative_count
        
        # Calculate sentiment scores (as percentages)
        total_scored_words = max(1, word_count)  # Avoid division by zero
        positive_score = positive_count / total_scored_words
        negative_score = negative_count / total_scored_words
        neutral_score = neutral_count / total_scored_words
        
        # Determine dominant sentiment
        if positive_score > negative_score and positive_score > neutral_score:
            dominant_sentiment = "positive"
        elif negative_score > positive_score and negative_score > neutral_score:
            dominant_sentiment = "negative"
        else:
            dominant_sentiment = "neutral"
        
        # Get top keywords (most frequent non-stopwords)
        word_freq = {}
        for word in filtered_words:
            if len(word) > 2:  # Only consider words longer than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Get top 10 keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Return results
        return {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "sentiment_scores": {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            },
            "dominant_sentiment": dominant_sentiment,
            "top_keywords": [word for word, _ in top_keywords]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {
            "word_count": 0,
            "sentence_count": 0,
            "sentiment_scores": {
                "positive": 0,
                "negative": 0,
                "neutral": 1.0
            },
            "dominant_sentiment": "neutral",
            "top_keywords": [],
            "error": str(e)
        }

def extract_key_topics(text: str, num_topics: int = 5) -> List[str]:
    """
    Extract key topics from the input text using NLTK.
    
    Args:
        text: Input text to analyze
        num_topics: Number of topics to extract
        
    Returns:
        list: Key topics from the text
    """
    try:
        # Tokenize text and remove stopwords
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_topics]
        
        return [topic for topic, _ in topics]
        
    except Exception as e:
        logger.error(f"Error extracting key topics: {str(e)}")
        return []
