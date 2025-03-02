#!/usr/bin/env python3
"""
CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric
A content generator with cross-platform adaptations
"""

import sys
import os
import re
import json
import logging
import time
import psutil
from typing import Dict, List, Any
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget, 
    QCheckBox, QFileDialog, QGroupBox, QMessageBox, QScrollArea,
    QComboBox, QStatusBar, QSplitter, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QDesktopServices, QIcon, QTextOption, QTextCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CANDOR")

# Import local modules
from src.utils.document_processor import extract_text_from_document, extract_text_from_url, analyze_sentiment, extract_key_topics
from src.models.content_generator import ContentGenerator
from src.models.platform_specs import get_platform_specs, get_platform_names
from src.utils.health_monitor import check_system_health
from src.utils.wandb_monitor import is_wandb_available, setup_wandb_monitoring

class GenerationWorker(QThread):
    """Worker thread for content generation to keep UI responsive"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, input_text: str, input_type: str, platforms: List[str], 
                 tone: str = "Informative", keywords: List[str] = None, 
                 writing_style: str = None):
        super().__init__()
        self.input_text = input_text
        self.input_type = input_type
        self.platforms = platforms
        self.tone = tone
        self.keywords = keywords if keywords else []
        self.writing_style = writing_style
        self.content_generator = ContentGenerator(use_wandb=is_wandb_available())
        
    def run(self):
        try:
            # Step 1: Process input based on type
            raw_content = ""
            self.progress.emit(10)
            
            if self.input_type == "text":
                raw_content = self.input_text
            elif self.input_type == "document":
                raw_content = self._process_document(self.input_text)
            elif self.input_type == "url":
                raw_content = extract_text_from_url(self.input_text)
            else:
                raise ValueError(f"Unknown input type: {self.input_type}")
            
            if not raw_content or raw_content.startswith("Error"):
                raise ValueError(f"Failed to process input: {raw_content}")
                
            self.progress.emit(30)
            
            # Step 2: Analyze sentiment
            sentiment_data = analyze_sentiment(raw_content)
            
            # If no keywords provided, extract them from the text
            if not self.keywords:
                self.keywords = extract_key_topics(raw_content, num_topics=5)
            
            self.progress.emit(50)
            
            # Step 3: Generate content for each platform
            results = {}
            
            total_platforms = len(self.platforms)
            for i, platform in enumerate(self.platforms):
                # Get platform specs
                platform_specs = get_platform_specs(platform)
                
                # Generate content for the platform
                generated_content = self.content_generator.generate_content(
                    input_text=raw_content,
                    platform=platform,
                    platform_specs=platform_specs,
                    tone=self.tone,
                    keywords=self.keywords,
                    writing_style=self.writing_style
                )
                
                results[platform] = generated_content
                
                # Update progress
                progress_value = 50 + int((i + 1) / total_platforms * 50)
                self.progress.emit(progress_value)
            
            # Add metadata
            results["metadata"] = {
                "sentiment": sentiment_data.get("dominant_sentiment", "neutral"),
                "keywords": self.keywords,
                "word_count": len(raw_content.split()),
                "character_count": len(raw_content),
                "generation_time": time.time()
            }
            
            self.progress.emit(100)
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Error in generation worker: {str(e)}")
            self.error.emit(str(e))
    
    def _process_document(self, file_path: str) -> str:
        """Process a document file"""
        try:
            # For direct file path
            with open(file_path, 'rb') as file:
                file_name = os.path.basename(file_path)
                file_extension = file_name.split('.')[-1].lower()
                
                # Create file-like object for extraction
                class FileObject:
                    def __init__(self, file_data, name):
                        self.name = name
                        self._data = file_data
                    
                    def getvalue(self):
                        return self._data
                
                file_data = file.read()
                file_obj = FileObject(file_data, file_name)
                
                # Extract text
                return extract_text_from_document(file_obj)
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return f"Error processing document: {str(e)}"


class SystemHealthWidget(QWidget):
    """Widget to display system health information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Health status label
        self.status_label = QLabel("System Status: Checking...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create metrics layout
        metrics_layout = QVBoxLayout()
        
        # CPU usage
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_bar)
        metrics_layout.addLayout(cpu_layout)
        
        # Memory usage
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(QLabel("Memory:"))
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        memory_layout.addWidget(self.memory_bar)
        metrics_layout.addLayout(memory_layout)
        
        # Disk usage
        disk_layout = QHBoxLayout()
        disk_layout.addWidget(QLabel("Disk:"))
        self.disk_bar = QProgressBar()
        self.disk_bar.setRange(0, 100)
        disk_layout.addWidget(self.disk_bar)
        metrics_layout.addLayout(disk_layout)
        
        layout.addLayout(metrics_layout)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Health Data")
        self.refresh_btn.clicked.connect(self.update_health_data)
        layout.addWidget(self.refresh_btn)
        
        # Initial update
        self.update_health_data()
    
    def update_health_data(self):
        """Update health data display"""
        health_data = check_system_health()
        
        # Update status
        status = health_data["status"]
        self.status_label.setText(f"System Status: {status}")
        
        # Set color based on status
        if status == "Healthy":
            self.status_label.setStyleSheet("color: green;")
        elif status == "Warning":
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setStyleSheet("color: red;")
        
        # Update metrics - convert float to int for progress bars
        self.cpu_bar.setValue(int(health_data["metrics"]["cpu_percent"]))
        self.memory_bar.setValue(int(health_data["metrics"]["memory_percent"]))
        self.disk_bar.setValue(int(health_data["metrics"]["disk_percent"]))


class CandorApp(QMainWindow):
    """Main application window for multi-platform content generator"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Platform Content Generator")
        self.setMinimumSize(1000, 800)
        
        # Set up main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Create a splitter for main content and sidebar
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Main content area (left side)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        # Sidebar (right side)
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setMaximumWidth(300)
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        
        # Add widgets to splitter
        self.main_splitter.addWidget(self.content_widget)
        self.main_splitter.addWidget(self.sidebar_widget)
        
        # Set up main layout
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.addWidget(self.main_splitter)
        
        # Add header
        header_label = QLabel("Multi-Platform Content Generator")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont("Arial", 14, QFont.Weight.Bold)
        header_label.setFont(header_font)
        self.content_layout.addWidget(header_label)
        
        subtitle_label = QLabel("Generate optimized content for multiple platforms")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(subtitle_label)
        
        # Input section
        input_group = QGroupBox("Input Content")
        input_layout = QVBoxLayout()
        
        # Text input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter your content here...")
        self.text_input.setMinimumHeight(150)
        input_layout.addWidget(QLabel("Direct Text Input:"))
        input_layout.addWidget(self.text_input)
        
        # Document upload button
        doc_layout = QHBoxLayout()
        doc_layout.addWidget(QLabel("Upload Document:"))
        self.doc_path_label = QLineEdit()
        self.doc_path_label.setReadOnly(True)
        self.doc_path_label.setPlaceholderText("No document selected")
        doc_layout.addWidget(self.doc_path_label)
        self.upload_btn = QPushButton("Browse...")
        self.upload_btn.clicked.connect(self.upload_document)
        doc_layout.addWidget(self.upload_btn)
        input_layout.addLayout(doc_layout)
        
        # URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL to analyze...")
        url_layout.addWidget(self.url_input)
        self.fetch_url_btn = QPushButton("Fetch")
        self.fetch_url_btn.clicked.connect(self.fetch_url_content)
        url_layout.addWidget(self.fetch_url_btn)
        input_layout.addLayout(url_layout)
        
        input_group.setLayout(input_layout)
        self.content_layout.addWidget(input_group)
        
        # Example topics
        self.examples_group = QGroupBox("Example Topics")
        examples_layout = QVBoxLayout()
        
        self.example_topics = [
            "AI Ethics in the Workplace",
            "The Future of Remote Work",
            "Data Privacy Concerns",
            "Blockchain Applications in Finance",
            "5G Technology Implications",
            "Cloud Computing Trends",
            "Cybersecurity Best Practices",
            "Digital Transformation Strategies"
        ]
        
        self.example_combo = QComboBox()
        self.example_combo.addItem("Select a topic...")
        self.example_combo.addItems(self.example_topics)
        examples_layout.addWidget(self.example_combo)
        
        self.use_example_btn = QPushButton("Use Example Topic")
        self.use_example_btn.clicked.connect(self.use_example_topic)
        examples_layout.addWidget(self.use_example_btn)
        
        self.examples_group.setLayout(examples_layout)
        self.content_layout.addWidget(self.examples_group)
        
        # Platform selection
        platform_group = QGroupBox("Target Platforms")
        platform_layout = QVBoxLayout()
        
        # Get available platforms from the module
        self.available_platforms = get_platform_names()
        if not self.available_platforms:
            # Fallback platforms
            self.available_platforms = [
                "Twitter", "LinkedIn", "Facebook", "Instagram", 
                "Blog", "Email Newsletter", "YouTube", "Medium", "Substack"
            ]
        
        # Create checkbox grid layout
        platforms_grid = QHBoxLayout()
        self.platform_checkboxes = {}
        
        for platform in self.available_platforms:
            checkbox = QCheckBox(platform)
            # Select some by default
            if platform in ["LinkedIn", "Twitter", "Medium", "Blog"]:
                checkbox.setChecked(True)
            platforms_grid.addWidget(checkbox)
            self.platform_checkboxes[platform] = checkbox
        
        platform_layout.addLayout(platforms_grid)
        platform_group.setLayout(platform_layout)
        self.content_layout.addWidget(platform_group)
        
        # Content parameters
        content_params_group = QGroupBox("Content Parameters")
        params_layout = QVBoxLayout()
        
        # Writing style
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Writing Style:"))
        self.style_combo = QComboBox()
        
        # Add available writing styles
        self.writing_styles = [
            {"id": "default", "name": "Default Style", "description": "Standard professional tone"},
            {"id": "pete_connor", "name": "C. Pete Connor", "description": "Satirical tech expert with data-driven insights"}
        ]
        
        # Check if custom templates exist
        custom_path = os.path.join(os.path.dirname(__file__), 'data', 'custom_templates.json')
        if os.path.exists(custom_path):
            self.writing_styles.append({
                "id": "custom", 
                "name": "Custom Templates", 
                "description": "User-defined templates"
            })
        
        for style in self.writing_styles:
            self.style_combo.addItem(style["name"])
        
        # Default to Pete Connor style
        default_index = self.style_combo.findText("C. Pete Connor")
        if default_index >= 0:
            self.style_combo.setCurrentIndex(default_index)
            
        style_layout.addWidget(self.style_combo)
        params_layout.addLayout(style_layout)
        
        # Style description
        self.style_description = QLabel(self.writing_styles[1]["description"])
        self.style_combo.currentIndexChanged.connect(self.update_style_description)
        params_layout.addWidget(self.style_description)
        
        # Tone selection
        tone_layout = QHBoxLayout()
        tone_layout.addWidget(QLabel("Content Tone:"))
        self.tone_combo = QComboBox()
        self.tone_combo.addItems([
            "Informative", "Professional", "Casual", 
            "Enthusiastic", "Thoughtful"
        ])
        tone_layout.addWidget(self.tone_combo)
        params_layout.addLayout(tone_layout)
        
        # Keywords input
        params_layout.addWidget(QLabel("Keywords (one per line):"))
        self.keywords_input = QTextEdit()
        self.keywords_input.setMaximumHeight(100)
        params_layout.addWidget(self.keywords_input)
        
        content_params_group.setLayout(params_layout)
        self.sidebar_layout.addWidget(content_params_group)
        
        # System health widget
        self.health_widget = SystemHealthWidget()
        self.sidebar_layout.addWidget(self.health_widget)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Content")
        self.generate_btn.clicked.connect(self.generate_content)
        self.generate_btn.setMinimumHeight(40)
        generate_font = QFont("Arial", 12, QFont.Weight.Bold)
        self.generate_btn.setFont(generate_font)
        self.content_layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.content_layout.addWidget(self.progress_bar)
        
        # Output tabs
        self.output_tabs = QTabWidget()
        
        # Add tabs for each platform (will be populated when generation is complete)
        self.content_layout.addWidget(self.output_tabs)
        
        # W&B status
        wandb_layout = QHBoxLayout()
        wandb_layout.addWidget(QLabel("Model Monitoring:"))
        self.wandb_status = QLabel()
        if is_wandb_available():
            self.wandb_status.setText("✅ W&B Monitoring: Active")
            self.wandb_status.setStyleSheet("color: green;")
        else:
            self.wandb_status.setText("❌ W&B Monitoring: Inactive")
            self.wandb_status.setStyleSheet("color: red;")
        wandb_layout.addWidget(self.wandb_status)
        self.sidebar_layout.addLayout(wandb_layout)
        
        # Add stretcher to bottom of sidebar
        self.sidebar_layout.addStretch()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Initialize state
        self.document_path = None
        self.worker = None
        
    def update_style_description(self):
        """Update the style description when selection changes"""
        current_style = self.style_combo.currentText()
        for style in self.writing_styles:
            if style["name"] == current_style:
                self.style_description.setText(style["description"])
                break
    
    def upload_document(self):
        """Handle document upload via file dialog"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 
            "Select Document", 
            "", 
            "Documents (*.txt *.pdf *.docx *.md);;All Files (*)"
        )
        
        if file_path:
            self.document_path = file_path
            self.doc_path_label.setText(os.path.basename(file_path))
            logger.info(f"Document selected: {file_path}")
    
    def fetch_url_content(self):
        """Fetch content from URL"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "No URL", "Please enter a URL to fetch content from.")
            return
        
        self.statusBar().showMessage(f"Fetching content from {url}...")
        
        try:
            # Get content from URL
            content = extract_text_from_url(url)
            
            # Check for errors
            if content.startswith("Error"):
                raise Exception(content)
            
            # Update text area with fetched content
            self.text_input.setText(content)
            self.statusBar().showMessage(f"Content fetched from {url}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error Fetching URL", f"Failed to fetch content: {str(e)}")
            self.statusBar().showMessage("Error fetching URL content")
    
    def use_example_topic(self):
        """Use the selected example topic"""
        selected_topic = self.example_combo.currentText()
        if selected_topic != "Select a topic...":
            self.text_input.setText(selected_topic)
    
    def generate_content(self):
        """Handle the generate button click"""
        # Get selected platforms
        selected_platforms = [platform for platform, checkbox in self.platform_checkboxes.items() 
                             if checkbox.isChecked()]
        
        if not selected_platforms:
            QMessageBox.warning(self, "No Platforms Selected", 
                               "Please select at least one target platform.")
            return
            
        # Determine input type and content
        input_type = "text"
        input_content = self.text_input.toPlainText().strip()
        
        if not input_content and self.document_path:
            input_type = "document"
            input_content = self.document_path
        elif not input_content and self.url_input.text().strip():
            input_type = "url"
            input_content = self.url_input.text().strip()
        elif not input_content:
            QMessageBox.warning(self, "No Input", 
                               "Please provide input content via text, document, or URL.")
            return
        
        # Get content parameters
        selected_tone = self.tone_combo.currentText()
        
        # Get writing style
        selected_style_name = self.style_combo.currentText()
        selected_style = None
        for style in self.writing_styles:
            if style["name"] == selected_style_name:
                selected_style = style["id"]
                break
        
        # Get keywords
        keywords_text = self.keywords_input.toPlainText()
        keywords = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
            
        # Update UI
        self.statusBar().showMessage("Generating content...")
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Create and start worker thread
        self.worker = GenerationWorker(
            input_content, input_type, selected_platforms,
            tone=selected_tone, keywords=keywords,
            writing_style=selected_style
        )
        self.worker.finished.connect(self.update_output)
        self.worker.error.connect(self.show_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_output(self, results):
        """Update the output tabs with generated content"""
        # Clear existing tabs
        self.output_tabs.clear()
        
        # Get metadata
        metadata = results.pop("metadata", {})
        
        # Add a tab for each platform
        for platform, content in results.items():
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Platform specifications
            platform_specs = get_platform_specs(platform)
            specs_label = QLabel(f"Character limit: {platform_specs.max_length} | Current length: {len(content)}")
            tab_layout.addWidget(specs_label)
            
            # Warning if content exceeds platform limit
            if len(content) > platform_specs.max_length:
                warning_label = QLabel(f"⚠️ Content exceeds the {platform} character limit by {len(content) - platform_specs.max_length} characters")
                warning_label.setStyleSheet("color: orange;")
                tab_layout.addWidget(warning_label)
            
            # Content display
            output_text = QTextEdit()
            output_text.setReadOnly(True)
            output_text.setPlainText(content)
            output_text.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            tab_layout.addWidget(output_text)
            
            # Button row
            button_layout = QHBoxLayout()
            
            # Copy button
            copy_btn = QPushButton("Copy to Clipboard")
            copy_btn.clicked.connect(lambda checked, editor=output_text: self.copy_to_clipboard(editor))
            button_layout.addWidget(copy_btn)
            
            # Save button
            save_btn = QPushButton("Save as Text File")
            save_btn.clicked.connect(lambda checked, platform=platform, content=content: self.save_to_file(platform, content))
            button_layout.addWidget(save_btn)
            
            tab_layout.addLayout(button_layout)
            
            # Add the tab
            self.output_tabs.addTab(tab, platform)
        
        # Add a metadata tab
        if metadata:
            meta_tab = QWidget()
            meta_layout = QVBoxLayout(meta_tab)
            
            # Metadata display
            meta_text = QTextEdit()
            meta_text.setReadOnly(True)
            
            # Format metadata
            meta_content = "Content Generation Metadata:\n\n"
            meta_content += f"Sentiment: {metadata.get('sentiment', 'unknown').capitalize()}\n"
            meta_content += f"Word Count: {metadata.get('word_count', 0)}\n"
            meta_content += f"Character Count: {metadata.get('character_count', 0)}\n"
            
            if 'keywords' in metadata and metadata['keywords']:
                meta_content += f"\nKeywords: {', '.join(metadata['keywords'])}\n"
                
            meta_text.setPlainText(meta_content)
            meta_layout.addWidget(meta_text)
            
            self.output_tabs.addTab(meta_tab, "Metadata")
        
        # Reset UI state
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Content generation complete")
        
        # Update system health
        self.health_widget.update_health_data()
    
    def save_to_file(self, platform, content):
        """Save content to a text file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Content",
            f"{platform.lower()}_content.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.statusBar().showMessage(f"Content saved to {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def show_error(self, error_message):
        """Display error message"""
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Error occurred during generation")
    
    def copy_to_clipboard(self, editor):
        """Copy the content of the text editor to clipboard"""
        editor.selectAll()
        editor.copy()
        editor.moveCursor(QTextCursor.MoveOperation.Start)
        self.statusBar().showMessage("Content copied to clipboard", 3000)


def main():
    app = QApplication(sys.argv)
    window = CandorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
