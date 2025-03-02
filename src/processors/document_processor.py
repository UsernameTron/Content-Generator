#!/usr/bin/env python3
"""
CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric
A satirical content generator with cross-platform adaptations

This application transforms regular content into satirical versions optimized for
different platforms like YouTube, Medium, LinkedIn, and Substack. It applies
sentiment analysis and the CANDOR transformation method to create engaging,
platform-specific satirical content.
"""

import sys
import os
import re
import logging
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget, 
    QCheckBox, QFileDialog, QGroupBox, QMessageBox, QScrollArea,
    QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QDesktopServices, QIcon

# Configure logging with timestamps and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CANDOR")

# Import processor modules
from src.processors.text_processor import process_text
from src.processors.document_processor import process_document
from src.processors.url_processor import process_url
from src.processors.sentiment_analyzer import analyze_sentiment
from src.processors.content_transformer import transform_content
from src.adapters.platform_adapter import adapt_for_platforms

class GenerationWorker(QThread):
    """
    Worker thread for content generation to keep UI responsive
    
    This class handles the content processing pipeline in a separate thread
    to prevent the UI from freezing during processing. It emits signals
    to update the UI when processing is complete or when errors occur.
    """
    # Define signals for communicating with the main thread
    finished = pyqtSignal(dict)  # Signal emitted when processing is complete
    error = pyqtSignal(str)      # Signal emitted when an error occurs
    progress = pyqtSignal(int, str)  # Signal for progress updates (percentage, message)
    
    def __init__(self, input_text, input_type, platforms):
        """
        Initialize the worker thread
        
        Args:
            input_text (str): The text to process or path to document/URL
            input_type (str): Type of input ("text", "document", or "url")
            platforms (list): List of target platforms for content adaptation
        """
        super().__init__()
        self.input_text = input_text
        self.input_type = input_type
        self.platforms = platforms
        
    def run(self):
        """Execute the content generation pipeline"""
        try:
            # Step 1: Process input based on type
            self.progress.emit(10, "Processing input...")
            if self.input_type == "text":
                raw_content = process_text(self.input_text)
            elif self.input_type == "document":
                raw_content = process_document(self.input_text)
            elif self.input_type == "url":
                raw_content = process_url(self.input_text)
            else:
                raise ValueError(f"Unknown input type: {self.input_type}")
            
            # Step 2: Analyze sentiment
            self.progress.emit(30, "Analyzing sentiment...")
            sentiment_data = analyze_sentiment(raw_content)
            
            # Step 3: Transform content using CANDOR method
            self.progress.emit(50, "Transforming content...")
            transformed_content = transform_content(raw_content, sentiment_data)
            
            # Step 4: Adapt for selected platforms
            self.progress.emit(70, "Adapting for platforms...")
            results = {}
            total_platforms = len(self.platforms)
            
            for i, platform in enumerate(self.platforms):
                platform_progress = 70 + (i / total_platforms) * 20
                self.progress.emit(int(platform_progress), f"Adapting for {platform}...")
                try:
                    platform_result = adapt_for_platforms(
                        transformed_content, 
                        sentiment_data, 
                        [platform]
                    )
                    results.update(platform_result)
                except Exception as platform_error:
                    # If a single platform fails, continue with others
                    logger.error(f"Error adapting for {platform}: {str(platform_error)}")
                    results[platform] = f"Error generating content for {platform}: {str(platform_error)}"
            
            # Complete
            self.progress.emit(100, "Generation complete!")
            self.finished.emit(results)
            
        except Exception as e:
            # Log the full exception for debugging
            logger.error(f"Error in generation worker: {str(e)}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))

class CandorApp(QMainWindow):
    """
    Main application window for CANDOR content generator
    
    This class defines the user interface and handles user interactions
    for the CANDOR content generation application.
    """
    
    def __init__(self):
        """Initialize the application window and UI components"""
        super().__init__()
        self.setWindowTitle("CANDOR - Satirical Content Generator")
        self.setMinimumSize(1000, 800)  # Slightly larger for better usability
        
        # Set up main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setSpacing(10)  # Add some spacing for better readability
        
        # Add header with application title
        header_label = QLabel("CANDOR: Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont("Arial", 14, QFont.Weight.Bold)
        header_label.setFont(header_font)
        self.layout.addWidget(header_label)
        
        # Add subtitle
        subtitle_label = QLabel("A satirical content generator with cross-platform adaptations")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(subtitle_label)
        
        # Input section
        input_group = QGroupBox("Input Content")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)  # Add spacing for better readability
        
        # Text input with label
        text_label = QLabel("Direct Text Input:")
        text_label.setToolTip("Enter the text you want to transform into satirical content")
        input_layout.addWidget(text_label)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter your content here...")
        self.text_input.setMinimumHeight(180)  # Slightly taller for better usability
        input_layout.addWidget(self.text_input)
        
        # Document upload with label and button
        doc_layout = QHBoxLayout()
        doc_label = QLabel("Upload Document:")
        doc_label.setToolTip("Select a document (PDF, DOCX, TXT) to transform")
        doc_layout.addWidget(doc_label)
        
        self.doc_path_label = QLineEdit()
        self.doc_path_label.setReadOnly(True)
        self.doc_path_label.setPlaceholderText("No document selected")
        doc_layout.addWidget(self.doc_path_label, stretch=1)
        
        self.upload_btn = QPushButton("Browse...")
        self.upload_btn.setToolTip("Click to select a document file")
        self.upload_btn.clicked.connect(self.upload_document)
        doc_layout.addWidget(self.upload_btn)
        
        input_layout.addLayout(doc_layout)
        
        # URL input with label
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        url_label.setToolTip("Enter a URL to extract and analyze content")
        url_layout.addWidget(url_label)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL to analyze...")
        url_layout.addWidget(self.url_input)
        
        input_layout.addLayout(url_layout)
        input_group.setLayout(input_layout)
        self.layout.addWidget(input_group)
        
        # Platform selection with checkboxes
        platform_group = QGroupBox("Target Platforms")
        platform_group.setToolTip("Select the platforms for which content will be generated")
        platform_layout = QHBoxLayout()
        platform_layout.setSpacing(15)  # More spacing between checkboxes
        
        # Create checkboxes for each platform
        self.platform_checkboxes = {}
        platforms = ["YouTube", "Medium", "LinkedIn", "Substack"]
        
        for platform in platforms:
            checkbox = QCheckBox(platform)
            checkbox.setChecked(True)  # Select all by default
            platform_layout.addWidget(checkbox)
            self.platform_checkboxes[platform] = checkbox
            
        platform_group.setLayout(platform_layout)
        self.layout.addWidget(platform_group)
        
        # Progress bar for generation status
        self.progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v")
        
        self.progress_label = QLabel("Ready")
        
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.progress_label)
        self.layout.addLayout(self.progress_layout)
        
        # Generate button with styling
        self.generate_btn = QPushButton("Generate Content")
        self.generate_btn.clicked.connect(self.generate_content)
        self.generate_btn.setMinimumHeight(45)  # Taller button for easier clicking
        generate_font = QFont("Arial", 12, QFont.Weight.Bold)
        self.generate_btn.setFont(generate_font)
        self.generate_btn.setToolTip("Click to generate satirical content for selected platforms")
        self.layout.addWidget(self.generate_btn)
        
        # Output tabs for displaying generated content
        self.output_tabs = QTabWidget()
        
        # Add a tab for each platform with text area and copy button
        for platform in platforms:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Text area for displaying generated content
            output_text = QTextEdit()
            output_text.setReadOnly(True)
            output_text.setPlaceholderText(f"Generated {platform} content will appear here")
            tab_layout.addWidget(output_text)
            
            # Copy button for easy copying
            copy_btn = QPushButton("Copy to Clipboard")
            copy_btn.setToolTip(f"Copy the {platform} content to clipboard")
            copy_btn.clicked.connect(lambda checked, editor=output_text: self.copy_to_clipboard(editor))
            tab_layout.addWidget(copy_btn)
            
            self.output_tabs.addTab(tab, platform)
            
        self.layout.addWidget(self.output_tabs)
        
        # Status bar at bottom of window
        self.statusBar().showMessage("Ready to generate satirical content")
        
        # Initialize state variables
        self.document_path = None
        self.worker = None
        
    def upload_document(self):
        """
        Handle document upload via file dialog
        
        Opens a file dialog allowing users to select a document,
        then updates the UI to reflect the selection.
        """
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
            self.statusBar().showMessage(f"Document selected: {os.path.basename(file_path)}")
            logger.info(f"Document selected: {file_path}")
    
    def generate_content(self):
        """
        Handle the generate button click
        
        Validates input, prepares the generation parameters,
        and starts the worker thread for content generation.
        """
        # Get selected platforms
        selected_platforms = [platform for platform, checkbox in self.platform_checkboxes.items() 
                             if checkbox.isChecked()]
        
        # Validate platform selection
        if not selected_platforms:
            QMessageBox.warning(
                self, 
                "No Platforms Selected", 
                "Please select at least one target platform."
            )
            return
            
        # Determine input type and content
        input_type = "text"
        input_content = self.text_input.toPlainText().strip()
        
        # Check if using document input
        if not input_content and self.document_path:
            input_type = "document"
            input_content = self.document_path
            
        # Check if using URL input
        elif not input_content and self.url_input.text().strip():
            input_type = "url"
            input_content = self.url_input.text().strip()
            
        # If no input is provided, show error message
        elif not input_content:
            QMessageBox.warning(
                self, 
                "No Input", 
                "Please provide input content via text, document, or URL."
            )
            return
            
        # Update UI to show processing state
        self.statusBar().showMessage("Generating content...")
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting generation...")
        
        # Create and configure worker thread
        self.worker = GenerationWorker(input_content, input_type, selected_platforms)
        self.worker.finished.connect(self.update_output)
        self.worker.error.connect(self.show_error)
        self.worker.progress.connect(self.update_progress)
        
        # Start the generation process
        self.worker.start()
    
    def update_progress(self, value, message):
        """
        Update the progress bar and label
        
        Args:
            value (int): Progress percentage (0-100)
            message (str): Status message to display
        """
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
    
    def update_output(self, results):
        """
        Update the output tabs with generated content
        
        Args:
            results (dict): Dictionary mapping platform names to generated content
        """
        # Populate each platform tab with its content
        for i in range(self.output_tabs.count()):
            platform = self.output_tabs.tabText(i)
            if platform in results:
                tab = self.output_tabs.widget(i)
                text_edit = tab.findChild(QTextEdit)
                if text_edit:
                    text_edit.setPlainText(results[platform])
        
        # Switch to the first tab with content
        if results and self.output_tabs.count() > 0:
            self.output_tabs.setCurrentIndex(0)
                    
        # Reset UI state to ready
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("Generation complete")
        self.statusBar().showMessage("Content generation complete")
        
    def show_error(self, error_message):
        """
        Display error message in a dialog box
        
        Args:
            error_message (str): Error message to display
        """
        QMessageBox.critical(
            self, 
            "Error", 
            f"An error occurred during content generation:\n\n{error_message}"
        )
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("Error occurred")
        self.statusBar().showMessage("Error occurred during generation")
        
    def copy_to_clipboard(self, editor):
        """
        Copy the content of the text editor to clipboard
        
        Args:
            editor (QTextEdit): The text editor containing content to copy
        """
        editor.selectAll()
        editor.copy()
        editor.moveCursor(editor.textCursor().Start)  # Reset cursor to start
        self.statusBar().showMessage("Content copied to clipboard", 3000)

def main():
    """
    Application entry point
    
    Creates and launches the CANDOR application window
    """
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = CandorApp()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

# Execute main function when script is run directly
if __name__ == "__main__":
    main()