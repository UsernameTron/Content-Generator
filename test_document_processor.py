#!/usr/bin/env python3
"""
Test script for document processor
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CANDOR.test")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import document processor
from src.processors.document_processor import process_document

def main():
    """Test document processing with various file types"""
    
    # Get test files directory
    test_dir = project_root / "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a sample text file if it doesn't exist
    txt_file = test_dir / "sample.txt"
    if not txt_file.exists():
        with open(txt_file, "w") as f:
            f.write("""Sample Corporate Text
            
Our innovative solution leverages cutting-edge technology to deliver synergistic outcomes across multiple verticals. By implementing strategic initiatives, we've achieved unprecedented growth and established a robust framework for sustainable success.

Key Performance Indicators:
- 27% increase in stakeholder engagement
- Optimized resource allocation across departments
- Enhanced cross-functional collaboration
            
Moving forward, we'll continue to drive operational excellence while maintaining our commitment to customer-centric approaches and agile methodologies. This paradigm shift will empower our teams to proactively address emerging challenges and capitalize on new opportunities in the dynamic marketplace.
""")
        print(f"Created sample text file at {txt_file}")
    
    # Process files
    print("\n\n=== TESTING TEXT FILE PROCESSING ===")
    if txt_file.exists():
        print(f"Processing text file: {txt_file}")
        try:
            content = process_document(str(txt_file))
            print("\nExtracted content:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing text file: {e}")
    
    # Check for PDF processing
    pdf_file = test_dir / "sample.pdf"
    if pdf_file.exists():
        print("\n\n=== TESTING PDF FILE PROCESSING ===")
        print(f"Processing PDF file: {pdf_file}")
        try:
            content = process_document(str(pdf_file))
            print("\nExtracted content:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing PDF file: {e}")
    else:
        print(f"\nNo PDF file found at {pdf_file}. To test PDF processing, please add a sample PDF file at this location.")
    
    # Check for DOCX processing
    docx_file = test_dir / "sample.docx"
    if docx_file.exists():
        print("\n\n=== TESTING DOCX FILE PROCESSING ===")
        print(f"Processing DOCX file: {docx_file}")
        try:
            content = process_document(str(docx_file))
            print("\nExtracted content:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing DOCX file: {e}")
    else:
        print(f"\nNo DOCX file found at {docx_file}. To test DOCX processing, please add a sample DOCX file at this location.")
    
    # Check for Markdown processing
    md_file = test_dir / "sample.md"
    if not md_file.exists():
        with open(md_file, "w") as f:
            f.write("""# Corporate Strategy Document

## Executive Summary

Our strategic initiatives have positioned us at the forefront of industry disruption, enabling seamless integration of disparate technologies within a unified ecosystem.

## Key Objectives

1. **Maximize Synergies** - Leverage cross-functional expertise to unlock value
2. **Drive Innovation** - Implement forward-thinking approaches to market challenges
3. **Optimize Resource Allocation** - Ensure efficient utilization of company assets

## Implementation Timeline

| Phase | Timeframe | Deliverables |
|-------|-----------|--------------|
| Discovery | Q1 | Stakeholder analysis, market assessment |
| Development | Q2-Q3 | MVP creation, feedback incorporation |
| Deployment | Q4 | Full-scale implementation, performance tracking |

Moving forward, we'll continue to pivot our approach based on real-time metrics and evolving market conditions.
""")
        print(f"Created sample markdown file at {md_file}")
    
    print("\n\n=== TESTING MARKDOWN FILE PROCESSING ===")
    if md_file.exists():
        print(f"Processing markdown file: {md_file}")
        try:
            content = process_document(str(md_file))
            print("\nExtracted content:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing markdown file: {e}")

if __name__ == "__main__":
    main()
