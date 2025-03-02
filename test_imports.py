#!/usr/bin/env python3
"""
Test script to diagnose import issues with the CANDOR application
"""

import os
import sys
import importlib
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("CANDOR: Import Test Utility")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print("="*80)

def test_import(module_name):
    """Test importing a specific module and report result"""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported: {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import: {module_name}")
        print(f"   Error: {str(e)}")
        print("   Traceback:")
        traceback.print_exc()
        return False

# List of modules to test
modules_to_test = [
    # Core modules
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    
    # Internal modules
    "src.models.platform_specs",
    "src.processors.text_processor",
    "src.processors.document_processor",
    "src.processors.url_processor",
    "src.processors.sentiment_analyzer",
    "src.processors.content_transformer",
    "src.adapters.platform_adapter",
]

# Test each module
print("Testing imports:")
success_count = 0

for module in modules_to_test:
    if test_import(module):
        success_count += 1
    print("-"*40)

print(f"Import test completed: {success_count}/{len(modules_to_test)} successful")

if success_count == len(modules_to_test):
    print("✅ All modules imported successfully!")
else:
    print("❌ Some modules failed to import. See above for details.")
    
print("="*80)
