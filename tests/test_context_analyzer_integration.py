#!/usr/bin/env python3
"""
Test the integration of PathEncoder with ContextAnalyzer

This test file verifies that the ContextAnalyzer correctly uses the PathEncoder
for path-based relationship encoding and context flattening.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the modules to test
from enhancement_module.context_analyzer import ContextAnalyzer
from enhancement_module.path_encoder import PathEncoder
from enhancement_module.reasoning_core import ReasoningCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_context_analyzer_with_path_encoder():
    """Test that ContextAnalyzer correctly uses PathEncoder for path-based relationship encoding."""
    # Create instances
    path_encoder = PathEncoder()
    reasoning_core = ReasoningCore()
    analyzer = ContextAnalyzer(reasoning_core=reasoning_core, path_encoder=path_encoder)
    
    # Create a sample hierarchical context for healthcare metrics
    healthcare_context = {
        'metadata': ['metric_id: M12345', 'date: 2025-03-01'],
        'values': {
            'current': 0.86,
            'target': 0.90,
            'baseline': 0.76,
            'historical': [0.74, 0.78, 0.80, 0.82]
        },
        'trends': ['Increasing by 2% quarterly', 'Accelerating improvement'],
        'components': ['Provider Communication', 'Facility Quality', 'Wait Times'],
        'factors': ['Staff Training', 'Process Optimization', 'Facility Updates'],
        'notes': ['Recent improvement influenced by new training program']
    }
    
    # STEP 1: Test flattening context with path encoding
    logger.info("Testing context flattening with path encoding...")
    flat_context = analyzer._flatten_context_hierarchy(healthcare_context)
    
    # Verify some expected encoded paths are present
    expected_paths = ['metadata.metric_id', 'values.current', 'components', 'factors']
    for path in expected_paths:
        matching = [p for p in flat_context if path in p]
        if matching:
            logger.info(f"✅ Found path containing '{path}': {matching[0]}")
        else:
            logger.error(f"❌ Could not find path containing '{path}'")
            return False
    
    # STEP 2: Test unflattening context back to hierarchy
    logger.info("\nTesting context unflattening...")
    recreated_hierarchy = analyzer.unflatten_context(flat_context)
    
    # Verify key elements are present in recreated hierarchy
    if 'values' in recreated_hierarchy and 'current' in recreated_hierarchy['values']:
        current_value = recreated_hierarchy['values']['current']
        logger.info(f"✅ Recreated hierarchy contains current value: {current_value}")
    else:
        logger.error("❌ Recreated hierarchy missing expected values")
        return False
    
    # STEP 3: Test relationship analysis
    logger.info("\nTesting relationship analysis...")
    relationships = analyzer._analyze_relationships(healthcare_context, {})
    
    # Verify relationships are present in updated structure
    if 'path_encoded_relationships' in relationships and relationships['path_encoded_relationships']:
        logger.info(f"✅ Found {len(relationships['path_encoded_relationships'])} path-encoded relationships")
        # Log a few examples
        for i, rel in enumerate(relationships['path_encoded_relationships'][:2]):
            logger.info(f"  Relationship {i+1}: {rel['type']} from {rel['source']} to {rel['target']}")
    else:
        logger.warning("⚠️ No path-encoded relationships detected, this may indicate an issue")
        
    # Check for other relationship types
    if 'value_relationships' in relationships and relationships['value_relationships']:
        logger.info(f"✅ Found {len(relationships['value_relationships'])} value relationships")
    
    # STEP 4: Test full metric context analysis
    logger.info("\nTesting full metric context analysis...")
    metric_data = {
        'name': 'Patient Satisfaction',
        'value': 0.86,
        'category': 'Customer Experience',
        'components': healthcare_context['components'],
        'factors': healthcare_context['factors'],
        'trends': healthcare_context['trends']
    }
    
    analysis_result = analyzer.analyze_metric_context(
        metric_data=metric_data, 
        query="What is driving the improvement in patient satisfaction?"
    )
    
    # Verify the analysis result
    if (analysis_result and 
            'enhanced_context' in analysis_result and 
            'confidence' in analysis_result):
        logger.info(f"✅ Analysis completed with confidence: {analysis_result['confidence']}")
        return True
    else:
        logger.error("❌ Analysis failed or missing expected components")
        return False

def main():
    """Run all tests."""
    logger.info("=== Testing ContextAnalyzer Integration with PathEncoder ===")
    result = test_context_analyzer_with_path_encoder()
    
    if result:
        logger.info("\n✅ All tests PASSED!")
        return 0
    else:
        logger.error("\n❌ Tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
