#!/usr/bin/env python3
"""
Test script for path-based relationship encoding.

This script tests the bidirectional conversion between hierarchical and 
flat context representations with preserved semantic relationships.
"""

import sys
import os
import logging
from typing import Dict, List, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancement_module.context_analyzer import ContextAnalyzer
from enhancement_module.reasoning_core import extract_path_relationships
from enhancement_module.path_encoder import PathEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_bidirectional_conversion():
    """Test bidirectional conversion with path-based relationship encoding."""
    # Create an analyzer instance
    analyzer = ContextAnalyzer()
    
    # Create a sample hierarchical context
    test_hierarchy = {
        'metadata': ['test_metric_id: M12345', 'date: 2025-03-15'],
        'values': {
            'current': 0.85,
            'target': 0.90,
            'baseline': 0.76,
            'historical': [0.74, 0.78, 0.80, 0.82]
        },
        'trends': ['Increasing by 2% quarterly', 'Accelerating improvement'],
        'components': ['Provider Communication', 'Facility Quality', 'Wait Times'],
        'factors': ['Staff Training', 'Process Optimization', 'Facility Updates'],
        'notes': ['Recent improvement influenced by new training program']
    }
    
    # STEP 1: Convert hierarchy to flat context with path-based encoding using analyzer
    logger.info("Converting hierarchy to flat context using analyzer...")
    flat_context = analyzer._flatten_context_hierarchy(test_hierarchy)
    logger.info(f"Generated {len(flat_context)} flat context items")
    
    # STEP 1B: Convert hierarchy using the new PathEncoder
    logger.info("Converting hierarchy to flat context using PathEncoder...")
    encoder = PathEncoder()
    encoded_paths = encoder.encode(test_hierarchy)
    logger.info(f"PathEncoder generated {len(encoded_paths)} encoded items")
    
    # Show a sample of the flat context
    logger.info("Sample of flat context items:")
    for i, item in enumerate(flat_context[:5]):
        logger.info(f"  {i}: {item}")
    
    # Count relationship encodings
    rel_count = sum(1 for item in flat_context if 'rel:' in item)
    logger.info(f"Generated {rel_count} relationship encodings")
    
    # STEP 2: Extract relationships from flat context
    logger.info("\nExtracting relationships from flat context...")
    relationships = extract_path_relationships(flat_context)
    logger.info(f"Extracted {len(relationships)} relationships")
    
    # Show the extracted relationships
    logger.info("Extracted relationships:")
    for i, rel in enumerate(relationships):
        logger.info(f"  {i}: {rel['source']} -> {rel['target']} ({rel['type']})")
    
    # STEP 3: Convert flat context back to hierarchy
    logger.info("\nConverting flat context back to hierarchy...")
    reconstructed_hierarchy = analyzer.unflatten_context(flat_context)
    
    # Verify the reconstructed hierarchy
    logger.info("Verifying reconstructed hierarchy...")
    success = verify_hierarchies(test_hierarchy, reconstructed_hierarchy)
    
    if success:
        logger.info("SUCCESS: Bidirectional conversion test passed!")
    else:
        logger.error("FAILURE: Bidirectional conversion test failed!")
    
    return success

def verify_hierarchies(original: Dict[str, Any], reconstructed: Dict[str, Any]) -> bool:
    """
    Verify that the reconstructed hierarchy matches the original.
    
    Args:
        original: Original hierarchy
        reconstructed: Reconstructed hierarchy
        
    Returns:
        True if verification passed, False otherwise
    """
    # Verify values - handle numeric types appropriately
    def compare_values(orig, recon):
        """Compare values accounting for numeric types"""
        if isinstance(orig, (int, float)) and isinstance(recon, (int, float)):
            # For numeric values, allow for minor differences due to formatting
            return abs(orig - recon) < 0.001
        elif isinstance(orig, (int, float)) and isinstance(recon, str):
            # If original is number and reconstructed is string, try to extract number
            try:
                recon_value = float(recon)
                return abs(orig - recon_value) < 0.001
            except ValueError:
                return False
        elif isinstance(recon, (int, float)) and isinstance(orig, str):
            # If reconstructed is number and original is string, try to extract number
            try:
                orig_value = float(orig)
                return abs(orig_value - recon) < 0.001
            except ValueError:
                return False
        else:
            # For strings or other types, use exact comparison
            return orig == recon
    
    # Check current value
    if not compare_values(original['values']['current'], reconstructed['values']['current']):
        logger.error(f"Current value mismatch: {original['values']['current']} vs {reconstructed['values']['current']}")
        return False
        
    # Check target value
    if not compare_values(original['values']['target'], reconstructed['values']['target']):
        logger.error(f"Target value mismatch: {original['values']['target']} vs {reconstructed['values']['target']}")
        return False
        
    # Check baseline value
    if not compare_values(original['values']['baseline'], reconstructed['values']['baseline']):
        logger.error(f"Baseline value mismatch: {original['values']['baseline']} vs {reconstructed['values']['baseline']}")
        return False
    
    # Verify that reconstructed has relationships
    if 'relationships' not in reconstructed or not reconstructed['relationships']:
        logger.error("Reconstructed hierarchy missing relationships")
        return False
    
    # Verify key list elements exist
    for key in ['trends', 'components', 'factors', 'notes']:
        if len(original[key]) != len(reconstructed[key]):
            logger.error(f"{key} length mismatch: {len(original[key])} vs {len(reconstructed[key])}")
            return False
    
    # Success if we reach this point
    return True

def test_reasoning_with_path_encoding():
    """Test reasoning with path-based relationship encoding."""
    from enhancement_module.reasoning_core import enhance_context_analysis
    
    # Create an analyzer instance
    analyzer = ContextAnalyzer()
    
    # Create a sample hierarchical context
    test_hierarchy = {
        'metadata': ['test_metric_id: M12345', 'date: 2025-03-15'],
        'values': {
            'current': 0.85,
            'target': 0.90,
            'baseline': 0.76,
            'historical': [0.74, 0.78, 0.80, 0.82]
        },
        'trends': ['Increasing by 2% quarterly', 'Accelerating improvement'],
        'components': ['Provider Communication', 'Facility Quality', 'Wait Times'],
        'factors': ['Staff Training', 'Process Optimization', 'Facility Updates'],
        'notes': ['Recent improvement influenced by new training program']
    }
    
    # STEP 1: Convert hierarchy to flat context with path-based encoding
    logger.info("Converting hierarchy to flat context...")
    flat_context = analyzer._flatten_context_hierarchy(test_hierarchy)
    
    # STEP 1B: Use the new PathEncoder
    logger.info("Using PathEncoder for healthcare context...")
    encoder = PathEncoder()
    
    # Create a healthcare context with patient experience data
    healthcare_context = {
        'patient': {
            'demographics': {
                'age': 65,
                'conditions': ['diabetes', 'hypertension']
            },
            'metrics': {
                'satisfaction': 0.86,
                'responsiveness': 0.83,
                'communication': 0.88
            }
        },
        'provider': {
            'quality': 0.90,
            'empathy': 0.87,
            'knowledge': 0.92
        },
        'facility': {
            'cleanliness': 0.95,
            'accessibility': 0.82,
            'wait_times': 0.78
        }
    }
    
    # Encode the healthcare context
    healthcare_encoded = encoder.encode(healthcare_context)
    
    # Add explicit relationships
    healthcare_encoded.append(encoder.encode_relationships(
        'provider.empathy', 'patient.metrics.satisfaction', 'direct_influence'
    ))
    healthcare_encoded.append(encoder.encode_relationships(
        'facility.wait_times', 'patient.metrics.satisfaction', 'inverse_correlation'
    ))
    
    # STEP 2: Use reasoning core with path-encoded context
    logger.info("\nApplying reasoning core to path-encoded context...")
    query = "What factors are contributing to the patient satisfaction improvements?"
    enhanced_context = enhance_context_analysis(flat_context, query)
    
    # Check confidence score
    logger.info(f"Confidence score: {enhanced_context['confidence']}")
    
    # Check that we have a reasonable confidence
    if enhanced_context['confidence'] >= 0.65:
        logger.info("SUCCESS: Reasoning with path encoding test passed!")
        return True
    else:
        logger.error("FAILURE: Reasoning with path encoding test failed!")
        return False

def test_standalone_path_encoder():
    """Test the standalone PathEncoder implementation."""
    # Create a PathEncoder instance
    encoder = PathEncoder()
    
    # Test with a healthcare metrics context
    test_context = {
        "patient": {
            "demographics": {
                "age": 65,
                "conditions": ["diabetes", "hypertension"]
            },
            "metrics": {
                "reasoning": 0.85,
                "adaptability": 0.78
            }
        }
    }
    
    # Encode the test context
    logger.info("Encoding test context with PathEncoder...")
    encoded = encoder.encode(test_context)
    for i, path in enumerate(encoded[:5]):
        logger.info(f"  Path {i+1}: {path}")
    
    # Decode the encoded paths
    logger.info("Decoding encoded paths...")
    decoded = encoder.decode(encoded)
    
    # Add a relationship and test relationship encoding
    logger.info("Testing relationship encoding...")
    rel = encoder.encode_relationships("patient.metrics.reasoning", "patient.metrics.adaptability", "correlation_high")
    logger.info(f"  Relationship: {rel}")
    encoded.append(rel)
    
    # Decode with relationship
    decoded_with_rel = encoder.decode(encoded)
    
    # Test relevance calculation
    logger.info("Testing relevance calculation...")
    relevance = encoder.calculate_path_encoding_relevance(encoded, "reasoning")
    for path, score in list(relevance.items())[:3]:
        logger.info(f"  Relevance for '{path}': {score}")
    
    # Verify original values were preserved in decoding
    if "patient" in decoded and "metrics" in decoded["patient"]:
        reasoning = decoded["patient"]["metrics"].get("reasoning")
        adaptability = decoded["patient"]["metrics"].get("adaptability")
        logger.info(f"  Decoded values - reasoning: {reasoning}, adaptability: {adaptability}")
        if reasoning == "0.85" and adaptability == "0.78":
            logger.info("\u2705 Values correctly preserved")
            return True
    
    logger.error("\u274c Values not correctly preserved in decoding")
    return False

def run_all_tests():
    """Run all path encoding tests."""
    logger.info("=== Running Path-Based Relationship Encoding Tests ===")
    
    bidirectional_test = test_bidirectional_conversion()
    reasoning_test = test_reasoning_with_path_encoding()
    encoder_test = test_standalone_path_encoder()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Bidirectional Conversion: {'PASSED' if bidirectional_test else 'FAILED'}")
    logger.info(f"Reasoning with Path Encoding: {'PASSED' if reasoning_test else 'FAILED'}")
    logger.info(f"Standalone PathEncoder: {'PASSED' if encoder_test else 'FAILED'}")
    logger.info(f"Overall Result: {'PASSED' if (bidirectional_test and reasoning_test and encoder_test) else 'FAILED'}")
    
    return bidirectional_test and reasoning_test and encoder_test

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
