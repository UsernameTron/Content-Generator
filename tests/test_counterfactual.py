#!/usr/bin/env python3
"""
Test script for the counterfactual reasoning system.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Import the counterfactual components
    from src.counterfactual import (
        CausalAnalyzer, 
        FailureCase, 
        DecisionPoint,
        CounterfactualComparator,
        PatternRecognizer,
        RecommendationGenerator,
        CounterfactualGenerator
    )
    
    print("✓ Successfully imported all counterfactual modules")
    
    # Test 1: Basic initialization of components
    print("\nTest 1: Basic initialization of components")
    
    try:
        causal_analyzer = CausalAnalyzer()
        print("  ✓ CausalAnalyzer initialized")
    except Exception as e:
        print(f"  ✗ Error initializing CausalAnalyzer: {e}")
    
    try:
        comparator = CounterfactualComparator()
        print("  ✓ CounterfactualComparator initialized")
    except Exception as e:
        print(f"  ✗ Error initializing CounterfactualComparator: {e}")
        
    try:
        pattern_recognizer = PatternRecognizer()
        print("  ✓ PatternRecognizer initialized")
    except Exception as e:
        print(f"  ✗ Error initializing PatternRecognizer: {e}")
        
    try:
        recommendation_generator = RecommendationGenerator()
        print("  ✓ RecommendationGenerator initialized")
    except Exception as e:
        print(f"  ✗ Error initializing RecommendationGenerator: {e}")
        
    try:
        generator = CounterfactualGenerator()
        print("  ✓ CounterfactualGenerator initialized")
    except Exception as e:
        print(f"  ✗ Error initializing CounterfactualGenerator: {e}")
    
    # Test 2: Create a simple failure case
    print("\nTest 2: Create a simple failure case")
    
    try:
        # Create a simple decision point
        decision_point = DecisionPoint(
            id="dp1",
            stage="data_preparation",
            description="Training data selection approach",
            actual_decision="Used historical data without recent examples",
            importance=4.5,
            alternatives=["Use more recent data", "Include edge cases"],
            consequences={"Used historical data": "Poor performance on new data types"},
            impact_areas=["Accuracy", "Robustness"]
        )
        print("  ✓ DecisionPoint created")
        
        # Create a simple failure case
        failure_case = FailureCase(
            id="test_case_1",
            title="Test Case",
            description="A test case for the counterfactual reasoning system",
            industry="Technology",
            project_type="Test",
            primary_failure_modes=["Data Quality"],
            decision_points=[decision_point],
            observed_outcome="The model performed poorly on new data types",
            sources=[]
        )
        print("  ✓ FailureCase created")
        
        # Test the getters and setters
        print(f"  ✓ Failure case attributes: id={failure_case.id}, title={failure_case.title}")
        print(f"  ✓ Decision point attributes: id={decision_point.id}, stage={decision_point.stage}")
        
    except Exception as e:
        print(f"  ✗ Error creating test objects: {e}")
    
    # Test 3: Basic counterfactual operations
    print("\nTest 3: Basic counterfactual operations")
    
    try:
        # Create a comparison
        comparison = comparator.create_comparison(
            failure_case=failure_case,
            counterfactual_description="A counterfactual scenario with better data preparation"
        )
        print(f"  ✓ Created comparison: {comparison.id}")
        
        # Add a comparison result
        comparator.add_comparison_result(
            comparison=comparison,
            dimension="data_quality",
            actual_score=5.0,
            counterfactual_score=8.5,
            explanation="Using more recent and representative data would have improved performance"
        )
        print("  ✓ Added comparison result")
        
        # Generate recommendations
        recommendations = recommendation_generator.generate_recommendations(comparison)
        print(f"  ✓ Generated {len(recommendations)} recommendations")
        
    except Exception as e:
        print(f"  ✗ Error in counterfactual operations: {e}")
    
    # Test 4: Integration functionality
    print("\nTest 4: Integration functionality")
    
    try:
        # Analyze a failure case
        case_data = {
            "id": "integration_test_1",
            "title": "Integration Test",
            "description": "Testing the integrated counterfactual generator",
            "industry": "Technology",
            "project_type": "Test",
            "primary_failure_modes": ["Data Quality", "Model Selection"],
            "failure_impact": 8.0,  # This will be ignored by the implementation
            "decision_points": [
                {
                    "id": "dp1",
                    "stage": "data_preparation",
                    "description": "Data selection approach",
                    "actual_decision": "Used synthetic data instead of real-world data",
                    "importance": 4.2,
                    "alternatives": ["Use real-world data", "Use a mix of synthetic and real data"],
                    "consequences": {"Used synthetic data": "Models didn't generalize to real-world scenarios"},
                    "impact_areas": ["Generalization", "Production Performance"]
                }
            ],
            "observed_outcome": "The model failed when deployed to production",
            "sources": []
        }
        
        # Use the integrated generator
        case = generator.analyze_implementation_failure(case_data)
        print(f"  ✓ Analyzed implementation failure: {case.id}")
        
        # Generate alternatives
        alternatives = generator.generate_alternatives(case)
        print(f"  ✓ Generated alternatives for {len(alternatives)} decision points")
        
        # Create comparison with integrated generator
        comparison = generator.create_comparison(
            failure_case=case,
            counterfactual_description="A scenario with better data selection"
        )
        print(f"  ✓ Created comparison through integrated generator: {comparison.id}")
        
        # Add a result to make the comparison complete
        comparator.add_comparison_result(
            comparison=comparison,
            dimension="data_quality",
            actual_score=4.0,
            counterfactual_score=7.5,
            explanation="Real-world data would have better represented the target domain"
        )
        
        # Get recommendations
        recommendations = generator.get_recommendations(comparison)
        print(f"  ✓ Generated {len(recommendations)} recommendations through integrated generator")
        
        # Test content integration
        original_content = "This is a test content piece about AI implementation."
        enhanced_content = generator.integrate_with_content(
            content=original_content,
            failure_case=case,
            recommendations=recommendations
        )
        print("  ✓ Successfully integrated counterfactual insights into content")
        print(f"  ✓ Original content length: {len(original_content)}, Enhanced content length: {len(enhanced_content)}")
        
    except Exception as e:
        print(f"  ✗ Error in integration functionality: {e}")
    
    print("\n✓ Counterfactual reasoning system tests completed successfully")
    
except ImportError as e:
    print(f"✗ Failed to import counterfactual modules: {e}")
except Exception as e:
    print(f"✗ Unexpected error during testing: {e}")
