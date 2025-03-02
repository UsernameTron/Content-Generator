#!/usr/bin/env python3
"""
Advanced test script for the counterfactual reasoning system.
"""

import unittest
import logging
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.counterfactual import (
    CounterfactualGenerator, 
    CausalAnalyzer,
    CounterfactualComparator, 
    PatternRecognizer,
    RecommendationGenerator,
    DecisionPoint,
    FailureCase
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCounterfactualAdvanced(unittest.TestCase):
    """Advanced test class for the counterfactual reasoning system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary test directories
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize the main generator with test directories
        self.generator = CounterfactualGenerator(
            data_dir=str(self.test_dir),
            output_dir=str(self.test_dir / "output")
        )
        
        # Get references to the components
        self.causal_analyzer = self.generator.causal_analyzer
        self.comparator = self.generator.comparator
        self.pattern_recognizer = self.generator.pattern_recognizer
        self.recommendation_generator = self.generator.recommendation_generator
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up test data
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_complex_failure_case(self):
        """Test the system with a complex failure case."""
        # Create a complex failure case with multiple decision points
        decision_points = [
            DecisionPoint(
                id="dp1",
                description="Data preprocessing approach",
                actual_decision="Minimal preprocessing with basic outlier removal",
                alternatives=["Extensive feature engineering", "Domain-specific preprocessing pipeline"],
                consequences={
                    "Minimal preprocessing with basic outlier removal": "Model struggled with noisy data",
                    "Extensive feature engineering": "Would likely improve model robustness",
                    "Domain-specific preprocessing pipeline": "Would address known data issues in this domain"
                },
                impact_areas=["model_performance", "data_quality"],
                importance=0.8,
                stage="data_preparation"
            ),
            DecisionPoint(
                id="dp2",
                description="Model selection strategy",
                actual_decision="Chose based on academic popularity",
                alternatives=["Industry-proven model for similar problems", "Custom model for specific requirements"],
                consequences={
                    "Chose based on academic popularity": "Model was theoretically sound but impractical",
                    "Industry-proven model for similar problems": "Would provide reliable baseline performance",
                    "Custom model for specific requirements": "Would address specific challenges in this domain"
                },
                impact_areas=["model_selection", "performance_optimization"],
                importance=0.9,
                stage="modeling"
            ),
            DecisionPoint(
                id="dp3",
                description="Validation strategy",
                actual_decision="Simple train/test split",
                alternatives=["Cross-validation", "Time-based validation", "Out-of-distribution validation"],
                consequences={
                    "Simple train/test split": "Failed to detect overfitting to specific data patterns",
                    "Cross-validation": "Would reveal variance in model performance",
                    "Time-based validation": "Would reveal temporal generalization issues",
                    "Out-of-distribution validation": "Would test real-world robustness"
                },
                impact_areas=["evaluation", "robustness"],
                importance=0.7,
                stage="validation"
            )
        ]
        
        failure_case = FailureCase(
            id="complex_case_1",
            title="ML Model Deployment Failure in Healthcare",
            description="A diagnostic model that performed well in lab settings failed when deployed in real clinical environments",
            industry="Healthcare",
            project_type="Diagnostic AI",
            primary_failure_modes=["overfitting", "data_drift", "insufficient_validation"],
            decision_points=decision_points,
            observed_outcome="The model's performance degraded significantly in real-world settings, requiring emergency rollback",
            sources=[
                {"type": "interview", "details": "Interview with project lead, June 2024"},
                {"type": "case_study", "details": "Internal post-mortem report"}
            ]
        )
        
        # Test the analysis
        logger.info("Testing causal analysis...")
        self.causal_analyzer.analyze_causal_relationships(failure_case)
        self.assertIn(failure_case.id, self.causal_analyzer.failure_cases)
        
        # Test alternative generation
        logger.info("Testing alternative generation...")
        alternatives = self.generator.generate_alternatives(failure_case.id)
        self.assertTrue(alternatives)
        self.assertEqual(len(alternatives), len(failure_case.decision_points))
        
        # Test comparison generation
        logger.info("Testing comparison generation...")
        comparison = self.generator.create_comparison(failure_case.id)
        self.assertIsNotNone(comparison)
        
        # Test recommendation generation
        logger.info("Testing recommendation generation...")
        recommendations = self.generator.generate_recommendations(comparison.id)
        self.assertTrue(recommendations)
        
        # Test content integration
        logger.info("Testing content integration...")
        original_content = """# Healthcare AI Implementation Report

This report discusses the challenges encountered in our diagnostic AI project.
"""
        
        # Test different integration styles
        for style in ["minimal", "summary", "detailed"]:
            enhanced_content = self.generator.integrate_with_content(
                original_content, 
                failure_case,
                recommendations,
                integration_style=style
            )
            self.assertGreater(len(enhanced_content), len(original_content))
            self.assertIn("counterfactual", enhanced_content.lower())
        
        # Test pattern recognition
        logger.info("Testing pattern recognition...")
        self.pattern_recognizer.add_failure_case(failure_case)
        patterns = self.pattern_recognizer.identify_patterns_in_case(failure_case.id)
        self.assertTrue(patterns)
        
        # Test report generation
        logger.info("Testing report generation...")
        report = self.generator.generate_report(failure_case.id)
        self.assertIn(failure_case.title, report)
        
        logger.info("Advanced tests completed successfully")
        
if __name__ == '__main__':
    print("Starting advanced tests for the counterfactual reasoning system...")
    unittest.main()
