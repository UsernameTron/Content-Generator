#!/usr/bin/env python3
"""
Example script for running counterfactual analysis on AI/ML implementation failures.
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counterfactual.counterfactual_generator import CounterfactualGenerator

def main():
    """Run a simple counterfactual analysis example."""
    print("Counterfactual Analysis Example")
    print("===============================")
    
    # Initialize the counterfactual generator
    generator = CounterfactualGenerator()
    
    # Example failure case data
    example_case = {
        "id": "example_case_1",
        "title": "NLP Model Deployment Failure in Customer Service",
        "description": "A natural language processing model for customer service automation failed to meet performance expectations in production, resulting in frequent escalations to human agents.",
        "industry": "Retail",
        "project_type": "NLP",
        "primary_failure_modes": ["Data Mismatch", "Model Selection"],
        "failure_impact": 8.5,
        "decision_points": [
            {
                "id": "dp1",
                "stage": "data_preparation",
                "description": "Training data selection approach",
                "actual_decision": "Used historical support tickets without recent data",
                "importance": 4.5
            },
            {
                "id": "dp2",
                "stage": "model_selection",
                "description": "Model architecture choice",
                "actual_decision": "Selected a complex transformer architecture without tuning for deployment constraints",
                "importance": 4.2
            },
            {
                "id": "dp3",
                "stage": "evaluation",
                "description": "Evaluation approach",
                "actual_decision": "Evaluated only on accuracy metrics without considering latency or resource usage",
                "importance": 3.8
            },
            {
                "id": "dp4",
                "stage": "deployment",
                "description": "Deployment strategy",
                "actual_decision": "Full rollout without phased approach or monitoring",
                "importance": 4.7
            }
        ]
    }
    
    # Step 1: Analyze implementation failure
    print("\nStep 1: Analyzing implementation failure...")
    failure_case = generator.analyze_implementation_failure(example_case)
    print(f"  ✓ Analysis complete for case: {failure_case.id}")
    
    # Step 2: Generate alternatives
    print("\nStep 2: Generating alternatives for critical decisions...")
    alternatives = generator.generate_alternatives(failure_case)
    print(f"  ✓ Generated alternatives for {len(alternatives)} decision points")
    
    # Step 3: Create comparison
    print("\nStep 3: Creating structured comparison...")
    comparison = generator.create_comparison(
        failure_case,
        "A counterfactual scenario where data preparation included recent examples, model selection considered deployment constraints, evaluation included operational metrics, and deployment used a phased approach with monitoring."
    )
    
    # Add example comparison results
    generator.comparator.add_comparison_result(
        comparison=comparison,
        dimension="data_quality",
        actual_score=5.0,
        counterfactual_score=8.5,
        explanation="The counterfactual approach with recent, more representative data would have better matched real-world use cases."
    )
    
    generator.comparator.add_comparison_result(
        comparison=comparison,
        dimension="model_suitability",
        actual_score=4.5,
        counterfactual_score=7.5,
        explanation="A model selected with deployment constraints in mind would have balanced accuracy and operational requirements better."
    )
    
    generator.comparator.add_comparison_result(
        comparison=comparison,
        dimension="evaluation_rigor",
        actual_score=3.5,
        counterfactual_score=8.0,
        explanation="Including latency and resource metrics would have identified deployment issues earlier."
    )
    
    generator.comparator.add_comparison_result(
        comparison=comparison,
        dimension="deployment_strategy",
        actual_score=2.0,
        counterfactual_score=9.0,
        explanation="A phased rollout with monitoring would have limited the impact of issues and allowed for corrective action."
    )
    
    generator.comparator.set_overall_assessment(
        comparison=comparison,
        assessment="The counterfactual approach demonstrates how a more comprehensive, production-oriented development process would have resulted in a more successful deployment, highlighting the importance of representative data, deployment-aware model selection, comprehensive evaluation, and cautious rollout strategies."
    )
    
    generator.comparator.save_comparison(comparison)
    print(f"  ✓ Created and saved structured comparison: {comparison.id}")
    
    # Step 4: Identify patterns
    print("\nStep 4: Identifying relevant patterns...")
    # For demo purposes, we create a pattern
    if generator.pattern_recognizer.patterns:
        patterns = generator.identify_patterns(failure_case)
    else:
        # Create a sample pattern for demonstration
        from src.counterfactual.pattern_recognition import FailurePattern
        pattern = FailurePattern(
            id="pattern_sample",
            name="Data Mismatch & Deployment Strategy",
            description="A pattern characterized by training on unrepresentative data and deploying without proper monitoring or phased approach.",
            failure_modes=["Data Mismatch", "Deployment Strategy"],
            decision_patterns=[
                {
                    "stage": "data_preparation",
                    "decision": "Used historical data without recent examples",
                    "description": "Data selection approach",
                    "frequency": 0.7,
                    "cases": [failure_case.id]
                },
                {
                    "stage": "deployment",
                    "decision": "Full rollout without phased approach",
                    "description": "Deployment strategy",
                    "frequency": 0.8,
                    "cases": [failure_case.id]
                }
            ],
            affected_industries=["Retail", "Healthcare"],
            affected_project_types=["NLP", "Recommendation Systems"],
            case_ids=[failure_case.id],
            severity=7.5,
            frequency=0.3
        )
        generator.pattern_recognizer.save_pattern(pattern)
        patterns = [pattern]
    
    print(f"  ✓ Identified {len(patterns)} relevant patterns")
    
    # Step 5: Generate recommendations
    print("\nStep 5: Generating recommendations...")
    recommendations = generator.get_recommendations(comparison)
    
    # Save recommendations
    for rec in recommendations:
        generator.recommendation_generator.save_recommendation(rec)
    
    print(f"  ✓ Generated and saved {len(recommendations)} recommendations")
    
    # Step 6: Generate insight report
    print("\nStep 6: Generating comprehensive insight report...")
    output_path = Path("output/counterfactual/reports/example_insight_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = generator.generate_insight_report(
        failure_case=failure_case,
        comparison=comparison,
        recommendations=recommendations,
        patterns=patterns,
        output_path=str(output_path)
    )
    
    print(f"  ✓ Generated insight report and saved to: {output_path}")
    
    # Step 7: Demonstrate content integration
    print("\nStep 7: Demonstrating integration with content...")
    original_content = """
# Analysis of Customer Service NLP Implementation Failure
    
This case study examines the failure of an NLP model implementation for customer service automation.
The system was designed to handle common customer inquiries but underperformed in production.
    
## Key Issues
    
The implementation faced several challenges in production, including mismatched training data,
high latency, and insufficient monitoring during rollout.
"""
    
    enhanced_content = generator.integrate_with_content(
        content=original_content,
        failure_case=failure_case,
        recommendations=recommendations
    )
    
    # Save enhanced content
    content_path = Path("output/counterfactual/examples/enhanced_content_example.md")
    content_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(content_path, 'w') as f:
        f.write(enhanced_content)
    
    print(f"  ✓ Enhanced content saved to: {content_path}")
    
    print("\n===============================")
    print("Counterfactual analysis complete!")
    print("===============================")
    print(f"\nOutput files can be found in: {Path('output/counterfactual').resolve()}")
    print("To view the insight report, open the markdown file in your favorite editor.")

if __name__ == "__main__":
    main()
