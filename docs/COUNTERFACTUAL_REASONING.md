# Counterfactual Reasoning Module

## Overview

The Counterfactual Reasoning Module enhances AI/ML implementation analysis by examining "what could have been done differently." This module helps users identify critical decision points in failed implementations, analyze alternatives, recognize patterns across industries, and generate actionable recommendations.

## Key Components

### 1. Causal Analysis

The causal analysis framework identifies critical decision points in AI/ML implementations and their causal relationships.

```python
from src.counterfactual import CausalAnalyzer, FailureCase

# Initialize the analyzer
analyzer = CausalAnalyzer()

# Create a failure case
failure_case = FailureCase(
    id="case_001",
    title="Recommendation System Deployment Failure",
    description="Description of the failure...",
    industry="E-commerce",
    project_type="Recommendation System",
    primary_failure_modes=["Data Quality", "Evaluation Metrics"],
    failure_impact=8.5,
    decision_points=[]  # Will be populated during analysis
)

# Analyze causal relationships
analyzer.analyze_causal_relationships(failure_case)

# Generate a report
report = analyzer.generate_case_report(failure_case)
```

### 2. Structured Comparison

The structured comparison module systematically contrasts actual implementation decisions with potential alternatives across various dimensions.

```python
from src.counterfactual import CounterfactualComparator

# Initialize the comparator
comparator = CounterfactualComparator()

# Create a comparison
comparison = comparator.create_comparison(
    failure_case=failure_case,
    counterfactual_description="Description of what could have been done differently..."
)

# Add comparison results for various dimensions
comparator.add_comparison_result(
    comparison=comparison,
    dimension="data_quality",
    actual_score=5.0,
    counterfactual_score=8.5,
    explanation="Using cleaner, more representative data would have significantly improved performance."
)

# Generate comparison report
report = comparator.generate_comparison_report(comparison)
```

### 3. Pattern Recognition

The pattern recognition module identifies recurring failure patterns across different industries and project types.

```python
from src.counterfactual import PatternRecognizer

# Initialize the recognizer
recognizer = PatternRecognizer()

# Identify patterns across all analyzed cases
patterns = recognizer.identify_patterns()

# Get patterns for a specific case
case_patterns = recognizer.get_pattern_by_case(failure_case.id)

# Visualize patterns
recognizer.visualize_pattern_network(output_path="pattern_network.png")
```

### 4. Recommendation Generation

The recommendation generator creates actionable insights based on counterfactual analysis.

```python
from src.counterfactual import RecommendationGenerator

# Initialize the generator
rec_generator = RecommendationGenerator()

# Generate recommendations from a comparison
recommendations = rec_generator.generate_recommendations(comparison)

# Prioritize recommendations
prioritized = rec_generator.prioritize_recommendations(recommendations)

# Generate recommendation report
report = rec_generator.generate_recommendation_report(prioritized)
```

### 5. Integrated Counterfactual Generator

The main interface that integrates all components for seamless analysis.

```python
from src.counterfactual import CounterfactualGenerator

# Initialize the generator
generator = CounterfactualGenerator()

# Analyze a failure case
failure_case = generator.analyze_implementation_failure(case_data)

# Generate alternatives
alternatives = generator.generate_alternatives(failure_case)

# Create comparison
comparison = generator.create_comparison(
    failure_case=failure_case,
    counterfactual_description="Description of what could have been done differently..."
)

# Identify patterns
patterns = generator.identify_patterns(failure_case)

# Get recommendations
recommendations = generator.get_recommendations(comparison)

# Generate comprehensive insight report
report = generator.generate_insight_report(
    failure_case=failure_case,
    comparison=comparison,
    recommendations=recommendations,
    patterns=patterns,
    output_path="insight_report.md"
)

# Integrate insights into content
enhanced_content = generator.integrate_with_content(
    content=original_content,
    failure_case=failure_case,
    recommendations=recommendations,
    integration_style="summary"  # Options: "summary", "detailed", "minimal"
)
```

## Using the Desktop Launcher

For ease of use, a desktop launcher is provided at `/Users/cpconnor/Desktop/counterfactual_analyzer.command`. This launcher is optimized for Apple Silicon and runs a demonstration of the counterfactual reasoning functionality.

To use the launcher:
1. Double-click the `counterfactual_analyzer.command` file on your Desktop
2. The terminal will open and run the demonstration
3. Output files will be saved to the `output/counterfactual` directory

## Integration with Content Generation

The counterfactual reasoning module seamlessly integrates with the existing content generation system, enhancing generated content with insights from counterfactual analysis.

```python
from src.models import ModelContentGenerator
from src.counterfactual import CounterfactualGenerator

# Initialize content generator and counterfactual generator
content_generator = ModelContentGenerator()
counterfactual_generator = CounterfactualGenerator()

# Generate base content
content = content_generator.generate("AI implementation failure in healthcare")

# Enhance with counterfactual insights
enhanced_content = counterfactual_generator.integrate_with_content(
    content=content,
    failure_case=analyzed_case,
    recommendations=recommendations
)
```

## Data Storage

By default, the module stores:
- Failure cases in `data/counterfactual/failure_cases`
- Comparisons in `data/counterfactual/comparisons`
- Patterns in `data/counterfactual/patterns`
- Recommendations in `data/counterfactual/recommendations`
- Reports and visualizations in `output/counterfactual`

## Requirements

The counterfactual reasoning module requires the following dependencies:
- `networkx` for causal graph analysis
- `pandas` for data manipulation
- `matplotlib` for visualizations
- Standard Python data science libraries

These are already included in the project's `requirements.txt` file.
