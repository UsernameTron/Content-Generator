# Healthcare Metrics Visualization User Guide

This guide explains how to use the enhanced visualization features for healthcare contradiction detection metrics analysis.

## Table of Contents
- [Selective Visualization Generation](#selective-visualization-generation)
- [Contradiction Performance Comparison](#contradiction-performance-comparison)
- [Trend Annotations](#trend-annotations)
- [HTML Reports](#html-reports)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)

## Selective Visualization Generation

You can now generate only specific visualizations instead of creating all visualization types. This improves performance and lets you focus on the metrics that matter most for your analysis.

### Command Line Usage

```bash
# Generate only specific visualization types
python scripts/visualize_metrics.py --results results.json --visualization-types contradiction_improvements healthcare_contradiction_types

# From pipeline script
python scripts/healthcare_cross_reference_pipeline.py --data data/healthcare --skip-evaluation --visualization-types contradiction_improvements contradiction_temporal_patterns
```

### Available Visualization Types

- `domain_scores`: Overview of performance across domains
- `radar_chart`: Radar chart of model capabilities
- `memory_usage`: Memory usage during evaluation
- `topic_performance`: Performance on specific topics
- `scenario_performance`: Performance on scenario-based evaluations
- `healthcare_combined_metrics`: Combined healthcare metrics 
- `healthcare_metrics_trends`: Trends in healthcare metrics over time
- `performance_gaps`: Gaps between current and target performance
- `healthcare_contradiction_types`: Performance by contradiction type
- `contradiction_temporal_patterns`: Temporal patterns in contradictions
- `contradiction_improvements`: Improvements in contradiction detection

## Contradiction Performance Comparison

Compare contradiction detection performance between two different result files to identify improvements or regressions.

### Command Line Usage

```bash
# Compare two result files and save the comparison visualization
python scripts/visualize_metrics.py --compare results1.json results2.json --compare-output comparison.png

# Example with specific file paths
python scripts/visualize_metrics.py --compare data/healthcare/pipeline_output/healthcare_eval_previous.json data/healthcare/pipeline_output/healthcare_eval_latest.json --compare-output comparison_results.png
```

### Understanding the Comparison Output

The comparison visualization shows:
- Side-by-side bar charts for key metrics
- Color-coded bars (green for improvements, red for regressions)
- Labels showing the absolute values and percentage changes
- Metrics compared include:
  - Contradiction detection accuracy
  - False positive rate
  - False negative rate
  - Precision
  - Recall
  - F1 score

## Trend Annotations

The system automatically detects and annotates significant changes in metrics over time. These annotations highlight important improvements or regressions.

### Features

- Color-coded annotations (green for improvements, red for regressions)
- Arrows indicating direction of change
- Percentage change labels
- Automatic threshold detection (default: 3% change is significant)

### Command Line Usage

```bash
# Generate trend visualization with annotations
python scripts/visualize_metrics.py --results results.json --visualization-types contradiction_improvements
```

## HTML Reports

HTML reports provide a comprehensive view of all generated visualizations with added context and explanations.

### Command Line Usage

```bash
# Generate HTML report with all visualizations
python scripts/visualize_metrics.py --results results.json --html-report

# Generate HTML report with specific visualization types
python scripts/visualize_metrics.py --results results.json --visualization-types contradiction_improvements healthcare_contradiction_types --html-report
```

### Report Components

- Summary section highlighting key findings
- Interactive navigation
- Embedded visualizations
- Contextual explanations
- Timestamp and metadata

## Error Handling

The system includes robust error handling for common issues:

- Missing files or directories
- Invalid JSON format
- Incompatible data structures
- Missing metrics or fields
- Permission issues

### Troubleshooting

If you encounter errors:

1. Check that input files exist and have correct permissions
2. Verify JSON files have the expected structure
3. Ensure output directories are writable
4. Check log messages for specific error details
5. For comparison features, ensure both input files have compatible metrics

## Examples

### Example 1: Track Contradiction Detection Improvements

```bash
python scripts/visualize_metrics.py --results data/healthcare/pipeline_output/healthcare_eval_latest.json --visualization-types contradiction_improvements
```

**Output**: A line chart showing contradiction detection metrics over time with annotated trend changes.

### Example 2: Compare Current vs Previous Performance

```bash
python scripts/visualize_metrics.py --compare data/healthcare/pipeline_output/healthcare_eval_previous.json data/healthcare/pipeline_output/healthcare_eval_latest.json --compare-output comparison.png
```

**Output**: A bar chart comparing contradiction metrics between two different versions.

### Example 3: Generate Focused HTML Report

```bash
python scripts/visualize_metrics.py --results data/healthcare/pipeline_output/healthcare_eval_latest.json --visualization-types healthcare_contradiction_types contradiction_temporal_patterns --html-report
```

**Output**: An HTML report focusing on contradiction types and temporal patterns.
