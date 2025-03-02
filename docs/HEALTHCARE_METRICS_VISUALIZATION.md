# Healthcare Metrics Visualization System

This document describes the Healthcare Metrics Visualization System, a comprehensive pipeline for tracking and analyzing healthcare cross-reference model performance over time.

## Overview

The Healthcare Metrics Visualization System consists of three core components:

1. **Healthcare Cross-Reference Pipeline**: Manages the end-to-end process of evaluation and visualization.
2. **Metrics Conversion Tool**: Transforms raw evaluation results into visualization-ready format.
3. **Visualization Engine**: Generates comprehensive visualizations and reports from processed metrics.

## Components

### 1. Healthcare Cross-Reference Pipeline

The pipeline in `healthcare_cross_reference_pipeline.py` orchestrates the entire process:
- Loads and runs the healthcare cross-reference model
- Processes evaluation data
- Generates visualizations
- Tracks metrics over time

```python
from scripts.healthcare_cross_reference_pipeline import run_pipeline

# Run the pipeline with metric history tracking
results = run_pipeline(
    model_path="models/healthcare_base",
    adapter_path="models/healthcare_lora",
    data_dir="data/healthcare",
    output_dir="output/healthcare",
    device="mps",  # Options: cpu, cuda, mps
    skip_evaluation=True,  # Use existing results if available
    metrics_history="output/healthcare/metrics_history.json"
)

# Access generated files
print(f"Results: {results['results']}")
print(f"Visualizations: {results['viz_output']}")
```

### 2. Metrics Conversion

The `convert_healthcare_results.py` script transforms raw evaluation data:
- Converts raw JSON evaluation data to visualization-ready format
- Calculates performance summaries and averages
- Prepares metrics for temporal analysis
- Structures data for multi-dimensional visualization

```python
from scripts.convert_healthcare_results import convert_results

# Convert results to visualization format
converted_data = convert_results(
    input_file="output/healthcare/healthcare_eval_latest.json",
    output_file="output/healthcare/healthcare_eval_latest_viz.json"
)
```

### 3. Visualization Engine

The `visualize_metrics.py` script provides comprehensive visualization:
- Generates temporal trend charts
- Creates performance comparison visualizations
- Produces gap analysis visualizations
- Generates comprehensive HTML reports
- Tracks metrics against target thresholds

```python
from scripts.visualize_metrics import generate_all_visualizations

# Generate visualizations from metrics data
visualizations = generate_all_visualizations(
    input_file="output/healthcare/healthcare_eval_latest_viz.json",
    output_dir="output/healthcare/visualizations",
    metrics_history="output/healthcare/metrics_history.json"
)
```

## Key Visualization Features

### 1. Temporal Trend Analysis
- Tracks performance metrics over time
- Visualizes progress toward target thresholds
- Identifies performance trends and patterns

### 2. Performance Gap Analysis
- Highlights gaps between current and target performance
- Visualizes performance improvements over time
- Identifies areas needing improvement

### 3. Contradiction Detection Metrics
- Tracks accuracy of contradiction detection
- Analyzes performance by category and domain
- Monitors progress toward target accuracy (0.75)

### 4. Evidence Ranking Performance
- Visualizes evidence ranking accuracy
- Tracks metrics by evidence type and domain
- Monitors progress toward target accuracy (0.80)

### 5. Contradiction Type Analysis
- Performance by contradiction type
- Domain-specific contradiction detection
- Medical specialty analysis

### 6. Temporal Contradiction Patterns
- Time gap analysis by contradiction type
- Medical domain temporal changes
- Timeline visualization of medical contradictions

## HTML Report Generation

The system generates comprehensive HTML reports with:
- Performance visualizations
- Evaluation details and metadata
- Detailed topic performance analysis
- Performance trend analysis
- Domain-specific visualizations

## Metrics History Tracking

The system maintains a metrics history for temporal analysis:
- Stores evaluation results with timestamps
- Tracks both current and target metrics
- Enables trend analysis and visualization
- Maintains consistent metrics schema over time

Example metrics history format:
```json
[
  {
    "timestamp": "2025-02-20 15:30:00",
    "metrics": {
      "contradiction_detection": {
        "accuracy": 0.55,
        "target": 0.75
      },
      "evidence_ranking": {
        "accuracy": 0.6,
        "target": 0.8
      }
    }
  }
]
```

## Usage

### Running the Complete Pipeline

```bash
python scripts/healthcare_cross_reference_pipeline.py \
    --data output/healthcare \
    --output output/healthcare/pipeline_output \
    --skip-evaluation \
    --metrics-history output/healthcare/metrics_history.json
```

### Generate Visualizations Only

```bash
python scripts/visualize_metrics.py \
    --results output/healthcare/pipeline_output/healthcare_eval_latest_viz.json \
    --output output/healthcare/visualizations \
    --metrics-history output/healthcare/metrics_history.json \
    --html
```

### Convert Results Only

```bash
python scripts/convert_healthcare_results.py \
    --input output/healthcare/pipeline_output/healthcare_eval_latest.json \
    --output output/healthcare/pipeline_output/healthcare_eval_latest_viz.json \
    --metrics output/healthcare/pipeline_output/healthcare_contradiction_metrics.json
```

### Contradiction Detection Improvements Tracking

The system now includes a new visualization for tracking contradiction detection improvements over time:

```bash
# Generate contradiction improvement visualization from existing metrics history
python scripts/visualize_metrics.py \
    --results output/healthcare/pipeline_output/healthcare_eval_latest_viz.json \
    --output output/healthcare/visualizations \
    --metrics-history output/healthcare/metrics_history.json \
    --contradiction-only
```

#### Metrics History

The contradiction improvement tracking works by:
1. Reading historical metrics from a metrics history JSON file (if available)
2. Generating synthetic metrics history for demonstration purposes (if no history file exists)
3. Creating a visualization showing improvement trends over time

The visualization tracks:
- Overall contradiction detection performance
- Performance by contradiction type (direct, temporal, methodological)
- Performance by medical domain

## Implementation Details

- **Visualization Library**: Uses matplotlib and seaborn for visualization generation
- **Error Handling**: Implements robust error handling with dedicated logging for visualization generation
- **Apple Silicon Optimization**: Optimized for performance on Apple Silicon via MPS
- **Fallback Mechanisms**: Built-in fallback to minimal history generation if synthetic history generation fails

## Features

### Healthcare Performance Metrics Visualization

* **Contradiction Detection Metrics**
  - Accuracy visualization by contradiction type
  - Performance across medical domains
  - Comparison against target thresholds

* **Evidence Ranking Analysis**
  - Precision/recall/F1 metrics
  - Ranking quality visualization
  - Source verification metrics

* **Temporal Performance Analysis**
  - Historical performance tracking
  - Trend line visualization
  - Regression analysis

* **Performance Gap Analysis**
  - Current vs. target metric visualization
  - Gap significance highlighting
  - Prioritization by gap size

* **Contradiction Type Analysis**
  - Performance by contradiction type
  - Domain-specific contradiction detection
  - Medical specialty analysis

* **Temporal Contradiction Patterns**
  - Time gap analysis by contradiction type
  - Medical domain temporal changes
  - Timeline visualization of medical contradictions

### Visualization Types

The system generates:
- Temporal trend charts
- Performance comparison visualizations
- Gap analysis visualizations
- Comprehensive HTML reports
- Contradiction type analysis visualizations
- Temporal contradiction patterns visualizations

## Visualization Gallery

### Healthcare Metrics Overview
![Healthcare Metrics Overview](../sample_images/healthcare_combined_metrics.png)

### Performance Gaps Analysis
![Performance Gaps](../sample_images/performance_gaps.png)

### Temporal Metrics Tracking
![Metrics Over Time](../sample_images/healthcare_metrics_over_time.png)

### Contradiction Analysis by Type and Domain
![Contradiction Types](../sample_images/healthcare_contradiction_types.png)

### Temporal Patterns in Medical Contradictions
![Contradiction Temporal Patterns](../sample_images/contradiction_temporal_patterns.png)
