# Healthcare Contradiction Detection Continuous Learning System

This project implements an adaptive continuous learning framework for healthcare contradiction detection. The system automatically improves model performance through iterative training and analysis of model outputs.

## Key Features

- **Performance Analysis**: Automatically identifies areas where the model needs improvement
- **Targeted Example Generation**: Creates new training examples focused on weak areas
- **Dataset Management**: Updates training datasets with new examples while maintaining history
- **Learning Tracking**: Monitors learning progress and visualizes improvements over time
- **Multi-cycle Learning**: Supports iterative learning cycles for continuous improvement

## System Architecture

The continuous learning system consists of several components:

1. **Metrics Visualizer**: Analyzes model performance and generates visualizations
2. **Healthcare Continuous Learning**: Core component that manages the learning cycle
3. **Example Generator**: Creates synthetic examples targeting model weaknesses
4. **Dataset Manager**: Maintains and updates training datasets
5. **Learning Tracker**: Records learning events and visualizes progress

## Getting Started

### Prerequisites

- Python 3.9+
- matplotlib
- rich

### Directory Structure

```
data/
  healthcare/
    contradiction_dataset/  # Contains contradiction examples
    training/               # Training data
    evaluation/             # Evaluation results
    learning_history.json   # Learning event history
```

## Usage

### Single Learning Cycle

Run a single continuous learning cycle:

```bash
python scripts/demonstrate_continuous_learning.py --examples 20 --accuracy 0.75
```

Options:
- `--data-dir`: Path to healthcare data directory (default: "data/healthcare")
- `--examples`: Number of examples to generate (default: 20)
- `--accuracy`: Simulated model accuracy (default: 0.75)

### Multiple Learning Cycles

Run multiple learning cycles and track progress:

```bash
python scripts/demonstrate_learning_progress.py --cycles 3 --examples 10
```

Options:
- `--data-dir`: Path to healthcare data directory (default: "data/healthcare")
- `--cycles`: Number of learning cycles to run (default: 3)
- `--examples`: Number of examples to generate per cycle (default: 10)
- `--accuracy`: Starting accuracy (default: 0.75)

### Generate Synthetic Data

Generate synthetic evaluation data:

```bash
python scripts/generate_synthetic_evaluation.py --output path/to/output.json --accuracy 0.75
```

### Interactive Learning Dashboard

Run the interactive dashboard to manage and visualize the learning system:

```bash
python scripts/interactive_learning_dashboard.py
```

The dashboard provides a comprehensive interface for:
- Running learning cycles
- Visualizing learning progress
- Viewing learning history
- Generating synthetic evaluation data
- Monitoring system metrics

For detailed instructions, see the [Dashboard Guide](docs/DASHBOARD_GUIDE.md).

### Initialize Dashboard Data

To initialize or reset the data structure used by the dashboard:

```bash
python scripts/initialize_dashboard_data.py
```

Options:
- `--data-dir`: Path to healthcare data directory (default: "data/healthcare")
- `--force`: Force reinitialization even if files exist

## Core Components

### HealthcareContinuousLearning Class

The `HealthcareContinuousLearning` class in `visualize_metrics.py` is the core component responsible for:

1. Analyzing model performance
2. Identifying improvement areas
3. Generating new training examples
4. Updating training datasets
5. Tracking learning progress

### Learning Cycle

A typical learning cycle includes:

1. **Evaluation**: Assess model performance on test data
2. **Analysis**: Identify areas for improvement (categories, domains)
3. **Example Generation**: Create new examples focusing on weak areas
4. **Dataset Update**: Add new examples to the training data
5. **Learning Event Tracking**: Record the learning event and metrics

## Example Generation

Examples are generated based on:

- **Performance Analysis**: Focuses on categories and domains with lower accuracy
- **Template-based Variation**: Uses existing examples as templates
- **Domain Knowledge**: Incorporates domain-specific knowledge
- **Similarity Checks**: Avoids generating duplicate examples

## Learning Progress Visualization

The system tracks and visualizes:

- **Accuracy Improvements**: How model accuracy changes over learning cycles
- **Example Count**: Number of examples generated per cycle
- **Improvement Areas**: Number of areas identified for improvement

## Future Enhancements

- Advanced NLP-based example generation
- Integration with real model training pipelines
- Automatic hyperparameter tuning
- Advanced example quality assessment
- Reinforcement learning for optimization

## Contributors

- AI Research Team

## License

This project is licensed under the MIT License - see the LICENSE file for details.
