# AI Model Evaluation Framework

A comprehensive, multi-domain evaluation system for assessing AI model performance across different knowledge domains and reasoning capabilities.

## Overview

This evaluation framework is designed to provide a standardized, repeatable way to assess the capabilities of language models across different knowledge domains and reasoning skills. The framework includes:

- Domain-specific knowledge assessment
- Cross-referencing capabilities
- Counterfactual reasoning abilities
- Memory usage tracking
- Performance metrics
- WandB integration for experiment tracking

## Components

### 1. Evaluation Manager

The `EvaluationManager` class in `comprehensive_evaluate.py` coordinates the evaluation process:

- Loads models and tokenizers
- Configures device settings (CPU, MPS, CUDA)
- Initializes evaluators
- Manages memory tracking
- Handles WandB integration
- Reports comprehensive results

### 2. Base Evaluator

The `BaseEvaluator` abstract class in `evaluators/base_evaluator.py` provides:

- Standard interface for all evaluators
- Response generation utilities
- Scoring mechanisms
- Result formatting

### 3. Domain-Specific Evaluators

The framework includes evaluators for assessing knowledge in specific domains:

- `CustomerExperienceEvaluator`: Customer service and experience knowledge
- `ArtificialIntelligenceEvaluator`: AI concepts and applications  
- `MachineLearningEvaluator`: ML algorithms, concepts and best practices

### 4. Reasoning Capability Evaluators

Special evaluators for assessing higher-order reasoning skills:

- `CrossReferencingEvaluator`: Ability to recall and connect information
- `CounterfactualEvaluator`: Ability to reason about hypothetical scenarios

### 5. Evaluation Data

JSON-based question repositories for each domain and capability:

- `customer_experience_questions.json`
- `artificial_intelligence_questions.json`
- `machine_learning_questions.json`
- `cross_reference_scenarios.json`
- `counterfactual_scenarios.json`

## Apple Silicon Optimizations

The framework includes special optimizations for Apple Silicon:

- Sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to prevent memory limits
- Enables `PYTORCH_ENABLE_MPS_FALLBACK=1` for compatibility
- Sets `TOKENIZERS_PARALLELISM=false` for stability
- Provides dynamic device selection based on availability
- Implements memory tracking to monitor resource usage

## Usage

### 1. Desktop Launcher

The easiest way to use the framework is with the desktop launcher:

```bash
# Run the desktop launcher
/Users/cpconnor/Desktop/run_evaluation.command
```

This provides an interactive setup with options for:
- Model adapter path selection
- Device selection (CPU/MPS)
- Evaluation domain selection
- Memory optimization settings
- WandB integration

### 2. Command Line

For more control, use the command line:

```bash
python comprehensive_evaluate.py --model_path /path/to/adapter --device mps
```

Available options:
- `--model_path`: Path to model adapter
- `--device`: Device to use (cpu, mps, cuda)
- `--batch_size`: Batch size for evaluation
- `--use_wandb`: Enable WandB integration
- `--skip_domains`: Domains to skip (comma-separated)
- `--wandb_project`: WandB project name
- `--wandb_entity`: WandB entity name
- `--track_memory`: Enable memory tracking

### 3. Programmatic Usage

```python
from comprehensive_evaluate import EvaluationManager

# Initialize the manager
manager = EvaluationManager(
    model_path="/path/to/adapter",
    device="mps",
    use_wandb=True,
    batch_size=1,
    skip_domains=["customer_experience"],
    track_memory=True
)

# Run evaluation
results = manager.run_evaluation()

# Access specific results
customer_experience_results = results.get("customer_experience", {})
```

## Adding New Evaluators

To add a new domain evaluator:

1. Create a new evaluator class in `evaluators/domain_evaluators.py`:

```python
class NewDomainEvaluator(BaseEvaluator):
    def __init__(self, manager):
        super().__init__(manager)
        self.domain_name = "new_domain"
        self.questions_file = "new_domain_questions.json"
        
    def score_response(self, response, reference_answer):
        # Custom scoring logic
        # ...
        return score, feedback
```

2. Create a questions JSON file in the `data` directory:

```json
[
    {
        "question": "What is X?",
        "reference_answer": "X is...",
        "context": "Optional context",
        "max_score": 5
    }
]
```

3. Update the `EvaluationManager.run_evaluation()` method to include your new evaluator.

## Testing

The framework includes a test script to validate its functionality:

```bash
python test_evaluation_framework.py
```

This validates:
- Evaluation data exists and is properly formatted
- Evaluator modules can be imported
- Main evaluation script functions correctly
- Desktop launcher is executable

## Memory Tracking

Memory usage is tracked throughout the evaluation process:

- Baseline memory usage before model loading
- Memory after model loading
- Memory during each evaluation phase
- Peak memory usage
- Memory usage over time (when WandB is enabled)

## WandB Integration

The framework can integrate with Weights & Biases for experiment tracking:

- Overall scores by domain
- Memory usage over time
- Individual question scores
- Response samples
- Device information
- Evaluation configurations

Enable with `--use_wandb` flag or in the desktop launcher.

## Limitations

- Currently requires pre-defined evaluation scenarios
- Dependent on model adapter availability
- Performance may vary across different hardware
- Designed primarily for language models
- Subjective scoring for complex reasoning tasks

## Future Enhancements

- Expand evaluation scenarios
- Add more domain-specific evaluators
- Improve cross-referencing logic
- Enhance counterfactual reasoning evaluation
- Add support for multi-modal models
- Implement more objective scoring metrics
- Add benchmark comparisons
