# C. Pete Connor Model Evaluation Framework

This comprehensive evaluation framework tests domain expertise across Customer Experience (CX), Artificial Intelligence (AI), and Machine Learning (ML), as well as special capabilities like cross-referencing and counterfactual reasoning.

## Features

- **Domain Knowledge Evaluation**: Tests understanding of CX, AI, and ML concepts
- **Cross-Referencing Capability**: Evaluates ability to synthesize information from multiple sources
- **Counterfactual Reasoning**: Tests ability to reason about alternative scenarios
- **Apple Silicon Optimization**: Memory-efficient processing on Apple Silicon
- **Weights & Biases Integration**: Visualize and track evaluation metrics
- **Memory Monitoring**: Proactive memory usage tracking and alerts

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- peft
- wandb (optional)
- rich
- psutil

## Usage

```bash
python comprehensive_evaluate.py --adapter-path /path/to/adapter --batch-size 3 --use-wandb
```

### Command-Line Options

- `--adapter-path`: Path to LoRA adapter folder (required)
- `--output-dir`: Directory for saving results (default: evaluation_results)
- `--temperature`: Temperature for generation (default: 0.7)
- `--batch-size`: Number of questions/scenarios per domain (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--force-cpu`: Force CPU usage even if GPU is available
- `--use-wandb`: Enable Weights & Biases integration
- `--wandb-project`: WandB project name (default: cpc-model-evaluation)
- `--skip-cross-reference`: Skip cross-referencing evaluation
- `--skip-counterfactual`: Skip counterfactual reasoning evaluation

## Output

The evaluation generates several outputs:

1. **Console Output**: Real-time progress and summary results
2. **JSON Results**: Detailed evaluation results saved to `evaluation_results/eval_results_TIMESTAMP.json`
3. **Memory Tracking**: CSV file with memory usage data
4. **WandB Dashboard**: Interactive visualizations (if enabled)

## Extending the Framework

### Adding New Domain Evaluators

1. Create a new evaluator class that inherits from `DomainKnowledgeEvaluator`
2. Implement the `get_default_questions` method
3. Add your evaluator to the `_setup_evaluators` method in `EvaluationManager`

### Custom Evaluation Data

You can provide custom evaluation questions by creating JSON files in the `data/evaluation/` directory:

- `customer_experience_questions.json`
- `artificial_intelligence_questions.json`
- `machine_learning_questions.json`
- `cross_reference_scenarios.json`
- `counterfactual_scenarios.json`

## Apple Silicon Optimizations

The evaluation framework includes special optimizations for Apple Silicon:

- Sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to disable upper memory limits
- Enables fallback operations with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Disables tokenizer parallelism
- Uses appropriate dtype based on device (float32 for MPS)
- Monitors memory usage and provides warnings at 70% and 85% usage

## Memory Monitoring

The framework actively monitors memory usage throughout the evaluation process:

- WARNING at >70% memory usage
- CRITICAL at >85% memory usage
- Memory tracking at key checkpoints:
  - Before model loading
  - After model loading
  - Before/after generation

## Weights & Biases Integration

When enabled with `--use-wandb`, the framework logs:

- Evaluation scores for each domain/capability
- Memory usage tracking
- Detailed metrics for each evaluator
- Configuration parameters
