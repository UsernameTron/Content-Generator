# CANDOR: Satirical Cross-Platform Content Generator

CANDOR (Cross-platform Adaptation for Natural-sounding Distributed Online Rhetoric) is a powerful application that generates platform-specific satirical content using C. Pete Connor's distinctive style. The system adapts a single message across multiple platforms while maintaining the core satirical tone.

## Features

- **Modern PyQt6 Interface**: Clean, intuitive UI with tabbed output display and clipboard integration
- **CANDOR Transformation Method**: Apply a systematic satirical transformation to any content
- **Multi-platform Support**: Tailor content for YouTube, Medium, LinkedIn, and Substack
- **Multiple Input Options**: Generate content from direct text input, document uploads (PDF, DOCX, TXT), or website URLs
- **Content Analysis**: Analyze sentiment and extract keywords for hashtag generation
- **Platform-Specific Adaptation**: Automatically format content according to platform requirements
- **Sentiment-Aware Generation**: Adjust satirical approach based on content sentiment
- **Corporate Jargon Detection**: Identify and transform business buzzwords into satirical alternatives
- **Multi-Format Output**: Generate different versions (subtle, base, exaggerated) for varying audience needs
- **Hardware Acceleration**: Optimized for Apple Silicon via MPS acceleration
- **Health Monitoring**: Track system resources and performance metrics
- **Desktop Launcher**: Easy one-click application startup with automatic environment setup
- **Counterfactual Reasoning**: Analyze AI/ML implementation failures through "what if" scenarios to derive insights
- **Healthcare Metrics Visualization**: Track and analyze healthcare cross-reference model performance over time
- **Automated Test Scheduling**: Schedule and run automated tests at regular intervals for continuous validation
- **Enhanced AI Performance Metrics**: Track reasoning, knowledge integration, and adaptability metrics
- **Path-Based Relationship Encoding**: Bidirectional conversion between hierarchical and flat context representations
- **Performance Trending**: Visualize trends in AI performance metrics over time
- **Customer Experience Metrics**: Track response time, satisfaction, and usability metrics

## PyQt UI Application

The CANDOR system now features a modern PyQt6-based user interface that provides a streamlined experience for content generation:

### Features

- **Tabbed Interface**: View generated content for each platform in separate tabs
- **Multiple Input Methods**: Text input, document upload, and URL processing
- **Content Analysis**: Automatic sentiment analysis and keyword extraction
- **Platform Selection**: Choose specific platforms for content adaptation
- **Copy to Clipboard**: One-click copying of generated content
- **Responsive Design**: Clean, professional interface with status updates

### CANDOR Method

The PyQt UI implements the CANDOR transformation method:

- **C**ontextualize the content for satire
- **A**mplify corporate jargon to highlight absurdity
- **N**eutralize PR-speak with candid alternatives
- **D**ramatize statistics and claims
- **O**verstate importance of trivial details
- **R**eframe from an irreverent perspective

### Quick Start

1. Double-click the `CANDOR_Launcher.command` on your Desktop
2. Enter your content via text, document, or URL
3. Select your target platforms
4. Click "Generate Content"
5. View and copy the results from the tabbed output area

## Installation

### Quick Install (macOS)

1. Double-click the `install.command` script in the project directory
2. The script will set up the virtual environment, install dependencies, and create desktop shortcuts
3. After installation, you can use the desktop shortcut `PeteConnorContentGenerator.command`

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/multi-platform-content-generator.git
cd multi-platform-content-generator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Set up data files:
```bash
python setup_data.py
```

5. Launch the application:
```bash
python app.py
```

### Docker Installation

1. Build and run using the provided script:
```bash
chmod +x build_and_run.sh
./build_and_run.sh
```

2. Or manually with Docker Compose:
```bash
docker-compose build
docker-compose up -d
```

3. To stop the containerized application:
```bash
docker-compose down
```

## Model Fine-Tuning

The system includes a fine-tuning pipeline to train models on C. Pete Connor's writing style:

1. Run the fine-tuning process:
```bash
# Make the script executable if needed
chmod +x run_finetune.command

# Run the script
./run_finetune.command
```

2. The fine-tuning process will:
   - Prepare training data from `writing_style.json`
   - Configure W&B monitoring for tracking training metrics
   - Train the model using LoRA (Low-Rank Adaptation) for efficiency
   - Save the fine-tuned model to `outputs/finetune/final`

3. Once fine-tuning is complete, the model-based content generator will automatically use the fine-tuned model.

**Note**: Fine-tuning requires significant computational resources and may take several hours.

## Usage

1. Start the Streamlit app:
```bash
cd src
streamlit run app_model.py
```
Or use the desktop launcher for one-click startup.

2. Open your browser at `http://localhost:8501`

3. Use the application:
   - Enter text directly, upload a document, or provide a URL
   - Select the target platform, writing style, and content tone
   - Generate and customize your content
   - View sentiment analysis results alongside your content

## Writing Styles

The application includes C. Pete Connor's distinctive writing style, characterized by:
- Data-driven, no-nonsense approach
- Sharp satire with genuine expertise
- Confident and slightly irreverent tone
- Skepticism of overblown AI claims and empty buzzwords
- Use of actual statistics and research

## Weights & Biases Integration

The application integrates with Weights & Biases (W&B) for monitoring content generation. To use this feature:

1. Create a W&B account at https://wandb.ai
2. Get your API key from https://wandb.ai/settings
3. Run `python setup_wandb.py` to configure your API key
4. The application will automatically log metrics and examples to your W&B project

## Project Structure

```
multi-platform-content-generator/
├── requirements.txt       # Project dependencies
├── setup_data.py          # Data initialization script
├── setup_wandb.py         # Weights & Biases setup script
├── README.md              # Project documentation
├── data/                  # Data directory (created by setup_data.py)
│   ├── custom_templates.json  # User-defined templates
│   ├── user_preferences.json  # User preferences
│   ├── examples/          # Example content directory
│   └── test/              # Test data directory
└── src/                   # Source code
    ├── app_model.py       # Streamlit application
    ├── models/            # Content generation models
    │   ├── content_generator.py  # Content generation logic
    │   ├── templates.py          # Content templates
    │   └── platform_specs.py     # Platform-specific parameters
    ├── cross_reference/   # Cross-reference functionality
    │   ├── vector_db.py         # Vector database for content storage
    │   ├── retrieval.py         # Content retrieval logic
    │   └── integration.py       # Cross-reference integration
    ├── counterfactual/    # Counterfactual reasoning functionality
    │   ├── causal_analysis.py         # Causal analysis framework
    │   ├── comparison.py             # Structured comparison methods
    │   ├── pattern_recognition.py    # Industry pattern recognition
    │   ├── recommendation.py         # Recommendation generation
    │   └── counterfactual_generator.py # Main integration interface
    └── utils/             # Utility functions
        ├── document_processor.py # Document processing utilities
        └── health_monitor.py     # System health monitoring
```

## Templates

The application uses templates to generate content. Templates are defined in `src/models/templates.py` and use placeholders like:

- `{main_point}`: The main point extracted from the input text
- `{supporting_points}`: Additional points from the input text
- `{topic}`: The main topic of the content
- `{hashtags}`: Automatically generated hashtags based on keywords
- `{question}`: An automatically generated question related to the content

You can add custom templates in the `data/custom_templates.json` file.

## Dependencies

- streamlit: Web application framework
- requests: HTTP requests for URL content fetching
- beautifulsoup4: HTML parsing for web scraping
- PyPDF2: PDF document processing
- python-docx: Word document processing
- nltk: Natural language processing for sentiment analysis and keyword extraction
- python-dotenv: Environment variable management
- psutil: System resource monitoring
- rich: Enhanced logging and terminal output
- wandb: Weights & Biases integration for monitoring

## System Health Monitoring

The application includes a health monitoring system that tracks:

- CPU usage
- Memory usage
- Disk usage
- Application performance

Warning thresholds are set at 70% for resources, and critical thresholds at 85%.

## Counterfactual Reasoning

The system includes a counterfactual reasoning module for analyzing AI/ML implementation failures through "what if" scenarios:

1. Launch the counterfactual analyzer:
```bash
# Run the desktop launcher
/Users/cpconnor/Desktop/counterfactual_analyzer.command
```

2. The counterfactual reasoning module provides:
   - **Causal Analysis**: Identify critical decision points in AI/ML implementations
   - **Structured Comparison**: Compare actual decisions with alternatives
   - **Pattern Recognition**: Identify recurring failure patterns across industries
   - **Recommendation Generation**: Get actionable insights from counterfactual analysis

3. Key components for integration:
```python
from src.counterfactual import CounterfactualGenerator

# Initialize the generator
generator = CounterfactualGenerator()

# Analyze a failure case
failure_case = generator.analyze_implementation_failure(case_data)

# Generate recommendations
recommendations = generator.get_recommendations(comparison)
```

For detailed documentation on the counterfactual reasoning module, see [COUNTERFACTUAL_REASONING.md](docs/COUNTERFACTUAL_REASONING.md).

## Automated Test Scheduling

The system includes an automated test scheduling system that allows you to run tests at regular intervals without manual intervention.

### Features

- **Configurable Intervals**: Schedule tests to run at specified intervals (default: 24 hours)
- **Comprehensive Logging**: All test results are logged for future reference
- **Multiple Testing Modes**: Support for both regular and testing dashboard modes
- **Desktop Launcher**: Easy-to-use launcher for starting the scheduler

### Using the Scheduler

1. Launch the scheduler using the desktop shortcut:
```bash
./healthcare_scheduler.command
```

2. Choose from the available options:
   - **Default Settings**: Runs tests every 24 hours in testing mode
   - **Custom Settings**: Configure interval, mode, configuration file, and data directory

3. The scheduler will run tests at the specified interval and log results to the `logs` directory

### Performance Summary Reports

The automated test scheduling system includes a powerful report generator that analyzes test logs to provide comprehensive performance insights:

- **Automated Analysis**: Generate detailed performance reports from test logs
- **Trend Detection**: Identify improvements or regressions in model performance
- **Category Analysis**: Break down performance by contradiction categories
- **Error Pattern Recognition**: Identify common error patterns and their frequency
- **Visualization**: ASCII-based visualization of performance trends
- **Actionable Recommendations**: Context-aware suggestions for improvement

Reports can be generated in three ways:
1. **Automatically**: The system periodically generates reports during scheduled test runs
2. **Via Desktop Launcher**: Use option #3 in the desktop launcher
3. **Via Command Line**: Run `./scripts/schedule_healthcare_tests.sh --report`

Example report sections:
```
## Performance Metrics

- Total accuracy measurements: 43
- Minimum accuracy: 0.63
- Maximum accuracy: 0.82
- First recorded accuracy: 0.67
- Latest recorded accuracy: 0.73

## Performance Trend

SIGNIFICANT IMPROVEMENT: Accuracy increased by 0.06

## Contradiction Detection Analysis

- medication_interaction: 11 occurrences
- treatment_protocol: 8 occurrences
- dosage_conflict: 8 occurrences
```

For detailed documentation on the report generator, see the `HEALTHCARE_SCHEDULER_GUIDE.md` on your Desktop.

### Command-Line Options

The scheduler can also be run directly from the command line:

```bash
./scripts/schedule_healthcare_tests.sh [options]
```

Available options:
- `-i, --interval HOURS`: Time between test runs (default: 24 hours)
- `-c, --config PATH`: Path to configuration file
- `-m, --mode MODE`: Dashboard mode (regular or testing)
- `-d, --data-dir PATH`: Path to data directory
- `-r, --report`: Generate a performance summary report from logs
- `-h, --help`: Show help message

For more detailed information, refer to the `HEALTHCARE_SCHEDULER_GUIDE.md` on your Desktop.

## Model Evaluation Framework

The system includes a comprehensive model evaluation framework for assessing AI model performance across different knowledge domains and reasoning capabilities:

### Framework Components

```
evaluators/
├── base_evaluator.py        # Abstract base class for all evaluators
├── domain_evaluators.py     # Domain knowledge evaluators
├── cross_reference_evaluator.py   # Cross-referencing capability evaluator
└── counterfactual_evaluator.py    # Counterfactual reasoning evaluator
comprehensive_evaluate.py    # Main evaluation script
run_evaluation.command       # Desktop launcher for evaluation
test_evaluation_framework.py # Test script for framework validation
```

### Key Features

- **Multi-Domain Evaluation**: Test model knowledge across multiple domains
- **Reasoning Assessment**: Evaluate cross-referencing and counterfactual reasoning
- **Apple Silicon Optimized**: Special optimizations for M-series chips
- **Memory Tracking**: Detailed memory usage monitoring
- **WandB Integration**: Experiment tracking and visualization
- **Interactive Setup**: Desktop launcher with configuration options

### Usage

1. Launch the evaluation framework:
```bash
# Run the desktop launcher
/Users/cpconnor/Desktop/run_evaluation.command
```

2. Or run directly with Python and custom options:
```python
python comprehensive_evaluate.py --model_path /path/to/adapter --device mps
```

3. Key components for integration:
```python
from comprehensive_evaluate import EvaluationManager

# Initialize the manager
manager = EvaluationManager(
    model_path="/path/to/adapter",
    device="mps",
    use_wandb=True
)

# Run evaluation
results = manager.run_evaluation()
```

For detailed documentation on the evaluation framework, see [EVALUATION_FRAMEWORK.md](docs/EVALUATION_FRAMEWORK.md).

## Cross-Reference Evaluation Framework

The repository includes a cross-reference evaluation framework specifically designed to test healthcare-related models:

1. Launch the evaluation framework:
```bash
python scripts/healthcare_cross_reference_pipeline.py --data data/healthcare --output output/healthcare
```

2. Examine evaluation results in the output directory.

3. Use the visualization tools to generate reports:
```bash
python scripts/visualize_metrics.py --input output/healthcare/healthcare_eval_latest_viz.json --output output/healthcare/visualizations
```

## Healthcare Metrics Visualization System

The Healthcare Metrics Visualization System provides specialized visualizations for healthcare cross-reference model performance. This system helps track, analyze, and visualize the performance of healthcare contradiction detection and evidence ranking over time.

Key features:
- **Comprehensive Healthcare Metrics Tracking**:
  - Contradiction detection accuracy
  - Evidence ranking performance
  - Performance gaps analysis
  - Historical metrics tracking
  
- **Specialized Healthcare Visualizations**:
  - Contradiction type performance analysis
  - Medical domain-specific metrics
  - Temporal contradiction pattern analysis
  - Time gap analysis for medical knowledge changes

- **Interactive HTML Reports**:
  - Combined visualization dashboards
  - Performance summaries
  - Trend analysis
  - Downloadable visualization assets

For detailed documentation, see [Healthcare Metrics Visualization Documentation](docs/HEALTHCARE_METRICS_VISUALIZATION.md).

To run the Healthcare Metrics Visualization:
1. Use the desktop launcher: `Healthcare_Metrics_Visualization.command`
2. Or directly via command line: `python scripts/visualize_metrics.py path/to/healthcare_eval_latest.json`

## Enhanced Performance Testing Framework

The Enhanced Performance Testing Framework provides comprehensive testing and monitoring for all aspects of the healthcare performance metrics system, including path-based relationship encoding and artificial intelligence metrics.

### Key Components

#### 1. Performance Testing Module
- Located at: `scripts/run_performance_tests.py`
- Tests AI performance across key dimensions:
  - **Reasoning**: Analytical capability in varied contexts (Baseline: 0.74, Target: 0.89)
  - **Knowledge Integration**: Cross-referencing and synthesis capability (Baseline: 0.78, Target: 0.91)
  - **Adaptability**: Context transformation flexibility (Baseline: 0.75, Target: 0.89)
- Generates performance trend visualizations
- Tracks progress toward performance targets

#### 2. Path Encoding Tests
- Located at: `tests/test_path_encoding.py`
- Tests bidirectional conversion between hierarchical and flat context representations
- Validates semantic relationship preservation
- Handles type markers and special cases
- Tests reasoning with path-encoded relationships

#### 3. Desktop Launchers

##### Healthcare Tests Launcher
- Located at: `/Users/cpconnor/Desktop/launch_healthcare_tests.command`
- Interactive menu for running different types of tests:
  - Standard healthcare tests
  - Metrics validation
  - Enhanced AI tests
  - Path encoding tests
  - Performance tests

##### Performance Tests Launcher
- Located at: `/Users/cpconnor/Desktop/launch_performance_tests.command`
- Streamlined interface focused on performance metrics:
  - Run all performance tests
  - Run path encoding tests
  - View performance trends
  - Browse test reports

#### 4. Automated Test Scheduler
- Enhanced scheduler with AI metrics tracking
- Extracts and reports on reasoning, knowledge integration, and adaptability metrics
- Generates detailed performance summaries
- Configurable test intervals

### Usage

#### Running Performance Tests

1. Use the desktop launcher:
```bash
/Users/cpconnor/Desktop/launch_performance_tests.command
```

2. Select the desired test option from the menu

3. View generated reports in the `reports` directory

#### Running Path Encoding Tests

1. Use either desktop launcher and select the path encoding option

2. Or run directly via command line:
```bash
python tests/test_path_encoding.py
```

#### Scheduling Automated Tests

1. Use the healthcare tests launcher and select the scheduling option

2. Configure test interval (hourly, daily, weekly, or custom)

3. Tests will run automatically at the specified intervals

For detailed information on the Enhanced Performance Testing Framework, see [TESTING.md](TESTING.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
