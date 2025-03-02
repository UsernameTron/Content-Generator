# Healthcare Contradiction Detection Dashboard

## Overview

The Healthcare Contradiction Detection Dashboard is a comprehensive tool designed to facilitate continuous learning and performance improvement for healthcare contradiction detection models. The dashboard provides an interactive interface for training, testing, and analyzing model performance across different healthcare domains and contradiction categories.

## Features

### Core Features

- **Interactive Learning Interface**: Train and improve the model through interactive learning cycles
- **Performance Tracking**: Monitor model performance across learning cycles
- **Domain-specific Analysis**: Track performance across different healthcare domains
- **Category-specific Analysis**: Analyze performance for different contradiction categories
- **Visualization Tools**: Visual representation of performance metrics and improvements

### Advanced Testing Features

- **Scenario Testing**: Test the model against predefined healthcare scenarios
- **Edge Case Detection**: Identify and test edge cases that may challenge the model
- **Stress Testing**: Evaluate model performance under high load conditions
- **Regression Testing**: Ensure new improvements don't negatively impact existing capabilities
- **Custom Test Cases**: Create and run custom test cases for specific requirements

### Configuration Preset System

- **Save and Load Configurations**: Save current dashboard configurations as presets and load them later
- **Preset Management**: Create, update, delete, import, and export configuration presets
- **Quick Configuration Switching**: Rapidly switch between different testing and analysis configurations
- **Configuration Comparison**: Compare different configurations to identify optimal settings
- **Preset Categories**: Organize presets by tags for easy retrieval
- **Batch Processing**: Apply the same configuration across multiple testing sessions

### Performance Analysis Features

- **Comprehensive Metrics**: Track accuracy, precision, recall, and F1 score
- **Improvement Tracking**: Monitor performance improvements across learning cycles
- **Category Comparison**: Compare performance across different contradiction categories
- **Domain Comparison**: Analyze performance differences between healthcare domains
- **Automated Recommendations**: Receive suggestions for improvement areas

## Dashboard Modes

The dashboard supports two operational modes:

### Regular Mode

The standard dashboard interface with core functionality:
- Interactive learning cycles
- Performance tracking
- Basic visualization
- Configuration management

### Testing Mode

Enhanced dashboard with advanced testing capabilities:
- All regular mode features
- Advanced testing module
- Detailed performance analysis
- Comprehensive reporting
- Regression detection

## Getting Started

### System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- macOS, Linux, or Windows operating system

### Installation

1. Clone the repository or download the source code
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure the dashboard using `dashboard_config.json`

### Launching the Dashboard

#### Using the Desktop Launcher

1. Double-click the `launch_healthcare_dashboard.command` file on your desktop
2. Select the desired mode (Regular or Testing)
3. The dashboard will launch in the selected mode

#### Using the Command Line

1. Navigate to the project directory
2. Run the dashboard with the desired mode:
   ```
   python scripts/interactive_learning_dashboard.py --mode regular
   ```
   or
   ```
   python scripts/interactive_learning_dashboard.py --mode testing
   ```

#### Using Environment Variables

You can also set the dashboard mode using an environment variable:
```
export DASHBOARD_MODE=testing
python scripts/interactive_learning_dashboard.py
```

## Configuration

The dashboard is highly configurable through the `dashboard_config.json` file. Key configuration options include:

### General Configuration

```json
{
  "app_name": "Healthcare Contradiction Detection",
  "version": "2.0.0",
  "log_level": "info",
  "data_directory": "data/healthcare",
  "reports_directory": "reports",
  "enable_advanced_testing": true,
  "enable_performance_tracking": true
}
```

### Testing Configuration

```json
{
  "testing": {
    "scenario_directory": "test_cases/scenarios",
    "edge_case_directory": "test_cases/edge_cases",
    "regression_directory": "test_cases/regression",
    "performance_directory": "test_cases/performance",
    "custom_directory": "test_cases/custom",
    "default_test_size": 100,
    "stress_test_size": 1000,
    "accuracy_threshold": 0.8,
    "precision_threshold": 0.75,
    "recall_threshold": 0.75,
    "f1_threshold": 0.75
  }
}
```

### Performance Configuration

```json
{
  "performance": {
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "track_categories": true,
    "track_domains": true,
    "improvement_threshold": 0.05,
    "regression_threshold": -0.02,
    "visualization_enabled": true,
    "report_format": "html"
  }
}
```

## Usage Guide

### Regular Mode

1. **Start a Learning Cycle**
   - Click "Start New Cycle" to begin a new learning cycle
   - The dashboard will load the current model state

2. **Generate Examples**
   - Click "Generate Examples" to create new training examples
   - Review and validate the generated examples

3. **Train the Model**
   - Click "Train Model" to train the model on the validated examples
   - Monitor the training progress

4. **Evaluate Performance**
   - Click "Evaluate" to assess model performance
   - Review performance metrics and visualizations

5. **Save Progress**
   - Click "Save Progress" to save the current model state
   - The dashboard will create a checkpoint for future reference

### Testing Mode

1. **Select Test Type**
   - Choose from Scenario, Edge Case, Stress, Regression, or Custom testing

2. **Configure Test Parameters**
   - Set test size, thresholds, and other parameters
   - Select specific categories or domains to test

3. **Run Tests**
   - Click "Run Tests" to execute the selected tests
   - Monitor test progress and results

4. **Analyze Results**
   - Review test results and performance metrics
   - Identify areas for improvement
   - Export test results for further analysis

### Configuration Preset System

1. **Accessing the Preset Manager**
   - From the main menu, select "Manage Configuration Presets"
   - The preset manager interface will be displayed

2. **Creating a New Preset**
   - Select "Create new preset from current configuration"
   - Enter a name, description, and optional tags for the preset
   - The current configuration will be saved as a new preset

3. **Applying a Preset**
   - Select "Apply preset to current configuration"
   - Choose a preset from the list of available presets
   - Confirm to apply the selected preset to the current configuration

4. **Managing Presets**
   - List available presets to view all saved configurations
   - Delete presets that are no longer needed
   - Export presets to share with other users
   - Import presets from external sources

5. **Using Preset Tags**
   - Organize presets with descriptive tags
   - Filter presets by tags to quickly find specific configurations
   - Use consistent tagging conventions for better organization

6. **Best Practices**
   - Create specialized presets for different testing scenarios
   - Use descriptive names and detailed descriptions
   - Regularly review and update presets to reflect current best practices
   - Export important presets as backups

### Batch Testing System

The Healthcare Contradiction Detection Dashboard includes a powerful batch testing system that allows you to run automated tests across multiple configuration presets. This system helps you compare performance metrics and identify optimal configurations for different use cases.

#### Accessing Batch Testing

You can access the batch testing functionality in two ways:

1. **From the Dashboard Launcher**:
   - Launch the dashboard launcher
   - Select "Run Batch Tests" from the main menu
   - Choose which presets to use for testing

2. **From the Command Line**:
   ```bash
   python scripts/run_batch_tests.py [options]
   ```

#### Command-Line Options

The batch testing script supports several command-line options:

- `--all`: Run tests with all available presets
- `--presets PRESETS`: Specify a comma-separated list of preset names to use
- `--output-dir DIR`: Specify a custom output directory for test results
- `--verbose`: Enable verbose output for detailed logging
- `--help`: Display help information

#### Test Results

After running batch tests, the system generates:

1. **Summary Report**: A concise overview of test results for all presets
2. **Detailed Reports**: Individual reports for each preset with comprehensive metrics
3. **Comparison Data**: Side-by-side comparison of key performance indicators

Results are stored in the `/reports/batch_tests/` directory by default, organized by timestamp.

#### Example Workflow

A typical batch testing workflow might look like this:

1. Create or modify configuration presets for specific testing scenarios
2. Run batch tests across these presets
3. Review the results to identify performance patterns
4. Refine presets based on test results
5. Repeat the process to optimize configurations

#### Integration with Performance Analyzer

The batch testing system integrates with the Enhanced Performance Analyzer to provide:

- Automated identification of improvement areas
- Priority-based recommendations
- Visual reporting of performance trends
- Regression detection across configurations

#### Best Practices

- Run batch tests after making significant changes to the system
- Use a diverse set of presets to ensure comprehensive testing
- Create specialized presets for specific testing scenarios
- Review results carefully to identify performance trends
- Automate batch testing as part of your development workflow

### Troubleshooting

### Common Issues

1. **Dashboard fails to start**
   - Ensure Python 3.9+ is installed
   - Verify all dependencies are installed
   - Check file permissions

2. **Configuration errors**
   - Validate JSON syntax in configuration files
   - Ensure all required directories exist
   - Check for correct file paths

3. **Performance issues**
   - Close resource-intensive applications
   - Increase memory allocation if possible
   - Reduce test batch sizes

4. **Visualization errors**
   - Ensure matplotlib is properly installed
   - Check for sufficient disk space for reports
   - Verify write permissions in reports directory

### Getting Help

For additional assistance:
- Check the logs in the `logs` directory
- Review the documentation in the `docs` directory
- Submit issues to the project repository

## Performance Analysis

The dashboard provides comprehensive performance analysis capabilities:

### Metrics Tracked

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall

### Analysis Views

- **Overall Performance**: Track overall model performance across learning cycles
- **Category Performance**: Analyze performance for different contradiction categories
- **Domain Performance**: Compare performance across healthcare domains
- **Improvement Analysis**: Measure performance improvements over time

### Reports

The dashboard generates detailed performance reports with:

- Performance metrics summary
- Visualizations of performance trends
- Identification of improvement areas
- Automated recommendations for enhancement
- Regression detection and alerts

## Advanced Testing

### Scenario Testing

Test the model against predefined healthcare scenarios:
- Medication contradictions
- Treatment plan conflicts
- Diagnostic inconsistencies
- Patient history discrepancies

### Edge Case Testing

Evaluate model performance on challenging edge cases:
- Rare medical conditions
- Complex medication interactions
- Ambiguous contradictions
- Boundary conditions

### Stress Testing

Assess model robustness under high load:
- Large batch processing
- Rapid request handling
- Extended operation periods
- Resource-constrained environments

### Regression Testing

Ensure new improvements don't break existing functionality:
- Historical test cases
- Previously fixed issues
- Core functionality verification
- Cross-domain validation

### Custom Testing

Create and run custom test cases for specific requirements:
- Domain-specific scenarios
- Category-focused tests
- Client-specific requirements
- Special use cases

## Best Practices

1. **Regular Learning Cycles**
   - Run learning cycles consistently to improve model performance
   - Validate examples carefully before training

2. **Comprehensive Testing**
   - Use a mix of test types for thorough evaluation
   - Create custom tests for specific use cases

3. **Performance Monitoring**
   - Regularly review performance reports
   - Address regression issues promptly

4. **Configuration Management**
   - Customize configuration for your specific needs
   - Back up configuration files before making changes

5. **Data Management**
   - Organize test cases in appropriate directories
   - Maintain a diverse set of examples across domains and categories

## Contributing

Contributions to the dashboard are welcome:
- Submit bug reports and feature requests
- Propose enhancements to existing functionality
- Contribute new test cases and scenarios
- Improve documentation and examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.
