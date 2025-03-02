# Enhanced Healthcare Performance Metrics Testing Framework

This document outlines the comprehensive testing framework for the Enhanced Healthcare Performance Metrics System, focusing on various performance areas including traditional metrics, Customer Experience, Artificial Intelligence capabilities, and Path-Based Relationship Encoding.

## Overview

The testing framework provides tools for:
- Running comprehensive test suites across all metric categories
- Scheduling automated tests at regular intervals
- Generating detailed performance reports
- Tracking progress against established baselines and targets

## Key Components

### 1. Test Runner (`scripts/run_tests.py`)

The test runner executes tests across multiple performance areas:

- **Traditional Metrics**
  - Categories (e.g., diagnosis, treatment, etc.)
  - Domains (e.g., cardiology, oncology, etc.)
  - Standard metrics (accuracy, precision, recall, F1 score)

- **Customer Experience Metrics**
  - Overall Score (baseline: 0.81, target: 0.91)
  - Response Time (baseline: 0.85, target: 0.93)
  - Satisfaction (baseline: 0.79, target: 0.88)
  - Usability (baseline: 0.82, target: 0.89)

- **Artificial Intelligence Metrics**
  - Overall Score (baseline: 0.76, target: 0.88)
  - Reasoning (baseline: 0.74, target: 0.86)
  - Knowledge Integration (baseline: 0.78, target: 0.89)
  - Adaptability (baseline: 0.75, target: 0.86)

### 2. Path Encoding Tests (`tests/test_path_encoding.py`)

The path encoding test suite validates the functionality of path-based relationship encoding:

- **Bidirectional Conversion**
  - Hierarchy flattening and reconstruction
  - Type marker handling (`p:`, `m:`, `rel:`)
  - Preservation of semantic relationships

- **Path-Based Representation**
  - Conversion of nested structures to path representations
  - Handling of numeric values and precision
  - Special case processing for enhanced reliability

### 3. Performance Testing (`scripts/run_performance_tests.py`)

The performance testing framework monitors critical AI metrics:

- **Reasoning Capability**
  - Baseline: 0.85
  - Target: 0.89
  - Tests logical reasoning in context transformation

- **Knowledge Integration**
  - Baseline: 0.88
  - Target: 0.91
  - Tests integration of information across contexts

- **Adaptability**
  - Baseline: 0.86
  - Target: 0.89
  - Tests adaptation to different content requirements

- **Visualization**
  - Generates trend charts showing progress over time
  - Creates comprehensive performance reports

### 4. Test Scheduler (`scripts/schedule_healthcare_tests.sh`)

The test scheduler:
- Runs tests at configurable intervals (default: 24 hours)
- Maintains detailed logs of test runs
- Generates performance summaries and trend analyses
- Tracks and reports on progress toward performance targets

### 5. Desktop Launchers

#### Healthcare Tests Launcher (`launch_healthcare_tests.command`)

The healthcare tests launcher provides an interactive menu for:
- Running individual test categories
- Starting the automated test scheduler
- Generating summary reports
- Viewing the latest test results

#### Performance Tests Launcher (`launch_performance_tests.command`)

The performance tests launcher provides an interactive menu for:
- Running comprehensive performance tests
- Running path encoding tests
- Viewing performance trends
- Browsing test reports

## Usage Instructions

### Running Healthcare Tests

1. **Execute the healthcare tests launcher**:
   ```
   ./launch_healthcare_tests.command
   ```

2. **Select a test option from the menu**:
   - Option 1: Run all tests
   - Option 2: Run only Customer Experience tests
   - Option 3: Run only Artificial Intelligence tests

### Running Performance and Path Encoding Tests

1. **Execute the performance tests launcher**:
   ```
   ./launch_performance_tests.command
   ```

2. **Select a test option from the menu**:
   - Option 1: Run all performance tests
   - Option 2: Run path encoding tests
   - Option 3: View performance trends
   - Option 4: View test reports

### Scheduling Automated Tests

1. **Using the desktop launcher**:
   - Select Option 4 to start the scheduler in a new terminal window

2. **Using the scheduler directly**:
   ```
   ./scripts/schedule_healthcare_tests.sh
   ```

3. **Scheduler Options**:
   - `-i, --interval HOURS`: Time between test runs (default: 24 hours)
   - `-c, --config PATH`: Path to configuration file
   - `-m, --mode MODE`: Dashboard mode (regular or testing)
   - `-d, --data-dir PATH`: Path to data directory
   - `-r, --report`: Generate a summary report from logs
   - `--no-enhanced-tests`: Skip running enhanced tests
   - `-h, --help`: Show help message

### Generating Reports

1. **Using the desktop launcher**:
   - Select Option 5 to generate a summary report

2. **Using the scheduler directly**:
   ```
   ./scripts/schedule_healthcare_tests.sh --report
   ```

## Interpreting Test Results

Test results include:

1. **Overall Metrics**: Aggregate performance across all categories and domains

2. **Category Results**: Performance metrics for each healthcare category

3. **Domain Results**: Performance metrics for each healthcare domain

4. **Customer Experience Results**: Detailed metrics on user experience factors with:
   - Current values
   - Baseline references
   - Target goals
   - Progress indicators

5. **Artificial Intelligence Results**: Detailed metrics on AI capabilities with:
   - Current values
   - Baseline references
   - Target goals
   - Progress indicators

6. **Path Encoding Results**: Verification of bidirectional conversion integrity:
   - Success/failure of each test case
   - Confidence scores for conversions
   - Special case handling performance

7. **Performance Test Results**: Comprehensive analysis of AI reasoning capabilities:
   - Trend visualization showing progress over time
   - Delta between current and target metrics
   - Historical performance logging

## Output Files

### Healthcare Tests
Test results are saved in JSON format in the `reports/tests` directory with filenames containing:
- Test type/preset
- Timestamp
- Example: `test_results_comprehensive_20231015_120530.json`

### Performance Tests
Performance test results are saved in the `reports` directory in several formats:
- JSON result files: `performance_results_YYYYMMDD_HHMMSS.json`
- Trend charts: `performance_trend.png`
- Log files: `performance_log_YYYYMMDD_HHMMSS.txt`

## Best Practices

1. **Regular Testing**: Schedule tests to run at least daily to track performance trends

2. **Report Review**: Regularly review generated reports to identify:
   - Areas of improvement
   - Regression issues
   - Stagnant metrics

3. **Target Adjustment**: Periodically review and adjust targets based on:
   - Current performance
   - Implementation capabilities
   - Business requirements

4. **Comprehensive Testing**: Run the full test suite before and after significant changes

5. **Path Encoding Verification**: Run path encoding tests after any changes to:
   - Context handling
   - Type management
   - Relationship modeling

6. **Performance Monitoring**: Track performance metrics over time to ensure progress toward targets

## Troubleshooting

If you encounter issues with the testing framework:

1. **Check Log Files**: Examine logs in the `logs` directory
2. **Verify Script Permissions**: Ensure all scripts have execution permissions
3. **Check Python Environment**: Verify all required dependencies are installed
4. **Directory Structure**: Confirm the expected directory structure is intact
5. **Virtual Environment**: If tests fail due to missing dependencies:
   ```
   cd /Users/cpconnor/CascadeProjects/multi-platform-content-generator
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy matplotlib networkx seaborn pandas rich
   ```
6. **Numeric Precision Issues**: If path encoding tests fail due to precision differences, adjust the allowed tolerance in `test_path_encoding.py`
