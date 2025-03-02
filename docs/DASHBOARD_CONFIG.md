# Healthcare Learning Dashboard Configuration

This document explains the configuration options for the Healthcare Learning Dashboard.

## Configuration File

The dashboard reads its configuration from `dashboard_config.json` in the project root directory. This file controls feature enablement and default settings.

## Configuration Options

### Features

The dashboard supports the following features that can be enabled or disabled:

```json
"features": {
  "batch_processing": true,
  "dataset_import": true,
  "performance_comparison": true,
  "advanced_testing": true
}
```

- **batch_processing**: When enabled, allows scheduling multiple learning cycles to run sequentially without manual intervention. This feature adds a new menu option to the dashboard for configuring and running batch processes.

- **dataset_import**: When enabled, provides a simplified interface for adding new contradiction examples through the dashboard rather than requiring direct file manipulation. This feature adds a dataset import option to the dashboard menu.

- **performance_comparison**: When enabled, implements detailed model performance comparison across learning cycles, showing clear metrics for contradiction detection improvement. This enhances the visualization and history viewing capabilities.

- **advanced_testing**: When enabled, adds comprehensive testing capabilities including:
  - Scenario-based testing with predefined healthcare contradiction cases
  - Edge case detection and handling
  - Stress testing with large batches of examples
  - Domain-specific validation for healthcare terminology
  - Automated regression testing across model versions
  - Custom test case creation and management
  - Performance benchmarking against baseline models
  - Test report generation with detailed metrics

### Default Settings

The dashboard uses these default settings for various operations:

```json
"default_settings": {
  "cycles": 5,
  "batch_size": 10,
  "evaluation_frequency": 2
}
```

- **cycles**: Default number of learning cycles to run in batch mode
- **batch_size**: Default number of examples to generate per cycle in batch mode
- **evaluation_frequency**: How often (in cycles) to perform a full evaluation during batch processing

## Enabling/Disabling Features

To enable or disable features:

1. Open `dashboard_config.json` in a text editor
2. Change the value of the feature from `true` to `false` to disable it
3. Save the file
4. Restart the dashboard if it's currently running

## Adding Custom Configuration

You can add custom configuration options by extending the JSON structure. The dashboard will ignore options it doesn't recognize, making the configuration file extensible for future enhancements.

## Example: Disabling Dataset Import

If you want to disable the dataset import feature while keeping other features enabled:

```json
{
  "dashboard": {
    "features": {
      "batch_processing": true,
      "dataset_import": false,
      "performance_comparison": true,
      "advanced_testing": true
    },
    "default_settings": {
      "cycles": 5,
      "batch_size": 10,
      "evaluation_frequency": 2
    }
  }
}
```

## Example: Changing Default Batch Settings

If you want to change the default number of cycles and batch size:

```json
{
  "dashboard": {
    "features": {
      "batch_processing": true,
      "dataset_import": true,
      "performance_comparison": true,
      "advanced_testing": true
    },
    "default_settings": {
      "cycles": 10,
      "batch_size": 5,
      "evaluation_frequency": 2
    }
  }
}
```
