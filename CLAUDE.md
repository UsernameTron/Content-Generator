# Multi-Platform Content Generator

## Useful Commands

### Testing and Performance
- Run performance tests: `python scripts/run_performance_tests.py --memory-tracking --parallel`
- Run healthcare tests: `python scripts/run_healthcare_learning.py`
- Run enhanced tests: `python enhancement_module/run_enhanced_tests.py`
- Memory tracking: `python monitor_resources.py`

### Evaluation
- Quick evaluation: `python comprehensive_evaluate.py --quick`
- Full evaluation: `python comprehensive_evaluate.py`
- Healthcare evaluation: `python scripts/healthcare_evaluation.py`
- Dashboard metrics: `python scripts/metrics_dashboard.py`

### System Management
- Check dashboard configuration: `python scripts/dashboard_config_reader.py`
- Run the application: `python app.py`
- Run UI dashboard: `sh launch_dashboard.sh`
- Run healthcare dashboard: `sh launch_healthcare_dashboard.command`

## Project Structure
- `src/` - Core application code
- `scripts/` - Utility scripts for testing and evaluation
- `enhancement_module/` - Performance enhancement modules
- `data/` - Training and evaluation data
- `output/` - Generated outputs and reports
- `config/` - Configuration files
- `reports/` - Test results and reporting data

## Environment
- Dashboard config location: `/config/dashboard_config.json`
- Default test batch size: 64
- Memory threshold (warning): 70%
- Memory threshold (critical): 85%

## Performance Notes
- Content generation is now cached to improve response time
- Path relationship handling is enhanced to support multiple formats
- Memory monitoring automatically saves metrics to CSV files
- For best parallel performance, use threading rather than multiprocessing