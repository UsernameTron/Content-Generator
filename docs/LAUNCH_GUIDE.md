# Healthcare Metrics System Quick-Start Guide

## Desktop Launchers

This system provides two convenient desktop launchers to help you quickly generate healthcare metrics visualizations:

### 1. Healthcare Visualizations Only

**File:** `Healthcare_Visualizations_Only.command` (on your Desktop)

**Purpose:** Generate visualizations from existing healthcare evaluation data without running the full pipeline.

**Best for:**
- Quick visualization of existing results
- Creating new reports from previously generated data
- Exploring different visualization options

**Usage:**
1. Double-click the file on your Desktop
2. The script will automatically:
   - Set up the environment
   - Generate all visualizations
   - Create an HTML report
3. Follow the interactive prompts to:
   - View the HTML report
   - Open the visualizations folder

### 2. Full Healthcare Metrics Pipeline

**File:** `Healthcare_Metrics_Full_Pipeline.command` (on your Desktop)

**Purpose:** Run the complete healthcare cross-reference pipeline and generate all visualizations.

**Best for:**
- Processing new healthcare data
- Generating fresh evaluation metrics
- Creating a complete set of visualizations and reports

**Usage:**
1. Double-click the file on your Desktop
2. The script will automatically:
   - Set up the environment
   - Run the healthcare cross-reference pipeline
   - Generate evaluation metrics
   - Create all visualizations
   - Build an HTML report
3. Follow the interactive prompts to:
   - View the HTML report
   - Open the visualizations folder

## Output Files

After running either launcher, you'll find the following outputs:

- **HTML Report:** `output/healthcare/visualizations/metrics_report.html`
- **Visualization Images:** `output/healthcare/visualizations/*.png`
- **Evaluation Data:** `output/healthcare/pipeline_output/healthcare_eval_latest.json`

## Troubleshooting

If you encounter any issues:

1. Make sure the scripts have execute permissions
   ```
   chmod +x ~/Desktop/Healthcare_Visualizations_Only.command
   chmod +x ~/Desktop/Healthcare_Metrics_Full_Pipeline.command
   ```

2. Ensure all required directories exist
   ```
   mkdir -p ~/CascadeProjects/multi-platform-content-generator/output/healthcare/visualizations
   mkdir -p ~/CascadeProjects/multi-platform-content-generator/data/healthcare
   ```

3. Check that all dependencies are installed
   ```
   cd ~/CascadeProjects/multi-platform-content-generator
   pip install -r requirements.txt
   ```

## For More Information

See the complete documentation in:
- `docs/HEALTHCARE_METRICS_SYSTEM_GUIDE.md` - Comprehensive system documentation
- `docs/HEALTHCARE_METRICS_VISUALIZATION.md` - Visualization methods details
