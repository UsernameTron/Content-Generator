# Launching CANDOR: Satirical Cross-Platform Content Generator

There are multiple ways to launch and use the application:

## 1. PyQt UI Application (New!)

The easiest way to use CANDOR is with the modern PyQt interface:

1. Double-click the `CANDOR_Launcher.command` file on your Desktop
2. The launcher will:
   - Set up a virtual environment if needed
   - Install required dependencies
   - Launch the PyQt6-based UI application
3. Use the intuitive interface to:
   - Input content via text, document, or URL
   - Select target platforms (YouTube, Medium, LinkedIn, Substack)
   - Generate platform-specific satirical content
   - Copy results directly to your clipboard

## 2. Quick Installation & Launch (Original Application)

For first-time setup of the original Streamlit application, run the installation script:

1. Double-click the `install.command` file in the project directory
2. The script will:
   - Create a virtual environment
   - Install all dependencies
   - Set up data files and configurations
   - Create a desktop shortcut for easy access
   - Offer to launch the application immediately

## 3. Using the Desktop Shortcut (Original Application)

After installation, you can use the desktop shortcut:

1. Double-click the `PeteConnorContentGenerator.command` file on your Desktop
2. This will launch the original application with the model-based content generator

## 4. Using the Model-Based App Launcher

You can also launch directly from the project directory:

1. Double-click the `launch_model_app.command` file in the project directory
2. The script will:
   - Activate the virtual environment
   - Check for the fine-tuned model
   - Launch the Streamlit web application with model-based generation

## 4. Manual Launch

If you prefer to launch the application manually:

1. Open a terminal
2. Navigate to the project directory:
   ```
   cd ~/CascadeProjects/multi-platform-content-generator
   ```
3. Create and activate a virtual environment (first time only):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies (first time only):
   ```
   pip install -r requirements.txt
   ```
5. Set up data files:
   ```
   python setup_data.py
   ```
6. Launch the model-based application:
   ```
   cd src
   streamlit run app_model.py
   ```

## 5. Fine-Tuning the Model

To fine-tune the model on C. Pete Connor's writing style:

1. Double-click the `run_finetune.command` file in the project directory
2. The script will:
   - Prepare training data from `writing_style.json`
   - Configure Weights & Bibes monitoring (if API key is available)
   - Run the fine-tuning process (may take several hours)
   - Save the fine-tuned model to the `outputs/finetune/final` directory

**Note**: Fine-tuning is resource-intensive and may take several hours to complete. You can continue to use the application with the template-based fallback while fine-tuning is in progress.

## 6. Training/Fine-Tuning the Model (Advanced)

For users who want to fine-tune the C. Pete Connor model:

1. Double-click the `Run_Pete_Connor_Overnight_Training.command` file on your Desktop
2. The script will:
   - Activate the virtual environment
   - Set up optimized environment variables for Apple Silicon
   - Start the training process with proper logging
   - Save checkpoints to the `outputs/finetune` directory

You can monitor the training progress by:
1. Double-clicking the `Monitor_Pete_Connor_Training.command` file on your Desktop
2. This will open a terminal window showing real-time training logs
3. You can also visit the Weights & Biases dashboard (if configured) for detailed metrics

## 7. Cross-Referencing Functionality

The content generator now includes powerful cross-referencing capabilities that allow it to leverage past content for improved generation:

### Setup Cross-Referencing Index

Before using cross-referencing, you need to index your existing content:

1. Double-click the `Pete_Connor_Cross_Reference_Indexing.command` file on your Desktop
2. The tool will:
   - Create a vector database of your existing content
   - Generate embeddings for efficient semantic search
   - Set up the necessary files for cross-referencing

### Using Cross-Referencing in Generation

Cross-referencing is automatically integrated into the content generation process:

1. Launch the application using any of the methods above
2. When generating content, the system will automatically:
   - Search for relevant past content
   - Incorporate reference examples into the generation process
   - Maintain consistency with your previous work

### Benefits of Cross-Referencing

- Improved content consistency across platforms
- More contextually relevant examples
- Better continuity in long-term content strategies
- Enhanced knowledge retention between generations

### Additional Documentation

For more detailed information about the cross-referencing system, see the [Cross-Referencing Documentation](docs/CROSS_REFERENCING.md).

## 8. Testing the Fine-Tuned Model

Once training is complete, you can test the model using several tools:

### Basic Testing
1. Double-click the `Test_Pete_Connor_Model.command` file on your Desktop
2. This will run a simple test script that generates responses for a few sample prompts
3. The results will be displayed in the terminal

### Interactive Testing
1. Double-click the `Pete_Connor_Interactive.command` file on your Desktop
2. This will start an interactive session where you can:
   - Type custom prompts
   - Get real-time responses from the model
   - Continue the conversation as long as needed
3. Type 'exit', 'quit', or 'q' to end the session

### Comprehensive Evaluation
1. Double-click the `Evaluate_Pete_Connor_Model.command` file on your Desktop
2. This will run a comprehensive evaluation across multiple categories:
   - Marketing content
   - Copywriting
   - Customer service responses
3. Results will be saved to the `evaluation_results` directory with metrics on:
   - Response quality
   - Generation time
   - Tokens per second
   - Output length

## 9. Model Fine-Tuning Documentation

For detailed information about the fine-tuning process, results, and optimization strategies:

1. Review the `FINE_TUNING_REPORT.md` file in the project directory
2. This report includes:
   - Executive summary of the fine-tuning process
   - Training configuration details
   - Apple Silicon optimization techniques
   - Evaluation results and analysis
   - Recommendations for further improvements

## 10. Using the Original Template-Based App (Legacy)

If you prefer to use the original template-based app with writing style selection:

1. Open a terminal
2. Navigate to the project directory:
   ```
   cd ~/CascadeProjects/multi-platform-content-generator
   ```
3. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```
4. Launch the original application:
   ```
   cd src
   streamlit run app.py
   ```

## 11. Dataset Enhancement Tool

To create or enhance the training dataset for the C. Pete Connor model:

1. Double-click the `Enhance_Pete_Connor_Dataset.command` file on your Desktop
2. The script offers two main options:
   - Create a new enhanced dataset with sample examples
   - Enhance an existing dataset and add sample examples
3. The enhanced dataset will be saved to the `dataset` directory in both JSON and JSONL formats
4. This improves the quality of your training data for better fine-tuning results

## 12. Training Status Check

To check the status of an ongoing or completed training process:

1. Double-click the `Pete_Connor_Training_Status.command` file on your Desktop
2. This will show comprehensive information about:
   - Current training progress and step count
   - Loss metrics and learning rates
   - Available checkpoints and their creation times
   - Estimated completion time for ongoing training
   - System resource usage
   - Weights & Biases integration status
   - Any errors or warnings that have occurred
3. This tool is especially useful for monitoring overnight training sessions

## 13. Model Deployment

After training is complete, you can deploy the model for easier integration:

1. Double-click the `Deploy_Pete_Connor_Model.command` file on your Desktop
2. This tool offers three deployment options:
   - Deploy using the latest checkpoint
   - Deploy using a specific checkpoint
   - Deploy and merge adapters (for faster inference)
3. The deployment process will:
   - Create a deployment package with the model and tokenizer
   - Generate example inference code for easy integration
   - Provide a README with usage instructions
   - Create model metadata for reference
4. The deployed model will be saved to the specified directory (default: `deployed_model`)
5. You can then use this deployed model in your applications following the included examples

## 14. Anti-Pattern Training

To eliminate AI-typical phrasing from the model's output, a specialized anti-pattern training module is available:

1. Double-click the `Run_Pete_Connor_Anti_Pattern_Training.command` file on your Desktop
2. The tool will:
   - Check for existing datasets or create new ones if needed
   - Set up a specialized training configuration focused on eliminating AI-typical phrases
   - Apply negative reinforcement to penalize common AI patterns such as:
     * Phrases like "game changer," "here's the kicker"
     * Formulaic transitions: "firstly," "secondly," "in conclusion" 
     * Symmetric sentence structures and repetitive patterns
   - Log all training progress to Weights & Biases

3. The anti-pattern training can be run:
   - After the main training has completed
   - In parallel with the main training (as a separate process)
   - Using intermediate checkpoints from the main training

4. After completion, the anti-pattern trained model will be saved in `outputs/anti_pattern` and can be used for content generation with more natural, authentic phrasing.

5. Performance metrics specific to anti-pattern training include:
   - Base loss (standard language modeling loss)
   - Pattern penalty (penalty applied for AI-typical phrasing)
   - Combined loss (total training signal)

## 15. Continuous Learning Mechanism

The Continuous Learning Mechanism allows the C. Pete Connor model to improve based on user feedback:

1. **Submitting Feedback**:
   - Double-click the `Pete_Connor_Feedback.command` file on your Desktop
   - The tool provides options to:
     * Submit new feedback on generated content
     * Rate outputs on a 1-5 scale
     * Add tags for specific issues (e.g., hallucination, voice mismatch)
     * Include inline annotations for specific text segments
     * Provide detailed comments about the content

2. **Feedback Categories**:
   - **Rating**: 1-5 stars overall assessment
   - **Tags**: Predefined categories for common issues
     * `hallucination`: Factually incorrect information
     * `voice_mismatch`: Not matching C. Pete Connor's voice
     * `content_quality`: General quality issues
     * `coherence`: Flow and logical structure problems
     * `platform_mismatch`: Not appropriate for the target platform
     * `audience_mismatch`: Not appropriate for the target audience
     * `verbosity`: Too wordy or too brief
     * `style_issue`: Other stylistic problems
   - **Annotations**: Specific comments on text segments
   - **Metadata**: Platform, audience, and domain information

3. **Training on Feedback**:
   - Double-click the `Run_Feedback_Training.command` file on your Desktop
   - The tool will:
     * Check if enough feedback has been collected (minimum 20 entries)
     * Prepare a training dataset with weighted examples based on feedback severity
     * Apply higher weights to negative examples for faster improvement
     * Convert annotations into corrected examples for fine-tuning
     * Train a specialized model variant focused on addressing common issues
     * Apply the same Apple Silicon optimizations used in other training modules

4. **Benefits of Continuous Learning**:
   - Gradual improvement based on actual usage
   - Targeted refinement of problematic patterns
   - Focus on specific platforms or audiences that need improvement
   - Personalization of the model to your specific content needs
   - Generation of training datasets from real-world usage

5. **Implementation Details**:
   - Feedback is stored in an SQLite database (`data/feedback.db`)
   - Training datasets are automatically generated in the `dataset` directory
   - Trained models are saved in the `models/feedback_training_[timestamp]` directory
   - Comprehensive logging tracks all feedback and training processes

6. **Usage Example**:
   ```python
   # After collecting feedback and training
   from src.models.model_content_generator import ModelContentGenerator
   
   # Load the feedback-trained model
   generator = ModelContentGenerator(model_path="models/feedback_training_20230601_120000")
   
   # Generate content with improved model
   improved_content = generator.generate_content(
       content="Key benefits of renewable energy", 
       platform="Blog"
   )
   ```

## 16. Audience Adaptation

The model can be trained to adapt content for different audience types, allowing for tailored messaging:

1. Double-click the `Run_Pete_Connor_Audience_Adaptation.command` file on your Desktop
2. The tool will:
   - Check for existing audience examples dataset or create a sample if needed
   - Configure training for audience adaptation using special audience tokens
   - Train the model to adapt content for three distinct audience types:
     * **Executive**: Concise, strategic, business-oriented content with metrics
     * **Practitioner**: Technical, implementation-focused content with specific details
     * **General Public**: Accessible, simplified content with analogies and examples
   - Log audience-specific metrics to Weights & Biases

3. The audience adaptation training:
   - Uses a lower learning rate (1e-5) to prevent catastrophic forgetting
   - Optionally applies LoRA adapters for efficient training
   - Monitors audience-appropriate complexity scores and jargon density

4. After completion, the audience-adapted model will be saved in `outputs/audience_adaptation` and can be used as follows:

```python
from src.models.model_content_generator import ModelContentGenerator

# Load the audience-adapted model
generator = ModelContentGenerator(model_dir='outputs/audience_adaptation')

# Generate content for different audience types
exec_content = generator.generate_content(
    content="AI implementation increases operational efficiency", 
    platform="LinkedIn", 
    audience="executive"
)

practitioner_content = generator.generate_content(
    content="AI implementation increases operational efficiency", 
    platform="LinkedIn", 
    audience="practitioner"
)

general_content = generator.generate_content(
    content="AI implementation increases operational efficiency", 
    platform="LinkedIn", 
    audience="general"
)
```

5. Performance metrics specific to audience adaptation include:
   - Audience-appropriate complexity scores
   - Jargon density across audience levels
   - Complexity and jargon ratios between audience types

## Model Evaluation Framework

The project includes a comprehensive model evaluation framework for assessing AI model performance across different domains and reasoning capabilities:

## Healthcare Performance Enhancement System

The Healthcare Performance Enhancement System provides targeted interventions to improve the performance of healthcare metrics detection, with a focus on contradiction detection, counterfactual reasoning, and cross-referencing capabilities.

### Running the Enhancement System

1. **Desktop Launcher (Recommended)**:
   - Double-click the `healthcare_enhancement_system.command` file on your Desktop
   - The launcher will:
     * Set up the necessary environment
     * Install required dependencies
     * Run all enhancement scripts
     * Generate comprehensive reports and visualizations

2. **Manual Launch**:
   - Open a terminal
   - Navigate to the project directory:
     ```
     cd ~/CascadeProjects/multi-platform-content-generator
     ```
   - Activate the virtual environment:
     ```
     source venv/bin/activate
     ```
   - Run the enhancement system:
     ```
     python run_enhancements.py
     ```

### Viewing Enhancement Results

1. **Enhancement Summary**:
   - Open `reports/enhancement_summary.md` for a comprehensive overview of all enhancements
   - This document provides details on current performance, projected improvements, and implementation strategies

2. **Improvement Plan**:
   - Open `reports/improvement_plan.md` for the detailed implementation roadmap
   - This document includes timeline, resource requirements, and expected outcomes

3. **Visualizations**:
   - View the generated visualizations in the `reports/enhancements/` directory
   - These visualizations show projected improvements for each enhancement area

### Using the Enhancement System

The Healthcare Performance Enhancement System is designed to be hardware-optimized for Apple Silicon systems and includes:

- **Automated Monitoring**: Real-time monitoring of system resources and performance metrics
- **Intelligent Batch Processing**: Optimized for the available memory and processing capabilities
- **MPS/Metal Acceleration**: Leverages Apple's Metal Performance Shaders for accelerated processing
- **Comprehensive Reporting**: Detailed analysis of current performance and projected improvements

### Using the Evaluation Framework

1. **Quick Launch**:
   - Double-click the `run_evaluation.command` file on your Desktop
   - This launches an interactive setup where you can configure:
     * Model adapter path
     * Evaluation domains to include/exclude
     * Device selection (CPU/MPS)
     * Memory optimization settings
     * W&B integration options

2. **Evaluation Domains**:
   - Customer Experience
   - Artificial Intelligence
   - Machine Learning
   - Cross-Referencing
   - Counterfactual Reasoning

3. **Advanced Options**:
   - Run `python comprehensive_evaluate.py --help` for all command-line options
   - Configure batch size, memory tracking, and more detailed settings

4. **Viewing Results**:
   - Results are displayed in the terminal with rich formatting
   - If W&B is enabled, detailed metrics and charts will be available in your W&B dashboard
   - Memory usage statistics are captured throughout the evaluation process

### Apple Silicon Optimizations

The framework includes special optimizations for Apple Silicon:
   - Sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to prevent memory limits
   - Enables `PYTORCH_ENABLE_MPS_FALLBACK=1` for compatibility
   - Sets `TOKENIZERS_PARALLELISM=false` for stability
   - Provides dynamic device selection based on availability

## Weights & Biases Integration

To use the Weights & Biases integration for monitoring content generation:

1. You'll need a W&B account (create one at https://wandb.ai if needed)
2. Run the setup script to configure your API key:
   ```
   python setup_wandb.py
   ```
3. The application will automatically log metrics and examples to your W&B project

## Troubleshooting

If you encounter any issues:

1. Make sure Python 3.8 or higher is installed
2. Check that all dependencies are installed correctly
3. Verify that the writing style JSON file is in the correct location
4. Look at the terminal output for error messages
5. For W&B issues, check that your API key is correctly set in the .env file

If problems persist, check the project's README.md for more detailed information.

## Model Evaluation Framework

The project includes a comprehensive evaluation framework for assessing model performance across different domains and capabilities:

### Quick Evaluation Launch

1. Double-click the `Model Evaluation.command` file on your Desktop
2. When prompted:
   - Select the adapter path (for a fine-tuned model)
   - Set the batch size (number of questions per domain)
   - Choose whether to enable W&B tracking
3. The evaluation will run automatically, testing:
   - Customer Experience knowledge
   - Artificial Intelligence knowledge
   - Machine Learning knowledge
   - Cross-referencing capabilities
   - Counterfactual reasoning

### Evaluation Results

Results are saved in:
- `evaluation_results/eval_results_[TIMESTAMP].json` - Detailed results in JSON format
- `evaluation_results/memory_tracking_[TIMESTAMP].csv` - Memory usage tracking
- W&B project dashboard (if enabled) with visualizations

For detailed information on the evaluation framework, please refer to the `README_EVALUATION.md` file.
