"""
Create custom W&B metrics callback for monitoring the C. Pete Connor model training.

This script creates a custom metrics callback for Weights & Biases to track:
1. Overall loss and convergence metrics
2. Domain-specific performance across CX, AI, and ML topics
3. Reduction in AI-typical phrasing ("humanity score")
4. Irony and humor effectiveness metrics
"""

import os
import argparse
import logging
import json
import re
from pathlib import Path
import numpy as np
import torch
from transformers import TrainerCallback
import wandb
from dotenv import load_dotenv
import time
import psutil
import platform
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CustomMetricsCallback(TrainerCallback):
    """Custom callback for logging specialized metrics to W&B during training."""
    
    def __init__(self, tokenizer, custom_loss_config, eval_dataset=None):
        """
        Initialize callback with model tokenizer and configuration.
        
        Args:
            tokenizer: Model tokenizer
            custom_loss_config: Configuration for custom metrics
            eval_dataset: Evaluation dataset
        """
        self.tokenizer = tokenizer
        self.penalized_phrases = custom_loss_config.get("penalized_phrases", [])
        self.rewarded_phrases = custom_loss_config.get("rewarded_phrases", [])
        self.eval_dataset = eval_dataset
        self.domains = ["customer_experience", "artificial_intelligence", "machine_learning"]
        self.domain_examples = self._prepare_domain_examples()
        self.last_log_time = None
        self.start_time = None
        self.device_type = None
        self.memory_metrics = {}  # For storing memory usage metrics
        
    def _prepare_domain_examples(self):
        """Prepare examples for each domain to evaluate domain-specific performance."""
        domain_examples = {}
        
        domain_prompts = {
            "customer_experience": [
                "Explain the reality of customer experience metrics:",
                "Describe the irony of NPS scores in modern businesses:",
                "Analyze how customer surveys are actually used:",
            ],
            "artificial_intelligence": [
                "Explain the reality of AI implementation in enterprises:",
                "Analyze the gap between AI marketing and actual capabilities:",
                "Describe the limitations of current generative AI models:",
            ],
            "machine_learning": [
                "Explain the challenges of ML model deployment in production:",
                "Describe the reality of ML model accuracy claims:",
                "Analyze the gap between ML research and practical applications:",
            ]
        }
        
        for domain, prompts in domain_prompts.items():
            domain_examples[domain] = prompts
            
        return domain_examples
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Log custom metrics during evaluation."""
        logger.info("Logging custom metrics to W&B")
        
        try:
            # Log domain-specific metrics
            self._log_domain_metrics(model)
            
            # Log humanity score metrics
            self._log_humanity_metrics(model)
            
            # Log irony and humor metrics
            self._log_style_metrics(model)
            
        except Exception as e:
            logger.error(f"Error logging custom metrics: {str(e)}")
            
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize training metrics for W&B tracking."""
        try:
            logger.info("Initializing training metrics for W&B")
            
            # Get device type for specialized logging
            if model is not None:
                self.device_type = next(model.parameters()).device.type
            
            # Initialize time tracking
            self.start_time = time.time()
            self.last_log_time = self.start_time
            
            # Count trainable parameters
            if model is not None:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                
                wandb.log({
                    "model/trainable_parameters": trainable_params,
                    "model/total_parameters": total_params,
                    "model/trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
                })
                
                # Log special metrics for MPS
                if self.device_type == "mps":
                    wandb.log({
                        "mps/device_active": True,
                        "mps/apple_silicon": True,
                    })
                    
                    # Try to get more detailed Apple Silicon info
                    try:
                        import platform
                        import subprocess
                        
                        # Get macOS version
                        mac_ver = platform.mac_ver()[0]
                        wandb.log({"mps/macos_version": mac_ver})
                        
                        # Try to get chip info
                        try:
                            chip_info = subprocess.check_output(
                                ["sysctl", "-n", "machdep.cpu.brand_string"]
                            ).decode("utf-8").strip()
                            wandb.log({"mps/chip_info": chip_info})
                        except:
                            pass
                    except:
                        pass
            
            logger.info("Successfully initialized training metrics")
        except Exception as e:
            logger.error(f"Error initializing training metrics: {str(e)}")
            
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log custom metrics during training."""
        if not logs:
            return
        
        try:
            # Extract performance metrics for detailed tracking
            metrics_dict = {}
            
            # Track time-based metrics
            current_time = time.time()
            if self.last_log_time is not None:
                time_elapsed = current_time - self.last_log_time
                metrics_dict["time/seconds_per_step"] = time_elapsed
                
                # Calculate step speed
                if "loss" in logs and hasattr(state, "global_step") and state.global_step > 0:
                    steps_per_second = 1.0 / time_elapsed if time_elapsed > 0 else 0
                    metrics_dict["time/steps_per_second"] = steps_per_second
                
                # Calculate total training time
                if self.start_time is not None:
                    total_elapsed = current_time - self.start_time
                    metrics_dict["time/total_hours"] = total_elapsed / 3600
                    metrics_dict["time/total_minutes"] = total_elapsed / 60
                    
                    # Estimate remaining time
                    if hasattr(state, "global_step") and state.global_step > 0 and hasattr(state, "max_steps") and state.max_steps > 0:
                        progress = state.global_step / state.max_steps
                        if progress > 0:
                            est_total_time = total_elapsed / progress
                            est_remaining = est_total_time - total_elapsed
                            metrics_dict["time/estimated_remaining_hours"] = est_remaining / 3600
                            metrics_dict["time/progress_percentage"] = progress * 100
            
            self.last_log_time = current_time
            
            # Enhanced loss tracking
            if "loss" in logs:
                loss = logs["loss"]
                metrics_dict["training/loss"] = loss
                metrics_dict["training/perplexity"] = np.exp(min(loss, 20))  # Cap at e^20 to avoid overflow
                
                # Track loss improvement over time
                if hasattr(self, "previous_loss"):
                    loss_change = self.previous_loss - loss
                    metrics_dict["training/loss_improvement"] = loss_change
                self.previous_loss = loss
            
            # Track evaluation metrics
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                metrics_dict["evaluation/loss"] = eval_loss
                metrics_dict["evaluation/perplexity"] = np.exp(min(eval_loss, 20))
                
                # Calculate overfitting signal (difference between train and eval loss)
                if "loss" in logs:
                    metrics_dict["training/overfitting_signal"] = eval_loss - logs["loss"]
            
            # Extract and track learning rate
            if "learning_rate" in logs:
                metrics_dict["training/learning_rate"] = logs["learning_rate"]
            
            # Track domain expertise metrics if they exist in logs
            domain_metrics = [
                "cx_expertise_score", 
                "ai_expertise_score",
                "satire_level_score",
                "domain_expertise/customer_experience",
                "domain_expertise/artificial_intelligence",
                "domain_expertise/machine_learning"
            ]
            
            for metric in domain_metrics:
                if metric in logs:
                    metrics_dict[f"expertise/{metric}"] = logs[metric]
            
            # Track gradient statistics
            if model is not None:
                grad_metrics = self._calculate_gradient_metrics(model)
                metrics_dict.update(grad_metrics)
            
            # Track MPS-specific metrics
            if self.device_type == "mps" and model is not None:
                # Check for any nan values in gradients which can cause issues on MPS
                has_nan_grads = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grads = True
                        break
                
                metrics_dict["mps/has_nan_gradients"] = has_nan_grads
                
                # Track training throughput metrics specific to Apple Silicon
                if "loss" in logs and time_elapsed > 0:
                    if hasattr(args, "per_device_train_batch_size") and hasattr(args, "gradient_accumulation_steps"):
                        effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                        tokens_per_batch = effective_batch_size * args.max_seq_length if hasattr(args, "max_seq_length") else 512
                        tokens_per_second = tokens_per_batch / time_elapsed
                        metrics_dict["mps/tokens_per_second"] = tokens_per_second
                        metrics_dict["mps/effective_batch_size"] = effective_batch_size
                
                # Add memory tracking for MPS
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    metrics_dict["memory/rss_gb"] = memory_info.rss / (1024 ** 3)  # GB
                    metrics_dict["memory/vms_gb"] = memory_info.vms / (1024 ** 3)  # GB
                    
                    # Get system memory info
                    system_memory = psutil.virtual_memory()
                    metrics_dict["system/memory_percent_used"] = system_memory.percent
                    metrics_dict["system/memory_available_gb"] = system_memory.available / (1024 ** 3)
                    
                    # Get CPU usage
                    metrics_dict["system/cpu_percent"] = psutil.cpu_percent(interval=0.1)
                    
                    # Alert if memory usage is high
                    if system_memory.percent > 85:
                        wandb.alert(
                            title="High Memory Usage",
                            text=f"System memory usage at {system_memory.percent}%. Training may become unstable.",
                            level=wandb.AlertLevel.WARNING
                        )
                except Exception as mem_error:
                    logger.warning(f"Error collecting memory metrics: {str(mem_error)}")
            
            # Add any existing logs
            metrics_dict.update(logs)
            
            # Log all metrics to W&B
            wandb.log(metrics_dict)
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            
    def _calculate_gradient_metrics(self, model):
        """Calculate gradient statistics for model parameters."""
        metrics = {}
        try:
            grad_norms = []
            layer_norms = {}
            
            # Calculate gradient norms by layer type
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    norm = param.grad.data.norm(2).item()
                    grad_norms.append(norm)
                    
                    # Group by parameter type for detailed tracking
                    param_type = name.split('.')[-1]
                    if param_type not in layer_norms:
                        layer_norms[param_type] = []
                    layer_norms[param_type].append(norm)
            
            if grad_norms:
                metrics["gradients/global_norm"] = np.mean(grad_norms)
                metrics["gradients/global_max"] = np.max(grad_norms)
                metrics["gradients/global_min"] = np.min(grad_norms)
                
                # Log layer-specific gradient statistics
                for layer_type, norms in layer_norms.items():
                    if norms:
                        metrics[f"gradients/{layer_type}_mean"] = np.mean(norms)
                        
                # Check for exploding/vanishing gradients
                if np.max(grad_norms) > 10:
                    metrics["gradients/exploding_gradient_signal"] = True
                if np.mean(grad_norms) < 0.0001:
                    metrics["gradients/vanishing_gradient_signal"] = True
        except Exception as e:
            logger.warning(f"Error calculating gradient metrics: {str(e)}")
        
        return metrics
    
    def _log_domain_metrics(self, model):
        """Log domain-specific performance metrics."""
        device = model.device
        model.eval()
        
        for domain, prompts in self.domain_examples.items():
            domain_losses = []
            
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    domain_losses.append(loss)
            
            # Log domain metrics
            wandb.log({
                f"eval/domain/{domain}/loss": np.mean(domain_losses),
                f"eval/domain/{domain}/perplexity": np.exp(np.mean(domain_losses))
            })
            
    def _log_humanity_metrics(self, model):
        """Log metrics related to AI-typical phrasing reduction."""
        # Count penalized phrases in generated text
        ai_phrase_count = 0
        total_phrases = max(1, len(self.penalized_phrases))
        
        # Generate text from sample prompts
        sample_prompts = [
            "Explain the future of customer experience:",
            "Describe how AI will transform business:",
            "Analyze the impact of machine learning on society:"
        ]
        
        device = model.device
        model.eval()
        
        for prompt in sample_prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=200,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Count AI buzzwords
            for phrase in self.penalized_phrases:
                if phrase.lower() in generated_text.lower():
                    ai_phrase_count += 1
        
        # Calculate metrics
        humanity_score = 1.0 - (ai_phrase_count / (total_phrases * len(sample_prompts)))
        sentence_diversity = self._calculate_sentence_diversity(sample_prompts, model)
        
        # Log metrics
        wandb.log({
            "eval/custom/ai_buzzword_score": ai_phrase_count / len(sample_prompts),
            "eval/custom/humanity_score": humanity_score,
            "eval/custom/sentence_diversity": sentence_diversity,
            "eval/custom/cliche_reduction": 1.0 - (ai_phrase_count / max(1, total_phrases))
        })
    
    def _calculate_sentence_diversity(self, prompts, model):
        """Calculate sentence structure diversity score."""
        device = model.device
        generated_texts = []
        
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=200,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                
            generated_texts.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
        
        # Calculate diversity based on sentence length variation
        sentences = []
        for text in generated_texts:
            sentences.extend([s.strip() for s in re.split(r'[.!?]', text) if s.strip()])
        
        if not sentences:
            return 0.0
            
        sentence_lengths = [len(s.split()) for s in sentences]
        
        if len(sentence_lengths) <= 1:
            return 0.0
            
        # Calculate coefficient of variation (higher is more diverse)
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        
        if mean_length == 0:
            return 0.0
            
        diversity = std_length / mean_length
        return min(1.0, diversity)  # Cap at 1.0
    
    def _log_style_metrics(self, model):
        """Log irony, humor, and sarcasm metrics."""
        device = model.device
        model.eval()
        
        # Sample prompts for style evaluation
        style_prompts = {
            "irony": [
                "Write an ironic comment about AI progress:",
                "Provide an ironic take on customer experience software:"
            ],
            "humor": [
                "Write a humorous explanation of machine learning:",
                "Create a funny take on customer satisfaction surveys:"
            ],
            "sarcasm": [
                "Write a sarcastic review of an AI product:",
                "Create a sarcastic comment about digital transformation:"
            ]
        }
        
        style_scores = {
            "irony": 0.0,
            "humor": 0.0,
            "sarcasm": 0.0
        }
        
        # Patterns that indicate each style
        style_patterns = {
            "irony": [
                r"(?i)ironic(ally)?",
                r"(?i)contrary to",
                r"(?i)yet somehow",
                r"(?i)supposedly",
                r"(?i)paradox(ically)?",
                r"(?i)in reality"
            ],
            "humor": [
                r"(?i)funny",
                r"(?i)humor(ous)?",
                r"(?i)amusing",
                r"(?i)ridiculous",
                r"(?i)absurd"
            ],
            "sarcasm": [
                r"(?i)sarcas(m|tic)",
                r"(?i)obviously",
                r"(?i)clearly",
                r"(?i)of course",
                r"(?i)surely",
                r"(?i)absolutely"
            ]
        }
        
        for style, prompts in style_prompts.items():
            style_count = 0
            max_count = len(style_patterns[style]) * len(prompts)
            
            for prompt in prompts:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_length=200,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.8
                    )
                    
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Count style patterns
                for pattern in style_patterns[style]:
                    if re.search(pattern, generated_text):
                        style_count += 1
            
            # Calculate style score
            style_scores[style] = style_count / max(1, max_count)
        
        # Log style metrics
        wandb.log({
            "eval/custom/irony_score": style_scores["irony"],
            "eval/custom/humor_score": style_scores["humor"],
            "eval/custom/sarcasm_score": style_scores["sarcasm"],
            "eval/custom/style_average": np.mean(list(style_scores.values()))
        })

    def _log_sample_generations(self, model):
        """Generate and log sample texts to WandB."""
        sample_prompts = [
            "What's the future of customer experience?",
            "How will AI transform business operations?",
            "Explain the impact of machine learning on retail."
        ]
        
        try:
            device = model.device
            model.eval()
            
            for i, prompt in enumerate(sample_prompts):
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_length=200,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Log sample text
                wandb.log({
                    f"samples/generation_{i+1}": wandb.Html(f"<strong>Prompt:</strong> {prompt}<br><br><strong>Generated:</strong> {generated_text}")
                })
                
        except Exception as e:
            logger.error(f"Error generating sample texts: {str(e)}")
    
def create_custom_metrics_callback(config_path="finetune_config.json", tokenizer=None, eval_dataset=None):
    """
    Create and return a custom metrics callback instance.
    
    Args:
        config_path: Path to configuration file
        tokenizer: Model tokenizer
        eval_dataset: Evaluation dataset
        
    Returns:
        CustomMetricsCallback instance
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        custom_loss_config = config.get("custom_loss_config", {})
        
        if tokenizer is None:
            logger.warning("Tokenizer not provided, metrics requiring tokenization will not work.")
            
        # Create callback
        callback = CustomMetricsCallback(tokenizer, custom_loss_config, eval_dataset)
        logger.info("Created custom metrics callback for W&B tracking")
        
        return callback
    except Exception as e:
        logger.error(f"Error creating custom metrics callback: {str(e)}")
        return None

def create_wandb_training_dashboard(run_id=None, project_name=None):
    """
    Create a custom W&B dashboard for monitoring Apple Silicon training performance.
    
    This function creates a comprehensive dashboard with panels for:
    1. Training Performance
    2. Apple Silicon Metrics
    3. Memory Usage
    4. Gradient Analysis
    5. Domain Expertise Tracking
    
    Args:
        run_id: Optional W&B run ID to attach this dashboard to
        project_name: Optional W&B project name
        
    Returns:
        Dashboard URL if successful, None otherwise
    """
    try:
        import wandb
        from wandb.apis import public
        
        # Get API handle
        api = public.Api()
        
        # Get current run if not specified
        if run_id is None and wandb.run is not None:
            run_id = wandb.run.id
            
        if project_name is None and wandb.run is not None:
            project_name = wandb.run.project
            
        if run_id is None or project_name is None:
            logger.warning("W&B run ID or project name not provided and no active run found")
            return None
            
        # Create dashboard specification
        dashboard_json = {
            "name": "Apple Silicon Training Dashboard",
            "description": "Comprehensive metrics for monitoring model training on Apple Silicon",
            "panels": [
                # Training Progress Panel
                {
                    "name": "Training Progress",
                    "panel_type": "line",
                    "fields": {
                        "title": "Training Progress & Loss",
                        "x_field": "_step",
                        "y_field": [
                            "training/loss",
                            "evaluation/loss",
                            "training/perplexity"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Loss / Perplexity"
                    }
                },
                # Training Time Estimation Panel
                {
                    "name": "Time Estimates",
                    "panel_type": "line",
                    "fields": {
                        "title": "Training Time Estimation",
                        "x_field": "_step",
                        "y_field": [
                            "time/progress_percentage",
                            "time/estimated_remaining_hours",
                            "time/total_hours"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Hours / Percent"
                    }
                },
                # Apple Silicon Performance Panel
                {
                    "name": "MPS Performance",
                    "panel_type": "line",
                    "fields": {
                        "title": "Apple Silicon Performance",
                        "x_field": "_step",
                        "y_field": [
                            "mps/tokens_per_second",
                            "time/steps_per_second"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Throughput"
                    }
                },
                # Memory Usage Panel
                {
                    "name": "Memory Usage",
                    "panel_type": "line",
                    "fields": {
                        "title": "Memory Utilization",
                        "x_field": "_step",
                        "y_field": [
                            "memory/rss_gb",
                            "memory/vms_gb",
                            "system/memory_percent_used"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Memory (GB / %)"
                    }
                },
                # Gradient Statistics Panel
                {
                    "name": "Gradient Analysis",
                    "panel_type": "line",
                    "fields": {
                        "title": "Gradient Statistics",
                        "x_field": "_step",
                        "y_field": [
                            "gradients/global_norm",
                            "gradients/global_max",
                            "gradients/global_min"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Gradient Norm"
                    }
                },
                # Learning Rate and Overfitting Panel
                {
                    "name": "Learning Dynamics",
                    "panel_type": "line",
                    "fields": {
                        "title": "Learning Rate & Overfitting",
                        "x_field": "_step",
                        "y_field": [
                            "training/learning_rate",
                            "training/overfitting_signal",
                            "training/loss_improvement"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Value"
                    }
                },
                # Domain Expertise Panel
                {
                    "name": "Domain Expertise",
                    "panel_type": "line",
                    "fields": {
                        "title": "Domain-Specific Performance",
                        "x_field": "_step",
                        "y_field": [
                            "expertise/cx_expertise_score",
                            "expertise/ai_expertise_score",
                            "expertise/satire_level_score"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Score"
                    }
                },
                # System Resources Panel
                {
                    "name": "System Resources",
                    "panel_type": "line",
                    "fields": {
                        "title": "System Resource Utilization",
                        "x_field": "_step",
                        "y_field": [
                            "system/cpu_percent",
                            "system/memory_available_gb",
                            "mps/has_nan_gradients"
                        ],
                        "x_axis_title": "Training Step",
                        "y_axis_title": "Value"
                    }
                },
                # Training Configuration Summary
                {
                    "name": "Training Config",
                    "panel_type": "summary",
                    "fields": {
                        "title": "Training Configuration"
                    }
                }
            ]
        }
        
        # Create the dashboard
        logger.info(f"Creating W&B dashboard for run {run_id} in project {project_name}")
        dashboard = api.create_dashboard(
            project_name,
            dashboard_json,
            entity=wandb.run.entity if wandb.run else None
        )
        
        dashboard_url = dashboard["url"]
        logger.info(f"Dashboard created successfully at: {dashboard_url}")
        
        return dashboard_url
    except Exception as e:
        logger.error(f"Error creating W&B dashboard: {str(e)}")
        return None

def create_wandb_alert_rules(run_id=None, project_name=None):
    """
    Create W&B alert rules for monitoring critical training metrics.
    
    Sets up alerts for:
    1. Memory usage exceeding thresholds
    2. NaN gradients detection
    3. Performance degradation
    4. Overfitting detection
    5. Training stalls
    
    Args:
        run_id: Optional W&B run ID to attach alerts to
        project_name: Optional W&B project name
        
    Returns:
        List of created alert IDs if successful, empty list otherwise
    """
    try:
        import wandb
        from wandb.apis import public
        
        # Get API handle
        api = public.Api()
        
        # Get current run if not specified
        if run_id is None and wandb.run is not None:
            run_id = wandb.run.id
            
        if project_name is None and wandb.run is not None:
            project_name = wandb.run.project
            
        if run_id is None or project_name is None:
            logger.warning("W&B run ID or project name not provided and no active run found")
            return []
            
        alert_ids = []
        entity = wandb.run.entity if wandb.run else None
            
        # Define alert configurations
        alert_configs = [
            # Memory threshold alerts
            {
                "name": "High Memory Usage Warning",
                "description": "Memory usage exceeds 70% of available RAM",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "system/memory_percent_used",
                    "op": "GREATER_THAN",
                    "value": 70.0
                },
                "frequency": "REGULAR",
                "duration": 120, # seconds
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "WARNING"
            },
            {
                "name": "Critical Memory Usage",
                "description": "Memory usage exceeds 85% of available RAM",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "system/memory_percent_used",
                    "op": "GREATER_THAN",
                    "value": 85.0
                },
                "frequency": "REGULAR",
                "duration": 60, # seconds
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "CRITICAL"
            },
            # Gradient issues
            {
                "name": "NaN Gradients Detected",
                "description": "NaN values detected in gradients",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "mps/has_nan_gradients",
                    "op": "EQUAL_TO",
                    "value": 1.0
                },
                "frequency": "ONCE",
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "CRITICAL"
            },
            {
                "name": "Exploding Gradients",
                "description": "Gradient norm is too high",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "gradients/global_max",
                    "op": "GREATER_THAN",
                    "value": 100.0
                },
                "frequency": "ONCE",
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "CRITICAL"
            },
            # Performance issues
            {
                "name": "Performance Degradation",
                "description": "Training speed dropped significantly",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "time/steps_per_second",
                    "op": "LESS_THAN",
                    "value": 0.05
                },
                "frequency": "REGULAR",
                "duration": 300, # seconds
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "WARNING"
            },
            # Training issues
            {
                "name": "Overfitting Detected",
                "description": "Significant gap between training and evaluation loss",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "training/overfitting_signal",
                    "op": "GREATER_THAN",
                    "value": 0.5
                },
                "frequency": "REGULAR",
                "duration": 600, # seconds
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "WARNING"
            },
            {
                "name": "Training Stalled",
                "description": "No improvement in loss for extended period",
                "filter": {
                    "run_id": run_id
                },
                "condition": {
                    "type": "metric",
                    "metric": "training/loss_improvement",
                    "op": "LESS_THAN",
                    "value": 0.001
                },
                "frequency": "REGULAR",
                "duration": 1200, # seconds
                "status": "ENABLED",
                "target_type": "EMAIL",
                "level": "WARNING"
            }
        ]
        
        # Create each alert
        for alert_config in alert_configs:
            try:
                logger.info(f"Creating W&B alert: {alert_config['name']}")
                alert = api.create_alert(
                    name=alert_config["name"],
                    description=alert_config["description"],
                    filter=alert_config["filter"],
                    condition=alert_config["condition"],
                    frequency=alert_config["frequency"],
                    status=alert_config["status"],
                    target_type=alert_config["target_type"],
                    entity=entity,
                    project=project_name,
                    level=alert_config["level"]
                )
                
                if "duration" in alert_config:
                    # Update duration if specified
                    alert.update(duration=alert_config["duration"])
                    
                alert_ids.append(alert.id)
                logger.info(f"Alert '{alert_config['name']}' created successfully")
                
            except Exception as e:
                logger.error(f"Error creating alert '{alert_config['name']}': {str(e)}")
        
        return alert_ids
    except Exception as e:
        logger.error(f"Error creating W&B alerts: {str(e)}")
        return []

def setup_wandb_for_apple_silicon(run_config=None, create_dashboard=True, create_alerts=True):
    """
    Set up W&B monitoring optimized for Apple Silicon training.
    
    This function:
    1. Initializes W&B with Apple Silicon-specific tags and configs
    2. Creates a comprehensive monitoring dashboard
    3. Sets up alert rules for critical metrics
    4. Returns the configured W&B run object
    
    Args:
        run_config (dict): Configuration for the W&B run
        create_dashboard (bool): Whether to create a monitoring dashboard
        create_alerts (bool): Whether to create alert rules
        
    Returns:
        wandb.Run: The configured W&B run object
    """
    try:
        import wandb
        import platform
        import torch
        import psutil
        import os
        import json
        import subprocess
        from pathlib import Path
        
        # Get system info
        system_info = {
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else platform.version(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device_type": "mps" if torch.backends.mps.is_available() else "cpu",
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        # Try to get more detailed Apple Silicon info
        if platform.system() == "Darwin" and system_info["device_type"] == "mps":
            try:
                # Get chip model
                chip_info = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]
                ).decode("utf-8").strip()
                system_info["chip_model"] = chip_info
                
                # Get MPS device details if available
                if hasattr(torch.mps, "get_device_properties"):
                    mps_props = torch.mps.get_device_properties(0)
                    if hasattr(mps_props, "name"):
                        system_info["mps_device_name"] = mps_props.name
                    if hasattr(mps_props, "total_memory"):
                        system_info["mps_memory_gb"] = round(mps_props.total_memory / (1024**3), 2)
            except Exception as e:
                logger.warning(f"Could not get detailed Apple Silicon info: {str(e)}")
        
        # Default configuration
        default_config = {
            "project": "apple-silicon-training",
            "name": f"training-{system_info['device_type']}",
            "tags": ["apple-silicon", "mps", "finetune"],
            "config": {
                "system": system_info,
                "training_params": {},
                "model_params": {}
            }
        }
        
        # Update with user config if provided
        if run_config:
            # Merge configs recursively
            if "config" in run_config and isinstance(run_config["config"], dict):
                for k, v in run_config["config"].items():
                    if k in default_config["config"] and isinstance(v, dict) and isinstance(default_config["config"][k], dict):
                        default_config["config"][k].update(v)
                    else:
                        default_config["config"][k] = v
                del run_config["config"]
            
            default_config.update(run_config)
        
        # Add Apple Silicon specific tags
        if system_info["device_type"] == "mps":
            if "tags" not in default_config:
                default_config["tags"] = []
            
            if "chip_model" in system_info:
                chip_tag = system_info["chip_model"].replace(" ", "-").lower()
                if "apple" not in chip_tag:
                    chip_tag = f"apple-{chip_tag}"
                default_config["tags"].append(chip_tag)
            
            default_config["tags"].extend(["mps", "apple-silicon"])
            
            # Ensure tags are unique
            default_config["tags"] = list(set(default_config["tags"]))
        
        # Initialize wandb
        logger.info(f"Initializing W&B with configuration: {json.dumps(default_config, indent=2)}")
        run = wandb.init(**default_config)
        
        # Set up dashboard if requested
        if create_dashboard:
            logger.info("Creating Apple Silicon training dashboard")
            dashboard_url = create_wandb_training_dashboard(
                run_id=run.id,
                project_name=run.project
            )
            if dashboard_url:
                logger.info(f"Dashboard created successfully at: {dashboard_url}")
                # Save dashboard URL in run config
                run.config.update({"dashboard_url": dashboard_url}, allow_val_change=True)
            else:
                logger.warning("Failed to create dashboard")
        
        # Set up alerts if requested
        if create_alerts:
            logger.info("Creating Apple Silicon training alerts")
            alert_ids = create_wandb_alert_rules(
                run_id=run.id,
                project_name=run.project
            )
            if alert_ids:
                logger.info(f"Created {len(alert_ids)} alert rules successfully")
                # Save alert IDs in run config
                run.config.update({"alert_ids": alert_ids}, allow_val_change=True)
            else:
                logger.warning("Failed to create alert rules")
        
        return run
    except Exception as e:
        logger.error(f"Error setting up W&B for Apple Silicon: {str(e)}")
        # Try to initialize a basic run as fallback
        try:
            return wandb.init(project="apple-silicon-training-fallback")
        except:
            logger.error("Failed to initialize W&B even in fallback mode")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom W&B metrics callback.")
    parser.add_argument("--config", type=str, default="finetune_config.json", help="Path to configuration file")
    parser.add_argument("--create-dashboard", action="store_true", help="Create W&B dashboard for monitoring")
    parser.add_argument("--create-alerts", action="store_true", help="Create W&B alert rules")
    parser.add_argument("--run-id", type=str, help="W&B run ID to attach dashboard/alerts to")
    parser.add_argument("--project", type=str, help="W&B project name")
    parser.add_argument("--test-full-setup", action="store_true", help="Test the complete W&B setup for Apple Silicon")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the W&B run when using --test-full-setup")
    
    args = parser.parse_args()
    
    # Test full setup if requested
    if args.test_full_setup:
        print("Testing complete W&B setup for Apple Silicon training...")
        
        # Create a config with the provided arguments
        run_config = {
            "project": args.project if args.project else "apple-silicon-test",
            "name": args.run_name if args.run_name else "test-setup"
        }
        
        # Set up W&B
        run = setup_wandb_for_apple_silicon(
            run_config=run_config,
            create_dashboard=args.create_dashboard,
            create_alerts=args.create_alerts
        )
        
        if run:
            print(f"W&B initialized successfully with run ID: {run.id}")
            print(f"View the run at: {run.url}")
            
            # Log some dummy metrics to test the dashboard
            print("Logging test metrics...")
            for i in range(20):
                metrics = {
                    "training/loss": 5.0 - (i * 0.2),
                    "evaluation/loss": 5.5 - (i * 0.15),
                    "time/steps_per_second": 0.5 + (i * 0.01),
                    "memory/rss_gb": 4.0 + (i * 0.1),
                    "system/memory_percent_used": 50 + (i * 0.5),
                    "mps/tokens_per_second": 100 + (i * 5),
                    "gradients/global_norm": 1.0 - (i * 0.02),
                    "expertise/cx_expertise_score": 0.5 + (i * 0.02)
                }
                run.log(metrics)
                
            print("Test metrics logged successfully!")
            print("Run finalization...")
            run.finish()
            print("W&B setup test completed successfully!")
            exit(0)
    
    # Just verify we can create the callback
    callback = create_custom_metrics_callback(args.config)
    if callback:
        print("Custom metrics callback created successfully.")
        print("Add this callback to your Trainer to log custom metrics during training.")
        print("Example usage:\n")
        print("from wandb_dashboards import create_custom_metrics_callback")
        print("callback = create_custom_metrics_callback(config_path, tokenizer, eval_dataset)")
        print("trainer = Trainer(..., callbacks=[callback])")
        
        # Create dashboard if requested
        if args.create_dashboard:
            print("\nCreating W&B dashboard...")
            dashboard_url = create_wandb_training_dashboard(run_id=args.run_id, project_name=args.project)
            if dashboard_url:
                print(f"Dashboard created successfully! View it at: {dashboard_url}")
            else:
                print("Failed to create dashboard. Check logs for details.")
                
        # Create alert rules if requested
        if args.create_alerts:
            print("\nCreating W&B alert rules...")
            alert_ids = create_wandb_alert_rules(run_id=args.run_id, project_name=args.project)
            if alert_ids:
                print(f"Created {len(alert_ids)} alert rules successfully!")
                for alert_id in alert_ids:
                    print(f"- Alert ID: {alert_id}")
            else:
                print("Failed to create alert rules. Check logs for details.")
    else:
        print("Error creating custom metrics callback. Check the logs for details.")
