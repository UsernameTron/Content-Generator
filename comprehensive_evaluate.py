#!/usr/bin/env python3
"""
Comprehensive evaluation script for the C. Pete Connor model.
Tests domain expertise, industry knowledge, cross-referencing, and counterfactual reasoning.
"""

import os
import sys
import json
import time
import torch
import logging
import argparse
import csv
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.rule import Rule
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
import wandb
from wandb.apis import public
from evaluators import BaseEvaluator

# Set up environment variables for Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("evaluator")

# Set up rich console for pretty output
console = Console()

class EvaluationManager:
    """
    Main coordinator for comprehensive model evaluation.
    
    This class manages the overall evaluation process, including:
    - Model loading and setup
    - Coordinating different evaluators
    - Result collection and aggregation
    - WandB integration and visualization
    - Memory management for Apple Silicon optimization
    """
    
    def __init__(self, args):
        """
        Initialize the evaluation manager.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.model = None
        self.tokenizer = None
        self.device = None
        self.evaluators = {}
        self.results = {}
        self.wandb_run = None
        self.start_time = time.time()
        self.memory_tracking = []
        
        # Initialize WandB if enabled
        if self.args.use_wandb:
            self._init_wandb()
        
        # Set up evaluators
        self._setup_evaluators()
        
    def _setup_evaluators(self):
        """Set up domain and industry-specific evaluators."""
        logger.info("Setting up evaluators")
        
        # Import evaluators
        from evaluators.domain_knowledge import CustomerExperienceEvaluator, ArtificialIntelligenceEvaluator, MachineLearningEvaluator
        from evaluators.cross_referencing import CrossReferencingEvaluator
        from evaluators.counterfactual import CounterfactualEvaluator
        
        # Initialize domain-specific evaluators
        self.evaluators["cx"] = CustomerExperienceEvaluator(self)
        self.evaluators["ai"] = ArtificialIntelligenceEvaluator(self)
        self.evaluators["ml"] = MachineLearningEvaluator(self)
        
        # Initialize special capability evaluators
        if not self.args.skip_cross_reference:
            self.evaluators["cross_reference"] = CrossReferencingEvaluator(self)
        
        if not self.args.skip_counterfactual:
            self.evaluators["counterfactual"] = CounterfactualEvaluator(self)
        
        logger.info(f"Initialized {len(self.evaluators)} evaluators")
        
    def _init_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        logger.info("Initializing WandB")
        
        # Set up wandb configuration
        wandb_config = {
            "adapter_path": self.args.model_path,
            "device": self.device,
            "batch_size": self.args.batch_size,
            "domains_evaluated": [],
            "temperature": self.args.temperature,
        }
        
        # Add selected evaluators to config
        for evaluator_name in self.evaluators:
            wandb_config["domains_evaluated"].append(evaluator_name)
        
        # Initialize wandb
        self.wandb_run = wandb.init(
            project="cpc-model-evaluation",
            name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=wandb_config,
            tags=["comprehensive", "evaluation"]
        )
        
        logger.info(f"WandB initialized: {self.wandb_run.name}")
    
    def load_model(self):
        """Load the fine-tuned model with adapters."""
        logger.info(f"Loading adapter config from {self.args.model_path}")
        
        # Load the adapter config
        adapter_config = PeftConfig.from_pretrained(self.args.model_path)
        
        # Determine device
        if torch.cuda.is_available() and self.args.device == "cuda":
            self.device = "cuda"
        elif torch.backends.mps.is_available() and self.args.device == "mps":
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Log memory before model loading
        self._track_memory("before_model_load")
        
        # Load the base model
        logger.info(f"Loading base model: {adapter_config.base_model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            torch_dtype=torch.float16 if self.device != "mps" else torch.float32,
            device_map=self.device,
            trust_remote_code=False,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_config.base_model_name_or_path)
        
        # Load the fine-tuned model with adapters
        logger.info("Loading LoRA adapters")
        self.model = PeftModel.from_pretrained(self.model, self.args.model_path)
        
        # Log memory after model loading
        self._track_memory("after_model_load")
        
        return self.model, self.tokenizer
    
    def _track_memory(self, checkpoint_name):
        """Track memory usage at different checkpoints."""
        if not self.args.track_memory:
            return
        
        mem = psutil.virtual_memory()
        gpu_mem = 0
        
        # Try to get GPU memory if available
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        
        memory_info = {
            "checkpoint": checkpoint_name,
            "timestamp": time.time() - self.start_time,
            "ram_used_gb": mem.used / (1024 ** 3),
            "ram_used_mb": mem.used / (1024 ** 2),
            "ram_percent": mem.percent,
            "gpu_used_gb": gpu_mem
        }
        
        self.memory_tracking.append(memory_info)
        logger.info(f"Memory at {checkpoint_name}: {mem.percent}% RAM used, {memory_info['ram_used_gb']:.2f}GB")
        
        # Check if we're approaching memory limits
        if mem.percent > 85:
            logger.warning("CRITICAL: Memory usage above 85%, risk of OOM errors")
        elif mem.percent > 70:
            logger.warning("WARNING: Memory usage above 70%, performance may be impacted")

    def generate_response(self, prompt, max_tokens=512):
        """Generate a response from the model."""
        logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        
        # Format the prompt with special tokens
        formatted_prompt = f"<|prompter|>{prompt}<|assistant|>"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Track memory before generation
        self._track_memory("before_generation")
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                generation_config=GenerationConfig(
                    temperature=self.args.temperature,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    max_length=max_tokens,
                    do_sample=True
                )
            )
        
        # Track memory after generation
        self._track_memory("after_generation")
        
        # Decode and clean response
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[1].strip()
            # Clean up any trailing special tokens
            if "<|" in response:
                response = response.split("<|")[0].strip()
        else:
            response = full_response
        
        return response
    
    def run_evaluation(self):
        """Run the comprehensive evaluation across all evaluators."""
        logger.info("Starting comprehensive evaluation")
        console.print(Rule("C. Pete Connor Model Evaluation", style="green"))
        
        # Load the model
        self.load_model()
        
        # Create a progress tracker
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Track overall progress
            overall_task = progress.add_task("[green]Running evaluations...", total=len(self.evaluators))
            
            # Run each evaluator
            for name, evaluator in self.evaluators.items():
                progress.update(overall_task, description=f"[green]Evaluating {name}...")
                
                try:
                    # Run the evaluation for this domain/capability
                    results = evaluator.evaluate()
                    
                    # Store results
                    self.results[name] = results
                    
                    # Log to wandb if enabled
                    if self.args.use_wandb and self.wandb_run:
                        self.wandb_run.log({f"{name}_score": results.get("score", 0)})
                        if "metrics" in results:
                            for metric, value in results["metrics"].items():
                                self.wandb_run.log({f"{name}_{metric}": value})
                    
                    logger.info(f"Completed evaluation for {name}")
                
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {str(e)}")
                    self.results[name] = {"error": str(e), "score": 0}
                
                # Mark this evaluator as done
                progress.update(overall_task, advance=1)
        
        # Generate and display a summary table
        self._display_results_summary()
        
        # Save results to file
        self._save_results()
        
        # Update wandb with memory tracking
        if self.args.use_wandb and self.wandb_run:
            self._log_memory_to_wandb()
        
        logger.info("Evaluation complete")
        
    def _display_results_summary(self):
        """Display a summary of evaluation results."""
        table = Table(title="Evaluation Results")
        table.add_column("Domain/Capability", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Details", style="yellow")
        
        for name, result in self.results.items():
            score = result.get("score", 0)
            details = result.get("summary", "No summary available")
            if "error" in result:
                details = f"ERROR: {result['error']}"
            
            table.add_row(name, f"{score:.2f}", details[:50] + "...")
        
        console.print(table)
        
    def _save_results(self):
        """Save evaluation results to disk."""
        if self.args.save_results:
            with open(self.args.save_results, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {self.args.save_results}")
        
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save JSON results
        results_file = results_dir / f"eval_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save memory tracking
        memory_file = results_dir / f"memory_tracking_{timestamp}.csv"
        with open(memory_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.memory_tracking[0].keys())
            writer.writeheader()
            writer.writerows(self.memory_tracking)
        
        logger.info(f"Results saved to {results_file}")
        
    def _log_memory_to_wandb(self):
        """Log memory tracking data to WandB."""
        if not self.args.use_wandb or not self.wandb_run:
            return
            
        logger.info("Logging memory usage data to WandB")
        
        # Extract data for plotting
        timestamps = [entry["timestamp"] for entry in self.memory_tracking]
        ram_usage = [entry["ram_percent"] for entry in self.memory_tracking]
        checkpoints = [entry["checkpoint"] for entry in self.memory_tracking]
        
        # Create table for memory tracking
        memory_table = wandb.Table(
            columns=["timestamp", "checkpoint", "ram_used_mb", "ram_percent"],
            data=[
                [entry["timestamp"], entry["checkpoint"], entry["ram_used_mb"], entry["ram_percent"]] 
                for entry in self.memory_tracking
            ]
        )
        
        # Log the data to wandb
        self.wandb_run.log({
            "memory_usage": memory_table,
            "memory_chart": wandb.plot.line_series(
                xs=[timestamps], 
                ys=[ram_usage],
                keys=["RAM %"],
                x_title="Time (s)",
                y_title="Usage (%)",
                title="Memory Usage During Evaluation"
            )
        })


# Setup command-line arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for C. Pete Connor model")
    
    # Required arguments
    parser.add_argument(
        "--model-path", 
        "--adapter-path", 
        type=str, 
        required=True,
        help="Path to the LoRA adapter folder"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--save-results", 
        type=str, 
        default="",
        help="Save results to the specified JSON file"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for text generation"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="mps",
        help="Device to use for evaluation (cpu, mps, or cuda)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # W&B integration
    parser.add_argument(
        "--use-wandb", 
        action="store_true",
        help="Enable Weights & Biases integration"
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default="cpc-model-evaluation",
        help="W&B project name"
    )
    parser.add_argument(
        "--track-memory", 
        action="store_true",
        help="Track memory usage during evaluation"
    )
    
    # Evaluation options
    parser.add_argument(
        "--skip-cross-reference", 
        action="store_true",
        help="Skip cross-referencing evaluation"
    )
    parser.add_argument(
        "--skip-counterfactual", 
        action="store_true",
        help="Skip counterfactual reasoning evaluation"
    )
    
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create the evaluation manager
    manager = EvaluationManager(args)
    
    # Run the evaluation
    try:
        manager.run_evaluation()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
    finally:
        # Clean up wandb if used
        if args.use_wandb and manager.wandb_run:
            manager.wandb_run.finish()
