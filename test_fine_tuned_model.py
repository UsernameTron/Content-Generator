"""
Test Fine-Tuned Model

This script tests the fine-tuned model by generating text samples across various domains.
It helps validate that the model has properly incorporated expertise in:
- Customer experience
- Artificial intelligence 
- Machine learning
while maintaining C. Pete Connor's satirical style.
"""

import os
import json
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import Trainer, TrainingArguments
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import print

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

def load_model(model_dir):
    """Load the fine-tuned model."""
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # Detect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        
        logger.info(f"Using device: {device}")
        
        # Check if the directory exists
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist")
            return None, None, None
        
        # Check if the directory has required files
        required_files = ["adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if missing_files:
            logger.warning(f"Model directory is missing files: {', '.join(missing_files)}")
            logger.warning("The model may be incomplete or still training")
        
        # Load model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,
                device_map=device
            )
            
            logger.info("Model loaded successfully")
            return model, tokenizer, device
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            return None, None, None
    except Exception as e:
        logger.error(f"Error during model loading process: {str(e)}")
        return None, None, None

def generate_text(model, tokenizer, prompt, max_length=200, num_samples=1, temperature=0.7):
    """Generate text using the model."""
    try:
        # Create generator pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Start timer to measure generation time
        start_time = time.time()
        
        # Generate text
        results = generator(prompt, num_return_sequences=num_samples)
        
        # Calculate generation time
        gen_time = time.time() - start_time
        
        # Extract results
        generated_texts = [res["generated_text"] for res in results]
        
        # Calculate performance metrics
        tokens_generated = sum(len(tokenizer.encode(text)) - len(tokenizer.encode(prompt)) for text in generated_texts)
        tokens_per_second = tokens_generated / gen_time if gen_time > 0 else 0
        
        performance_metrics = {
            "generation_time_seconds": gen_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
        }
        
        logger.debug(f"Generated {tokens_generated} tokens in {gen_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec)")
        
        return generated_texts, performance_metrics
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}", exc_info=True)
        return [f"Error: {str(e)}"], {"error": str(e)}

def load_metadata(model_dir):
    """Load model metadata."""
    metadata_path = Path(model_dir) / "model_metadata.json"
    try:
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
                return metadata
        else:
            logger.warning(f"No metadata found at {metadata_path}")
            
            # Try to create basic metadata if it doesn't exist
            parent_dir = Path(model_dir).parent
            alternative_path = parent_dir / "model_metadata.json"
            
            if alternative_path.exists():
                with open(alternative_path, "r") as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded metadata from {alternative_path}")
                    return metadata
            
            logger.warning("Creating basic metadata structure")
            return {
                "model_name": Path(model_dir).name,
                "training_complete": "final" in model_dir.lower(),
                "specialization_domains": [
                    "customer experience", 
                    "artificial intelligence",
                    "machine learning"
                ]
            }
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}

def measure_inference_performance(model, tokenizer, device, iterations=5):
    """Measure model inference performance."""
    console.print("[bold]Measuring inference performance...[/bold]")
    
    try:
        # Create sample prompts of different lengths
        sample_prompts = [
            "The future of customer service is",  # Short
            "When implementing AI solutions for customer experience, the most important consideration is",  # Medium
            "The relationship between machine learning, artificial intelligence, and customer experience can be understood by examining how these technologies impact the way companies interact with their customers across multiple touchpoints. Specifically,"  # Long
        ]
        
        results = []
        
        for prompt in track(sample_prompts, description="Testing prompts..."):
            prompt_length = len(tokenizer.encode(prompt))
            prompt_results = []
            
            # Run multiple iterations to get average performance
            for i in range(iterations):
                # Generate with different temperatures to test variability
                temperature = 0.7 if i % 2 == 0 else 0.9
                
                # Generate text
                _, metrics = generate_text(
                    model, 
                    tokenizer, 
                    prompt, 
                    max_length=prompt_length + 100,
                    temperature=temperature
                )
                
                if "error" not in metrics:
                    prompt_results.append(metrics)
            
            # Calculate averages if we have valid results
            if prompt_results:
                avg_time = sum(res["generation_time_seconds"] for res in prompt_results) / len(prompt_results)
                avg_tokens_per_sec = sum(res["tokens_per_second"] for res in prompt_results) / len(prompt_results)
                
                results.append({
                    "prompt_length": prompt_length,
                    "avg_generation_time": avg_time,
                    "avg_tokens_per_second": avg_tokens_per_sec
                })
        
        # Create a table of performance results
        if results:
            table = Table(title="Inference Performance")
            table.add_column("Prompt Length", style="cyan")
            table.add_column("Avg. Generation Time", style="green")
            table.add_column("Tokens/Second", style="magenta")
            
            for res in results:
                table.add_row(
                    str(res["prompt_length"]),
                    f"{res['avg_generation_time']:.2f}s",
                    f"{res['avg_tokens_per_second']:.2f}"
                )
            
            console.print(table)
            
            # Calculate overall averages
            overall_avg_time = sum(res["avg_generation_time"] for res in results) / len(results)
            overall_avg_tokens = sum(res["avg_tokens_per_second"] for res in results) / len(results)
            
            return {
                "device": device,
                "prompt_results": results,
                "overall_avg_generation_time": overall_avg_time,
                "overall_avg_tokens_per_second": overall_avg_tokens
            }
        else:
            logger.warning("No valid performance results collected")
            return {"error": "No valid results"}
    except Exception as e:
        logger.error(f"Error measuring performance: {str(e)}", exc_info=True)
        return {"error": str(e)}

def test_domain_expertise(model, tokenizer):
    """Test model's expertise across different domains."""
    test_prompts = {
        "Customer Experience": [
            "The trouble with most customer experience metrics is that",
            "A truly effective Voice of Customer program should focus on",
            "The biggest misconception about customer journey mapping is"
        ],
        "Artificial Intelligence": [
            "Large Language Models are often praised, but their critical flaw is",
            "The ethical implications of AI deployment in customer service are",
            "Prompt engineering is becoming a valuable skill because"
        ],
        "Machine Learning": [
            "The most overrated machine learning technique that executives love is",
            "The gap between academic machine learning and production systems exists because",
            "Feature engineering remains important despite deep learning because"
        ]
    }
    
    results = {}
    all_metrics = []
    
    for domain, prompts in test_prompts.items():
        domain_results = []
        domain_metrics = []
        console.rule(f"Testing {domain} Expertise")
        
        for prompt in prompts:
            console.print(f"[bold blue]Prompt:[/bold blue] {prompt}")
            
            generated_texts, metrics = generate_text(model, tokenizer, prompt)
            generated = generated_texts[0]
            domain_results.append(generated)
            domain_metrics.append(metrics)
            
            # Extract the model's continuation (remove the prompt)
            continuation = generated[len(prompt):].strip()
            
            print(Panel(
                continuation,
                title="Generated Response",
                border_style="green"
            ))
            
            # Display metrics for this generation
            if "error" not in metrics:
                console.print(f"[dim]Generation time: {metrics['generation_time_seconds']:.2f}s, "
                             f"Tokens/sec: {metrics['tokens_per_second']:.2f}[/dim]")
            
            print("\n")
        
        results[domain] = domain_results
        all_metrics.append({"domain": domain, "metrics": domain_metrics})
    
    return results, all_metrics

def find_checkpoints(output_dir):
    """Find available checkpoints in the output directory."""
    try:
        checkpoints = []
        
        # Check for final model
        final_dir = os.path.join(output_dir, "final")
        if os.path.exists(final_dir):
            checkpoints.append(("final", final_dir))
        
        # Check for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(output_dir) 
                         if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
        
        # Sort checkpoints by step number
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        
        # Add checkpoints to the list
        for checkpoint in checkpoint_dirs:
            checkpoint_dir = os.path.join(output_dir, checkpoint)
            checkpoints.append((checkpoint, checkpoint_dir))
        
        return checkpoints
    except Exception as e:
        logger.error(f"Error finding checkpoints: {str(e)}")
        return []

def main():
    """Main function to test the fine-tuned model."""
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--model-dir", default="outputs/finetune/final", 
                      help="Directory containing the model to test")
    parser.add_argument("--checkpoint", default=None,
                      help="Specific checkpoint to test (e.g., 'checkpoint-500' or 'final')")
    parser.add_argument("--output-dir", default="outputs/finetune",
                      help="Output directory where checkpoints are stored")
    parser.add_argument("--performance-test", action="store_true",
                      help="Run performance tests")
    args = parser.parse_args()
    
    console.rule("[bold]Testing Fine-Tuned Model[/bold]")
    
    # Determine which model to test
    model_dir = args.model_dir
    output_dir = args.output_dir
    
    # Find available checkpoints
    if args.checkpoint or not os.path.exists(model_dir):
        checkpoints = find_checkpoints(output_dir)
        
        if not checkpoints:
            logger.error(f"No checkpoints found in {output_dir}")
            return
        
        console.print(f"[bold]Found {len(checkpoints)} checkpoints:[/bold]")
        for i, (name, path) in enumerate(checkpoints):
            console.print(f"{i+1}. {name}")
        
        # Select checkpoint if specified
        if args.checkpoint:
            matching = [c for c in checkpoints if args.checkpoint in c[0]]
            if matching:
                model_dir = matching[0][1]
                console.print(f"[bold]Testing checkpoint: {matching[0][0]}[/bold]")
            else:
                logger.error(f"Checkpoint {args.checkpoint} not found")
                return
        else:
            # Use the latest checkpoint
            model_dir = checkpoints[-1][1]
            console.print(f"[bold]Testing latest checkpoint: {checkpoints[-1][0]}[/bold]")
    
    # Load metadata
    metadata = load_metadata(model_dir)
    if metadata:
        console.print("[bold]Model Metadata:[/bold]")
        for key, value in metadata.items():
            if isinstance(value, dict):
                console.print(f"[bold]{key}:[/bold]")
                for k, v in value.items():
                    console.print(f"  {k}: {v}")
            else:
                console.print(f"[bold]{key}:[/bold] {value}")
        console.print("\n")
    
    # Load model
    model, tokenizer, device = load_model(model_dir)
    if model is None or tokenizer is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run performance tests if requested
    if args.performance_test:
        console.rule("[bold]Performance Testing[/bold]")
        performance_results = measure_inference_performance(model, tokenizer, device)
        
        if "error" not in performance_results:
            console.print(Panel(
                f"Device: {performance_results['device']}\n"
                f"Average Generation Time: {performance_results['overall_avg_generation_time']:.2f}s\n"
                f"Average Tokens/Second: {performance_results['overall_avg_tokens_per_second']:.2f}",
                title="Performance Summary",
                border_style="blue"
            ))
    
    # Test domain expertise
    console.rule("[bold]Domain Expertise Testing[/bold]")
    results, metrics = test_domain_expertise(model, tokenizer)
    
    # Update metadata with test results if possible
    try:
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_device": device,
            "average_generation_time": np.mean([m["generation_time_seconds"] for dom in metrics 
                                              for m in dom["metrics"] if "generation_time_seconds" in m]),
            "average_tokens_per_second": np.mean([m["tokens_per_second"] for dom in metrics 
                                                for m in dom["metrics"] if "tokens_per_second" in m]),
        }
        
        # Only update if we have permission to write to the directory
        test_results_path = os.path.join(model_dir, "test_results.json")
        try:
            with open(test_results_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Saved test results to {test_results_path}")
        except Exception as e:
            logger.warning(f"Could not save test results: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}")
    
    console.rule("[bold]Testing Complete[/bold]")

if __name__ == "__main__":
    main()
