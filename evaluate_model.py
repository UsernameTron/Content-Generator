#!/usr/bin/env python3
"""
Comprehensive evaluation script for the fine-tuned C. Pete Connor model
"""

import os
import torch
import logging
import json
import csv
import time
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel, PeftConfig

# Set up logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("evaluate_model")
console = Console()

# Environment variables for Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Test prompts across different categories
TEST_PROMPTS = [
    # Content Marketing
    {"category": "Marketing", "prompt": "Write a compelling Instagram caption for a new coffee shop that emphasizes community and quality beans."},
    {"category": "Marketing", "prompt": "Create a Twitter post announcing a limited-time sale on premium headphones."},
    {"category": "Marketing", "prompt": "Draft a LinkedIn post highlighting our company's commitment to sustainability initiatives."},
    
    # Copywriting
    {"category": "Copywriting", "prompt": "Write a product description for a luxury watch that emphasizes craftsmanship and heritage."},
    {"category": "Copywriting", "prompt": "Create compelling bullet points for the features section of a smart home security system."},
    {"category": "Copywriting", "prompt": "Write the opening paragraph for an email newsletter about summer travel destinations."},
    
    # Customer Service
    {"category": "Customer Service", "prompt": "Write a response to a customer who received a damaged product in the mail."},
    {"category": "Customer Service", "prompt": "Craft a reply to a customer asking about your return policy for clothing items."},
    {"category": "Customer Service", "prompt": "Create a message thanking a customer for their 5-star review and asking for a referral."},
]

def load_model(adapter_path):
    """Load the fine-tuned model with LoRA adapters"""
    logger.info(f"Loading adapter config from {adapter_path}")
    
    # Load the adapter config
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model
    logger.info(f"Loading base model: {adapter_config.base_model_name_or_path}")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        adapter_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_config.base_model_name_or_path)
    
    # Load the fine-tuned model with adapters
    logger.info("Loading LoRA adapters")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate a response using the fine-tuned model"""
    
    # Format the prompt
    formatted_prompt = f"<|prompter|>{prompt}<|assistant|>"
    
    # Prepare inputs and send to device
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    
    # Track generation start time
    start_time = time.time()
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                num_beams=1,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=False,
                do_sample=True
            ),
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Calculate generation time
    generation_time = time.time() - start_time
        
    # Get full response text
    full_response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[1].strip()
        # Clean up any trailing special tokens or unrelated text
        if "<|" in response:
            response = response.split("<|")[0].strip()
    else:
        response = "Model did not generate a proper response format. Raw output: " + full_response
    
    # Calculate tokens
    input_tokens = len(inputs.input_ids[0])
    output_tokens = len(output[0]) - input_tokens
    
    return {
        "response": response,
        "generation_time": generation_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / generation_time if generation_time > 0 else 0
    }

def evaluate_model(model, tokenizer):
    """Run evaluation on test prompts"""
    console.rule("[bold blue]Model Evaluation")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Evaluating model on test prompts...", total=len(TEST_PROMPTS))
        
        for test_case in TEST_PROMPTS:
            category = test_case["category"]
            prompt = test_case["prompt"]
            
            progress.update(task, description=f"[cyan]Testing {category} prompt...")
            
            # Generate response
            generation_result = generate_response(model, tokenizer, prompt)
            
            # Add to results
            results.append({
                "category": category,
                "prompt": prompt,
                "response": generation_result["response"],
                "generation_time": generation_result["generation_time"],
                "input_tokens": generation_result["input_tokens"],
                "output_tokens": generation_result["output_tokens"],
                "tokens_per_second": generation_result["tokens_per_second"]
            })
            
            progress.update(task, advance=1)
    
    return results

def display_results(results):
    """Display evaluation results in a table"""
    console.rule("[bold green]Evaluation Results")
    
    # Performance metrics table
    perf_table = Table(title="Performance Metrics")
    perf_table.add_column("Category", style="cyan")
    perf_table.add_column("Avg. Generation Time", style="green")
    perf_table.add_column("Avg. Output Tokens", style="green")
    perf_table.add_column("Avg. Tokens/Second", style="green")
    
    # Group by category
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {
                "count": 0,
                "total_time": 0,
                "total_tokens": 0
            }
        
        categories[cat]["count"] += 1
        categories[cat]["total_time"] += result["generation_time"]
        categories[cat]["total_tokens"] += result["output_tokens"]
    
    # Calculate averages for each category
    for cat, data in categories.items():
        avg_time = data["total_time"] / data["count"]
        avg_tokens = data["total_tokens"] / data["count"]
        avg_tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
        
        perf_table.add_row(
            cat,
            f"{avg_time:.2f}s",
            f"{avg_tokens:.1f}",
            f"{avg_tokens_per_sec:.1f}"
        )
    
    console.print(perf_table)
    
    # Individual results
    for i, result in enumerate(results):
        console.rule(f"[bold blue]Test Case {i+1}: {result['category']}")
        console.print(f"[bold cyan]Prompt:[/bold cyan] {result['prompt']}")
        console.print(f"[bold green]Response:[/bold green] {result['response']}")
        console.print(f"[bold yellow]Metrics:[/bold yellow] {result['generation_time']:.2f}s, {result['output_tokens']} tokens, {result['tokens_per_second']:.1f} tokens/sec")
        console.print()

def save_results(results, output_path):
    """Save evaluation results to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Save JSON
    json_path = os.path.join("evaluation_results", f"eval_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    csv_path = os.path.join("evaluation_results", f"eval_results_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {json_path} and {csv_path}")

def main():
    adapter_path = os.environ.get("ADAPTER_PATH", "outputs/finetune/final")
    
    # Print header
    console.rule("[bold blue]C. Pete Connor Model Evaluation")
    console.print(f"[bold]Evaluating fine-tuned model with adapters from:[/bold] {adapter_path}")
    
    # Load the model
    model, tokenizer = load_model(adapter_path)
    
    # Run evaluation
    results = evaluate_model(model, tokenizer)
    
    # Display results
    display_results(results)
    
    # Save results
    save_results(results, "evaluation_results")
    
    console.rule("[bold blue]Evaluation Complete")

if __name__ == "__main__":
    main()
