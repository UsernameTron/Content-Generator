#!/usr/bin/env python3
"""
Interactive test script for the fine-tuned C. Pete Connor model
"""

import os
import torch
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
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
logger = logging.getLogger("interactive_test")
console = Console()

# Environment variables for Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    console.rule("[bold green]Generating Response")
    
    # Format the prompt
    formatted_prompt = f"<|prompter|>{prompt}<|assistant|>"
    
    rprint(Panel(prompt, title="[bold cyan]Your Prompt", border_style="cyan"))
    console.print("[bold yellow]Thinking...[/bold yellow]")
    
    # Prepare inputs and send to device
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                num_beams=1,
                no_repeat_ngram_size=3,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=False,
                do_sample=True
            ),
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Get full response text
    full_response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[1].strip()
        # Clean up any trailing special tokens or unrelated text
        if "<|" in response:
            response = response.split("<|")[0].strip()
    else:
        response = "Model did not generate a proper response format. Here's the raw output:\n" + full_response
    
    rprint(Panel(response, title="[bold green]C. Pete Connor's Response", border_style="green"))
    
    return response

def interactive_mode(model, tokenizer):
    """Run an interactive session with the model"""
    console.rule("[bold blue]Interactive C. Pete Connor Model Test")
    console.print("[bold]Enter 'exit', 'quit', or 'q' to end the session[/bold]")
    
    while True:
        prompt = Prompt.ask("\n[bold cyan]Enter your prompt[/bold cyan]")
        
        if prompt.lower() in ['exit', 'quit', 'q']:
            break
            
        generate_response(model, tokenizer, prompt)

def main():
    adapter_path = os.environ.get("ADAPTER_PATH", "outputs/finetune/final")
    
    # Print header
    console.rule("[bold blue]C. Pete Connor Model Interactive Test")
    console.print(f"[bold]Loading fine-tuned model with adapters from:[/bold] {adapter_path}")
    
    # Load the model
    model, tokenizer = load_model(adapter_path)
    
    # Run interactive mode
    interactive_mode(model, tokenizer)
    
    console.rule("[bold blue]Test Complete")

if __name__ == "__main__":
    main()
