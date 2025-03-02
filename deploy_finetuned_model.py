#!/usr/bin/env python3
"""
Deployment utility for the fine-tuned C. Pete Connor model.

This script prepares the model for deployment by:
1. Converting the fine-tuned model to a deployment-ready format
2. Creating a simple inference API endpoint
3. Providing model metadata and usage examples
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime
import logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
import torch

# Try to import required libraries, install if missing
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
except ImportError:
    Console().print("[yellow]Installing required dependencies...[/yellow]")
    os.system(f"{sys.executable} -m pip install transformers peft")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class ModelDeployer:
    def __init__(self, adapter_path, output_dir, merge_adapters=False):
        self.adapter_path = adapter_path
        self.output_dir = output_dir
        self.merge_adapters = merge_adapters
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self):
        """Determine the device to use"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """Load the model with adapter weights"""
        # Set environment variables for Apple Silicon
        if self.device == "mps":
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load adapter config
        console.print("[bold cyan]Loading adapter configuration...[/bold cyan]")
        adapter_config = PeftConfig.from_pretrained(self.adapter_path)
        
        # Load base model and tokenizer
        console.print(f"[bold cyan]Loading base model: {adapter_config.base_model_name_or_path}[/bold cyan]")
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
            trust_remote_code=False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_config.base_model_name_or_path)
        
        # Load adapters
        console.print("[bold cyan]Loading LoRA adapters...[/bold cyan]")
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        # Optionally merge adapters with base model
        if self.merge_adapters:
            console.print("[bold yellow]Merging adapters with base model...[/bold yellow]")
            model = model.merge_and_unload()
        
        console.print(f"[bold green]Successfully loaded model on {self.device} device[/bold green]")
        
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer
    
    def create_deployment_package(self):
        """Create a deployment package with the model, tokenizer, and metadata"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save model metadata
        metadata = {
            "model_name": "C. Pete Connor Fine-tuned Model",
            "base_model": self.model.config.name_or_path if hasattr(self.model, "config") else "N/A",
            "adapter_path": self.adapter_path,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "merged_adapters": self.merge_adapters,
            "device": self.device,
            "usage_format": "<|prompter|>{YOUR_PROMPT}<|assistant|>",
            "response_format": "The model will respond after the <|assistant|> token",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "max_length": 512
            }
        }
        
        with open(os.path.join(self.output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save example inference code
        example_code = '''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def load_model(model_path):
    """Load the C. Pete Connor fine-tuned model"""
    # Set environment variables for Apple Silicon (if applicable)
    import os
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device="cpu", temperature=0.7):
    """Generate a response from the model"""
    # Format the prompt with special tokens
    formatted_prompt = f"<|prompter|>{prompt}<|assistant|>"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            generation_config=GenerationConfig(
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2,
                max_length=512,
                do_sample=True
            )
        )
    
    # Decode and clean response
    full_response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[1].strip()
        # Clean up any trailing special tokens
        if "<|" in response:
            response = response.split("<|")[0].strip()
    else:
        response = full_response
    
    return response

# Example usage
if __name__ == "__main__":
    model_path = "./deployed_model"  # Path to the deployed model
    model, tokenizer, device = load_model(model_path)
    
    # Example prompt
    prompt = "Write a compelling product description for a new smartphone with advanced camera features."
    
    # Generate response
    response = generate_response(model, tokenizer, prompt, device)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
'''
        
        with open(os.path.join(self.output_dir, "example_inference.py"), "w") as f:
            f.write(example_code)
        
        # Save README with usage instructions
        readme = f'''# C. Pete Connor Fine-tuned Model

## Model Information
- **Model Name**: C. Pete Connor Fine-tuned Model
- **Base Model**: {self.model.config.name_or_path if hasattr(self.model, "config") else "N/A"}
- **Created At**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Usage Instructions

### Loading the Model
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variables for Apple Silicon (if applicable)
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "./deployed_model",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("./deployed_model")

# Format your prompt
prompt = "Write a compelling product description for a new smartphone."
formatted_prompt = f"<|prompter|>{prompt}<|assistant|>"

# Generate
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)

# Decode and extract response
response = tokenizer.decode(outputs[0])
assistant_response = response.split("<|assistant|>")[1].strip()
print(assistant_response)
```

## Examples
See `example_inference.py` for a complete working example.

## Recommended Parameters
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Repetition Penalty**: 1.2
- **Max Length**: 512

## Contact
For questions or support, contact the model developer.
'''
        
        with open(os.path.join(self.output_dir, "README.md"), "w") as f:
            f.write(readme)
        
        console.print(f"[bold green]Successfully created deployment package in {self.output_dir}[/bold green]")
    
    def save_deployment_model(self):
        """Save the model for deployment"""
        console.print("[bold cyan]Saving model for deployment...[/bold cyan]")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Saving model...", total=3)
            
            # Save model
            self.model.save_pretrained(self.output_dir)
            progress.update(task, advance=1)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.output_dir)
            progress.update(task, advance=1)
            
            # Copy config files
            source_configs = [
                os.path.join(self.adapter_path, "adapter_config.json"),
                os.path.join(self.adapter_path, "special_tokens_map.json"),
                os.path.join(self.adapter_path, "tokenizer_config.json")
            ]
            
            for config_file in source_configs:
                if os.path.exists(config_file):
                    shutil.copy(config_file, self.output_dir)
            
            progress.update(task, advance=1)
        
        console.print(f"[bold green]Successfully saved model to {self.output_dir}[/bold green]")
    
    def deploy(self):
        """Run the deployment process"""
        console.rule("[bold blue]C. Pete Connor Model Deployment")
        
        try:
            # Load the model
            self.load_model()
            
            # Create deployment package
            self.create_deployment_package()
            
            # Save the model
            self.save_deployment_model()
            
            # Display successful deployment message
            console.print(Panel(
                f"[bold green]Model successfully deployed to: {self.output_dir}[/bold green]\n\n"
                f"The deployment package includes:\n"
                f"- Model weights and configuration\n"
                f"- Tokenizer and special tokens\n"
                f"- README with usage instructions\n"
                f"- Example inference code\n"
                f"- Model metadata\n\n"
                f"You can now use the model in your applications by loading it from this directory.",
                title="Deployment Complete",
                border_style="green"
            ))
            
            return True
        except Exception as e:
            console.print(f"[bold red]Error during deployment: {str(e)}[/bold red]")
            logger.error(f"Deployment failed: {str(e)}", exc_info=True)
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy the fine-tuned C. Pete Connor model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="outputs/finetune/final",
        help="Path to the LoRA adapter folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployed_model",
        help="Output directory for the deployed model"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge adapters with the base model for deployment"
    )
    
    args = parser.parse_args()
    
    deployer = ModelDeployer(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        merge_adapters=args.merge
    )
    
    deployer.deploy()

if __name__ == "__main__":
    main()
