#!/usr/bin/env python3
"""
Simplified training script for running on Apple Silicon
with optimized memory settings and reduced complexity.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
import wandb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Environment variable configuration for Apple Silicon
# Set these before importing torch or any other libraries
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit for memory allocations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for unsupported operations
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism

class StyleDataset(torch.utils.data.Dataset):
    """
    Simple dataset for fine-tuning the model.
    """
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.examples = []
        
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        logger.info(f"Loading {len(lines)} examples from {file_path}")
        
        for line in lines:
            try:
                item = json.loads(line)
                
                # Extract text from various formats
                if "messages" in item:
                    # ChatML format
                    messages = item["messages"]
                    text = ""
                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        text += f"{role.upper()}: {content}\n\n"
                elif "text" in item:
                    # Simple text format
                    text = item["text"]
                elif "prompt" in item and "response" in item:
                    # Prompt-response format
                    text = f"USER: {item['prompt']}\n\nASSISTANT: {item['response']}"
                else:
                    logger.warning(f"Unknown data format: {item.keys()}")
                    continue
                
                # Tokenize the text
                encodings = tokenizer(
                    text, 
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                self.examples.append({
                    "input_ids": encodings["input_ids"][0],
                    "attention_mask": encodings["attention_mask"][0],
                })
                
            except Exception as e:
                logger.warning(f"Error processing line: {e}")
        
        logger.info(f"Successfully loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    """
    Main training function with simplified process.
    """
    # Load configuration
    config_file = "finetune_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Set random seed
    set_seed(42)  # Use a fixed seed for reproducibility
    
    # Initialize wandb
    logger.info("Initializing W&B")
    wandb.init(
        project="pete-connor-cx-ai-expert",
        name=f"simplified-training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config
    )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'],
        use_fast=True,
        trust_remote_code=False,
    )
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading base model: {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    
    # Skip 8-bit preparation, just use the model as-is with LoRA
    logger.info("Configuring model for efficient training on Apple Silicon")
    
    # Configure LoRA
    logger.info("Setting up LoRA configuration")
    lora_config = LoraConfig(
        r=config["training_config"]["lora_r"],
        lora_alpha=config["training_config"]["lora_alpha"],
        target_modules=config["training_config"]["target_modules"],
        lora_dropout=config["training_config"]["lora_dropout"],
        bias=config["training_config"]["bias"],
        task_type=config["training_config"]["task_type"],
    )
    
    # Get PEFT model
    logger.info("Applying LoRA adapters to model")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("Preparing datasets")
    train_dataset = StyleDataset(
        config["data_config"]["train_file"],
        tokenizer,
        max_length=config["data_config"]["max_seq_length"]
    )
    
    eval_dataset = StyleDataset(
        config["data_config"]["validation_file"],
        tokenizer,
        max_length=config["data_config"]["max_seq_length"]
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create training arguments
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=config["training_config"]["training_args"]["output_dir"],
        num_train_epochs=config["training_config"]["training_args"]["num_train_epochs"],
        per_device_train_batch_size=1,  # Hard-coded for Apple Silicon
        per_device_eval_batch_size=1,   # Hard-coded for Apple Silicon
        gradient_accumulation_steps=16, # Hard-coded for Apple Silicon
        evaluation_strategy="steps",
        eval_steps=config["training_config"]["training_args"]["eval_steps"],
        logging_dir="./logs",
        logging_steps=config["training_config"]["training_args"]["logging_steps"],
        save_strategy="steps",
        save_steps=config["training_config"]["training_args"]["save_steps"],
        learning_rate=config["training_config"]["training_args"]["learning_rate"],
        weight_decay=config["training_config"]["training_args"]["weight_decay"],
        fp16=False,  # Disable mixed precision for MPS
        bf16=False,  # Disable bf16 for MPS
        max_grad_norm=config["training_config"]["training_args"]["max_grad_norm"],
        max_steps=config["training_config"]["training_args"]["max_steps"],
        warmup_ratio=config["training_config"]["training_args"]["warmup_ratio"],
        group_by_length=False,  # Disable for simplicity
        report_to="wandb",
        run_name=f"simplified-training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_total_limit=3,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting model fine-tuning")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        
    # Save the final model
    logger.info("Saving model to outputs/finetune/final")
    trainer.save_model("outputs/finetune/final")
    logger.info("Model saving complete.")
    
    # Save the PEFT adapter config
    logger.info("Saving adapter config")
    model.save_pretrained("outputs/finetune/final")
    
    # Close wandb run
    wandb.finish()
    
    logger.info("Fine-tuning completed")

if __name__ == "__main__":
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs/finetune", exist_ok=True)
    
    # Start training
    print("=" * 80)
    print(f"Starting simplified Apple Silicon training at {datetime.now()}")
    print("=" * 80)
    
    start_time = time.time()
    main()
    end_time = time.time()
    
    print("=" * 80)
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
    print("=" * 80)
