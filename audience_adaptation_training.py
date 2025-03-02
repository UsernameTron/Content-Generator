#!/usr/bin/env python3
"""
Audience Adaptation Training Script for C. Pete Connor Model

This script trains the model to adapt content to different audience types:
- Executive: Concise, strategic, business-oriented content with metrics
- Practitioner: Technical, implementation-focused content with specific details
- General Public: Accessible, simplified content with analogies and examples

The script uses audience-specific conditioning tokens and a specialized LoRA adapter.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import wandb
from peft import PeftModel, LoraConfig, get_peft_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/audience_adaptation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Environment variables for Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Audience conditioning tokens
AUDIENCE_TOKENS = {
    "executive": "[EXEC]",
    "practitioner": "[PRAC]",
    "general": "[GEN]"
}

class AudienceAdaptationDataset(Dataset):
    """
    Dataset for audience adaptation training using examples formatted for different audience types.
    """
    def __init__(self, data_file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load the audience examples
        logger.info(f"Loading audience examples from {data_file_path}")
        with open(data_file_path, "r") as f:
            data = json.load(f)
        
        # Process each example
        for item in data:
            # Extract examples for each audience type
            for audience_type in ["executive", "practitioner", "general"]:
                if audience_type in item:
                    text = item[audience_type]
                    
                    # Add to examples list
                    self.examples.append({
                        "text": text,
                        "audience": audience_type
                    })
        
        logger.info(f"Loaded {len(self.examples)} audience-specific examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create item with input_ids and attention_mask
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone()
        }
        
        return item

def calculate_jargon_density(text: str, jargon_list: List[str]) -> float:
    """
    Calculate the density of technical jargon in the text.
    
    Args:
        text: The text to analyze
        jargon_list: List of technical terms/jargon to check for
        
    Returns:
        float: Jargon density (percentage of words that are jargon)
    """
    words = text.lower().split()
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    jargon_count = sum(1 for word in words if word in jargon_list)
    return jargon_count / total_words * 100

def calculate_complexity_score(text: str) -> float:
    """
    Calculate a complexity score based on sentence length, word length, and structure.
    
    Args:
        text: The text to analyze
        
    Returns:
        float: Complexity score (higher means more complex)
    """
    # Split text into sentences and words
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if not sentences:
        return 0.0
    
    # Calculate average sentence length
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Calculate average word length
    words = [w for s in sentences for w in s.split()]
    if not words:
        return 0.0
    
    avg_word_length = sum(len(w) for w in words) / len(words)
    
    # Calculate complexity score (simple weighted formula)
    # Higher values indicate more complex text
    complexity_score = (0.7 * avg_sentence_length) + (0.3 * avg_word_length)
    
    return complexity_score

def add_audience_tokens_to_tokenizer(tokenizer):
    """
    Add audience-specific tokens to the tokenizer vocabulary.
    
    Args:
        tokenizer: Hugging Face tokenizer
        
    Returns:
        tokenizer: Updated tokenizer with audience tokens
    """
    special_tokens = {"additional_special_tokens": list(AUDIENCE_TOKENS.values())}
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer

def train_audience_adaptation_model(
    base_model_path: str,
    data_path: str,
    output_dir: str,
    use_lora: bool = True,
    wandb_logging: bool = True,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
):
    """
    Train a model for audience adaptation using conditioning tokens.
    
    Args:
        base_model_path: Path to the base model or adapter
        data_path: Path to audience examples JSON
        output_dir: Directory to save the trained model
        use_lora: Whether to use LoRA adapters
        wandb_logging: Whether to log to Weights & Biases
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for training
    """
    logger.info("Initializing audience adaptation training")
    
    # Initialize W&B if requested
    if wandb_logging:
        wandb.init(
            project="pete-connor-cx-ai-expert",
            name=f"audience-adaptation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "base_model": base_model_path,
                "use_lora": use_lora,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
            }
        )
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Add audience tokens to tokenizer
    tokenizer = add_audience_tokens_to_tokenizer(tokenizer)
    
    # Load model
    logger.info(f"Loading model from {base_model_path}")
    if os.path.exists(os.path.join(base_model_path, "adapter_config.json")):
        # This is a LoRA adapter path
        logger.info("Detected LoRA adapter path")
        # Load the original base model from the adapter config
        from peft import PeftConfig
        adapter_config = PeftConfig.from_pretrained(base_model_path)
        original_model_path = adapter_config.base_model_name_or_path
        
        logger.info(f"Loading original model: {original_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        
        # Load the adapter
        logger.info(f"Loading adapter from {base_model_path}")
        model = PeftModel.from_pretrained(model, base_model_path)
    else:
        # This is a regular model path
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
    
    # Resize token embeddings to account for new audience tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA if requested
    if use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    # Prepare dataset
    logger.info("Preparing audience adaptation dataset")
    dataset = AudienceAdaptationDataset(data_path, tokenizer)
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Define metrics for evaluation
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        # Convert predictions to text
        prediction_texts = [tokenizer.decode(pred) for pred in predictions]
        
        # Define jargon lists for different audience types
        technical_jargon = [
            "implementation", "architecture", "framework", "api", "endpoint",
            "infrastructure", "deployment", "protocol", "algorithm", "middleware",
            "authentication", "authorization", "encryption", "tokenization", "validation",
            "regression", "configuration", "parameter", "optimization", "throughput",
            "latency", "bandwidth", "scalability", "resiliency", "redundancy"
        ]
        
        # Calculate metrics
        complexity_scores = [calculate_complexity_score(text) for text in prediction_texts]
        jargon_densities = [calculate_jargon_density(text, technical_jargon) for text in prediction_texts]
        
        # Calculate averages per audience type
        exec_indices = [i for i, text in enumerate(prediction_texts) if "[EXEC]" in text]
        prac_indices = [i for i, text in enumerate(prediction_texts) if "[PRAC]" in text]
        gen_indices = [i for i, text in enumerate(prediction_texts) if "[GEN]" in text]
        
        # Calculate metrics by audience type
        exec_complexity = sum(complexity_scores[i] for i in exec_indices) / max(len(exec_indices), 1)
        prac_complexity = sum(complexity_scores[i] for i in prac_indices) / max(len(prac_indices), 1)
        gen_complexity = sum(complexity_scores[i] for i in gen_indices) / max(len(gen_indices), 1)
        
        exec_jargon = sum(jargon_densities[i] for i in exec_indices) / max(len(exec_indices), 1)
        prac_jargon = sum(jargon_densities[i] for i in prac_indices) / max(len(prac_indices), 1)
        gen_jargon = sum(jargon_densities[i] for i in gen_indices) / max(len(gen_indices), 1)
        
        # Return metrics
        return {
            "exec_complexity": exec_complexity,
            "prac_complexity": prac_complexity,
            "gen_complexity": gen_complexity,
            "exec_jargon_density": exec_jargon,
            "prac_jargon_density": prac_jargon,
            "gen_jargon_density": gen_jargon,
            "complexity_ratio_exec_gen": exec_complexity / max(gen_complexity, 0.1),
            "complexity_ratio_prac_gen": prac_complexity / max(gen_complexity, 0.1),
            "jargon_ratio_prac_gen": prac_jargon / max(gen_jargon, 0.1)
        }
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=50,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(device == "cuda"),
        bf16=False,
        report_to="wandb" if wandb_logging else "none",
        disable_tqdm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Start training
    logger.info("Starting audience adaptation training")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Close W&B if it was used
    if wandb_logging:
        wandb.finish()
    
    logger.info("Audience adaptation training complete")
    return model, tokenizer

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Audience Adaptation Training for C. Pete Connor Model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/finetune/final",
        help="Path to the base model or adapter"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/audience_examples.json",
        help="Path to audience examples (JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/audience_adaptation",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA adapters"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check if audience data file exists
    if not os.path.exists(args.data):
        logger.error(f"Audience examples file not found: {args.data}")
        logger.info("Please create an audience examples file before running this script.")
        sys.exit(1)
    
    # Train the model
    train_audience_adaptation_model(
        base_model_path=args.base_model,
        data_path=args.data,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        wandb_logging=not args.no_wandb,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

if __name__ == "__main__":
    main()
