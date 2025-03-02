#!/usr/bin/env python3
"""
Anti-Pattern Training Script for C. Pete Connor Model

This script implements a negative reinforcement approach to eliminate AI-typical
phrasing from model outputs. It works as a complementary process to the main
fine-tuning, focusing on penalizing unwanted patterns during training.

Features:
1. Detection and penalization of banned phrases
2. Identification of symmetric sentence structures
3. Penalization of formulaic transitions
4. Custom loss function that incorporates pattern penalties
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
        logging.FileHandler(f"logs/anti_pattern_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Environment variables for Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define AI-typical phrases to penalize
AI_TYPICAL_PHRASES = [
    "game changer",
    "here's the kicker",
    "as an ai",
    "as an assistant",
    "i'm happy to help",
    "i'd be happy to",
    "i'd be glad to",
    "thanks for asking",
    "i hope this helps",
    "let me know if you need anything else",
    "hope this answers your question",
    "let's dive in",
    "let's break this down",
    "breaking it down",
    "unpacking this",
    "simply put",
    "to put it simply",
    "in a nutshell",
    "the bottom line is",
    "absolutely",
    "certainly",
    "definitely",
    "without a doubt",
    "first and foremost",
    "it's worth noting",
    "it's important to note",
    "it's crucial to understand",
]

# Define formulaic transitions to penalize
FORMULAIC_TRANSITIONS = [
    "firstly",
    "secondly",
    "thirdly",
    "lastly",
    "in conclusion",
    "to summarize",
    "to sum it up",
    "in summary",
    "to wrap up",
    "moving on to",
    "turning to",
    "shifting gears to",
    "on one hand",
    "on the other hand",
    "furthermore",
    "moreover",
    "additionally",
    "in addition",
]

class AntiPatternDataset(Dataset):
    """
    Dataset that includes both positive examples and negative (anti-pattern) examples
    with appropriate labels for contrastive learning.
    """
    def __init__(self, positive_file_path: str, negative_file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.labels = []
        
        # Load positive examples (good content)
        logger.info(f"Loading positive examples from {positive_file_path}")
        with open(positive_file_path, "r") as f:
            positive_data = json.load(f)
            for item in positive_data:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                elif isinstance(item, str):
                    text = item
                else:
                    continue
                
                self.examples.append(text)
                self.labels.append(0)  # 0 indicates positive example
        
        # Load or generate negative examples (content with AI-typical patterns)
        if os.path.exists(negative_file_path):
            logger.info(f"Loading negative examples from {negative_file_path}")
            with open(negative_file_path, "r") as f:
                negative_data = json.load(f)
                for item in negative_data:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                    elif isinstance(item, str):
                        text = item
                    else:
                        continue
                    
                    self.examples.append(text)
                    self.labels.append(1)  # 1 indicates negative example
        else:
            logger.warning(f"Negative examples file {negative_file_path} not found.")
            logger.info("Generating synthetic negative examples from positive samples")
            
            # Generate synthetic negative examples by adding AI-typical phrases
            # to copies of the positive examples
            negative_examples = []
            for text in self.examples[:]:
                # Add 2-3 random AI-typical phrases
                num_phrases = np.random.randint(2, 4)
                phrases = np.random.choice(AI_TYPICAL_PHRASES, num_phrases, replace=False)
                
                sentences = text.split('.')
                for phrase in phrases:
                    if len(sentences) > 3:
                        # Insert phrase into a random sentence
                        idx = np.random.randint(1, len(sentences) - 1)
                        sentences[idx] = f" {phrase}, {sentences[idx].strip()}"
                
                # Add 1-2 formulaic transitions
                num_transitions = np.random.randint(1, 3)
                transitions = np.random.choice(FORMULAIC_TRANSITIONS, num_transitions, replace=False)
                
                for transition in transitions:
                    if len(sentences) > 3:
                        idx = np.random.randint(1, len(sentences) - 1)
                        sentences[idx] = f" {transition}, {sentences[idx].strip()}"
                
                negative_text = '. '.join(sentences)
                negative_examples.append(negative_text)
                
            # Add generated negative examples to dataset
            for text in negative_examples:
                self.examples.append(text)
                self.labels.append(1)  # 1 indicates negative example
            
            # Save generated negative examples for future use
            with open(negative_file_path, "w") as f:
                json.dump(negative_examples, f, indent=2)
        
        logger.info(f"Dataset loaded with {len(self.examples)} examples "
                   f"({sum(1 for l in self.labels if l == 0)} positive, "
                   f"{sum(1 for l in self.labels if l == 1)} negative)")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        label = self.labels[idx]
        
        # Format with appropriate special tokens for discrimination task
        if label == 0:  # Positive example
            formatted_text = f"<|good_text|>{text}"
        else:  # Negative example
            formatted_text = f"<|bad_text|>{text}"
        
        # Tokenize the text
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to appropriate format for training
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        
        return item

def detect_ai_typical_phrases(text: str) -> List[Tuple[str, int]]:
    """
    Detect AI-typical phrases in the text and return them with their positions.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of tuples containing (phrase, position)
    """
    text_lower = text.lower()
    matches = []
    
    for phrase in AI_TYPICAL_PHRASES:
        start_pos = 0
        while start_pos < len(text_lower):
            pos = text_lower.find(phrase, start_pos)
            if pos == -1:
                break
            matches.append((phrase, pos))
            start_pos = pos + len(phrase)
    
    return matches

def detect_formulaic_transitions(text: str) -> List[Tuple[str, int]]:
    """
    Detect formulaic transitions in the text and return them with their positions.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of tuples containing (transition, position)
    """
    text_lower = text.lower()
    matches = []
    
    for transition in FORMULAIC_TRANSITIONS:
        start_pos = 0
        while start_pos < len(text_lower):
            pos = text_lower.find(transition, start_pos)
            if pos == -1:
                break
            matches.append((transition, pos))
            start_pos = pos + len(transition)
    
    return matches

def detect_symmetric_structures(text: str) -> List[str]:
    """
    Detect symmetric sentence structures in the text.
    
    This detects patterns like repeated sentence beginnings or parallel structures.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of detected symmetric structure patterns
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 3:
        return []
    
    patterns = []
    
    # Check for repeated sentence beginnings
    beginnings = [s.split()[:3] for s in sentences if len(s.split()) >= 3]
    beginning_counts = {}
    
    for i, beginning in enumerate(beginnings):
        beginning_tuple = tuple(beginning)
        if beginning_tuple in beginning_counts:
            patterns.append(f"Repeated beginning: {' '.join(beginning)}")
        else:
            beginning_counts[beginning_tuple] = i
    
    # Check for parallel structures (sentences of similar length in sequence)
    lengths = [len(s.split()) for s in sentences]
    for i in range(len(lengths) - 2):
        if abs(lengths[i] - lengths[i+1]) <= 2 and abs(lengths[i+1] - lengths[i+2]) <= 2:
            patterns.append(f"Parallel structure at sentences {i+1}-{i+3}")
    
    return patterns

class AntiPatternLoss(torch.nn.Module):
    """
    Custom loss function that penalizes outputs containing AI-typical patterns.
    """
    def __init__(self, tokenizer, base_loss_fn=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_loss_fn = base_loss_fn
        self.phrase_penalty = 0.2
        self.transition_penalty = 0.1
        self.symmetric_penalty = 0.15
    
    def forward(self, logits, labels, input_ids=None, attention_mask=None):
        # Calculate the base loss
        base_loss = self.base_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # If we don't have input_ids for penalty calculation, just return base loss
        if input_ids is None:
            return base_loss
        
        # Calculate penalties based on the predicted outputs
        batch_size = input_ids.size(0)
        penalty = 0.0
        
        # Get the most likely output sequence for each item in the batch
        with torch.no_grad():
            outputs = torch.argmax(logits, dim=-1)
        
        # Process each item in the batch
        for i in range(batch_size):
            # Decode tokens to text
            text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # Apply penalties if this is labeled as positive text but contains negative patterns
            if labels[i][0] == 0:  # Positive example should not have AI patterns
                # Check for AI-typical phrases
                phrase_matches = detect_ai_typical_phrases(text)
                if phrase_matches:
                    penalty += self.phrase_penalty * len(phrase_matches) / batch_size
                
                # Check for formulaic transitions
                transition_matches = detect_formulaic_transitions(text)
                if transition_matches:
                    penalty += self.transition_penalty * len(transition_matches) / batch_size
                
                # Check for symmetric structures
                symmetric_patterns = detect_symmetric_structures(text)
                if symmetric_patterns:
                    penalty += self.symmetric_penalty * len(symmetric_patterns) / batch_size
        
        # Return combined loss
        return base_loss + penalty

def train_anti_pattern_model(
    base_model_path: str,
    positive_data_path: str,
    negative_data_path: str,
    output_dir: str,
    use_lora: bool = True,
    wandb_logging: bool = True,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
):
    """
    Train a model with anti-pattern detection to avoid AI-typical phrasing.
    
    Args:
        base_model_path: Path to the base model or adapter
        positive_data_path: Path to positive examples (JSON)
        negative_data_path: Path to negative examples (JSON), will be generated if not found
        output_dir: Directory to save the trained model
        use_lora: Whether to use LoRA adapters
        wandb_logging: Whether to log to Weights & Biases
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for training
    """
    logger.info("Initializing anti-pattern training")
    
    # Initialize W&B if requested
    if wandb_logging:
        wandb.init(
            project="pete-connor-cx-ai-expert",
            name=f"anti-pattern-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    
    # Ensure special tokens for our task
    special_tokens = {"additional_special_tokens": ["<|good_text|>", "<|bad_text|>"]}
    tokenizer.add_special_tokens(special_tokens)
    
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
    
    # Resize token embeddings to account for new special tokens
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
    logger.info("Preparing anti-pattern dataset")
    dataset = AntiPatternDataset(positive_data_path, negative_data_path, tokenizer)
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create custom loss function
    custom_loss_fn = AntiPatternLoss(tokenizer)
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="no",
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
        remove_unused_columns=False,
    )
    
    # Create trainer with custom loss function
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            labels = inputs.get("labels")
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.get("logits")
            
            loss = custom_loss_fn(logits, labels, input_ids, attention_mask)
            
            return (loss, outputs) if return_outputs else loss
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting anti-pattern training")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Close W&B if it was used
    if wandb_logging:
        wandb.finish()
    
    logger.info("Anti-pattern training complete")
    return model, tokenizer

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Anti-Pattern Training for C. Pete Connor Model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/finetune/final",
        help="Path to the base model or adapter"
    )
    parser.add_argument(
        "--positive_data",
        type=str,
        default="dataset/positive_examples.json",
        help="Path to positive examples (JSON)"
    )
    parser.add_argument(
        "--negative_data",
        type=str,
        default="dataset/negative_examples.json",
        help="Path to negative examples (JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/anti_pattern",
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
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check if dataset directories exist
    os.makedirs(os.path.dirname(args.positive_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.negative_data), exist_ok=True)
    
    # Check if positive data file exists
    if not os.path.exists(args.positive_data):
        logger.error(f"Positive data file not found: {args.positive_data}")
        logger.info("Please create a positive examples file before running this script.")
        sys.exit(1)
    
    # Train the model
    train_anti_pattern_model(
        base_model_path=args.base_model,
        positive_data_path=args.positive_data,
        negative_data_path=args.negative_data,
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
