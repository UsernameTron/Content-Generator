"""
Fine-tuning script for C. Pete Connor's satirical tech expert style model.

This script fine-tunes a base language model with LoRA to generate content 
in C. Pete Connor's distinctive writing style.
"""

import os
import json
import logging
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Union

# Set numpy compatibility for wandb
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    HfArgumentParser,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import transformers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class StyleDataset(Dataset):
    """Dataset for fine-tuning with C. Pete Connor's writing style."""
    
    def __init__(self, tokenizer, file_path, block_size=512):
        """
        Initialize dataset from JSONL file.
        
        Args:
            tokenizer: Tokenizer for the model
            file_path: Path to the JSONL data file
            block_size: Maximum sequence length
        """
        self.examples = []
        logger.info(f"Loading data from {file_path}")
        
        with open(file_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # Handle the new data format with messages
                    if "messages" in item:
                        # Extract user prompt and assistant response from messages
                        messages = item["messages"]
                        conversation = ""
                        
                        for message in messages:
                            if message["role"] == "user":
                                conversation += f"USER: {message['content']}\n\n"
                            elif message["role"] == "assistant":
                                conversation += f"ASSISTANT: {message['content']}\n\n"
                        
                        text = conversation.strip()
                    # Handle old format or direct text
                    elif "text" in item:
                        text = item["text"]
                    # Handle prompt/response format
                    elif "prompt" in item and "response" in item:
                        text = f"USER: {item['prompt']}\n\nASSISTANT: {item['response']}"
                    else:
                        logger.warning(f"Unknown data format: {item.keys()}")
                        continue
                    
                    tokenized = tokenizer(
                        text,
                        truncation=True,
                        max_length=block_size,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    # Do NOT set requires_grad for input tensors, the model will handle this
                    self.examples.append({
                        "input_ids": tokenized["input_ids"][0],
                        "attention_mask": tokenized["attention_mask"][0],
                    })
                except Exception as e:
                    logger.warning(f"Error processing line: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class CustomStyleLoss(torch.nn.Module):
    """Custom loss function to enforce C. Pete Connor's writing style."""
    
    def __init__(
        self, 
        tokenizer, 
        penalized_phrases: List[str], 
        rewarded_phrases: List[str],
        penalty_weight: float = 0.5,
        reward_weight: float = 0.5
    ):
        """
        Initialize custom loss module.
        
        Args:
            tokenizer: Model tokenizer
            penalized_phrases: List of phrases to penalize
            rewarded_phrases: List of phrases to reward
            penalty_weight: Weight for the penalty component
            reward_weight: Weight for the reward component
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.base_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Tokenize phrases to penalize
        self.penalized_ids = [
            tokenizer.encode(phrase, add_special_tokens=False)
            for phrase in penalized_phrases
        ]
        
        # Tokenize phrases to reward
        self.rewarded_ids = [
            tokenizer.encode(phrase, add_special_tokens=False)
            for phrase in rewarded_phrases
        ]
        
        self.penalty_weight = penalty_weight
        self.reward_weight = reward_weight
    
    def forward(self, logits, labels):
        """
        Custom forward pass with style-enforcing loss.
        
        Args:
            logits: Model prediction logits
            labels: Target labels
            
        Returns:
            Loss with style enforcement components
        """
        # Standard cross-entropy loss
        base_loss = self.base_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Get predicted token IDs
        predicted = torch.argmax(logits, dim=-1)
        
        # Initialize penalty and reward tensors
        penalty = torch.zeros_like(base_loss)
        reward = torch.zeros_like(base_loss)
        
        # Convert to CPU for phrase matching
        predicted_cpu = predicted.cpu().numpy()
        
        # Apply penalties for generic phrases
        for phrase_ids in self.penalized_ids:
            for i in range(len(predicted_cpu) - len(phrase_ids) + 1):
                if np.array_equal(predicted_cpu[i:i+len(phrase_ids)], phrase_ids):
                    penalty[i:i+len(phrase_ids)] += self.penalty_weight
        
        # Apply rewards for stylistic phrases
        for phrase_ids in self.rewarded_ids:
            for i in range(len(predicted_cpu) - len(phrase_ids) + 1):
                if np.array_equal(predicted_cpu[i:i+len(phrase_ids)], phrase_ids):
                    reward[i:i+len(phrase_ids)] -= self.reward_weight
        
        # Combine loss components
        final_loss = base_loss + penalty - reward
        
        # Log components to W&B
        if wandb.run is not None:
            wandb.log({
                "base_loss": base_loss.mean().item(),
                "style_penalty": penalty.mean().item(),
                "style_reward": reward.mean().item(),
                "total_loss": final_loss.mean().item()
            })
        
        return final_loss.mean()

class CustomDataCollator:
    """
    Custom data collator that prepares tensors for the model.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mlm = False
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        
        # Set labels equal to input_ids
        labels = input_ids.clone()
        # Ignore padding token in loss calculation
        labels[labels == self.pad_token_id] = -100
        
        # Do NOT set requires_grad for input tensors, the model will handle this
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class MPSDataCollator:
    """
    Custom data collator specifically optimized for MPS (Apple Silicon).
    
    This collator ensures all tensors are properly moved to the MPS device
    and explicitly sets requires_grad where needed for training stability.
    """
    def __init__(self, base_collator, device):
        self.base_collator = base_collator
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized MPSDataCollator for device: {device}")
    
    def __call__(self, features):
        # Get base batch from original collator
        batch = self.base_collator(features)
        
        # Process each tensor in the batch
        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # Move tensor to correct device
                tensor = v.to(self.device)
                
                # For input_ids and attention_mask, no gradients needed
                if k in ["input_ids", "attention_mask"]:
                    tensor.requires_grad_(False)
                
                # For other tensors like position_ids, set requires_grad
                # to maintain compatibility with gradient checkpointing
                processed_batch[k] = tensor
            else:
                processed_batch[k] = v
                
        return processed_batch

def prepare_datasets(config, tokenizer):
    """
    Prepare training and validation datasets.
    
    Args:
        config: Configuration dictionary
        tokenizer: Model tokenizer
        
    Returns:
        train_dataset, eval_dataset
    """
    train_file = config["data_config"]["train_file"]
    validation_file = config["data_config"]["validation_file"]
    
    logger.info(f"Preparing training dataset from {train_file}")
    train_dataset = StyleDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=config["data_config"]["max_seq_length"]
    )
    
    logger.info(f"Preparing validation dataset from {validation_file}")
    eval_dataset = StyleDataset(
        tokenizer=tokenizer,
        file_path=validation_file,
        block_size=config["data_config"]["max_seq_length"]
    )
    
    return train_dataset, eval_dataset

def setup_apple_silicon_environment():
    """
    Configure environment variables for optimal performance on Apple Silicon.
    """
    # Essential environment variables for MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper memory limit
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallbacks for unsupported operations
    
    # Performance optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism issues
    
    # Additional optimizations
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Use all CPU cores for PyTorch ops
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())  # Use all cores for MKL
    
    logger.info("Applied Apple Silicon optimization settings")
    logger.debug(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")
    logger.debug(f"PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    logger.debug(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    logger.debug(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

def init_wandb(config):
    """Initialize Weights & Biases for training monitoring."""
    wandb_config = config["wandb_config"]
    
    logger.info(f"Initializing W&B project: {wandb_config['project']}")
    
    # Load .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env loading")
    
    # Check if WANDB_API_KEY is set
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY environment variable not set. Running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        
        # Create wandb directory if it doesn't exist
        os.makedirs("wandb", exist_ok=True)
        logger.info("Created local wandb directory for offline logging")
    else:
        # If we have an API key but it's the placeholder, still go to offline mode
        if os.environ.get("WANDB_API_KEY") == "your_api_key_here":
            logger.warning("WANDB_API_KEY is set to placeholder value. Running in offline mode.")
            os.environ["WANDB_MODE"] = "offline"
        else:
            logger.info("WANDB_API_KEY found, running in online mode")
            # If WANDB_MODE was explicitly set to offline in .env, respect that
            if os.environ.get("WANDB_MODE") == "offline":
                logger.info("WANDB_MODE explicitly set to offline in environment")
    
    # Set additional environment variables for better compatibility
    os.environ["WANDB_SILENT"] = "true"  # Reduce console output
    os.environ["WANDB_NOTEBOOK_NAME"] = "finetune_model.py"  # For better run organization
    
    try:
        # Generate a unique run name with timestamp
        from datetime import datetime
        
        wandb_run_name = f"{wandb_config['name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Add specialization information to the WandB config
        wandb_config_with_specializations = config.copy()
        wandb_config_with_specializations["specializations"] = {
            "customer_experience": {
                "keywords": ["customer journey", "touchpoints", "engagement", "satisfaction metrics", 
                             "customer retention", "experience design", "CX metrics", "VoC", "service design"],
                "weight": 0.35
            },
            "artificial_intelligence": {
                "keywords": ["machine learning", "neural networks", "deep learning", "LLMs", "prompt engineering", 
                             "transformer models", "model training", "data bias", "AI ethics"],
                "weight": 0.35
            },
            "machine_learning": {
                "keywords": ["supervised learning", "unsupervised learning", "reinforcement learning", 
                             "model evaluation", "feature engineering", "overfitting", "hyperparameters"],
                "weight": 0.30
            },
            "satirical_style": {
                "keywords": ["satire", "irony", "sarcasm", "humor", "mockery", "parody", 
                             "critique", "exaggeration", "wit"],
                "weight": 0.50  # High importance for maintaining Pete Connor's style
            }
        }
        
        # Try to import wandb, give clear error if not installed
        try:
            import wandb
        except ImportError:
            logger.error("wandb package not installed. Install with: pip install wandb")
            logger.info("Continuing without WandB tracking...")
            return None
        
        # Initialize wandb with more robust error handling
        try:
            run = wandb.init(
                project=wandb_config["project"],
                name=wandb_run_name,
                tags=wandb_config["tags"],
                notes=wandb_config["notes"],
                config=wandb_config_with_specializations,
                resume="allow"
            )
            
            # Add custom metrics for model specialization
            wandb.define_metric("cx_expertise_score")
            wandb.define_metric("ai_expertise_score")
            wandb.define_metric("satire_level_score")
            wandb.define_metric("domain_expertise/customer_experience")
            wandb.define_metric("domain_expertise/artificial_intelligence")
            wandb.define_metric("domain_expertise/machine_learning")
            
            logger.info("Successfully initialized W&B")
            
            # Log starting message
            wandb.log({"status": "started"})
            
            return run
        except Exception as e:
            logger.error(f"Error initializing W&B: {str(e)}")
            logger.info("Continuing without WandB tracking...")
            return None
    except Exception as e:
        logger.error(f"Error in WandB setup: {str(e)}")
        logger.info("Continuing without WandB tracking...")
        return None

def fine_tune_model(config_path):
    """
    Main fine-tuning function.
    
    Args:
        config_path: Path to the configuration JSON file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize W&B
    wandb_run = init_wandb(config)
    
    # Comprehensive device detection and configuration
    training_successful = False
    try:
        # Determine hardware capabilities and optimal device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device for training: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Log device metrics to WandB
            if wandb_run:
                wandb.log({
                    "device/type": "cuda",
                    "device/name": torch.cuda.get_device_name(0),
                    "device/memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
                })
            
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Detect Apple Silicon model and capabilities
            import platform
            import subprocess
            from psutil import virtual_memory
            
            # Configure environment for Apple Silicon
            setup_apple_silicon_environment()
            
            # Get Apple Silicon model details
            try:
                sysctl_output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
                apple_model = sysctl_output if "Apple" in sysctl_output else "Apple Silicon"
                
                # Get memory information
                mem = virtual_memory()
                total_memory_gb = mem.total / (1024**3)
                available_memory_gb = mem.available / (1024**3)
                
                logger.info(f"Detected {apple_model} with {total_memory_gb:.1f} GB RAM ({available_memory_gb:.1f} GB available)")
                
                # Check if we have enough memory to proceed with training
                if available_memory_gb < 8:
                    logger.warning("Low available memory detected. Training may be unstable.")
                    logger.warning("Consider closing other applications to free up memory.")
                    
                    if wandb_run:
                        wandb.alert(
                            title="Low Memory Warning",
                            text=f"Only {available_memory_gb:.1f} GB of memory available. Training may be unstable.",
                            level=wandb.AlertLevel.WARNING
                        )
                
                # Dynamically adjust training parameters based on available resources
                if "training_args" in config["training_config"]:
                    # Adjust batch size and gradient accumulation based on available memory
                    if available_memory_gb < 12:
                        logger.info("Limited memory - using minimal batch size and higher gradient accumulation")
                        config["training_config"]["training_args"]["per_device_train_batch_size"] = 1
                        config["training_config"]["training_args"]["gradient_accumulation_steps"] = 16
                    elif available_memory_gb < 24:
                        logger.info("Medium memory - using balanced batch size and gradient accumulation")
                        config["training_config"]["training_args"]["per_device_train_batch_size"] = 1
                        config["training_config"]["training_args"]["gradient_accumulation_steps"] = 8
                    else:
                        logger.info("High memory - using optimal batch size and gradient accumulation")
                        config["training_config"]["training_args"]["per_device_train_batch_size"] = 2
                        config["training_config"]["training_args"]["gradient_accumulation_steps"] = 4
                
                # Log detailed MPS environment settings
                logger.info("MPS Environment Configuration:")
                logger.info(f"- PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Not Set')}")
                logger.info(f"- PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'Not Set')}")
                logger.info(f"- TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'Not Set')}")
                
                # Log to WandB
                if wandb_run:
                    wandb.log({
                        "device/type": "mps",
                        "device/name": apple_model,
                        "device/total_memory_gb": total_memory_gb,
                        "device/available_memory_gb": available_memory_gb,
                        "device/memory_percent_used": mem.percent,
                        "training/batch_size": config["training_config"]["training_args"]["per_device_train_batch_size"],
                        "training/gradient_accumulation": config["training_config"]["training_args"]["gradient_accumulation_steps"]
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting Apple Silicon details: {e}")
            
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for training on Apple Silicon")
            
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for training (this may be slow)")
            
            import platform
            import psutil
            
            # Get CPU information
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=True)
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            logger.info(f"CPU: {cpu_info}")
            logger.info(f"CPU Cores: {cpu_count}")
            logger.info(f"RAM: {ram_gb:.1f} GB")
            
            # Log to WandB
            if wandb_run:
                wandb.log({
                    "device/type": "cpu",
                    "device/name": cpu_info,
                    "device/cores": cpu_count,
                    "device/memory_gb": ram_gb
                })
        
        # Verify device is properly configured
        if device.type == "mps":
            # Verify MPS is functioning correctly
            try:
                test_tensor = torch.ones(1, 1).to(device)
                test_result = test_tensor + test_tensor
                logger.info("MPS device test successful")
            except Exception as e:
                logger.error(f"MPS device test failed: {e}")
                logger.warning("Falling back to CPU due to MPS initialization failure")
                device = torch.device("cpu")
                
                if wandb_run:
                    wandb.alert(
                        title="MPS Fallback",
                        text=f"Failed to initialize MPS device: {e}. Falling back to CPU.",
                        level=wandb.AlertLevel.WARNING
                    )
    except Exception as e:
        logger.error(f"Error during device setup: {e}")
        logger.warning("Falling back to CPU")
        device = torch.device("cpu")
        
        if wandb_run:
            wandb.alert(
                title="Device Setup Error",
                text=f"Error during device setup: {e}. Falling back to CPU.",
                level=wandb.AlertLevel.ERROR
            )
    
    # Load model and tokenizer
    logger.info(f"Loading base model: {config['base_model']}")
    
    # Load tokenizer with appropriate settings
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        if wandb_run:
            wandb.alert(
                title="Tokenizer Loading Error",
                text=f"Failed to load tokenizer: {str(e)}",
                level=wandb.AlertLevel.ERROR
            )
        raise
    
    # Adjust training args based on device
    training_args_config = config["training_config"]["training_args"].copy()
    
    # Disable fp16 on MPS and CPU
    if device.type != "cuda":
        if "fp16" in training_args_config:
            training_args_config["fp16"] = False
            logger.info("Disabled FP16 as it's only supported on CUDA devices")
    
    # Add checkpoint resume capability
    if Path(training_args_config["output_dir"]).exists():
        previous_checkpoints = sorted(
            [d for d in Path(training_args_config["output_dir"]).iterdir() 
             if d.is_dir() and d.name.startswith("checkpoint")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        if previous_checkpoints:
            latest_checkpoint = previous_checkpoints[-1]
            training_args_config["resume_from_checkpoint"] = str(latest_checkpoint)
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
    
    # Enhanced error handling for model loading
    logger.info(f"Loading model on {device} device")
    model = None
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Try loading with different settings based on device type
            if device.type == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            elif device.type == "mps":
                # Special handling for MPS (Apple Silicon)
                # First load model to CPU to configure it properly before moving to MPS
                logger.info("Loading model to CPU first for Apple Silicon compatibility")
                model = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Force model to train mode and enable gradient tracking
                model.train()
                
                # Explicitly enable gradients for base model params
                for name, param in model.named_parameters():
                    param.requires_grad = True
                    logger.debug(f"Param {name} requires_grad set to {param.requires_grad}")
                
                logger.info("Model configured on CPU, now moving to MPS device")
                model = model.to(device)
            else:
                # CPU fallback
                model = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.float32
                )
                model = model.to(device)
                
            logger.info("Model loaded successfully")
            break
        except Exception as e:
            retry_count += 1
            logger.error(f"Error loading model (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                if wandb_run:
                    wandb.alert(
                        title="Model Loading Failed",
                        text=f"Failed to load model after {max_retries} attempts: {str(e)}",
                        level=wandb.AlertLevel.ERROR
                    )
                raise RuntimeError(f"Failed to load model after {max_retries} attempts") from e
            
            logger.info(f"Retrying model loading with simplified configuration...")
            time.sleep(2)  # Brief pause before retry
    
    # Prepare model for training
    logger.info("Preparing model for kbit training")
    model = prepare_model_for_kbit_training(model)
    
    # Ensure model has requires_grad set for parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters")
    lora_config = LoraConfig(
        r=config["training_config"]["lora_r"],
        lora_alpha=config["training_config"]["lora_alpha"],
        lora_dropout=config["training_config"]["lora_dropout"],
        bias=config["training_config"]["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=config["training_config"]["target_modules"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    logger.info("LoRA adapters successfully applied to model")
    
    # Ensure trainable parameters require gradients
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        logger.error("No trainable parameters found. Check LoRA configuration.")
        raise ValueError("No trainable parameters found in the model.")
    else:
        logger.info(f"Number of trainable parameters: {trainable_params}")
    
    # Memory usage logging
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Log model size and parameters
    model_size = sum(p.numel() for p in model.parameters()) / 1_000_000
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    logger.info(f"Model size: {model_size:.2f}M parameters")
    logger.info(f"Trainable parameters: {trainable_params:.2f}M ({trainable_params/model_size*100:.2f}%)")
    
    # Prepare datasets with error handling
    try:
        train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        if wandb_run:
            wandb.alert(
                title="Dataset Preparation Error",
                text=f"Failed to prepare datasets: {str(e)}",
                level=wandb.AlertLevel.ERROR
            )
        raise
    
    # Import our custom W&B callback
    try:
        from wandb_dashboards import create_custom_metrics_callback
        custom_callback = create_custom_metrics_callback(config_path, tokenizer, eval_dataset)
        callbacks = [custom_callback] if custom_callback else []
        logger.info("Custom W&B metrics callback loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load custom W&B callback: {str(e)}")
        callbacks = []
    
    # Custom style loss (commented out for now as it needs more testing)
    custom_loss = CustomStyleLoss(
        tokenizer=tokenizer,
        penalized_phrases=config["custom_loss_config"]["penalized_phrases"],
        rewarded_phrases=config["custom_loss_config"]["rewarded_phrases"],
        penalty_weight=config["custom_loss_config"]["penalty_weight"],
        reward_weight=config["custom_loss_config"]["reward_weight"]
    )
    
    # Create training arguments with improved handling
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(**training_args_config)
    
    # Initialize trainer with callbacks
    logger.info("Initializing trainer")
    
    # Configure optimizer based on device
    if device.type == "mps":
        # Special optimizer config for Apple Silicon
        logger.info("Configuring AdamW optimizer with explicit gradient tracking for MPS")
        
        # Make sure all trainable parameters have requires_grad=True
        trainable_params = []
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                trainable_params.append(param)
        
        # Create optimizer with only trainable params
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )
        
        # Create custom scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * training_args.max_steps),
            num_training_steps=training_args.max_steps
        )
        
        # Custom data collator
        data_collator = MPSDataCollator(CustomDataCollator(tokenizer=tokenizer), device)
        
        # Create trainer with custom optimizer and scheduler
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler)
        )
    else:
        # Standard trainer for other devices
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=CustomDataCollator(tokenizer=tokenizer),
            callbacks=callbacks
        )
    
    # Enable checkpointing at timed intervals (every 2 hours = 7200 seconds)
    if "save_steps" in training_args_config and training_args_config["save_steps"] > 0:
        logger.info(f"Checkpoints will be saved every {training_args_config['save_steps']} steps")
    
    # Custom loss integration (commented out for stability)
    # trainer.compute_loss = lambda model, inputs, return_outputs=False, **kwargs: custom_loss(
    #     model(**inputs).logits, inputs["input_ids"]
    # )
    
    # Start training with improved error handling
    logger.info("Starting model fine-tuning")
    training_successful = False
    
    # Verify gradients before training
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if trainable_count == 0:
        logger.error("No trainable parameters detected. Training cannot proceed.")
        if wandb_run:
            wandb.alert(
                title="Training Error",
                text="No trainable parameters detected. Check model configuration.",
                level=wandb.AlertLevel.ERROR
            )
        raise ValueError("No trainable parameters found. Cannot train the model.")
    else:
        logger.info(f"Found {trainable_count} trainable parameters. Training can proceed.")
    
    try:
        # Run a test batch through the model to verify gradients
        if train_dataset:
            if device.type == "mps":
                logger.info("Running MPS-specific gradient verification test")
                
                # Get a batch using our data collator to ensure correct device placement
                test_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=1,
                    collate_fn=data_collator
                )
                test_batch = next(iter(test_loader))
                
                # Create an optimizer just for testing gradient flow
                test_optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=1e-5
                )
                test_optimizer.zero_grad()
                
                # Enable gradients and run forward pass
                with torch.enable_grad():
                    outputs = model(**test_batch)
                    loss = outputs.loss
                    
                    # Log the loss value
                    logger.info(f"Test batch loss on MPS: {loss.item()}")
                    
                    # Attempt backward pass to check for gradients
                    try:
                        loss.backward()
                        
                        # Check if any parameters received gradients
                        grad_count = sum(1 for p in model.parameters() 
                                       if p.requires_grad and p.grad is not None)
                        
                        if grad_count > 0:
                            logger.info(f"Success! {grad_count} parameters received gradients")
                        else:
                            logger.warning("No parameters received gradients, but continuing anyway")
                            logger.info("This is common with MPS and training may still work")
                    except Exception as e:
                        logger.warning(f"Backward pass test failed: {str(e)}")
                        logger.info("Will attempt training anyway as this may be MPS-specific")
                
                # Clean up
                test_optimizer.zero_grad()
                del test_optimizer, test_batch, test_loader
            else:
                # For non-MPS devices, use the CPU test approach
                logger.info("Running test batch to verify gradients (on CPU for stability)")
                # Create CPU test model
                test_model = model.cpu()
                for param in test_model.parameters():
                    if param.requires_grad:
                        # Double-check requires_grad
                        param.requires_grad_(True)
                        
                # Get a batch
                batch = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=1)))
                
                # Convert batch to tensors on CPU
                batch = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Enable gradients
                with torch.enable_grad():
                    # Run test forward pass
                    outputs = test_model(**batch)
                    loss = outputs.loss
                    
                    # Check if loss has gradient
                    if not hasattr(loss, 'grad_fn') or loss.grad_fn is None:
                        logger.warning("Test loss has no gradient function, but will proceed with training anyway.")
                        logger.info("This warning is common with MPS device and may not affect actual training.")
                    else:
                        logger.info("Test batch successfully produced loss with gradient.")
                
                # Move model back to original device
                test_model = test_model.to(device)
                del test_model
            
            logger.info("Starting actual training...")
        
        # Now start actual training
        if device.type == "mps":
            logger.info("Starting training with MPS-specific optimizations")
            
            # For Apple Silicon, we use a custom training loop with safeguards
            def mps_safe_training():
                # Start training with MPS safeguards
                try:
                    trainer.train()
                    return True
                except RuntimeError as e:
                    error_str = str(e)
                    # Check for common MPS errors
                    if "cuDNN error" in error_str or "CUDA error" in error_str:
                        logger.warning("CUDA-related error with MPS device, likely a compatibility issue")
                        return False
                    elif "NaN" in error_str or "nan" in error_str:
                        logger.warning("NaN detected in computation, may be fixable with different hyperparameters")
                        return False
                    elif "device" in error_str and "MPS" in error_str:
                        logger.warning("MPS-specific device error occurred")
                        return False
                    else:
                        # Unknown error, re-raise
                        raise
            
            # Attempt training with MPS
            mps_success = mps_safe_training()
            
            if not mps_success:
                logger.warning("MPS training failed, attempting fallback to CPU")
                
                # Move model to CPU for fallback training
                logger.info("Moving model to CPU for fallback training")
                model = model.cpu()
                
                # Update data collator for CPU
                data_collator = CustomDataCollator(tokenizer=tokenizer)
                
                # Re-initialize trainer for CPU
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    callbacks=callbacks
                )
                
                # Attempt CPU training
                try:
                    trainer.train()
                    logger.info("CPU fallback training completed successfully")
                    training_successful = True
                except Exception as e:
                    logger.error(f"CPU fallback training also failed: {str(e)}")
                    if wandb_run:
                        wandb.alert(
                            title="Training Failed",
                            text=f"Both MPS and CPU training attempts failed: {str(e)}",
                            level=wandb.AlertLevel.ERROR
                        )
                    raise
            else:
                logger.info("MPS training completed successfully!")
                training_successful = True
        else:
            # Standard training for non-MPS devices
            trainer.train()
            training_successful = True
            logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoint...")
        # Save checkpoint on keyboard interrupt
        try:
            interrupt_save_dir = Path(training_args_config["output_dir"]) / "user_interrupted"
            interrupt_save_dir.mkdir(exist_ok=True, parents=True)
            trainer.save_model(interrupt_save_dir)
            logger.info(f"Saved interrupt checkpoint to {interrupt_save_dir}")
        except Exception as e:
            logger.error(f"Error saving interrupt checkpoint: {str(e)}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.warning("Training was interrupted. Saving checkpoint of the current state.")
        
        if wandb_run:
            wandb.alert(
                title="Training Error",
                text=f"Error during training: {str(e)}",
                level=wandb.AlertLevel.ERROR
            )
    
    # Save the model (even if training was interrupted)
    try:
        # Save the final model
        output_dir = Path(training_args_config["output_dir"])
        final_model_dir = output_dir / "final"
        final_model_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Saving model to {final_model_dir}")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Save LoRA adapter separately
        lora_dir = output_dir / "lora_adapter"
        lora_dir.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(lora_dir)
        
        logger.info("Model saving complete.")
        
        # Save metadata
        from datetime import datetime
        
        # Get a serializable version of lora_config
        try:
            lora_config_dict = {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "bias": lora_config.bias,
                "task_type": str(lora_config.task_type),
                "target_modules": [str(mod) for mod in lora_config.target_modules]
            }
        except Exception as e:
            logger.warning(f"Error converting LoRA config to dict: {str(e)}")
            lora_config_dict = {"error": "Could not serialize LoRA config"}
        
        # Create metadata dictionary with proper JSON serializable values
        metadata = {
            "model_name": config["base_model"],
            "training_complete": training_successful,
            "lora_config": lora_config_dict,
            "custom_loss_config": config["custom_loss_config"],
            "timestamp": datetime.now().isoformat(),
            "specializations": ["customer experience", "artificial intelligence", "machine learning"],
            "satire_level": "high",
            "training_device": str(device),
            "total_examples": len(train_dataset) if 'train_dataset' in locals() else 0
        }
        
        with open(output_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Fine-tuning process complete.")
        
        # Log final status to W&B
        if wandb_run:
            wandb.log({"training_complete": int(training_successful)})
            wandb.log({"status": "completed" if training_successful else "failed"})
            if training_successful:
                wandb.alert(
                    title="Training Complete",
                    text="Model training completed successfully!",
                    level=wandb.AlertLevel.INFO
                )
            
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        logger.error("Could not save the trained model. Check the logs for details.")
        
        if wandb_run:
            wandb.alert(
                title="Model Saving Error",
                text=f"Failed to save model: {str(e)}",
                level=wandb.AlertLevel.ERROR
            )
    
    # Cleanup
    if wandb_run is not None:
        wandb.finish()
    
    logger.info("Fine-tuning completed")
    return training_successful

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with C. Pete Connor's writing style")
    parser.add_argument("--config", type=str, default="finetune_config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    fine_tune_model(args.config)
