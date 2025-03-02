"""
Weights & Biases integration for model monitoring.
"""

import logging
import os
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import wandb, but don't fail if it's not available
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed. W&B monitoring will be disabled.")

def is_wandb_available() -> bool:
    """
    Check if Weights & Biases is properly configured.
    
    Returns:
        bool: True if W&B is available, False otherwise
    """
    return _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY") is not None

def setup_wandb_monitoring(json_data: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Set up W&B monitoring for the training process.
    
    Args:
        json_data: The writing style JSON data
        
    Returns:
        tuple: (callback, examples_table)
    """
    logger.info("Setting up Weights & Biases monitoring")
    
    # Check if W&B is available
    if not is_wandb_available():
        logger.warning("W&B not available or API key not set. W&B monitoring disabled.")
        return None, None
    
    try:
        # Initialize W&B
        wandb.init(
            project="multi-platform-content-generator",
            config={
                "model_name": json_data.get("model_parameters", {}).get("name", "content-generator"),
                "style_type": "satirical-tech-expert",
                "platforms": ["LinkedIn", "Twitter", "Medium", "Substack", "Instagram", "Facebook"]
            }
        )
        
        # Define metrics to track
        wandb.define_metric("platform_adaptation", summary="mean")
        wandb.define_metric("content_engagement", summary="mean")
        
        # Create examples table
        examples_table = wandb.Table(
            columns=["platform", "prompt", "content", "within_limit"]
        )
        
        logger.info("W&B monitoring setup completed")
        return None, examples_table
        
    except Exception as e:
        logger.error(f"Error setting up W&B monitoring: {str(e)}")
        return None, None

def log_generation_example(platform: str, 
                          prompt: str, 
                          generated_content: str, 
                          examples_table: Optional[Any] = None) -> None:
    """
    Log a generation example to W&B.
    
    Args:
        platform: The target platform
        prompt: The generation prompt
        generated_content: The generated content
        examples_table: The W&B Table to log to
    """
    # Check if W&B is configured
    if not is_wandb_available():
        logger.debug("W&B not configured, skipping logging")
        return
    
    try:
        # Check if wandb is initialized
        if wandb.run is None:
            logger.warning("W&B run not initialized. Initializing now.")
            wandb.init(project="multi-platform-content-generator")
        
        # Create examples table if not provided
        if examples_table is None:
            examples_table = wandb.Table(
                columns=["platform", "prompt", "content", "within_limit"]
            )
        
        # Check if content is within platform character limits
        from src.models.platform_specs import get_platform_specs
        platform_specs = get_platform_specs(platform)
        max_length = platform_specs.max_length
        within_limit = len(generated_content) <= max_length
        
        # Log to examples table
        examples_table.add_data(platform, prompt, generated_content, within_limit)
        
        # Log the example
        wandb.log({
            "generation_examples": examples_table,
            "platform_adaptation": 0.8,  # Simplified scoring
            "content_length": len(generated_content),
            "within_limit": within_limit
        })
        
        logger.info(f"Logged generation example for {platform} to W&B")
        
    except Exception as e:
        logger.error(f"Error logging example to W&B: {str(e)}")

def log_training_metrics(epoch: int, metrics: Dict[str, float]) -> None:
    """
    Log training metrics to W&B.
    
    Args:
        epoch: Current training epoch
        metrics: Dictionary of metrics to log
    """
    if not is_wandb_available():
        logger.debug("W&B not configured, skipping metrics logging")
        return
    
    try:
        # Check if wandb is initialized
        if wandb.run is None:
            logger.warning("W&B run not initialized. Skipping metrics logging.")
            return
        
        # Add epoch to metrics
        metrics["epoch"] = epoch
        
        # Log metrics
        wandb.log(metrics)
        logger.info(f"Logged training metrics for epoch {epoch} to W&B")
        
    except Exception as e:
        logger.error(f"Error logging metrics to W&B: {str(e)}")
