# C. Pete Connor Model Fine-Tuning Instructions

## Overview
This document explains how to run the overnight fine-tuning process for the C. Pete Connor model. The training has been optimized for Apple Silicon devices, specifically the M4 Pro with 48GB of unified memory.

## Quick Start
For the simplest experience, use the desktop launcher:

1. Double-click `Run_Pete_Connor_Overnight_Training.command` on your Desktop
2. The training will start automatically, optimized for your Apple Silicon hardware
3. Monitor progress on the Weights & Biases dashboard

## Monitoring Training
- **Weights & Biases Dashboard**: https://wandb.ai/cpeteconnor-fiverr/pete-connor-cx-ai-expert
- **Log Files**: Check the `logs` directory for detailed training logs

## Memory Optimization
The training process has been specifically optimized for Apple Silicon by:

1. Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (disables upper memory limit)
2. Using minimal batch sizes (1 for both training and evaluation)
3. Implementing higher gradient accumulation (16 steps)
4. Reducing maximum sequence length to 1024
5. Using 8-bit quantization for the base model
6. Using LoRA adapters for efficient fine-tuning

## Training Configuration
Key settings:
- **Model**: Mistral 7B base
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Custom dataset in `data/training_data.jsonl`
- **Validation Data**: Sample in `data/validation_data.jsonl`
- **Output Directory**: `outputs/finetune/`

## Advanced Usage
If you need to customize the training process:

### Edit Configuration
Modify `finetune_config.json` to adjust:
- Learning rate
- Number of epochs
- Save frequency
- LoRA parameters

### Custom Training Script
For custom modifications, edit:
- `apple_silicon_training.py`: Simplified, memory-optimized training script
- `finetune_model.py`: Original full-featured training script

## Troubleshooting

### Out of Memory Errors
If you encounter memory issues:
1. Further reduce batch size or gradient accumulation in `finetune_config.json`
2. Reduce the model size or use a more efficient LoRA configuration
3. Make sure no other memory-intensive applications are running

### Training Not Progressing
If training seems stuck:
1. Check log files in the `logs` directory
2. View the Weights & Biases dashboard for metrics
3. Restart the training process with `Run_Pete_Connor_Overnight_Training.command`

## Support
For additional support, refer to:
- Hugging Face documentation for transformer models
- PyTorch documentation for MPS (Metal Performance Shaders) device
