# Fine-Tuning Monitoring and WandB Integration

This document explains the monitoring tools available for tracking the fine-tuning process and integrating with Weights & Biases (WandB).

## 🔍 Available Monitoring Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `check_fine_tuning_progress.py` | Monitors training progress, checkpoints and model status | `python check_fine_tuning_progress.py --monitor` |
| `check_wandb_status.py` | Verifies WandB configuration and connection | `python check_wandb_status.py` |
| `fix_wandb_auth.py` | Fixes WandB authentication issues | `python fix_wandb_auth.py` |
| `fix_wandb_and_continue_training.py` | Restarts training with proper WandB integration | `python fix_wandb_and_continue_training.py` |
| `create_wandb_dashboard.py` | Creates a WandB dashboard for monitoring metrics | `python create_wandb_dashboard.py` |
| `capture_logs.py` | Captures and saves logs from running processes | `python capture_logs.py` |
| `monitor_resources.py` | Monitors system resources during training | `python monitor_resources.py` |

## 📊 WandB Integration

Weights & Biases is used for tracking fine-tuning metrics in real-time. To set up WandB:

1. **Get your API key**: Go to [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. **Update your `.env` file**: Add your API key to the `.env` file:
   ```
   WANDB_API_KEY=your_api_key_here
   ```
3. **Fix authentication**: Run `python fix_wandb_auth.py` if you encounter any issues
4. **Create dashboard**: Run `python create_wandb_dashboard.py` to create a monitoring dashboard

## 🔄 Training Workflow

### Starting Training
```bash
python finetune_model.py
```

### Monitoring Training
```bash
# Monitor progress with checkpoint tracking
python check_fine_tuning_progress.py --monitor --interval 5

# Monitor system resources
python monitor_resources.py

# Capture logs from the process
python capture_logs.py
```

### If Training Fails or WandB Integration Issues Occur
```bash
# Fix WandB authentication
python fix_wandb_auth.py

# Check WandB status
python check_wandb_status.py

# Restart training with fixed WandB integration
python fix_wandb_and_continue_training.py
```

## 📈 Fine-Tuning Metrics

The following metrics are tracked during fine-tuning:

- **Training Loss**: How well the model is learning
- **Learning Rate**: The current learning rate schedule
- **Validation Loss**: How well the model generalizes
- **Training Speed**: Steps per second and total runtime
- **GPU/MPS Memory Usage**: Memory consumption during training
- **System Resources**: CPU, RAM, and thermal metrics

## 🧠 Apple Silicon Optimization

For Apple Silicon (M1/M2/M3/M4) devices, the training uses:

- **MPS (Metal Performance Shaders)**: For GPU acceleration
- **Resource Monitoring**: To prevent thermal throttling
- **Batch Size Optimization**: To maximize use of unified memory

## 📁 Log Files

Logs are stored in the `logs/` directory:
- `finetune.log`: Main fine-tuning log
- `finetune_pid*.log`: Process-specific logs captured during training
- `resources_*.log`: System resource monitoring logs

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **WandB Authentication Failure**
   - Run `python fix_wandb_auth.py` to update your API key
   - Check internet connection

2. **Training Process Hangs**
   - Check resource monitors for memory pressure
   - Consider reducing batch size in `finetune_config.json`

3. **MPS/Metal Errors**
   - Ensure you're using a compatible PyTorch version
   - Verify accelerate package is installed

4. **Checkpoint Loading Failures**
   - Check that previous checkpoints are complete
   - Verify disk space is adequate

## 📚 Further Resources

- [WandB Documentation](https://docs.wandb.ai/)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [HuggingFace Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
