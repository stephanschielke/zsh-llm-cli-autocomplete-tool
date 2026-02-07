# Qwen3 LoRA Training Guide

Complete guide for training a LoRA adapter on Qwen3-1.7B for CLI command completion.

## Overview

This training pipeline fine-tunes Qwen3-1.7B using LoRA (Low-Rank Adaptation) to improve CLI command completion capabilities.

## Training Steps

### 1. Data Preparation & Splitting

Split training data into train (70%), validation (15%), and test (15%) sets:

```bash
python -m training.prepare_and_split_data \
    --input src/training/zsh_training_data.jsonl \
    --output-dir src/training/data_splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### 2. Convert to Axolotl Format

Convert split data to Axolotl-compatible format:

```bash
python -m training.convert_to_axolotl_format \
    --split-data \
    --input src/training/data_splits \
    --output src/training/data_splits_axolotl
```

### 3. Train LoRA Adapter

Train the LoRA adapter using Axolotl:

```bash
python -m training.qwen3_axolotl_trainer \
    --base-model Qwen/Qwen3-1.7B \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --learning-rate 2e-4 \
    --epochs 3
```

**LoRA Parameters:**
- `lora-r`: Rank of LoRA matrices (default: 16)
- `lora-alpha`: LoRA alpha scaling factor (default: 32)
- `lora-dropout`: Dropout rate for LoRA layers (default: 0.05)
- `learning-rate`: Learning rate (default: 2e-4)
- `epochs`: Number of training epochs (default: 3)

**Low Memory Mode:**
```bash
python -m training.qwen3_axolotl_trainer --low-memory
```

### 4. Evaluate on Test Set

Evaluate the trained adapter:

```bash
python -m training.evaluate_model \
    --adapter zsh-lora-output \
    --base-model Qwen/Qwen3-1.7B \
    --test-file src/training/data_splits_axolotl/test_axolotl.jsonl \
    --output evaluation_report.json
```

### 5. Upload to HuggingFace

Upload the trained adapter to HuggingFace Hub:

```bash
python -m training.upload_to_hf \
    --adapter zsh-lora-output \
    --repo-id your-username/zsh-assistant-lora \
    --base-model Qwen/Qwen3-1.7B
```

## Complete Workflow (One Command)

Run the complete workflow with a single command:

```bash
python -m training.train_workflow \
    --base-model Qwen/Qwen3-1.7B \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --learning-rate 2e-4 \
    --epochs 3
```

**Skip Steps:**
```bash
# Skip data preparation (use existing splits)
python -m training.train_workflow --skip-data-prep

# Skip evaluation
python -m training.train_workflow --skip-evaluation
```

## LoRA Configuration for Qwen3

The training script uses optimized LoRA parameters for Qwen3:

- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Rank (r)**: 16 (balance between capacity and efficiency)
- **Alpha**: 32 (2x rank for good scaling)
- **Dropout**: 0.05 (prevent overfitting)

## Training Time Estimates

- **GPU (NVIDIA)**: 20-40 minutes
- **Apple Silicon (M1/M2)**: 30-60 minutes
- **CPU**: 2-4 hours (not recommended)

## Output Files

After training, you'll find:

- `zsh-lora-output/`: LoRA adapter directory
  - `adapter_config.json`: LoRA configuration
  - `adapter_model.safetensors`: Trained weights
  - `training_args.bin`: Training arguments
- `evaluation_report.json`: Test set evaluation results
- `qwen3_axolotl_config.yml`: Training configuration

## Troubleshooting

### Out of Memory

Use low memory mode:
```bash
python -m training.qwen3_axolotl_trainer --low-memory
```

### Training Fails

1. Check dependencies: `pip install axolotl torch transformers accelerate datasets peft`
2. Verify data format: Check that Axolotl format files exist
3. Check GPU/CPU availability

### Evaluation Fails

1. Ensure test data exists: `src/training/data_splits_axolotl/test_axolotl.jsonl`
2. Check adapter path: `zsh-lora-output/`
3. Verify base model is accessible

## Next Steps

After training:

1. **Test locally**: `python -m training.test_adapter`
2. **Review evaluation**: Check `evaluation_report.json`
3. **Upload to HuggingFace**: Use `upload_to_hf.py`
4. **Use in production**: Load adapter with PEFT library

