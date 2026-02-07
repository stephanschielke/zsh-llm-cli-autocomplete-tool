---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2-0.5B
tags:
- axolotl
- base_model:adapter:Qwen/Qwen2-0.5B
- lora
- transformers
datasets:
- /Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/src/training/data_splits_axolotl/train_axolotl.jsonl
pipeline_tag: text-generation
model-index:
- name: Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/zsh-lora-output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.13.1`
```yaml
base_model: Qwen/Qwen2-0.5B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true
datasets:
- path: /Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/src/training/data_splits_axolotl/train_axolotl.jsonl
  type: alpaca
dataset_prepared_path: /Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/last_run_prepared
val_set_size: 0.0
output_dir: /Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/zsh-lora-output
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
lora_modules_to_save:
- embed_tokens
- lm_head
sequence_len: 512
sample_packing: true
pad_to_sequence_len: true
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 1
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0002
warmup_steps: 50
load_in_8bit: false
load_in_4bit: true
bf16: false
fp16: false
tf32: false
gradient_checkpointing: true
group_by_length: false
logging_steps: 10
save_steps: 200
save_safetensors: true
save_total_limit: 3
early_stopping_patience: null
early_stopping_threshold: null
wandb_mode: disabled
fp8: false
val_dataset:
  path: /Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/src/training/data_splits_axolotl/val_axolotl.jsonl
  type: alpaca

```

</details><br>

# Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/zsh-lora-output

This model is a fine-tuned version of [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) on the /Users/duoyun/Desktop/zsh-llm-cli-autocomplete-tool/src/training/data_splits_axolotl/train_axolotl.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 50
- training_steps: 3

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.10.0
- Datasets 4.5.0
- Tokenizers 0.22.1