---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
- lora
- zsh
- cli
- autocomplete
- command-line
- peft
---

# duoyuncloud/zsh-assistant-lora

This is a LoRA (Low-Rank Adaptation) adapter for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) fine-tuned for Zsh CLI command autocomplete.

## Model Details

### Base Model
- **Base Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Model Type:** Causal Language Model
- **Task:** CLI Command Completion

### LoRA Configuration
- **LoRA Rank (r):** 8
- **LoRA Alpha:** 16
- **LoRA Dropout:** 0.1
- **Target Modules:** k_proj, q_proj, o_proj, gate_proj, v_proj, down_proj, up_proj

### Training Configuration
- **Epochs:** 3
- **Learning Rate:** 0.0002
- **Batch Size:** 1
- **Training Data:** 277 CLI command completion samples (Git, Docker, NPM, Python, Kubernetes, etc.)

## Usage

### Using with PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "duoyuncloud/zsh-assistant-lora")

# Use for inference
prompt = "Input: git comm\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Using with Ollama

This adapter is designed to work with the [zsh-llm-cli-autocomplete-tool](https://github.com/duoyuncloud/zsh-llm-cli-autocomplete-tool) project.

After importing to Ollama, use it as:
```bash
ollama run zsh-assistant
```

## Training Data

The model was fine-tuned on 277 command completion pairs covering:
- Git commands (status, add, commit, push, pull, etc.)
- Docker commands (run, build, ps, exec, etc.)
- NPM/Node commands (install, run, start, etc.)
- Python commands (-m, -c, pip, etc.)
- Kubernetes commands (get, apply, delete, etc.)
- System commands (ls, cd, mkdir, etc.)

## Limitations

- This adapter is specifically fine-tuned for CLI command completion tasks
- Performance may vary for other use cases
- The base model's limitations also apply

## Citation

If you use this adapter, please cite:

```bibtex
@misc{duoyuncloud_zsh_assistant_lora},
  title={Zsh CLI Autocomplete LoRA Adapter},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/duoyuncloud/zsh-assistant-lora}}
}
```

## License

This adapter inherits the license from the base model [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
