# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-13 14:05:00 UTC
**Commit:** 0b00b1c
**Branch:** main

## OVERVIEW
AI-powered Zsh plugin with LoRA fine-tuning and personalized command predictions. Uses local LLMs via Ollama to provide intelligent command completions that learn from your workflow.

## STRUCTURE
```
./
├── bin/                  # Wrapper script for the CLI
├── config/               # Configuration files (default.yaml, custom.yaml)
├── docs/                 # Documentation
├── scripts/              # Installation and utility scripts (install.sh, run_training.sh, etc.)
├── src/
│   ├── model_completer/  # Main Python package (CLI, daemon, training)
│   ├── training/         # Training data generation and LoRA fine-tuning
│   └── scripts/          # Zsh plugin source
├── tests/                # Test directory (minimal)
└── zsh-lora-output/      # LoRA adapter output (gitignored)
```

## WHERE TO LOOK
| Task                          | Location                                 | Notes                                                                 |
|-------------------------------|------------------------------------------|-----------------------------------------------------------------------|
| Install / setup               | ./install.sh                             | One-click installer                                                   |
| Run the CLI                   | ./bin/model-completer or `model-completer` | Command-line interface                                                |
| Zsh plugin                    | ./src/scripts/zsh_autocomplete.plugin.zsh | Main Zsh plugin file                                                  |
| Training data generation      | ./src/training/prepare_zsh_data.py       | Generates training data for LoRA fine-tuning                          |
| LoRA training                 | ./src/training/                          | Contains training scripts                                             |
| Model import to Ollama        | ./src/model_completer/cli.py --import-to-ollama | Imports LoRA adapter to Ollama                                        |
| Configuration                 | ./config/default.yaml                    | Default configuration (overridden by custom.yaml)                     |
| Daemon (for low latency)      | ./scripts/run_completion_daemon.sh       | Runs the completion daemon for faster Tab completion                  |
| Test the merged model         | ./scripts/chat_merged_model.py           | Talk to the merged model directly                                     |
| Check system status           | ai-completion-status                     | Command to check system status                                        |

## CODE MAP
| Symbol                     | Type     | Location                     | Refs | Role                                  |
|----------------------------|----------|------------------------------|------|---------------------------------------|
| model_completer.cli:main   | function | src/model_completer/cli.py   | 1    | Entry point for the CLI               |
| EnhancedCompleter          | class    | src/model_completer/completer.py | 5  | Main completion logic                 |
| OllamaClient               | class    | src/model_completer/ollama.py | 3    | Ollama API communication              |
| TrainingDataManager        | class    | src/training/prepare_zsh_data.py | 2  | Training data preparation             |
| LoRATrainer                | class    | src/training/lora_trainer.py | 4    | LoRA fine-tuning                      |
| ZshAutocompletePlugin      | class    | src/scripts/zsh_autocomplete.plugin.zsh | 1 | Zsh plugin implementation             |

## CONVENTIONS
- Uses `mise` for task management and `uv` for package management (see mise.toml)
- Dependencies split: runtime (requirements.txt) and training (requirements-training.txt)
- Configuration via YAML files in config/ directory
- All models served locally via Ollama (no external API calls)

## ANTI-PATTERNS (THIS PROJECT)
- Do not commit model outputs (zsh-lora-output/, zsh-model-merged/) - they are large and gitignored
- Do not modify .sisyphus/ directory (unclear purpose, may be safe to remove)
- Avoid committing the =0.43.1 file (pip error output, already ignored by git?)

## UNIQUE STYLES
- Hybrid Python/Shell project: Python for backend/training, Shell for installation and Zsh plugin
- Uses LoRA (Low-Rank Adaptation) for efficient model fine-tuning
- Command history stored locally for personalization (never leaves your machine)

## COMMANDS
```
bash
# Install dependencies (runtime)
mise run install-dependencies

# Install with training dependencies
INSTALL_TRAINING_DEPS=1 mise run install-dependencies

# Setup Ollama and download models
mise run setup-ollama

# Generate training data
mise run model-completer:generate-data

# Train LoRA model
mise run model-completer:train

# Import LoRA adapter to Ollama
mise run model-completer:lora:import-to-ollama

# Run the completion daemon (for low latency)
mise run model-completer:daemon

# Test the merged model directly
mise run model-completer:test

# Check system status
mise run model-completer:status
```

## NOTES
- First run can take 5-15 minutes (downloading models). Later sessions use the cached model.
- The pre-trained model is automatically downloaded and set up during `./install.sh`.
- For lower latency, run the completion daemon so each Tab doesn't start a new Python process.
- Smart commit messages analyze your git diff to generate descriptive commit messages.
- The project uses 100% local processing - no data leaves your machine.