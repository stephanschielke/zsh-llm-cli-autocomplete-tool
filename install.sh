#!/bin/bash
# One-click install: merged model (base + adapter from Hugging Face) for Zsh autocomplete.
# Run from project root: ./install.sh
# Then: source ~/.zshrc

echo "Zsh AI Autocomplete - One-Click Install (merged model)"
echo "========================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_feature() { echo -e "${PURPLE}[FEATURE]${NC} $1"; }
print_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create install log
INSTALL_LOG="$SCRIPT_DIR/install.log"
echo "=== Zsh AI Autocomplete Installation Log ===" > "$INSTALL_LOG"
echo "Started at: $(date)" >> "$INSTALL_LOG"
echo "" >> "$INSTALL_LOG"

print_info "Starting complete setup in: $SCRIPT_DIR"
print_info "Installation log: $INSTALL_LOG"

# Function to log commands
log_command() {
    echo ">>> $1" >> "$INSTALL_LOG"
    eval "$1" >> "$INSTALL_LOG" 2>&1
    local status=$?
    echo ">>> Exit code: $status" >> "$INSTALL_LOG"
    return $status
}

# Function to run command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    
    print_info "$description..."
    echo "=== $description ===" >> "$INSTALL_LOG"
    
    if log_command "$cmd"; then
        print_success "$description"
        return 0
    else
        print_error "$description failed (see $INSTALL_LOG)"
        return 1
    fi
}

# ============================================================================
# PHASE 1: SYSTEM CHECKS & VIRTUAL ENVIRONMENT
# ============================================================================

print_step "Phase 1: System checks and environment setup..."

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        print_info "Please install Python 3.8+ and try again"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is required but not installed"
        exit 1
    fi
    print_success "pip3 found"
    
    # Check Zsh
    if ! command -v zsh &> /dev/null; then
        print_error "Zsh is required but not installed"
        print_info "Please install Zsh and try again"
        exit 1
    fi
    ZSH_VERSION=$(zsh --version | cut -d' ' -f2)
    print_success "Zsh $ZSH_VERSION found"
    
    # Check curl (for Ollama installation)
    if ! command -v curl &> /dev/null; then
        print_error "curl is required for Ollama installation"
        exit 1
    fi
    print_success "curl found"
    
    # Check git (for training data)
    if ! command -v git &> /dev/null; then
        print_warning "git not found - some features may be limited"
    else
        print_success "git found"
    fi
}

check_requirements

# Create virtual environment
print_info "Setting up Python virtual environment..."
if [[ ! -d "$SCRIPT_DIR/venv" ]]; then
    if run_command "python3 -m venv '$SCRIPT_DIR/venv'" "Creating virtual environment"; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        print_info "Trying without virtual environment..."
        USE_VENV=0
    fi
else
    print_info "Virtual environment already exists"
    USE_VENV=1
fi

if [[ $USE_VENV -eq 1 ]]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    PIP_CMD="$SCRIPT_DIR/venv/bin/pip"
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
    print_success "Using virtual environment"
else
    PIP_CMD="pip3"
    PYTHON_CMD="python3"
    print_warning "Using --user installs (no virtual environment)"
fi

# ============================================================================
# PHASE 2: INSTALL PYTHON DEPENDENCIES
# ============================================================================

print_step "Phase 2: Installing Python dependencies..."

# Install core dependencies
print_info "Installing core Python dependencies..."
CORE_DEPS=(
    "requests>=2.28.0"
    "pyyaml>=6.0"
    "argcomplete>=2.0.0"
    "python-dotenv>=0.19.0"
    "prompt-toolkit>=3.0.0"
    "huggingface_hub>=0.16.0"
    "gguf>=0.1.0"
)

for dep in "${CORE_DEPS[@]}"; do
    if run_command "$PIP_CMD install '$dep'" "Installing $dep"; then
        print_success "Installed $dep"
    else
        print_warning "Failed to install $dep - continuing..."
    fi
done

# Install model-import dependencies (required: download + merge LoRA with base model)
print_info "Installing model dependencies (for downloading and using the fine-tuned model)..."
IMPORT_DEPS=(
    "torch>=2.0.0"
    "transformers>=4.30.0"
    "peft>=0.4.0"
)
for dep in "${IMPORT_DEPS[@]}"; do
    if run_command "$PIP_CMD install '$dep'" "Installing $dep"; then
        print_success "Installed $dep"
    else
        print_warning "Failed to install $dep - model import may fail"
    fi
done

# Optional: full training dependencies (skip by default for faster one-click install)
if [[ "${INSTALL_TRAINING_DEPS:-0}" = "1" ]]; then
    print_info "Installing full training dependencies (axolotl, etc.)..."
    TRAINING_DEPS=("accelerate>=0.20.0" "datasets>=2.12.0" "bitsandbytes>=0.41.0" "axolotl")
    for dep in "${TRAINING_DEPS[@]}"; do
        run_command "$PIP_CMD install '$dep'" "Installing $dep" || true
    done
else
    print_info "Skipping full training deps (set INSTALL_TRAINING_DEPS=1 to install axolotl for LoRA training)"
fi

# Install the package in development mode
print_info "Installing core package..."
if run_command "$PIP_CMD install -e ." "Installing package in development mode"; then
    print_success "Core package installed successfully"
else
    print_warning "Development mode install failed, trying regular install..."
    if run_command "$PIP_CMD install ." "Installing package regularly"; then
        print_success "Package installed regularly"
    else
        print_error "All installation methods failed"
        print_info "Creating manual setup..."
    fi
fi

# ============================================================================
# PHASE 3: SETUP OLLAMA AND MODELS
# ============================================================================

print_step "Phase 3: Setting up Ollama and models..."

# Use the existing Python modules to setup Ollama
print_info "Setting up Ollama using existing modules..."
if run_command "$PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from model_completer.ollama_manager import create_ollama_manager

manager = create_ollama_manager()
if not manager.is_installed():
    print('Installing Ollama...')
    if manager.install():
        print('Ollama installed successfully')
    else:
        print('Failed to install Ollama')
        sys.exit(1)
else:
    print('Ollama already installed')

if not manager.is_running():
    print('Starting Ollama server...')
    if manager.start_server():
        print('Ollama server started')
    else:
        print('Failed to start Ollama server')
        sys.exit(1)
else:
    print('Ollama server already running')

print('Setting up default models...')
if manager.setup_default_models():
    print('Default models setup completed')
else:
    print('Failed to setup default models')
\"" "Setting up Ollama and models"; then
    print_success "Ollama and models setup completed"
else
    print_warning "Ollama setup had issues - check logs"
fi

# ============================================================================
# PHASE 4: TRAINING DATA (optional, skip for one-click autocomplete-only)
# ============================================================================
if [[ "${GENERATE_TRAINING_DATA:-0}" = "1" ]]; then
    print_step "Phase 4: Generating training data (optional)..."
    run_command "$PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from model_completer.training import TrainingDataManager
TrainingDataManager().generate_training_data(500)
\"" "Generating training data" || true
else
    print_info "Skipping training data (set GENERATE_TRAINING_DATA=1 to generate)"
fi

# ============================================================================
# PHASE 5: INSTALL ZSH PLUGIN
# ============================================================================

print_step "Phase 5: Installing Zsh plugin..."

install_zsh_plugin() {
    local plugin_file="$SCRIPT_DIR/src/scripts/zsh_autocomplete.plugin.zsh"
    
    if [[ ! -f "$plugin_file" ]]; then
        print_error "Plugin file not found: $plugin_file"
        return 1
    fi
    
    # Add to .zshrc if not already there
    if ! grep -q "zsh_autocomplete.plugin.zsh" ~/.zshrc 2>/dev/null; then
        echo "" >> ~/.zshrc
        echo "# Zsh AI Autocomplete Plugin" >> ~/.zshrc
        echo "source \"$plugin_file\"" >> ~/.zshrc
        print_success "Added plugin to ~/.zshrc"
    else
        print_info "Plugin already configured in ~/.zshrc"
    fi
}

install_zsh_plugin

# ============================================================================
# PHASE 6: CREATE CONFIGURATION
# ============================================================================

print_step "Phase 6: Setting up configuration..."

print_info "Creating configuration using existing modules..."
if run_command "$PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from model_completer.utils import load_config
import os
import yaml

# Create config directory
config_dir = os.path.expanduser('~/.config/model-completer')
os.makedirs(config_dir, exist_ok=True)

# Create default config if it doesn't exist
config_file = os.path.join(config_dir, 'config.yaml')
if not os.path.exists(config_file):
    default_config = {
        'ollama': {
            'url': 'http://localhost:11434',
            'timeout': 10
        },
        'model': 'zsh-assistant',
        'fallback_model': 'qwen2:0.5b',
        'hf_lora_repo': 'duoyuncloud/zsh-cli-lora',  # Pre-trained LoRA adapter (Qwen2-0.5B)
        'cache': {
            'enabled': True,
            'ttl': 3600
        },
        'logging': {
            'level': 'INFO',
            'file': '~/.cache/model-completer/logs.txt'
        },
        'ui': {
            'enabled': True,
            'max_suggestions': 5,
            'show_confidence': True
        },
        'blacklist': [
            'rm -rf /',
            'dd if=/dev/random',
            ':(){:|:&};:',
            'mkfs',
            'fdisk'
        ],
        'training': {
            'enabled': True,
            'data_path': 'src/training/zsh_training_data.jsonl',
            'output_path': 'zsh-lora-output',
            'max_examples': 500
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f'Configuration created: {config_file}')
else:
    print('Configuration already exists')
\"" "Creating configuration"; then
    print_success "Configuration created"
else
    print_warning "Configuration creation had issues"
fi

# ============================================================================
# PHASE 7: DOWNLOAD AND IMPORT MERGED MODEL (base + adapter from Hugging Face)
# ============================================================================

print_step "Phase 7: Downloading and importing merged model (base + adapter from Hugging Face)..."

# Ensure Ollama is running before checking for model
print_info "Ensuring Ollama server is running..."
if run_command "$PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from model_completer.ollama_manager import create_ollama_manager
manager = create_ollama_manager()
if not manager.is_running():
    print('Starting Ollama server...')
    if manager.start_server():
        import time
        time.sleep(3)  # Wait for server to start
        print('Ollama server started')
    else:
        print('Failed to start Ollama server')
        sys.exit(1)
else:
    print('Ollama server already running')
\"" "Ensuring Ollama is running"; then
    print_success "Ollama server ready"
else
    print_warning "Ollama server may not be running, model check may fail"
fi

# Get HF repo ID so we load base+adapter from Hugging Face into Ollama
HF_REPO=$($PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')
from model_completer.utils import load_config
try:
    config = load_config()
    repo = config.get('hf_lora_repo', '')
    if repo:
        print(repo)
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")

if [ -z "$HF_REPO" ]; then
    HF_REPO=$($PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')
import yaml
import os
try:
    config_file = os.path.join('config', 'default.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
            repo = config.get('hf_lora_repo', '')
            if repo:
                print(repo)
    print('')
except:
    print('')
" 2>/dev/null || echo "")
fi

if [ -z "$HF_REPO" ]; then
    print_warning "No Hugging Face repo configured. Model will not be downloaded."
    print_info "Set hf_lora_repo in config (e.g. duoyuncloud/zsh-cli-lora) and run install again"
else
    print_info "Loading base + adapter from Hugging Face and importing to Ollama: $HF_REPO"
    print_info "This merges the LoRA adapter with the base model and loads the result into Ollama (5-15 min)..."
    
    if run_command "$PYTHON_CMD -c \"
import sys
sys.path.insert(0, 'src')
from model_completer.ollama_lora_import import import_lora_to_ollama
import logging
logging.basicConfig(level=logging.INFO)

print('📥 Starting model download and import...')
print('   This includes:')
print('   1. Downloading LoRA adapter from Hugging Face')
print('   2. Downloading base model (Qwen2-0.5B)')
print('   3. Merging adapter with base model')
print('   4. Importing to Ollama as zsh-assistant')
print('')
print('   This may take 5-15 minutes depending on your internet speed...')
print('')

success = import_lora_to_ollama(hf_repo_id='$HF_REPO', use_merged_model=True)
if success:
    print('')
    print('✅ Model (base+adapter) imported to Ollama! Use zsh-assistant for autocomplete.')
    sys.exit(0)
else:
    print('')
    print('❌ Failed to import model')
    print('   You can try running manually: python -m model_completer.cli --import-to-ollama')
    sys.exit(1)
\"" "Download and import base+adapter from Hugging Face to Ollama"; then
    print_success "Model (base+adapter from Hugging Face) loaded into Ollama successfully!"
else
    print_warning "Model import had issues. Check install.log"
    print_info "Retry: python -m model_completer.cli --import-to-ollama"
fi
fi

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print_success "🎉 One-click install finished!"
echo ""
echo "  Merged model (base + adapter) is in Ollama as: zsh-assistant"
echo ""
echo "  Reload your shell, then use Tab for autocomplete:"
echo "    source ~/.zshrc"
echo ""
echo "  Try: git comm[Tab]   docker run[Tab]   npm run[Tab]"
echo ""

# Make script executable for future
chmod +x "$SCRIPT_DIR/install.sh"

# Setup PATH for model-completer command
print_step "Setting up PATH for model-completer command..."

# Add to ~/.zshrc if not already there
if ! grep -q "zsh-llm-cli-autocomplete-tool.*PATH" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# Model CLI Autocomplete PATH setup" >> ~/.zshrc
    echo "export PATH=\"$SCRIPT_DIR/venv/bin:\$PATH\"" >> ~/.zshrc
    echo "export PATH=\"$SCRIPT_DIR/bin:\$PATH\"" >> ~/.zshrc
    print_success "Added model-completer to PATH in ~/.zshrc"
else
    print_info "PATH already configured in ~/.zshrc"
fi

echo "  Next: source ~/.zshrc   (then Tab-complete any command)"
echo ""