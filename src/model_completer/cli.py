#!/usr/bin/env python3
"""CLI interface for AI command completion."""

import argparse
import sys
import os
import re
from typing import Dict, Optional

# Add the src directory to the path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..')
sys.path.insert(0, os.path.abspath(src_dir))

from model_completer.utils import load_config, setup_logging
from model_completer.client import OllamaClient


# Same instruction as chat_merged_model.py — model just completes the command.
_COMPLETION_SYSTEM = (
    "You are a shell command completion assistant. "
    "Given a partial command, reply with ONLY the completed full command on one line. "
    "No explanation, no prefix like 'Complete:' or 'The command is:', no markdown. Just the command."
)


def _fast_completion(command: str, url: str, model: str, timeout: int = 3) -> Optional[str]:
    """Merged model only. System prompt + partial command -> one completed line."""
    import requests
    data = {
        "model": model,
        "system": _COMPLETION_SYSTEM,
        "prompt": command.strip(),
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 48, "num_ctx": 512},
    }
    try:
        r = requests.post(f"{url.rstrip('/')}/api/generate", json=data, timeout=timeout)
        if r.status_code != 200:
            return None
        text = (r.json().get("response") or "").strip()
        if not text or len(text) <= len(command):
            return None
        # First line that looks like a command (extends input or same base command)
        prefix = command.rstrip()
        first = (command.split() or [""])[0]
        for line in text.split("\n"):
            line = line.strip().replace("```", "").strip()
            if not line or len(line) <= len(command):
                continue
            if any(line.startswith(x) for x in ("Complete", "Output", "The ", "Sure,", "Here")):
                continue
            if prefix and line.startswith(prefix):
                return line
            if first and line.startswith(first):
                return line
        return text.split("\n")[0].strip() if text else None
    except Exception:
        return None


def get_ai_completion(command: str, config: Optional[Dict] = None) -> str:
    """Get AI completion using enhanced completer with personalization."""
    from model_completer.utils import load_config
    from model_completer.enhanced_completer import EnhancedCompleter
    if config is None:
        config = load_config()
    completer = EnhancedCompleter(
        ollama_url=config.get('ollama', {}).get('url', 'http://localhost:11434'),
        model=config.get('model', 'zsh-assistant'),
        config=config
    )
    return completer.get_completion(command)


def main():
    parser = argparse.ArgumentParser(description='AI Command Completion - Simple Tab completion with personalized predictions')
    parser.add_argument('command', nargs='?', help='Command to complete')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--test', action='store_true', help='Test completions')
    parser.add_argument('--train', action='store_true', help='Start LoRA training')
    parser.add_argument('--generate-data', action='store_true', help='Generate training data')
    parser.add_argument('--import-to-ollama', action='store_true', help='Import fine-tuned LoRA model to Ollama')
    parser.add_argument('--upload-to-hf', metavar='REPO_ID', help='Upload LoRA adapter to Hugging Face (e.g., username/model-name)')
    parser.add_argument('--hf-token', help='Hugging Face API token (or set HF_TOKEN env var)')
    parser.add_argument('--hf-private', action='store_true', help='Create private Hugging Face repository')
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    # Silent mode if called with a command (from Zsh plugin)
    silent = args.command is not None
    setup_logging(config, silent=silent)
    
    # Initialize Ollama client for model listing
    client = OllamaClient(config.get('ollama', {}).get('url', 'http://localhost:11434'))
    
    if args.list_models:
        if client.is_server_available():
            models = client.get_available_models()
            if models:
                print("Available models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("No models found")
        else:
            print("Could not connect to Ollama server")
    elif args.train:
        from model_completer.training import create_trainer
        print("🚀 Starting LoRA training...")
        trainer = create_trainer()
        data_file = "src/training/zsh_training_data.jsonl"
        success = trainer.train(data_file)
        if success:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed")
            sys.exit(1)
    elif args.generate_data:
        print("📊 Generating training data...")
        from model_completer.training import TrainingDataManager
        data_manager = TrainingDataManager()
        data_file = data_manager.generate_training_data()
        print(f"✅ Training data generated: {data_file}")
    elif args.import_to_ollama:
        print("📦 Importing fine-tuned LoRA model to Ollama...")
        from model_completer.ollama_lora_import import import_lora_to_ollama
        import logging
        
        # Enable more verbose logging for debugging
        logging.basicConfig(level=logging.INFO)
        
        # Get HF repo ID from config if available
        hf_repo_id = config.get('hf_lora_repo', '')
        if hf_repo_id:
            print(f"   Using pre-trained model from Hugging Face: {hf_repo_id}")
        else:
            print("   No HF repo configured, will use local adapter if available")
        
        # Check Ollama first
        try:
            import subprocess
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("❌ Ollama is not installed or not working")
                print("   Install it from: https://ollama.ai")
                sys.exit(1)
        except FileNotFoundError:
            print("❌ Ollama is not installed")
            print("   Install it from: https://ollama.ai")
            sys.exit(1)
        
        # Check if Ollama server is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                print("⚠️  Ollama server may not be running")
                print("   Starting Ollama server...")
                subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                import time
                time.sleep(3)
        except Exception as e:
            print(f"⚠️  Cannot connect to Ollama server: {e}")
            print("   Try starting it manually: ollama serve")
        
        # Force merge when HF repo is provided to use actual LoRA adapter
        if import_lora_to_ollama(hf_repo_id=hf_repo_id if hf_repo_id else None, force_merge=bool(hf_repo_id)):
            print("✅ Model imported to Ollama successfully!")
            print("   You can now use 'zsh-assistant' model for completions")
        else:
            print("❌ Failed to import model to Ollama")
            print("   Troubleshooting:")
            print("   1. Check if Ollama is running: ollama list")
            print("   2. If model already exists, remove it: ollama rm zsh-assistant")
            print("   3. Check logs above for specific error messages")
            print("   4. Make sure hf_lora_repo is set in config.yaml")
            sys.exit(1)
    elif args.upload_to_hf:
        print(f"📤 Uploading LoRA adapter to Hugging Face: {args.upload_to_hf}")
        from model_completer.hf_uploader import upload_lora_to_hf
        import os
        
        # Set token if provided
        if args.hf_token:
            os.environ['HF_TOKEN'] = args.hf_token
        
        if upload_lora_to_hf(
            repo_id=args.upload_to_hf,
            token=args.hf_token,
            private=args.hf_private
        ):
            print(f"✅ Successfully uploaded to: https://huggingface.co/{args.upload_to_hf}")
        else:
            print("❌ Failed to upload to Hugging Face")
            print("   Make sure:")
            print("   1. LoRA training is completed (run --train first)")
            print("   2. You have a Hugging Face account and token")
            print("   3. Run: huggingface-cli login (or set HF_TOKEN env var)")
            sys.exit(1)
    elif args.test:
        print("Testing merged model (zsh-assistant) only:")
        url = config.get('ollama', {}).get('url', 'http://localhost:11434')
        model = config.get('model', 'zsh-assistant')
        for cmd in ["git ad", "git comm", "docker run", "npm run", "kubectl get"]:
            completion = _fast_completion(cmd, url, model)
            print(f"  {cmd} -> {completion or cmd}")
    elif args.command:
        # Tab: merged model only, no enhanced completer
        url = config.get('ollama', {}).get('url', 'http://localhost:11434')
        model = config.get('model', 'zsh-assistant')
        completion = _fast_completion(args.command, url, model)
        if completion and len(completion) > len(args.command):
            print(completion.split('\n')[0].strip(), flush=True)
        else:
            print(args.command, flush=True)
    else:
        print("AI Command Completer - Simple Tab completion with personalized predictions")
        print("Usage: model-completer 'git comm'")
        print("Options:")
        print("  --test            Test the system")
        print("  --list-models     List available models")
        print("  --train           Start LoRA training")
        print("  --generate-data   Generate training data")
        print("  --import-to-ollama Import trained model to Ollama")
        print("  --upload-to-hf REPO_ID  Upload LoRA adapter to Hugging Face")
        print("  --hf-token TOKEN  Hugging Face API token (or set HF_TOKEN env var)")
        print("  --hf-private      Create private Hugging Face repository")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Suppress traceback for Zsh plugin usage
        if len(sys.argv) > 1 and sys.argv[1] and not sys.argv[1].startswith('--'):
            # Called with a command from Zsh plugin - return original command on error
            print(sys.argv[1], flush=True)
            sys.exit(0)
        else:
            # Called directly - show error
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
