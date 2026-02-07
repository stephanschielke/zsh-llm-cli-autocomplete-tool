#!/usr/bin/env python3
"""
Upload trained LoRA adapter to HuggingFace Hub.
"""

import os
import sys
from pathlib import Path
import argparse

try:
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceUploader:
    """Upload LoRA adapter to HuggingFace Hub."""
    
    def __init__(self, adapter_path: str):
        self.adapter_path = Path(adapter_path)
        self.api = HfApi() if HF_AVAILABLE else None
    
    def check_adapter(self) -> bool:
        """Check if adapter directory exists and is valid."""
        if not self.adapter_path.exists():
            print(f"❌ Adapter directory not found: {self.adapter_path}")
            return False
        
        # Check for required files
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = []
        
        for file in required_files:
            if not (self.adapter_path / file).exists():
                missing.append(file)
        
        if missing:
            print(f"❌ Missing required files: {', '.join(missing)}")
            return False
        
        print(f"✅ Adapter found at: {self.adapter_path}")
        return True
    
    def upload(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Upload LoRA adapter for CLI completion"
    ) -> bool:
        """
        Upload adapter to HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/repo-name")
            private: Whether to make repository private
            commit_message: Commit message for upload
        """
        if not HF_AVAILABLE:
            print("❌ huggingface_hub not available")
            print("💡 Install with: pip install huggingface_hub")
            return False
        
        print("=" * 70)
        print("📤 Uploading LoRA Adapter to HuggingFace Hub")
        print("=" * 70)
        print(f"Repository: {repo_id}")
        print(f"Adapter:   {self.adapter_path}")
        print("=" * 70)
        
        # Check if logged in
        try:
            user = self.api.whoami()
            print(f"✅ Logged in as: {user.get('name', 'unknown')}")
        except Exception:
            print("🔐 Not logged in. Please login:")
            print("   huggingface-cli login")
            print("   or")
            print("   from huggingface_hub import login; login()")
            try:
                login()
                user = self.api.whoami()
                print(f"✅ Logged in as: {user.get('name', 'unknown')}")
            except Exception as e:
                print(f"❌ Login failed: {e}")
                return False
        
        # Create repository if it doesn't exist
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"✅ Repository ready: {repo_id}")
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                print("❌ Unauthorized. Please check your HuggingFace token.")
                return False
            raise
        
        # Upload files
        print("\n📤 Uploading files...")
        try:
            self.api.upload_folder(
                folder_path=str(self.adapter_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )
            print(f"✅ Upload completed!")
            print(f"\n🔗 Repository URL: https://huggingface.co/{repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_readme(self, repo_id: str, base_model: str = "Qwen/Qwen3-1.7B") -> str:
        """Generate README content for HuggingFace repository."""
        readme = f"""---
license: mit
base_model: {base_model}
tags:
- lora
- cli
- command-completion
- zsh
- qwen3
---

# CLI Command Completion LoRA Adapter

This is a LoRA (Low-Rank Adaptation) adapter fine-tuned on Qwen3-1.7B for CLI command completion tasks.

## Model Details

- **Base Model**: {base_model}
- **Adapter Type**: LoRA
- **Task**: CLI Command Completion
- **Training Data**: Zsh command completion pairs

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details

This adapter was trained for CLI command completion, helping users complete partial Zsh commands intelligently.

## License

MIT License
"""
        return readme


def main():
    parser = argparse.ArgumentParser(
        description='Upload LoRA adapter to HuggingFace Hub'
    )
    parser.add_argument(
        '--adapter', '-a',
        default='zsh-lora-output',
        help='Path to adapter directory'
    )
    parser.add_argument(
        '--repo-id', '-r',
        required=True,
        help='HuggingFace repository ID (e.g., username/repo-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private'
    )
    parser.add_argument(
        '--base-model', '-b',
        default='Qwen/Qwen3-1.7B',
        help='Base model name for README'
    )
    parser.add_argument(
        '--commit-message', '-m',
        default='Upload LoRA adapter for CLI completion',
        help='Commit message'
    )
    
    args = parser.parse_args()
    
    # Resolve adapter path
    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        adapter_path = project_root / adapter_path
    
    uploader = HuggingFaceUploader(str(adapter_path))
    
    if not uploader.check_adapter():
        sys.exit(1)
    
    success = uploader.upload(
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        # Optionally create README
        readme_content = uploader.create_readme(args.repo_id, args.base_model)
        readme_path = adapter_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"\n📝 README.md created at: {readme_path}")
        print("💡 You can upload it manually or include it in the next upload")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

