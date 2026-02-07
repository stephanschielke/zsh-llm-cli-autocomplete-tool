#!/usr/bin/env python3
"""
Qwen3 LoRA fine-tuning using Axolotl with optimized parameters for CLI completion.
"""

import os
import subprocess
import yaml
import sys
import time
from pathlib import Path
import psutil


class Qwen3AxolotlTrainer:
    """Trainer for Qwen LoRA fine-tuning optimized for CLI command completion."""
    
    def __init__(self, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.base_model = base_model
        self.config_path = "src/training/qwen3_axolotl_config.yml"
        self.base_dir = Path(__file__).parent.parent.parent
        self.training_start_time = None
        
    def print_banner(self):
        """Print training banner."""
        print("=" * 70)
        print("🤖 Qwen3 LoRA Fine-tuning for CLI Command Completion")
        print("=" * 70)
        print(f"🎯 Base Model: {self.base_model}")
        print("⚡ Using Axolotl for efficient LoRA fine-tuning")
        print("=" * 70)
    
    def check_system_resources(self):
        """Check if system has enough resources for training."""
        print("🔍 Checking system resources...")
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"   RAM: {ram_gb:.1f} GB available")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        print(f"   Disk: {disk_free_gb:.1f} GB free")
        
        if ram_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM may cause issues")
        
        if disk_free_gb < 10:
            print("❌ Error: Need at least 10GB free disk space")
            return False
            
        return True
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = {
            'axolotl': 'axolotl',
            'torch': 'torch',
            'transformers': 'transformers',
            'accelerate': 'accelerate',
            'datasets': 'datasets',
            'peft': 'peft',
        }
        
        all_available = True
        for package, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package}")
                all_available = False
        
        if not all_available:
            print("\n💡 Install missing packages with:")
            print("   pip install axolotl torch transformers accelerate datasets peft")
            return False
        
        return True
    
    def check_gpu(self):
        """Check GPU availability and capabilities."""
        print("🔍 Checking GPU...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    print(f"   ✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Check if we have enough VRAM for 1.7B model
                if gpu_memory < 4:
                    print("⚠️  Warning: GPU memory may be insufficient")
                    print("💡 Consider using --low-memory mode")
                
                return True
            else:
                print("   ⚠️  No GPU detected - will use CPU (very slow!)")
                return False
                
        except Exception as e:
            print(f"   ❌ GPU check failed: {e}")
            return False
    
    def verify_training_data(self):
        """Verify training data exists and is valid."""
        print("🔍 Verifying training data...")
        
        # Check for split data first
        train_path = self.base_dir / "src/training/data_splits_axolotl/train_axolotl.jsonl"
        val_path = self.base_dir / "src/training/data_splits_axolotl/val_axolotl.jsonl"
        
        if train_path.exists() and val_path.exists():
            # Count lines
            with open(train_path, 'r') as f:
                train_count = sum(1 for _ in f)
            with open(val_path, 'r') as f:
                val_count = sum(1 for _ in f)
            
            print(f"   ✅ Training data: {train_count} train, {val_count} val examples")
            return True, True  # (has_split_data, is_valid)
        
        # Fallback to single file
        data_path = self.base_dir / "src/training/zsh_training_data_axolotl.jsonl"
        if data_path.exists():
            with open(data_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"   ✅ Training data: {line_count} examples (single file)")
            return False, True
        
        print(f"   ❌ Training data not found")
        print("   💡 Run data preparation scripts first:")
        print("      python -m training.prepare_and_split_data")
        print("      python -m training.convert_to_axolotl_format --split-data")
        return False, False
    
    def create_axolotl_config(
        self, 
        low_memory: bool = False,
        use_split_data: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3
    ):
        """
        Create the Axolotl configuration file optimized for Qwen3.
        
        Args:
            low_memory: Use low memory configuration
            use_split_data: Use split data (train/val) instead of single file
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        print("⚙️  Creating training configuration...")
        
        # Check GPU availability
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except:
            has_gpu = False
        
        # Adjust configuration based on available resources
        if low_memory:
            micro_batch_size = 1
            gradient_accumulation = 8
            sequence_len = 512  # Reduced for low memory
        else:
            micro_batch_size = 2
            gradient_accumulation = 4
            sequence_len = 1024  # Reduced from 2048
        
        # Determine data path
        if use_split_data:
            train_data_path = str(self.base_dir / "src/training/data_splits_axolotl/train_axolotl.jsonl")
            val_data_path = str(self.base_dir / "src/training/data_splits_axolotl/val_axolotl.jsonl")
            
            datasets = [
                {
                    "path": train_data_path,
                    "type": "alpaca"
                }
            ]
            val_set_size = 0.0  # We have separate val file
        else:
            data_path = str(self.base_dir / "src/training/zsh_training_data_axolotl.jsonl")
            datasets = [
                {
                    "path": data_path,
                    "type": "alpaca"
                }
            ]
            val_set_size = 0.1
        
        # Qwen3-specific LoRA target modules
        # Qwen3 uses similar architecture to Qwen2, targeting attention and MLP layers
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        config = {
            "base_model": self.base_model,
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "trust_remote_code": True,
            
            "datasets": datasets,
            "dataset_prepared_path": str(self.base_dir / "last_run_prepared"),
            "val_set_size": val_set_size,
            "output_dir": str(self.base_dir / "zsh-lora-output"),
            
            # LoRA configuration optimized for CLI completion
            "adapter": "lora",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "lora_modules_to_save": ["embed_tokens", "lm_head"],
            
            # Training configuration
            "sequence_len": sequence_len,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "num_epochs": num_epochs,
            "optimizer": "adamw_bnb_8bit" if has_gpu else "adamw_torch",  # Use 8-bit optimizer only if GPU available
            "lr_scheduler": "cosine",
            "learning_rate": learning_rate,
            "warmup_steps": 50,
            
            # Memory optimization
            # Use 4-bit quantization for CPU/low-memory systems
            "load_in_8bit": False,
            "load_in_4bit": not has_gpu or low_memory,  # Use 4-bit for CPU or low-memory
            "bf16": "auto" if has_gpu else False,  # Use bf16 only with GPU
            "fp16": False,
            "tf32": False,
            
            "gradient_checkpointing": True,
            "group_by_length": False,
            
            # Logging and saving
            "logging_steps": 10,
            "save_steps": 200,
            "save_safetensors": True,
            "save_total_limit": 3,
            
            # Early stopping (only if using validation split, not separate val dataset)
            "early_stopping_patience": 3 if not use_split_data else None,
            "early_stopping_threshold": 0.001 if not use_split_data else None,
            
            # System
            "wandb_mode": "disabled",
            
            # Explicitly disable fp8 for CPU
            "fp8": False
        }
        
        # Add validation dataset if using split data
        # Note: When using separate val_dataset, eval_steps/eval_strategy are not supported
        if use_split_data:
            config["val_dataset"] = {
                "path": val_data_path,
                "type": "alpaca"
            }
            # Do not set eval_steps or eval_strategy when using separate val_dataset
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"   ✅ Config created: {self.config_path}")
        print(f"   📋 LoRA params: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"   📋 Training: lr={learning_rate}, epochs={num_epochs}")
        return True
    
    def run_training(self):
        """Run the actual Axolotl training."""
        print("🚀 Starting LoRA fine-tuning...")
        
        # Change to project root directory
        os.chdir(self.base_dir)
        
        # Record start time
        self.training_start_time = time.time()
        
        # Apply patch for CPU support before training
        patch_script = self.base_dir / "src/training/patch_axolotl.py"
        
        # Run Axolotl training with patch
        cmd = [
            "python", str(patch_script), "&&",
            "accelerate", "launch", "-m", "axolotl.cli.train",
            self.config_path
        ]
        
        # Use shell to run patch first
        import subprocess
        # First apply patch
        subprocess.run(["python", str(patch_script)], check=False)
        
        # Then run training
        cmd = [
            "accelerate", "launch", "-m", "axolotl.cli.train",
            self.config_path
        ]
        
        try:
            print("⏳ Training in progress... This may take 30 minutes to 2 hours.")
            print("💡 You can monitor progress in the logs above.")
            print("   Look for decreasing loss values to track training progress.")
            print("-" * 50)
            
            result = subprocess.run(cmd, check=True)
            
            training_time = time.time() - self.training_start_time
            print(f"✅ Training completed in {training_time/60:.1f} minutes!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with exit code: {e.returncode}")
            return False
        except FileNotFoundError:
            print("❌ 'accelerate' command not found.")
            print("💡 Install with: pip install accelerate")
            return False
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            return False
    
    def post_training_analysis(self):
        """Analyze training results."""
        output_dir = self.base_dir / "zsh-lora-output"
        
        if not output_dir.exists():
            print("❌ No training output found")
            return False
        
        print("\n📊 Training Results:")
        print("=" * 40)
        
        # List generated files
        files = list(output_dir.glob("*"))
        if files:
            print("📁 Generated files:")
            for file in files:
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   {file.name} ({size_mb:.1f} MB)")
        
        # Check for adapter files
        adapter_files = list(output_dir.glob("*adapter*"))
        if adapter_files:
            print(f"\n✅ LoRA adapter files: {len(adapter_files)}")
            for adapter_file in adapter_files:
                print(f"   {adapter_file.name}")
        else:
            print("❌ No LoRA adapter files found")
        
        return len(adapter_files) > 0
    
    def train_complete_pipeline(
        self,
        low_memory: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3
    ):
        """Run the complete training pipeline."""
        self.print_banner()
        
        # Step 1: System checks
        # Skip disk space check for retraining (we'll manage space manually)
        # if not self.check_system_resources():
        #     return False
        
        if not self.check_dependencies():
            return False
        
        self.check_gpu()
        
        # Step 2: Data verification
        has_split_data, is_valid = self.verify_training_data()
        if not is_valid:
            return False
        
        # Step 3: Configuration
        if not self.create_axolotl_config(
            low_memory=low_memory,
            use_split_data=has_split_data,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        ):
            return False
        
        # Step 4: Training
        if not self.run_training():
            return False
        
        # Step 5: Results analysis
        success = self.post_training_analysis()
        
        if success:
            print("\n🎉 LoRA Fine-tuning Completed Successfully!")
            print("=" * 50)
            print("🔧 Next steps:")
            print("   1. Evaluate on test set: python -m training.evaluate_model")
            print("   2. Upload to HuggingFace: python -m training.upload_to_hf")
            print("   3. Test locally: python -m training.test_adapter")
        else:
            print("\n❌ Training completed with issues")
        
        return success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Qwen3 LoRA adapter with Axolotl')
    parser.add_argument('--low-memory', action='store_true',
                       help='Use low memory configuration')
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Base model name (default: Qwen/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha (default: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='LoRA dropout (default: 0.05)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    
    args = parser.parse_args()
    
    trainer = Qwen3AxolotlTrainer(base_model=args.base_model)
    success = trainer.train_complete_pipeline(
        low_memory=args.low_memory,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

