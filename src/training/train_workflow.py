#!/usr/bin/env python3
"""
Complete training workflow for Qwen3 LoRA fine-tuning.
Orchestrates data preparation, training, evaluation, and optimization.
"""

import sys
import subprocess
from pathlib import Path
import argparse


class TrainingWorkflow:
    """Complete training workflow orchestrator."""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.scripts_dir = self.base_dir / "src" / "training"
    
    def print_banner(self):
        """Print workflow banner."""
        print("=" * 70)
        print("🚀 Qwen3 LoRA Training Workflow for CLI Completion")
        print("=" * 70)
        print("This workflow will:")
        print("  1. Prepare and split training data")
        print("  2. Convert data to Axolotl format")
        print("  3. Train LoRA adapter")
        print("  4. Evaluate on test set")
        print("=" * 70)
    
    def run_step(self, step_name: str, script_path: Path, args: list = None) -> bool:
        """Run a workflow step."""
        print(f"\n{'=' * 70}")
        print(f"📋 Step: {step_name}")
        print(f"{'=' * 70}")
        
        cmd = ["python3", str(script_path)]
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(cmd, cwd=self.base_dir, check=True)
            print(f"✅ {step_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {step_name} failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"❌ Python3 not found")
            return False
    
    def step1_prepare_data(self, input_file: str = None) -> bool:
        """Step 1: Prepare and split data."""
        if input_file is None:
            input_file = str(self.scripts_dir / "zsh_training_data.jsonl")
        
        script = self.scripts_dir / "prepare_and_split_data.py"
        
        args = [
            "--input", input_file,
            "--output-dir", str(self.scripts_dir / "data_splits"),
            "--train-ratio", "0.7",
            "--val-ratio", "0.15",
            "--test-ratio", "0.15"
        ]
        
        return self.run_step("Data Preparation & Splitting", script, args)
    
    def step2_convert_format(self) -> bool:
        """Step 2: Convert to Axolotl format."""
        script = self.scripts_dir / "convert_to_axolotl_format.py"
        
        args = [
            "--split-data",
            "--input", str(self.scripts_dir / "data_splits"),
            "--output", str(self.scripts_dir / "data_splits_axolotl")
        ]
        
        return self.run_step("Data Format Conversion", script, args)
    
    def step3_train(
        self,
        base_model: str = "Qwen/Qwen3-1.7B",
        low_memory: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3
    ) -> bool:
        """Step 3: Train LoRA adapter."""
        script = self.scripts_dir / "qwen3_axolotl_trainer.py"
        
        args = [
            "--base-model", base_model,
            "--lora-r", str(lora_r),
            "--lora-alpha", str(lora_alpha),
            "--lora-dropout", str(lora_dropout),
            "--learning-rate", str(learning_rate),
            "--epochs", str(num_epochs)
        ]
        
        if low_memory:
            args.append("--low-memory")
        
        return self.run_step("LoRA Training", script, args)
    
    def step4_evaluate(self, base_model: str = "Qwen/Qwen3-1.7B") -> bool:
        """Step 4: Evaluate on test set."""
        script = self.scripts_dir / "evaluate_model.py"
        
        args = [
            "--adapter", str(self.base_dir / "zsh-lora-output"),
            "--base-model", base_model,
            "--test-file", str(self.scripts_dir / "data_splits_axolotl" / "test_axolotl.jsonl"),
            "--output", str(self.base_dir / "evaluation_report.json")
        ]
        
        return self.run_step("Model Evaluation", script, args)
    
    def run_complete_workflow(
        self,
        base_model: str = "Qwen/Qwen3-1.7B",
        low_memory: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        skip_data_prep: bool = False,
        skip_evaluation: bool = False
    ) -> bool:
        """Run complete training workflow."""
        self.print_banner()
        
        # Step 1: Prepare data
        if not skip_data_prep:
            if not self.step1_prepare_data():
                print("\n❌ Workflow failed at data preparation step")
                return False
        else:
            print("\n⏭️  Skipping data preparation (using existing splits)")
        
        # Step 2: Convert format
        if not skip_data_prep:
            if not self.step2_convert_format():
                print("\n❌ Workflow failed at format conversion step")
                return False
        else:
            print("\n⏭️  Skipping format conversion (using existing converted data)")
        
        # Step 3: Train
        if not self.step3_train(
            base_model=base_model,
            low_memory=low_memory,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        ):
            print("\n❌ Workflow failed at training step")
            return False
        
        # Step 4: Evaluate
        if not skip_evaluation:
            if not self.step4_evaluate(base_model=base_model):
                print("\n⚠️  Evaluation failed, but training completed")
                # Don't fail workflow if evaluation fails
        else:
            print("\n⏭️  Skipping evaluation")
        
        print("\n" + "=" * 70)
        print("🎉 Training Workflow Completed!")
        print("=" * 70)
        print("\n📁 Output files:")
        print(f"   LoRA adapter: {self.base_dir / 'zsh-lora-output'}")
        if not skip_evaluation:
            print(f"   Evaluation report: {self.base_dir / 'evaluation_report.json'}")
        print("\n🔧 Next steps:")
        print("   1. Review evaluation report")
        print("   2. Test adapter: python -m training.test_adapter")
        print("   3. Upload to HuggingFace: python -m training.upload_to_hf")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete Qwen3 LoRA training workflow'
    )
    parser.add_argument(
        '--base-model',
        default='Qwen/Qwen3-1.7B',
        help='Base model name (default: Qwen/Qwen3-1.7B)'
    )
    parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Use low memory configuration'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank (default: 16)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha (default: 32)'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout (default: 0.05)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs (default: 3)'
    )
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation (use existing splits)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step'
    )
    
    args = parser.parse_args()
    
    workflow = TrainingWorkflow()
    success = workflow.run_complete_workflow(
        base_model=args.base_model,
        low_memory=args.low_memory,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        skip_data_prep=args.skip_data_prep,
        skip_evaluation=args.skip_evaluation
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

