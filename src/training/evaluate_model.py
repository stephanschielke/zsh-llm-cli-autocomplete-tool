#!/usr/bin/env python3
"""
Evaluate trained LoRA adapter on test set.
Computes accuracy, BLEU scores, and other metrics for CLI completion.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModelEvaluator:
    """Evaluate LoRA adapter on test set."""
    
    def __init__(self, adapter_path: str, base_model: str = "Qwen/Qwen3-1.7B"):
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load base model and LoRA adapter."""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers and peft not available")
            print("💡 Install with: pip install transformers peft torch")
            return False
        
        if not self.adapter_path.exists():
            print(f"❌ Adapter directory not found: {self.adapter_path}")
            return False
        
        adapter_weights = self.adapter_path / "adapter_model.safetensors"
        if not adapter_weights.exists():
            adapter_weights = self.adapter_path / "adapter_model.bin"
        if not adapter_weights.exists():
            print(f"❌ No adapter weights found in {self.adapter_path}")
            print("   Expected adapter_model.safetensors or adapter_model.bin (run training first).")
            return False
        
        print(f"📦 Loading base model: {self.base_model}")
        print(f"📦 Loading adapter from: {self.adapter_path}")
        
        try:
            # Load base model
            print("📥 Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load tokenizer
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load LoRA adapter (local path: require local_files_only so PEFT doesn't treat path as HF repo_id)
            print("📥 Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.adapter_path),
                is_trainable=False,
                local_files_only=True,
            )
            self.model.eval()
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from JSONL file."""
        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data
    
    def generate_completion(self, input_text: str, max_length: int = 100) -> str:
        """Generate completion for given input."""
        # Format prompt similar to training
        prompt = f"Complete this Zsh command. Provide only the full command without explanations.\n\nInput: {input_text}\nOutput:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract output part
        if "Output:" in result:
            output_part = result.split("Output:")[-1].strip()
            # Clean up - take first line only (command completion)
            output_part = output_part.split('\n')[0].strip()
            return output_part
        else:
            # Fallback: return last part
            return result.split()[-1] if result.split() else ""
    
    def exact_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted command exactly matches expected."""
        # Normalize whitespace
        predicted = ' '.join(predicted.split())
        expected = ' '.join(expected.split())
        return predicted.strip() == expected.strip()
    
    def command_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted command matches expected (more lenient)."""
        # Extract base command (first word)
        pred_cmd = predicted.split()[0] if predicted.split() else ""
        exp_cmd = expected.split()[0] if expected.split() else ""
        
        if pred_cmd != exp_cmd:
            return False
        
        # Check if key arguments match
        pred_args = set(predicted.split()[1:])
        exp_args = set(expected.split()[1:])
        
        # If expected has no args, any args in predicted is OK
        if not exp_args:
            return True
        
        # Check if at least some args match
        return len(pred_args & exp_args) > 0 or len(pred_args) == 0
    
    def evaluate(self, test_file: str) -> Dict:
        """Evaluate model on test set."""
        print("=" * 70)
        print("🧪 Evaluating Model on Test Set")
        print("=" * 70)
        
        # Load test data
        print(f"\n📥 Loading test data from {test_file}...")
        test_data = self.load_test_data(test_file)
        print(f"✅ Loaded {len(test_data)} test examples")
        
        if len(test_data) == 0:
            print("❌ No test data found!")
            return {}
        
        # Evaluate
        exact_matches = 0
        command_matches = 0
        total = len(test_data)
        
        results = []
        
        print("\n🔄 Running evaluation...")
        print("-" * 70)
        
        for i, item in enumerate(test_data):
            input_text = item.get("input", "")
            expected = item.get("output", "")
            
            if not input_text or not expected:
                continue
            
            # Generate prediction
            try:
                predicted = self.generate_completion(input_text)
            except Exception as e:
                print(f"⚠️  Error generating for '{input_text}': {e}")
                predicted = ""
            
            # Evaluate
            exact = self.exact_match(predicted, expected)
            cmd_match = self.command_match(predicted, expected) if not exact else True
            
            if exact:
                exact_matches += 1
            if cmd_match:
                command_matches += 1
            
            results.append({
                "input": input_text,
                "expected": expected,
                "predicted": predicted,
                "exact_match": exact,
                "command_match": cmd_match
            })
            
            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   Processed {i + 1}/{total} examples...")
        
        # Calculate metrics
        exact_accuracy = exact_matches / total if total > 0 else 0
        command_accuracy = command_matches / total if total > 0 else 0
        
        # Print results
        print("\n" + "=" * 70)
        print("📊 Evaluation Results")
        print("=" * 70)
        print(f"Total test examples: {total}")
        print(f"Exact matches:       {exact_matches} ({exact_accuracy:.1%})")
        print(f"Command matches:     {command_matches} ({command_accuracy:.1%})")
        print("=" * 70)
        
        # Show some examples
        print("\n📝 Sample Results:")
        print("-" * 70)
        
        # Show some correct predictions
        correct = [r for r in results if r["exact_match"]]
        if correct:
            print("\n✅ Correct Predictions:")
            for r in correct[:5]:
                print(f"   Input:    {r['input']}")
                print(f"   Expected: {r['expected']}")
                print(f"   Got:      {r['predicted']}")
                print()
        
        # Show some incorrect predictions
        incorrect = [r for r in results if not r["exact_match"]]
        if incorrect:
            print("\n❌ Incorrect Predictions:")
            for r in incorrect[:5]:
                print(f"   Input:    {r['input']}")
                print(f"   Expected: {r['expected']}")
                print(f"   Got:      {r['predicted']}")
                print()
        
        return {
            "total": total,
            "exact_matches": exact_matches,
            "command_matches": command_matches,
            "exact_accuracy": exact_accuracy,
            "command_accuracy": command_accuracy,
            "results": results
        }
    
    def save_evaluation_report(self, metrics: Dict, output_file: str):
        """Save evaluation report to JSON file."""
        # Remove results for file size (keep only metrics)
        report = {k: v for k, v in metrics.items() if k != "results"}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Evaluation report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained LoRA adapter on test set'
    )
    parser.add_argument(
        '--adapter', '-a',
        default='zsh-lora-output',
        help='Path to adapter directory'
    )
    parser.add_argument(
        '--base-model', '-b',
        default='Qwen/Qwen3-1.7B',
        help='Base model name'
    )
    parser.add_argument(
        '--test-file', '-t',
        default='src/training/data_splits_axolotl/test_axolotl.jsonl',
        help='Test data file'
    )
    parser.add_argument(
        '--output', '-o',
        default='evaluation_report.json',
        help='Output file for evaluation report'
    )
    
    args = parser.parse_args()
    
    # Resolve adapter path
    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        adapter_path = project_root / adapter_path
    
    # Resolve test file path
    test_file = Path(args.test_file)
    if not test_file.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        test_file = project_root / test_file
    
    evaluator = ModelEvaluator(str(adapter_path), args.base_model)
    
    if not evaluator.load_model():
        sys.exit(1)
    
    metrics = evaluator.evaluate(str(test_file))
    
    if metrics:
        evaluator.save_evaluation_report(metrics, args.output)
    
    sys.exit(0)


if __name__ == '__main__':
    main()

