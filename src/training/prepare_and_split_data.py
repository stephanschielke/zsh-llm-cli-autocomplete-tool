#!/usr/bin/env python3
"""
Prepare and split training data for Qwen3 LoRA fine-tuning.
Splits data into train (70%), validation (15%), and test (15%) sets.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict


class DataPreparer:
    """Prepare and split training data for CLI completion fine-tuning."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_data(self, input_file: str) -> List[Dict[str, str]]:
        """Load training data from JSONL file."""
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping invalid JSON line: {e}")
        return data
    
    def balance_data_by_category(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Balance data by command category to ensure fair distribution."""
        # Categorize commands
        categories = defaultdict(list)
        
        for item in data:
            cmd = item.get("input", "").split()[0] if item.get("input") else ""
            category = self._categorize_command(cmd)
            categories[category].append(item)
        
        # Print category distribution
        print("\n📊 Data Distribution by Category:")
        for cat, items in sorted(categories.items()):
            print(f"   {cat}: {len(items)} examples")
        
        return data
    
    def _categorize_command(self, cmd: str) -> str:
        """Categorize command for balancing."""
        cmd_lower = cmd.lower()
        
        if cmd_lower.startswith('git'):
            return 'git'
        elif cmd_lower.startswith('docker'):
            return 'docker'
        elif cmd_lower.startswith('npm') or cmd_lower.startswith('yarn'):
            return 'npm'
        elif cmd_lower.startswith('python') or cmd_lower.startswith('pip'):
            return 'python'
        elif cmd_lower.startswith('kubectl'):
            return 'kubernetes'
        elif cmd_lower in ['ls', 'cd', 'cp', 'mv', 'rm', 'mkdir', 'find', 'grep']:
            return 'system'
        elif cmd_lower.startswith('curl') or cmd_lower.startswith('wget'):
            return 'network'
        else:
            return 'other'
    
    def split_data(
        self, 
        data: List[Dict[str, str]], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: List of training examples
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Verify ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Shuffle data
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = shuffled[:train_end]
        val_data = shuffled[train_end:val_end]
        test_data = shuffled[val_end:]
        
        return train_data, val_data, test_data
    
    def save_split(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        output_dir: str
    ):
        """Save split data to separate files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save train set
        train_file = output_path / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save validation set
        val_file = output_path / "val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save test set
        test_file = output_path / "test.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Data split saved to {output_dir}/")
        print(f"   Train: {len(train_data)} examples ({train_file})")
        print(f"   Val:   {len(val_data)} examples ({val_file})")
        print(f"   Test:  {len(test_data)} examples ({test_file})")
    
    def prepare_complete(
        self,
        input_file: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """Complete data preparation and splitting pipeline."""
        print("=" * 70)
        print("📊 Data Preparation and Splitting for Qwen3 LoRA Training")
        print("=" * 70)
        
        # Load data
        print(f"\n📥 Loading data from {input_file}...")
        data = self.load_data(input_file)
        print(f"✅ Loaded {len(data)} examples")
        
        if len(data) == 0:
            print("❌ No data loaded!")
            return False
        
        # Balance data
        print("\n⚖️  Balancing data by category...")
        data = self.balance_data_by_category(data)
        
        # Split data
        print(f"\n✂️  Splitting data ({train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test)...")
        train_data, val_data, test_data = self.split_data(
            data, train_ratio, val_ratio, test_ratio
        )
        
        # Save splits
        self.save_split(train_data, val_data, test_data, output_dir)
        
        # Print statistics
        print("\n📈 Split Statistics:")
        print(f"   Total examples: {len(data)}")
        print(f"   Train set:     {len(train_data)} ({len(train_data)/len(data):.1%})")
        print(f"   Val set:       {len(val_data)} ({len(val_data)/len(data):.1%})")
        print(f"   Test set:      {len(test_data)} ({len(test_data)/len(data):.1%})")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare and split training data for Qwen3 LoRA fine-tuning'
    )
    parser.add_argument(
        '--input', '-i',
        default='src/training/zsh_training_data.jsonl',
        help='Input JSONL file with training data'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='src/training/data_splits',
        help='Output directory for split data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    preparer = DataPreparer(seed=args.seed)
    success = preparer.prepare_complete(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()

