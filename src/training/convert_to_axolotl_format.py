#!/usr/bin/env python3
"""
Convert training data to Axolotl-compatible format.
Supports both single file and split data (train/val/test).
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse


def convert_to_axolotl_format(
    input_file: str,
    output_file: str,
    system_prompt: str = None
):
    """
    Convert training data to Axolotl's alpaca format.
    
    Args:
        input_file: Input JSONL file with {input, output} format
        output_file: Output JSONL file in Axolotl format
        system_prompt: Optional custom system prompt
    """
    if system_prompt is None:
        system_prompt = (
            "You are a Zsh shell expert specialized in CLI command completion. "
            "Given a partial command, provide the complete, executable command. "
            "Always respond with only the full command without explanations or additional text."
        )
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    axolotl_data = []
    
    for item in data:
        # Format for instruction fine-tuning
        axolotl_item = {
            "instruction": "Complete this Zsh command. Provide only the full command without explanations.",
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "system": system_prompt
        }
        axolotl_data.append(axolotl_item)
    
    # Save in Axolotl format
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in axolotl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(axolotl_data)} examples to Axolotl format")
    print(f"📁 Output: {output_file}")


def convert_splits(
    input_dir: str,
    output_dir: str,
    system_prompt: str = None
):
    """
    Convert split data (train/val/test) to Axolotl format.
    
    Args:
        input_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        output_dir: Output directory for converted files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        input_file = input_path / f"{split}.jsonl"
        if input_file.exists():
            output_file = output_path / f"{split}_axolotl.jsonl"
            print(f"\n🔄 Converting {split} set...")
            convert_to_axolotl_format(str(input_file), str(output_file), system_prompt)
        else:
            print(f"⚠️  {split}.jsonl not found, skipping...")


def main():
    parser = argparse.ArgumentParser(
        description='Convert training data to Axolotl-compatible format'
    )
    parser.add_argument(
        '--input', '-i',
        help='Input JSONL file (or directory for split data)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSONL file (or directory for split data)'
    )
    parser.add_argument(
        '--split-data',
        action='store_true',
        help='Convert split data (train/val/test) instead of single file'
    )
    parser.add_argument(
        '--system-prompt',
        help='Custom system prompt (optional)'
    )
    
    args = parser.parse_args()
    
    # Default paths
    if not args.input:
        if args.split_data:
            args.input = 'src/training/data_splits'
        else:
            args.input = 'src/training/zsh_training_data.jsonl'
    
    if not args.output:
        if args.split_data:
            args.output = 'src/training/data_splits_axolotl'
        else:
            args.output = 'src/training/zsh_training_data_axolotl.jsonl'
    
    if args.split_data:
        convert_splits(args.input, args.output, args.system_prompt)
    else:
        convert_to_axolotl_format(args.input, args.output, args.system_prompt)


if __name__ == "__main__":
    main()
