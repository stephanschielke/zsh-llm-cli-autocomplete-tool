#!/usr/bin/env bash
# Upload the trained LoRA adapter to Hugging Face.
# 1. Log in first (one-time): huggingface-cli login
# 2. Replace YOUR_HF_USERNAME with your Hugging Face username below, then run.

set -e
cd "$(dirname "$0")"

REPO_ID="${HF_REPO_ID:-YOUR_HF_USERNAME/zsh-cli-lora}"

if [[ "$REPO_ID" == *"YOUR_HF_USERNAME"* ]]; then
  echo "❌ Set your Hugging Face repo ID first:"
  echo "   export HF_REPO_ID=your_username/zsh-cli-lora"
  echo "   or edit this script and replace YOUR_HF_USERNAME"
  echo ""
  echo "Then run: ./upload_to_huggingface.sh"
  exit 1
fi

echo "📤 Uploading adapter to https://huggingface.co/$REPO_ID"
venv/bin/python -m training.upload_to_hf \
  --adapter zsh-lora-output \
  --repo-id "$REPO_ID" \
  --base-model Qwen/Qwen2-0.5B \
  "$@"
