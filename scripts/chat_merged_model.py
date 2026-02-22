#!/usr/bin/env python3
"""
Interact with the merged model (zsh-assistant) directly. Model only — no extra logic.
Usage:
  python scripts/chat_merged_model.py "git ad"
  python scripts/chat_merged_model.py              # interactive loop
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

def ask(partial_command: str) -> str:
    from model_completer.utils import load_config
    from model_completer.ollama_complete import complete, SYSTEM
    config = load_config()
    url = config.get("ollama", {}).get("url", "http://localhost:11434")
    model = config.get("model", "zsh-assistant")
    out = complete(partial_command.strip(), url, model, timeout=30)
    return out or ""


def main():
    import requests
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
    except Exception as e:
        print("Ollama not reachable:", e, "\nStart with: ollama serve", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print("Partial command:", repr(cmd))
        from model_completer.ollama_complete import SYSTEM
        print("System:", SYSTEM[:80] + "...")
        print()
        t0 = time.perf_counter()
        out = ask(cmd)
        t1 = time.perf_counter()
        print("Model response:", repr(out))
        print(f"({t1 - t0:.2f}s)")
        return

    # Interactive: model only, no extra logic
    print("Merged model (zsh-assistant). System: complete the command, one line only.")
    print("Type a partial command, or 'quit'.")
    print()
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        t0 = time.perf_counter()
        try:
            out = ask(line)
            t1 = time.perf_counter()
            print("  ->", out or "(empty)")
            print(f"  ({t1 - t0:.2f}s)")
        except Exception as e:
            print("  Error:", e)
        print()


if __name__ == "__main__":
    main()
