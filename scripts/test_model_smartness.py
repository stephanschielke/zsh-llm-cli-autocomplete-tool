#!/usr/bin/env python3
"""
Test if the merged model (zsh-assistant in Ollama) is smart enough.
Uses the same completion path as Tab (fast path) and evaluates on a test set.
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Curated cases: (input, expected_output) for key scenarios
CURATED_CASES = [
    ("git comm", "git commit -m \"message\""),
    ("git stat", "git status"),
    ("git pull", "git pull origin main"),
    ("git push", "git push origin main"),
    ("docker run", "docker run -it image"),
    ("docker build", "docker build -t name ."),
    ("npm run", "npm run build"),
    ("npm install", "npm install"),
    ("python -m", "python -m http.server 8000"),
    ("kubectl get", "kubectl get pods"),
    ("kubectl apply", "kubectl apply -f file.yaml"),
    ("curl ", "curl -X GET url"),
    ("vim ", "vim file.txt"),
]


def command_match(pred: str, expected: str, input_prefix: str) -> bool:
    """True if predicted is a valid completion: extends the input prefix and same base command."""
    if not pred or not expected:
        return False
    pred = pred.strip()
    prefix = input_prefix.strip()
    if not prefix:
        return pred.split()[0] == expected.strip().split()[0] if expected.split() else False
    # Completion must start with the user's input (they're completing that prefix)
    if not pred.startswith(prefix) and not pred.startswith(prefix.rstrip()):
        return False
    if len(pred) <= len(prefix):
        return False
    # Same base command as expected
    p0 = pred.split()[0] if pred.split() else ""
    e0 = expected.strip().split()[0] if expected.strip().split() else ""
    return p0 == e0


def exact_match(pred: str, expected: str) -> bool:
    n = lambda s: " ".join(s.strip().split())
    return n(pred) == n(expected)


def main():
    from model_completer.cli import _fast_completion
    from model_completer.utils import load_config

    print("Testing merged model (zsh-assistant) smartness")
    print("=" * 60)

    config = load_config()
    url = config.get("ollama", {}).get("url", "http://localhost:11434")
    model = config.get("model", "zsh-assistant")

    # Check Ollama
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=2)
        if r.status_code != 200:
            print("Ollama not responding. Start with: ollama serve")
            sys.exit(1)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not any("zsh-assistant" in n for n in models):
            print("Model zsh-assistant not found. Run: python -m model_completer.cli --import-to-ollama")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot reach Ollama: {e}")
        sys.exit(1)

    # Load test cases: curated + optional JSONL
    cases = list(CURATED_CASES)
    test_file = PROJECT_ROOT / "src" / "training" / "data_splits" / "test.jsonl"
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        inp = obj.get("input", "").strip()
                        out = obj.get("output", "").strip()
                        if inp and out and (inp, out) not in cases:
                            cases.append((inp, out))
                    except json.JSONDecodeError:
                        pass
    # Cap so run stays quick
    cases = cases[:35]
    print(f"Running {len(cases)} test cases (same path as Tab completion)...\n")

    exact_ok = 0
    command_ok = 0
    results = []

    for i, (inp, expected) in enumerate(cases):
        pred = _fast_completion(inp, url, model)
        if pred is None:
            pred = ""
        ex = exact_match(pred, expected)
        cm = command_match(pred, expected, inp)
        if ex:
            exact_ok += 1
        if cm:
            command_ok += 1
        results.append({
            "input": inp,
            "expected": expected,
            "predicted": pred,
            "exact": ex,
            "command_ok": cm,
        })

    total = len(results)
    exact_pct = 100 * exact_ok / total if total else 0
    command_pct = 100 * command_ok / total if total else 0

    print("Results")
    print("-" * 60)
    print(f"Exact match:   {exact_ok}/{total} ({exact_pct:.0f}%)")
    print(f"Command match: {command_ok}/{total} ({command_pct:.0f}%)")
    print()

    # Show a few good and a few bad
    good = [r for r in results if r["exact"]]
    ok_cmd = [r for r in results if r["command_ok"] and not r["exact"]]
    bad = [r for r in results if not r["command_ok"]]

    if good:
        print("Sample exact matches:")
        for r in good[:5]:
            print(f"  {r['input']!r} -> {r['predicted']!r}")
        print()
    if ok_cmd:
        print("Sample command matches (same intent, different wording):")
        for r in ok_cmd[:5]:
            print(f"  {r['input']!r}")
            print(f"    expected: {r['expected']!r}")
            print(f"    got:      {r['predicted']!r}")
        print()
    if bad:
        print("Sample misses:")
        for r in bad[:8]:
            print(f"  {r['input']!r}")
            print(f"    expected: {r['expected']!r}")
            print(f"    got:      {r['predicted']!r}")
        print()

    # Verdict
    print("=" * 60)
    if command_pct >= 50:
        print("Verdict: Model is smart enough for practical Tab completion.")
        print("  (Command match >= 50%. Exact match can be improved with more training.)")
    elif command_pct >= 30:
        print("Verdict: Model is partly there; good for common commands.")
    else:
        print("Verdict: Completions are often off. Consider more training or better prompt.")
    print()


if __name__ == "__main__":
    main()
