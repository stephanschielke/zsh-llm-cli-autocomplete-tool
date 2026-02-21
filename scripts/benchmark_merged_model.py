#!/usr/bin/env python3
"""Benchmark merged model (zsh-assistant) latency: Ollama raw vs full CLI path."""

import os
import sys
import time
import requests

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

OLLAMA_URL = "http://localhost:11434"
MODEL = "zsh-assistant"
TRIALS = 3


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code != 200:
            print("Ollama not responding (status %s)" % r.status_code)
            return False
        models = r.json().get("models", [])
        names = [m.get("name", "") for m in models]
        if not any("zsh-assistant" in n for n in names):
            print("Model zsh-assistant not found. Run: python -m model_completer.cli --import-to-ollama")
            return False
        return True
    except Exception as e:
        print("Ollama not reachable:", e)
        return False


def bench_ollama_raw(prompt: str, num_predict: int = 30):
    """Measure raw Ollama /api/generate latency (no Python completer)."""
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "num_ctx": 512,
        },
    }
    start = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=data, timeout=60)
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            return elapsed, r.json().get("response", "").strip()
        return elapsed, None
    except Exception as e:
        return time.perf_counter() - start, str(e)


def bench_cli(command: str):
    """Measure full path: Python CLI with command (like Zsh plugin)."""
    cli_path = os.path.join(PROJECT_ROOT, "src", "model_completer", "cli.py")
    python = sys.executable
    if os.path.isfile(os.path.join(PROJECT_ROOT, "venv", "bin", "python3")):
        python = os.path.join(PROJECT_ROOT, "venv", "bin", "python3")
    cmd = [python, "-W", "ignore::UserWarning", "-W", "ignore::DeprecationWarning",
           "-u", cli_path, command]
    start = time.perf_counter()
    import subprocess
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": os.path.join(PROJECT_ROOT, "src")},
        )
        elapsed = time.perf_counter() - start
        return elapsed, (out.stdout or "").strip() if out.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return 15.0, None
    except Exception as e:
        return time.perf_counter() - start, str(e)


def main():
    print("Merged model (zsh-assistant) latency benchmark")
    print("=" * 50)
    if not check_ollama():
        sys.exit(1)
    print()

    # Minimal prompt (what we'd want for speed)
    minimal_prompt = "Complete this command: git comm\n\nOutput ONLY the complete command, nothing else."
    # Current style (longer prompt from enhanced completer)
    long_prompt = (
        "Complete this command: git comm\n"
        "Context: Git branch: main | Has uncommitted changes\n"
        "Output ONLY the complete command, nothing else. No explanations."
    )

    # 1) Warm-up: one raw request so model is loaded
    print("Warm-up (load model)...")
    t, _ = bench_ollama_raw(minimal_prompt)
    print("  Warm-up request: %.2fs" % t)
    print()

    # 2) Raw Ollama - minimal prompt
    print("Raw Ollama (minimal prompt), %d trials:" % TRIALS)
    raw_times = []
    for i in range(TRIALS):
        t, resp = bench_ollama_raw(minimal_prompt)
        raw_times.append(t)
        print("  trial %d: %.2fs  -> %s" % (i + 1, t, (resp[:60] + "..." if resp and len(resp) > 60 else resp)))
    print("  avg: %.2fs  min: %.2fs" % (sum(raw_times) / len(raw_times), min(raw_times)))
    print()

    # 3) Raw Ollama - fewer tokens
    print("Raw Ollama (num_predict=15), %d trials:" % TRIALS)
    for i in range(TRIALS):
        t, resp = bench_ollama_raw(minimal_prompt, num_predict=15)
        raw_times.append(t)
        print("  trial %d: %.2fs" % (i + 1, t))
    print("  avg: %.2fs" % (sum(raw_times[-TRIALS:]) / TRIALS))
    print()

    # 4) Full CLI path (like Tab)
    print("Full CLI path (git comm), %d trials:" % TRIALS)
    cli_times = []
    for i in range(TRIALS):
        t, out = bench_cli("git comm")
        cli_times.append(t)
        print("  trial %d: %.2fs  -> %s" % (i + 1, t, (out[:60] + "..." if out and len(out) > 60 else out)))
    print("  avg: %.2fs  min: %.2fs" % (sum(cli_times) / len(cli_times), min(cli_times)))
    print()

    # Summary
    raw_avg = sum(raw_times[:TRIALS]) / TRIALS
    cli_avg = sum(cli_times) / len(cli_times)
    cli_min = min(cli_times)
    print("Summary:")
    print("  Ollama (minimal prompt): ~%.2fs" % raw_avg)
    print("  Full CLI (Tab path):     ~%.2fs avg, %.2fs min" % (cli_avg, cli_min))
    if cli_avg > raw_avg * 1.2:
        print("  -> Extra time is Python startup + parsing (fast path avoids heavy completer)")


if __name__ == "__main__":
    main()
