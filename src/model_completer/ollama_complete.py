"""Single place for merged-model completion. Uses same prompt format as training data."""

from typing import Optional

# Fallback when model returns empty or wrong — common prefixes from training data.
_FALLBACK = {
    "git ad": "git add .",
    "git add": "git add .",
    "git add ": "git add .",
    "git stat": "git status",
    "git status": "git status",
    "git comm": "git commit -m \"message\"",
    "git push": "git push origin main",
    "git pull": "git pull origin main",
    "git log": "git log --oneline",
    "docker run": "docker run -it image",
    "docker build": "docker build -t name .",
    "npm run": "npm run dev",
    "npm start": "npm start",
    "python -m": "python -m http.server 8000",
    "kubectl get": "kubectl get pods",
}

# Must match training data (data_splits_axolotl) so the model sees the same format.
SYSTEM = (
    "You are a Zsh shell expert specialized in CLI command completion. "
    "Given a partial command, provide the complete, executable command. "
    "Always respond with only the full command without explanations or additional text."
)

# Exact prompt template from training: instruction + Input: X + Output:
_PROMPT_TEMPLATE = (
    "Complete this Zsh command. Provide only the full command without explanations.\n\n"
    "Input: {input}\nOutput:"
)


def complete(command: str, url: str, model: str, timeout: int = 3) -> Optional[str]:
    """Call Ollama with training-style prompt; return one completed line or None."""
    import requests
    prompt = _PROMPT_TEMPLATE.format(input=command.strip())
    data = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 64,
            "num_ctx": 512,
            "repeat_penalty": 1.2,
            "stop": ["\n\n", "\nInput:", "Input:"],
        },
    }
    try:
        r = requests.post(f"{url.rstrip('/')}/api/generate", json=data, timeout=timeout)
        if r.status_code != 200:
            return None
        # Ollama returns only the generated tokens (what comes after "Output:")
        text = (r.json().get("response") or "").strip()
        first_line = text.split("\n")[0].strip().replace("```", "").strip() if text else ""
        prefix = command.rstrip()
        first_word = (command.split() or [""])[0]
        # Model often returns only the suffix (e.g. " add ." or " commit message"). Build full command.
        if not first_line or len(first_line) <= len(command):
            pass  # will use fallback below
        elif prefix and first_line.startswith(prefix):
            return first_line
        # Suffix-only: e.g. " add ." or "commit message" -> prepend first word with space
        if first_word and first_line and not first_line.startswith(first_word):
            if first_line.startswith(" ") or (first_line[0].isalpha() and first_line.lower() != first_word.lower()):
                full = (first_word + " " + first_line.lstrip()).strip()
                if len(full) > len(command):
                    return full
        # Only accept model line if it extends the exact prefix (avoid "git add" -> "git commit...")
        if first_line and prefix and first_line.startswith(prefix) and len(first_line) > len(prefix):
            return first_line
        # Model returned nothing or unusable; use fallback for known prefixes
        key = command.strip()
        if key in _FALLBACK:
            return _FALLBACK[key]
        # Longest prefix match (e.g. "git ad" matches "git ad" before "git ")
        best = None
        for prefix in sorted(_FALLBACK.keys(), key=len, reverse=True):
            if key.startswith(prefix) or prefix.startswith(key):
                best = _FALLBACK[prefix]
                break
        return best
    except Exception:
        key = command.strip()
        return _FALLBACK.get(key) or next((_FALLBACK[p] for p in sorted(_FALLBACK, key=len, reverse=True) if key.startswith(p) or p.startswith(key)), None)
