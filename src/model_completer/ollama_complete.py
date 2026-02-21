"""Single place for merged-model completion: system prompt + Ollama call. Used by CLI and daemon."""

from typing import Optional

SYSTEM = (
    "You are a shell command completion assistant. "
    "Given a partial command, reply with ONLY the completed full command on one line. "
    "No explanation, no prefix like 'Complete:' or 'The command is:', no markdown. Just the command."
)


def complete(command: str, url: str, model: str, timeout: int = 3) -> Optional[str]:
    """Call Ollama with system + prompt; return one completed line or None."""
    import requests
    data = {
        "model": model,
        "system": SYSTEM,
        "prompt": command.strip(),
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 48, "num_ctx": 512},
    }
    try:
        r = requests.post(f"{url.rstrip('/')}/api/generate", json=data, timeout=timeout)
        if r.status_code != 200:
            return None
        text = (r.json().get("response") or "").strip()
        if not text or len(text) <= len(command):
            return None
        prefix = command.rstrip()
        first = (command.split() or [""])[0]
        for line in text.split("\n"):
            line = line.strip().replace("```", "").strip()
            if not line or len(line) <= len(command):
                continue
            if any(line.startswith(x) for x in ("Complete", "Output", "The ", "Sure,", "Here")):
                continue
            if prefix and line.startswith(prefix):
                return line
            if first and line.startswith(first):
                return line
        return text.split("\n")[0].strip() if text else None
    except Exception:
        return None
