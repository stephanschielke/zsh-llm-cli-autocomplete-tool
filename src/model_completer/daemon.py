"""
Completion daemon: long-lived process so Tab is fast (no Python startup per keystroke).
Like Cursor/Copilot: one server, reuse connection, low latency.
Run: python -m model_completer.daemon   (or scripts/run_completion_daemon.sh)
Then the Zsh plugin will use it when available (curl POST to port).
"""

import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# Project layout
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, "..")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

from model_completer.utils import load_config
from model_completer.ollama_complete import complete

DEFAULT_PORT = 11435


class CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/complete" and not self.path.startswith("/complete?"):
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        try:
            command = body.decode("utf-8", errors="replace").strip()
        except Exception:
            command = ""
        if not command:
            self._send("", 400)
            return
        config = load_config()
        url = config.get("ollama", {}).get("url", "http://localhost:11434")
        model = config.get("model", "zsh-assistant")
        timeout = config.get("ollama", {}).get("timeout", 10)
        timeout = min(int(timeout), 5)
        result = complete(command, url, model, timeout=timeout)
        self._send(result or command)

    def do_GET(self):
        if self.path == "/health":
            self._send("ok")
            return
        self.send_error(404)

    def _send(self, text: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # quiet by default


def main():
    port = int(os.environ.get("MODEL_COMPLETION_DAEMON_PORT", DEFAULT_PORT))
    server = HTTPServer(("127.0.0.1", port), CompletionHandler)
    print(f"Completion daemon listening on http://127.0.0.1:{port}/complete", file=sys.stderr)
    print("Tab completion will use this for low latency. Ctrl+C to stop.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
