#!/usr/bin/env python3
"""Run a temporary Cloudflare Quick Tunnel to the local proxy."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from proxy.config import Config  # noqa: E402

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def main() -> int:
    executable = shutil.which("cloudflared")
    if not executable:
        print("cloudflared is not installed or not on PATH. See the README install steps.", file=sys.stderr)
        return 2

    cfg = Config.load()
    origin = f"http://127.0.0.1:{cfg.port}"
    command = [executable, "tunnel", "--url", origin]
    print(f"Starting a temporary Cloudflare Quick Tunnel to {origin} (Ctrl+C to stop).", flush=True)
    try:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
    except OSError as exc:
        print(f"Could not start cloudflared: {exc}", file=sys.stderr)
        return 2

    tunnel_path = ROOT / "tunnel.url"

    def relay_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            match = URL_RE.search(line)
            if match:
                tunnel_path.write_text(match.group(0) + "\n", encoding="utf-8")
                print(f"\nQuick Tunnel URL: {match.group(0)}", flush=True)
                print("Set this as Cursor's OpenAI Base URL (append /v1 if requested).", flush=True)

    reader = threading.Thread(target=relay_output, daemon=True)
    reader.start()
    try:
        code = proc.wait()
        reader.join(timeout=2)
        if code:
            print(f"cloudflared exited with status {code}", file=sys.stderr)
        return code
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
