#!/usr/bin/env python3
"""Create local configuration and an API key for Cursor -> PrefixRelay."""
from __future__ import annotations

import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
EXAMPLE = ROOT / "config.example.yaml"
TOKEN = ROOT / "secrets" / "proxy.token"


def main() -> int:
    if not CONFIG.exists():
        shutil.copy2(EXAMPLE, CONFIG)
        print(f"Created {CONFIG.name} from config.example.yaml")
    else:
        print(f"Keeping existing {CONFIG.name}")

    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    if not TOKEN.exists():
        TOKEN.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
        try:
            TOKEN.chmod(0o600)
        except OSError:
            pass  # Windows ACLs, not POSIX mode bits, control access there.
        print(f"Created {TOKEN.relative_to(ROOT)}")
    else:
        print(f"Keeping existing {TOKEN.relative_to(ROOT)}")

    print("\nUse the token as Cursor's OpenAI API key. Set your provider key")
    print("in the environment named by api_key_env in config.yaml.")
    print("Review config.yaml and replace the example model IDs before use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
