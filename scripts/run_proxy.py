#!/usr/bin/env python3
"""Run the local PrefixRelay HTTP server using config.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from proxy.config import Config  # noqa: E402


if __name__ == "__main__":
    cfg = Config.load()
    uvicorn.run("proxy.app:app", host=cfg.host, port=cfg.port, access_log=False)
