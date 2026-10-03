"""Tiny .env loader (no dependency). Real environment variables always win."""

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> bool:
    candidates = (
        [path] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    )
    for p in candidates:
        if p and p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return True
    return False
