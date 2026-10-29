"""Scan tracked files for secret-shaped lines. Prints filenames only, never the line."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(password|passwd|secret|api_key)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def tracked_files() -> list:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / rel for rel in out.splitlines() if rel]


def scan() -> dict:
    hits = {}
    for path in tracked_files():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".stl", ".obj", ".glb"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        found = [p.pattern for p in PATTERNS if p.search(text)]
        if found:
            hits[str(path.relative_to(ROOT))] = len(found)
    return hits


def main() -> int:
    hits = scan()
    if not hits:
        print("no secret-shaped lines in tracked text")
        return 0
    for name, n in sorted(hits.items()):
        print(f"{n} pattern(s): {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
