"""Write artifacts/baseline/manifest.json from the local checkout. No network. No hardware."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def _versions() -> dict:
    import importlib.metadata as meta
    out = {}
    for name in ("mujoco", "numpy", "scipy", "fastapi", "pydantic", "mink"):
        try:
            out[name] = meta.version(name)
        except meta.PackageNotFoundError:
            out[name] = None
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    tracked = _git("ls-files").splitlines()
    hashed = {}
    for rel in tracked:
        if not rel.endswith((".py", ".md", ".html", ".toml")):
            continue
        path = ROOT / rel
        if path.is_file() and path.stat().st_size < 2_000_000:
            hashed[rel] = _sha256(path)
    payload = {
        "physical_validation": False,
        "hardware_motion": False,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
        "published_randomized_rates_commit": "32b5d08",
        "published_randomized_rates": {"1": "19/24", "2": "10/24", "3": "10/24", "4": "5/24", "5": "3/24"},
        "source_sha256": hashed,
    }
    dest = ROOT / "artifacts" / "baseline" / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
