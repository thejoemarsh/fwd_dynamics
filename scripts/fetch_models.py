"""Download upstream OpenSim models we depend on into data/models/recipes/.

Run once after `uv sync`. Idempotent — skips already-downloaded files.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = REPO_ROOT / "data" / "models" / "recipes"

MODELS = {
    "LaiUhlrich2022_full_body.osim": (
        "https://raw.githubusercontent.com/stanfordnmbl/opencap-core/main/"
        "opensimPipeline/Models/LaiUhlrich2022.osim"
    ),
}


def fetch_one(name: str, url: str, dest_dir: Path) -> Path:
    dest = dest_dir / name
    if dest.exists():
        print(f"  [skip] {name} already at {dest}")
        return dest
    print(f"  [get ] {name} <- {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f"         wrote {len(data):,} bytes -> {dest}")
    return dest


def main() -> int:
    print(f"Fetching models into {RECIPES_DIR}")
    for name, url in MODELS.items():
        fetch_one(name, url, RECIPES_DIR)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
