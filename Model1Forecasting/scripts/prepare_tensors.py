"""One-time: parse ocean_cube_sequences.csv into a cached dense grid .npz."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.grid import build_grid, save_grid  # noqa: E402


def main() -> None:
    bundle = build_grid()
    save_grid(bundle)


if __name__ == "__main__":
    main()
