"""Fail fast with a clear message if the wrong Python interpreter is used.

The repo ships a `.venv` with numpy, pandas, etc. Running `python run_pipeline.py`
with system Python (no venv) raises ModuleNotFoundError — this module catches that
before heavier imports and points users to activate `.venv` or use `.venv/bin/python`.
"""

from __future__ import annotations


def ensure_scientific_stack() -> None:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError as e:
        name = e.name or "numpy"
        raise SystemExit(
            f"Missing dependency '{name}'. Use the project virtual environment.\n\n"
            "  cd Model1DataEngineering\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python run_pipeline.py\n\n"
            "Or run without activating:\n"
            "  .venv/bin/python run_pipeline.py\n"
        ) from e
