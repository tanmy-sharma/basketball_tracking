"""Repo-relative locations. Outputs can be redirected with SRC_OUTPUTS (tests do)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


def outputs_dir() -> Path:
    return Path(os.environ.get("SRC_OUTPUTS") or OUTPUTS)


def resolve_path(p: Path | str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def add_tools_to_path() -> Path:
    """Make `import trackid` work without running from tools/."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return Path(tools)
