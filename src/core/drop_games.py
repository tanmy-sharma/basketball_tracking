"""Remove games from every config, for a run where a video is missing.  `python -m src.core.drop_games --games IMG_0105`

Rewrites the YAML files in place, so every entry point sees the same reduced games and its config hash changes with
them. The notebooks use this on their own unpacked copy; the repo's configs are never meant to be changed by it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.core.paths import ROOT


def drop_from_text(text: str, games: list[str]) -> str:
    """Drop games from YAML flow lists like `train: [A, B]` and `- [A, B]`; a fold left empty loses its line."""
    for g in games:
        text = re.sub(rf"\b{re.escape(g)}\b,?[ \t]*", "", text)
    text = re.sub(r",[ \t]*\]", "]", text)
    return re.sub(r"^[ \t]*-[ \t]*\[\][ \t]*\n", "", text, flags=re.M)


def drop_games(games: list[str], root: Path) -> list[Path]:
    changed = []
    for p in sorted(root.rglob("*.yaml")):
        old = p.read_text()
        new = drop_from_text(old, games)
        if new != old:
            p.write_text(new)
            changed.append(p)
    return changed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "src/configs")
    a = ap.parse_args(argv)
    changed = drop_games(a.games, a.root)
    print(f"removed {a.games} from {len(changed)} config files")


if __name__ == "__main__":
    main()
