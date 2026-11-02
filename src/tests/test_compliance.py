"""Hard rules: no banned libraries or models anywhere in the repo, and nothing under src/ is gitignored.

Checks pick up other packages' files automatically as they appear.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.constants import FORBIDDEN_MODULES
from src.core.paths import ROOT

SRC = ROOT / "src"
BANNED_MODEL_SHA256 = "e4923f56224586f9eafe3526eb7a34cbad5848525e9f12f496c250e8afe1b1f1"
BANNED_LEN = 6
LOCK_BANNED = ("mmcv", "mmdet", "mmpose", "mmocr", "mmengine", "ultralytics")
ENTRY_MODULES = ("src.train", "src.core.loop", "src.core.run", "src.dataio.select", "src.dataio.labels",
                 "src.dataio.video", "src.crops.extract", "src.crops.pose", "src.classifier.plugin",
                 "src.classifier.predict", "src.recognizer.plugin", "src.recognizer.predict",
                 "src.weak.folds", "src.weak.finalize", "src.evaluate.classifier",
                 "src.detector.plugin", "src.detector.predict", "src.detector.evaluate",
                 "src.tracker.ucmc", "src.tracker.evaluate", "src.tracker.run_ucmc")


def py_files() -> list[Path]:
    files = [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]
    nb = ROOT / "colab" / "build_src_notebooks.py"
    return files + ([nb] if nb.exists() else [])


def top(name: str) -> str:
    return name.split(".")[0]


def test_no_forbidden_imports_ast():
    bad = []
    forbidden = set(FORBIDDEN_MODULES)
    for p in py_files():
        for node in ast.walk(ast.parse(p.read_text(), str(p))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and getattr(node.func, "attr", getattr(node.func, "id", "")) in ("import_module", "__import__")):
                names = [node.args[0].value]
            bad += [f"{p.relative_to(ROOT)}:{node.lineno} imports {n}" for n in names if top(n) in forbidden]
    assert not bad, bad


def test_forbidden_list_keeps_required_entries():
    assert {"mmdet", "mmcv", "mmengine", "mmpose", "mmocr", "doctr", "strhub", "ultralytics",
            "groundingdino"} <= set(FORBIDDEN_MODULES)


def test_entry_modules_do_not_pull_forbidden_or_rfdetr():
    present = []
    for m in ENTRY_MODULES:
        try:
            if importlib.util.find_spec(m) is not None:
                present.append(m)
        except ModuleNotFoundError:
            pass
    assert "src.train" in present
    code = (
        "import importlib, json, sys\n"
        f"mods = {present!r}\n"
        "failed = {}\n"
        "for m in mods:\n"
        "    try:\n        importlib.import_module(m)\n"
        "    except Exception as e:\n        failed[m] = repr(e)[:300]\n"
        f"bad = sorted({{t for t in map(lambda n: n.split('.')[0], sys.modules) if t in {tuple(FORBIDDEN_MODULES)!r}}})\n"
        "print(json.dumps({'failed': failed, 'bad': bad, 'rfdetr': 'rfdetr' in sys.modules}))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["failed"] == {}, out["failed"]
    assert out["bad"] == [], f"forbidden modules in sys.modules: {out['bad']}"
    assert not out["rfdetr"], "rfdetr must stay optional: no import at module level"


def test_uv_lock_has_no_banned_packages():
    lock = (ROOT / "uv.lock").read_text()
    names = set(re.findall(r'^name = "([^"]+)"', lock, re.M))
    assert not [n for n in names if n.lower().replace("_", "-") in LOCK_BANNED], names & set(LOCK_BANNED)


def tracked_files() -> list[Path]:
    r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    return [ROOT / f for f in r.stdout.splitlines() if (ROOT / f).is_file()]


def test_banned_model_name_absent_from_repo():
    hits = []
    for p in tracked_files():
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.lower().splitlines(), 1):
            for tok in re.findall(r"[a-z]+", line):
                for i in range(len(tok) - BANNED_LEN + 1):
                    if hashlib.sha256(tok[i:i + BANNED_LEN].encode()).hexdigest() == BANNED_MODEL_SHA256:
                        hits.append(f"{p.relative_to(ROOT)}:{n}")
    assert not hits, sorted(set(hits))


def test_nothing_under_src_is_gitignored():
    files = [str(p.relative_to(ROOT)) for p in SRC.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(files),
                       capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    assert r.stdout.strip() == "", f"gitignored: {r.stdout.split()}"
    assert not [p for p in SRC.rglob("*") if p.is_dir() and p.name in ("data", "outputs")]
