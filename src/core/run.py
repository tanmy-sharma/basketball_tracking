"""Run directory: config, manifest, log, metrics and the atomic `done.json` sentinel."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.core.config import Config, config_hash
from src.core.paths import ROOT, outputs_dir
from src.core.plugin import FitResult


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_version() -> dict:
    """CODE_VERSION.json (Colab bundle) wins; otherwise git hash and dirty flag."""
    cv = ROOT / "CODE_VERSION.json"
    if cv.exists():
        return json.loads(cv.read_text())
    status = _git("status", "--porcelain")
    return {"git_hash": _git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "bundle_sha256": None}


def _versions() -> dict:
    out = {}
    for pkg in ("torch", "transformers", "rfdetr"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["gpu"] = None
    return out


def resolve_run_dir(cfg: Config) -> Path:
    return outputs_dir() / "runs" / f"{cfg.run_name}__{config_hash(cfg)}"


class Run:
    def __init__(self, cfg: Config, dir: Path, hash: str) -> None:
        self.cfg, self.dir, self.hash = cfg, Path(dir), hash

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(line + "\n")

    def log_metrics(self, row: dict) -> None:
        p = self.dir / "metrics.csv"
        df = pd.read_csv(p) if p.exists() else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        tmp = p.with_name(p.name + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)

    def manifest(self) -> dict:
        p = self.dir / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def update_manifest(self, **fields) -> None:
        m = self.manifest()
        m.update(fields)
        _write_json(self.dir / "manifest.json", m)

    def finish(self, result: FitResult) -> None:
        try:
            best = str(Path(result.best_path).relative_to(self.dir))
        except ValueError:
            best = str(result.best_path)
        _write_json(self.dir / "summary.json", {"best_path": best, "metric_name": result.metric_name,
                                                "metric": result.metric, "extra": result.extra})
        self.update_manifest(finished=now())
        tmp = self.dir / "done.json.tmp"
        tmp.write_text(json.dumps({"finished": now(), "run": self.dir.name}))
        os.replace(tmp, self.dir / "done.json")

    def load_result(self) -> FitResult:
        s = json.loads((self.dir / "summary.json").read_text())
        return FitResult(self.dir / s["best_path"], s["metric_name"], s["metric"], s["extra"])

    @property
    def is_done(self) -> bool:
        return (self.dir / "done.json").exists()


def open_run(cfg: Config, argv: list[str] | None = None) -> Run:
    h = config_hash(cfg)
    d = resolve_run_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    run = Run(cfg, d, h)
    _write_json(d / "config.json", cfg.model_dump(mode="json"))
    argv = list(sys.argv if argv is None else argv)
    if run.manifest():
        run.update_manifest(resumed=now(), argv=argv)
    else:
        cv = code_version()
        run.update_manifest(run_name=cfg.run_name, config_hash=h, config=cfg.model_dump(mode="json"),
                            seed=cfg.seed, argv=argv, git_hash=cv.get("git_hash"), dirty=cv.get("dirty"),
                            bundle_sha256=cv.get("bundle_sha256"), versions=_versions(),
                            weight_sources=[], input_files={}, frame_counts={}, started=now())
    return run
