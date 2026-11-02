"""General entry point: config -> run dir -> plugin.fit.  `python -m src.train --config PATH [--set a.b=c ...]`"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from src.core.config import Config, load_config
from src.core.plugin import FitResult, get_plugin
from src.core.run import open_run


def run_training(cfg: Config, argv: list[str] | None = None) -> FitResult:
    run = open_run(cfg, argv)
    if run.is_done:
        run.log("already done, skipping")
        return run.load_result()
    plugin = get_plugin(cfg.plugin)(cfg)
    result = plugin.fit(run)
    run.update_manifest(weight_sources=[asdict(w) for w in plugin.weight_sources()])
    run.finish(result)
    run.log(f"done: {result.metric_name}={result.metric}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="a.b=c")
    a = ap.parse_args(argv)
    run_training(load_config(a.config, a.overrides), argv)


if __name__ == "__main__":
    main()
