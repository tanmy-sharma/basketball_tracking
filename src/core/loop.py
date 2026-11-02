"""Generic phase-based training loop: warmup+cosine per step, bf16 on cuda, early stop, resume."""
from __future__ import annotations

import contextlib
import math
import random

import numpy as np
import torch

from src.core.checkpoint import load_checkpoint, save_checkpoint
from src.core.config import PhaseCfg
from src.core.constants import NUMBERS
from src.core.plugin import FitResult, LoopPlugin
from src.core.run import Run


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _schedule(total: int, warmup_frac: float):
    warm = int(total * warmup_frac)

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return f


def _better(monitor: str, new: float, best: float | None) -> bool:
    if best is None:
        return True
    return new < best if "loss" in monitor else new > best


def _cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _meta(plugin: LoopPlugin) -> dict:
    cfg = plugin.cfg
    head = cfg.model.head if cfg.model else ""
    meta = {"head": head, "backbone": cfg.model.backbone.name if cfg.model else "",
            "classes": list(NUMBERS) if head == "number" else ["ineligible", "eligible"] if head else []}
    meta.update(plugin.checkpoint_meta())
    return meta


def fit_loop(plugin: LoopPlugin, run: Run) -> FitResult:
    """Early stopping applies only in phases with unfreeze_blocks > 0 and ends the whole run."""
    cfg, tcfg = run.cfg, run.cfg.train
    if tcfg is None or not tcfg.phases:
        raise ValueError("fit_loop needs train.phases")
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    use_amp = device.type == "cuda"
    model = plugin.build_model().to(device)
    train_loader, val_loader = plugin.build_loaders(run)
    if len(train_loader) == 0:
        raise ValueError("empty train loader")
    meta, monitor = _meta(plugin), tcfg.monitor
    best_path, last_path = run.dir / "best.pt", run.dir / "last.pt"

    phase_idx, epochs_done, best, bad, stopped, n_epochs = 0, 0, None, 0, False, 0
    resume = None
    if last_path.exists():
        resume = load_checkpoint(last_path, full=True)
        model.load_state_dict(resume["state_dict"])
        phase_idx, epochs_done = resume["phase_index"], resume["epoch_in_phase"]
        best, bad, stopped = resume["best_metric"], resume["bad_epochs"], resume.get("stopped", False)
        n_epochs = resume["epoch"]
        random.setstate(resume["rng"]["python"])
        np.random.set_state(resume["rng"]["numpy"])
        torch.set_rng_state(resume["rng"]["torch"])
        run.log(f"resuming at phase {phase_idx} epoch {epochs_done} (global {n_epochs})")
    best_epoch = resume["best_epoch"] if resume else 0

    def payload(phase: PhaseCfg, metric: float) -> dict:
        return {"format_version": 1, "plugin": cfg.plugin, **meta, "state_dict": _cpu_state(model),
             "config": cfg.model_dump(mode="json"), "epoch": n_epochs, "phase": phase.name,
             "metric_name": monitor, "metric": metric}

    while phase_idx < len(tcfg.phases) and not stopped:
        phase = tcfg.phases[phase_idx]
        if epochs_done >= phase.epochs:
            phase_idx, epochs_done, bad = phase_idx + 1, 0, 0
            continue
        groups = plugin.configure_phase(model, phase)
        params = [p for g in groups for p in g["params"]]
        if not params:
            raise ValueError(f"phase {phase.name!r} has no trainable parameters")
        opt = torch.optim.AdamW(groups, weight_decay=tcfg.weight_decay)
        total = phase.epochs * len(train_loader)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(total, phase.warmup_frac))
        if resume is not None and epochs_done > 0:
            opt.load_state_dict(resume["optimizer"])
            sched.load_state_dict(resume["scheduler"])
        resume = None
        run.log(f"phase {phase.name}: {phase.epochs} epochs, {sum(p.numel() for p in params)} trainable params")

        for ep in range(epochs_done, phase.epochs):
            model.train()
            plugin.set_train_mode(model, phase)
            tot, n = 0.0, 0
            for batch in train_loader:
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
                with ctx:
                    loss = plugin.loss(model, batch, device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss in phase {phase.name} epoch {ep}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
                opt.step()
                sched.step()
                tot, n = tot + loss.item(), n + 1
            model.eval()
            with torch.no_grad():
                metrics = plugin.evaluate(model, val_loader, device)
            if monitor not in metrics:
                raise KeyError(f"monitor {monitor!r} not in evaluate() output {sorted(metrics)}")
            n_epochs += 1
            metric = float(metrics[monitor])
            run.log_metrics({"epoch": n_epochs, "phase": phase.name, "epoch_in_phase": ep + 1,
                             "train_loss": tot / max(n, 1), **metrics})
            run.log(f"epoch {n_epochs} [{phase.name}] train_loss={tot / max(n, 1):.4f} {monitor}={metric:.4f}")
            if _better(monitor, metric, best):
                best, bad, best_epoch = metric, 0, n_epochs
                save_checkpoint(best_path, payload(phase, metric))
            elif phase.unfreeze_blocks > 0:
                bad += 1
                stopped = bad >= tcfg.patience
            last = payload(phase, metric)
            last.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), phase_index=phase_idx,
                        epoch_in_phase=ep + 1, best_metric=best, bad_epochs=bad, stopped=stopped,
                        best_epoch=best_epoch,
                        rng={"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state()})
            save_checkpoint(last_path, last)
            if stopped:
                run.log(f"early stop: no {monitor} gain for {tcfg.patience} epochs")
                break
        phase_idx, epochs_done, bad = phase_idx + 1, 0, 0

    return FitResult(best_path, monitor, float(best), {"epochs": n_epochs, "best_epoch": best_epoch,
                                                       "stopped_early": stopped, "device": str(device)})
