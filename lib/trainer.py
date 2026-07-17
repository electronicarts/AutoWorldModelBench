# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Step-based training loop for all world models.

Uses itertools.cycle for infinite iteration over the training set.
Supports AMP (automatic mixed precision) for GPU throughput.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaselineTrainer:
    """Step-based training with AMP for RSSM, D3PM, and MaskGIT."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        warmup_steps: int = 500,
        max_steps: int = 30_000,
        eval_every: int = 1000,
        patience: int = 10,
        output_dir: Path = Path("outputs"),
        device: str = "cuda",
        use_amp: bool = True,
        max_train_seconds: float | None = None,
        model_type: str = "dreamer",
        val_tensors: dict | None = None,
        eval_tensors: dict | None = None,
        eval_max_batches: int = 16,
        final_eval_horizons: list[int] | None = None,
        eval_batch_size: int = 64,
        eval_timeout_per_horizon: float = 120.0,
        context_len: int = 8,
    ):
        self.context_len = context_len
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.max_train_seconds = max_train_seconds
        self.eval_every = eval_every
        self.patience = patience
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr * 0.01
        )
        self.base_lr = lr

        # AMP
        self.use_amp = use_amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # torch.compile disabled — requires triton which isn't installed

        # Unified evaluator params
        self.model_type = model_type
        self.val_tensors = val_tensors
        self.eval_tensors = eval_tensors  # longer-window tensors for h50 eval
        self.eval_max_batches = eval_max_batches
        self.final_eval_horizons = final_eval_horizons or [1, 10, 20]
        self.eval_batch_size = eval_batch_size
        self.eval_timeout_per_horizon = eval_timeout_per_horizon

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: list[dict] = []

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train(self) -> dict:
        """Run step-based training. Returns best metrics dict."""
        n_params = sum(p.numel() for p in self.model.parameters())
        time_info = f" | time_limit={self.max_train_seconds}s" if self.max_train_seconds else ""
        print(f"Training {self.model.__class__.__name__} | "
              f"params={n_params:,} | device={self.device} | "
              f"max_steps={self.max_steps}{time_info} | amp={self.use_amp}")

        t0 = time.time()
        running_loss = 0.0
        n_accum = 0

        train_iter = itertools.chain.from_iterable(
            itertools.repeat(self.train_loader)
        )

        self.model.train()
        for batch in train_iter:
            if self.global_step >= self.max_steps:
                break

            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if self.global_step < self.warmup_steps:
                self._set_lr(self.base_lr * self.global_step / max(1, self.warmup_steps))

            # Forward with AMP
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss_dict = self.loss_fn(self.model, batch)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()

            running_loss += loss.item()
            n_accum += 1
            self.global_step += 1

            # Wall-clock time limit
            if self.max_train_seconds is not None and time.time() - t0 >= self.max_train_seconds:
                print(f"Time limit reached ({time.time() - t0:.0f}s) at step {self.global_step}")
                break

            if self.global_step % self.eval_every == 0:
                val_metrics = self._validate()
                val_loss = val_metrics.get("loss", float("inf"))
                train_avg = running_loss / max(n_accum, 1)

                self.history.append({
                    "step": self.global_step,
                    "train_loss": train_avg,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })

                elapsed = time.time() - t0
                steps_per_sec = self.global_step / elapsed
                eta = (self.max_steps - self.global_step) / max(steps_per_sec, 0.01)
                print(f"  step {self.global_step:>6d}/{self.max_steps} | "
                      f"train={train_avg:.4f} val={val_loss:.4f} | "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"{steps_per_sec:.1f} step/s ETA {eta:.0f}s")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        self.output_dir / "best_model.pt",
                    )
                else:
                    self.patience_counter += 1

                running_loss = 0.0
                n_accum = 0
                self.model.train()

                if self.patience_counter >= self.patience:
                    print(f"Early stopping at step {self.global_step}")
                    break

        elapsed = time.time() - t0
        print(f"Training complete in {elapsed:.0f}s ({self.global_step} steps)")

        best_path = self.output_dir / "best_model.pt"
        if not best_path.exists():
            # Time ran out before first eval checkpoint — save current model
            torch.save(self.model.state_dict(), best_path)
            print("No eval checkpoint before time limit; saving current model")
        self.model.load_state_dict(
            torch.load(best_path, weights_only=True), strict=False
        )

        val_metrics = self._validate()

        # Run unified evaluation (architecture-agnostic metrics)
        eval_metrics = None
        # Use eval_tensors (longer windows) if available, else val_tensors
        eval_data = self.eval_tensors if self.eval_tensors is not None else self.val_tensors
        if eval_data is not None:
            try:
                from evaluator import UnifiedEvaluator
                evaluator = UnifiedEvaluator(
                    model=self.model,
                    model_type=self.model_type,
                    val_tensors=eval_data,
                    device=self.device,
                    max_batches=self.eval_max_batches,
                    horizons=self.final_eval_horizons,
                    eval_batch_size=self.eval_batch_size,
                    eval_timeout_per_horizon=self.eval_timeout_per_horizon,
                    context_len=self.context_len,
                )
                eval_metrics = evaluator.evaluate_all()
                parts = []
                for hk in sorted(eval_metrics.keys()):
                    if hk.startswith("h") and isinstance(eval_metrics[hk], dict):
                        comp = eval_metrics[hk].get("composite")
                        if comp is not None:
                            parts.append(f"{hk}={comp:.4f}")
                print(f"Eval: {' '.join(parts)} ({eval_metrics['eval_time_seconds']:.1f}s)")
            except Exception as e:
                eval_metrics = {"error": str(e)}
                print(f"Eval error: {e}")

        # Determine why training stopped
        if self.max_train_seconds and elapsed >= self.max_train_seconds:
            stopped_by = "time_limit"
        elif self.patience_counter >= self.patience:
            stopped_by = "early_stopping"
        else:
            stopped_by = "max_steps"

        result = {
            "best_val_loss": self.best_val_loss,
            "final_val_metrics": val_metrics,
            "eval_metrics": eval_metrics,
            "total_steps": self.global_step,
            "elapsed_seconds": elapsed,
            "max_train_seconds": self.max_train_seconds,
            "stopped_by": stopped_by,
            "history": self.history,
        }
        with open(self.output_dir / "training_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return result

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metrics_accum: dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss_dict = self.loss_fn(self.model, batch)
            total_loss += loss_dict["loss"].item()
            for k, v in loss_dict.items():
                if k != "loss" and isinstance(v, (float, int, torch.Tensor)):
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    metrics_accum[k] = metrics_accum.get(k, 0) + val
            n_batches += 1

            if n_batches >= 50:
                break

        result = {"loss": total_loss / max(n_batches, 1)}
        for k, v in metrics_accum.items():
            result[k] = v / max(n_batches, 1)
        return result
