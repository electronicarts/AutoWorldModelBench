# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Architecture-agnostic evaluation for all world-model baselines.

Computes shared prediction-quality metrics (position L1, alive F1,
terminal F1, composite score) at multiple rollout horizons.
Works with all 5 model families via a thin adapter layer.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ======================================================================
# Metrics
# ======================================================================

def position_l1(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Mean L1 position error over mutable entities.

    Args:
        pred: [..., 3+] predicted state (pos_x, pos_y, alive, ...).
        true: [..., D_s] ground-truth state.
        mask: [...] bool mask for valid mutable entities.

    Returns:
        Scalar mean |pred_pos - true_pos| (sum over x,y).
    """
    err = (pred[..., :2] - true[..., :2]).abs().sum(-1)  # [...] L1 per entity
    if mask.sum() == 0:
        return 0.0
    return (err * mask.float()).sum().item() / mask.float().sum().item()


def alive_f1(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """F1 score for alive/dead binary classification.

    Args:
        pred: [..., 3+] predicted state (channel 2 = alive).
        true: [..., D_s] ground-truth state (channel 2 = alive).
        mask: [...] bool mask for valid mutable entities.
    """
    pred_alive = (pred[..., 2] > 0.5).bool()
    true_alive = (true[..., 2] > 0.5).bool()
    return _binary_f1(pred_alive, true_alive, mask)


def terminal_f1(
    pred_terminal: torch.Tensor,
    true_terminal: torch.Tensor,
) -> float:
    """F1 score for terminal flag prediction.

    Args:
        pred_terminal: [...] terminal logits (pre-sigmoid).
        true_terminal: [...] ground-truth terminal flags (0/1).
    """
    pred_bin = (torch.sigmoid(pred_terminal) > 0.5).bool()
    true_bin = (true_terminal > 0.5).bool()
    mask = torch.ones_like(pred_bin)
    return _binary_f1(pred_bin, true_bin, mask)


def composite_score(pos_l1: float, alive_f: float) -> float:
    """Composite = 0.9 * (1 - position_l1) + 0.1 * alive_f1."""
    return 0.9 * max(0.0, 1.0 - pos_l1) + 0.1 * alive_f


def _binary_f1(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute micro-F1 for binary classification under mask."""
    pred_flat = pred[mask].bool()
    true_flat = true[mask].bool()
    if pred_flat.numel() == 0:
        return 1.0

    tp = (pred_flat & true_flat).sum().float()
    fp = (pred_flat & ~true_flat).sum().float()
    fn = (~pred_flat & true_flat).sum().float()

    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)

    if (tp + fp) == 0 and (tp + fn) == 0:
        return 1.0  # no positives in pred or true — perfect
    if tp == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.item()


# ======================================================================
# Model adapters
# ======================================================================

class ModelAdapter(ABC):
    """Thin wrapper normalizing predict_step signatures."""

    @abstractmethod
    def predict_one_step(
        self,
        registry: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (next_state [B,N,3+G], terminal_logit [B])."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any recurrent state between sequences."""



class RecurrentAdapter(ModelAdapter):
    """For Dreamer — carries hidden state h across steps."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.h: torch.Tensor | None = None

    def predict_one_step(self, registry, state, action, globals_, entity_mask):
        next_state, terminal, self.h = self.model.predict_step(
            registry, state, action, globals_, entity_mask, h=self.h,
        )
        return next_state, terminal

    def reset(self):
        self.h = None


class StatelessAdapter(ModelAdapter):
    """For AR-Transformer, D3PM, MaskGIT — no hidden state."""

    def __init__(self, model: nn.Module):
        self.model = model

    def predict_rollout(self, registry, states, actions, globals_, entity_mask):
        return self.model.predict_rollout(
            registry, states, actions, globals_, entity_mask,
        )

    def predict_one_step(self, registry, state, action, globals_, entity_mask):
        next_state, terminal = self.model.predict_step(
            registry, state, action, globals_, entity_mask,
        )
        return next_state, terminal

    def reset(self):
        pass


def make_adapter(model: nn.Module, model_type: str) -> ModelAdapter:
    """Factory: select adapter based on model_type string."""
    if model_type == "dreamer":
        return RecurrentAdapter(model)
    return StatelessAdapter(model)


# ======================================================================
# Unified evaluator
# ======================================================================

class UnifiedEvaluator:
    """Architecture-agnostic evaluation at multiple horizons.

    Computes position_l1, alive_f1, terminal_f1, composite at h=1
    (teacher-forced) and h=H (open-loop rollout).
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        val_tensors: dict[str, torch.Tensor],
        device: str = "cuda",
        max_batches: int = 16,
        horizons: list[int] | None = None,
        eval_batch_size: int = 64,
        eval_timeout_per_horizon: float | None = None,
        context_len: int = 8,
    ):
        self.adapter = make_adapter(model, model_type)
        self.model = model
        self.val_tensors = val_tensors
        self.device = device
        self.max_batches = max_batches
        self.horizons = horizons or [1, 10, 20]
        self.eval_batch_size = eval_batch_size
        self.eval_timeout_per_horizon = eval_timeout_per_horizon
        self.context_len = context_len
        self.is_recurrent = model_type == "dreamer"

    def _iter_batches(self):
        """Yield batches from val_tensors with eval_batch_size."""
        n_samples = next(iter(self.val_tensors.values())).shape[0]
        bs = self.eval_batch_size
        n_batches = 0
        for start in range(0, n_samples, bs):
            if n_batches >= self.max_batches:
                break
            batch = {
                k: v[start : start + bs].to(self.device)
                for k, v in self.val_tensors.items()
            }
            yield batch
            n_batches += 1

    def _get_mask(self, batch: dict) -> torch.Tensor:
        """Combined mutable & entity mask [B, N]."""
        return batch["mutable_mask"] & batch["entity_mask"]

    @torch.no_grad()
    def evaluate_teacher_forced(self) -> dict[str, float]:
        """Teacher-forced (h=1): predict each frame from ground-truth input.

        For recurrent models, hidden state accumulates across the window
        using ground-truth states (not predictions).
        """
        self.model.eval()
        total_pos_err = 0.0
        total_entities = 0
        all_pred_alive = []
        all_true_alive = []
        all_pred_terminal = []
        all_true_terminal = []

        use_amp = self.device == "cuda"
        for batch in self._iter_batches():
            B, W, N, D_s = batch["states"].shape
            mask = self._get_mask(batch)  # [B, N]

            self.adapter.reset()
            for t in range(W - 1):
                state = batch["states"][:, t]         # [B, N, D_s]
                action = batch["actions"][:, t]       # [B, D_a]
                globals_ = batch["globals"][:, t]     # [B, D_g]
                true_next = batch["target_states"][:, t]  # [B, N, D_s]
                true_term = batch["terminals"][:, t]  # [B]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_state, pred_terminal = self.adapter.predict_one_step(
                        batch["registry"], state, action, globals_,
                        batch["entity_mask"],
                    )

                # Position L1 accumulation
                err = (pred_state[..., :2] - true_next[..., :2]).abs().sum(-1)
                total_pos_err += (err * mask.float()).sum().item()
                total_entities += mask.float().sum().item()

                # Alive accumulation (masked)
                all_pred_alive.append((pred_state[..., 2] > 0.5)[mask])
                all_true_alive.append((true_next[..., 2] > 0.5)[mask])

                # Terminal accumulation
                all_pred_terminal.append(pred_terminal)
                all_true_terminal.append(true_term)

        pos_l1 = total_pos_err / max(total_entities, 1.0)

        pred_alive_cat = torch.cat(all_pred_alive).bool()
        true_alive_cat = torch.cat(all_true_alive).bool()
        alive_mask = torch.ones_like(pred_alive_cat)
        alive_f = _binary_f1(pred_alive_cat, true_alive_cat, alive_mask)

        pred_term_cat = torch.cat(all_pred_terminal)
        true_term_cat = torch.cat(all_true_terminal)
        term_f = terminal_f1(pred_term_cat, true_term_cat)

        comp = composite_score(pos_l1, alive_f)

        return {
            "position_l1": round(pos_l1, 6),
            "alive_f1": round(alive_f, 6),
            "terminal_f1": round(term_f, 6),
            "composite": round(comp, 6),
        }

    @torch.no_grad()
    def evaluate_rollout(self, horizon: int) -> dict[str, float]:
        """Open-loop rollout with immutable pinning.

        Routes to recurrent (Dreamer) or stateless (AR-Trans/D3PM/MaskGIT)
        evaluation path. Both pin immutable entities to ground truth each step.
        Stateless models use a sliding W-frame context window matching training.
        """
        if self.is_recurrent:
            return self._evaluate_rollout_recurrent(horizon)
        return self._evaluate_rollout_stateless(horizon)

    def _collect_metrics(self, total_pos_err, total_entities,
                         all_pred_alive, all_true_alive,
                         all_pred_terminal, all_true_terminal) -> dict[str, float]:
        """Aggregate accumulated metrics into a result dict."""
        if total_entities == 0:
            return {"position_l1": float("nan"), "alive_f1": float("nan"),
                    "terminal_f1": float("nan"), "composite": float("nan")}

        pos_l1 = total_pos_err / max(total_entities, 1.0)
        pred_alive_cat = torch.cat(all_pred_alive).bool()
        true_alive_cat = torch.cat(all_true_alive).bool()
        alive_mask = torch.ones_like(pred_alive_cat)
        alive_f = _binary_f1(pred_alive_cat, true_alive_cat, alive_mask)
        pred_term_cat = torch.cat(all_pred_terminal)
        true_term_cat = torch.cat(all_true_terminal)
        term_f = terminal_f1(pred_term_cat, true_term_cat)
        comp = composite_score(pos_l1, alive_f)
        return {"position_l1": round(pos_l1, 6), "alive_f1": round(alive_f, 6),
                "terminal_f1": round(term_f, 6), "composite": round(comp, 6)}

    @torch.no_grad()
    def _evaluate_rollout_recurrent(self, horizon: int) -> dict[str, float]:
        """Recurrent models (Dreamer): predict_one_step loop with hidden state."""
        self.model.eval()
        total_pos_err = 0.0
        total_entities = 0
        all_pred_alive, all_true_alive = [], []
        all_pred_terminal, all_true_terminal = [], []

        use_amp = self.device == "cuda"
        h_start = time.time()
        for batch in self._iter_batches():
            if self.eval_timeout_per_horizon is not None:
                if time.time() - h_start > self.eval_timeout_per_horizon:
                    break
            B, W, N, D_s = batch["states"].shape
            if W - 1 < horizon:
                continue
            mask = self._get_mask(batch)
            mutable = batch["mutable_mask"]
            immutable = ~mutable & batch["entity_mask"]

            self.adapter.reset()
            current_state = batch["states"][:, 0]

            for step in range(horizon):
                action = batch["actions"][:, step]
                globals_ = batch["globals"][:, step]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_state, pred_terminal = self.adapter.predict_one_step(
                        batch["registry"], current_state, action, globals_,
                        batch["entity_mask"],
                    )

                true_state = batch["states"][:, step + 1]
                true_term = batch["terminals"][:, step]

                # Pin immutable entities to ground truth
                imm = immutable.unsqueeze(-1)
                pred_state = torch.where(
                    imm.expand_as(pred_state),
                    true_state[..., :pred_state.shape[-1]],
                    pred_state,
                )

                if step == horizon - 1:
                    err = (pred_state[..., :2] - true_state[..., :2]).abs().sum(-1)
                    total_pos_err += (err * mask.float()).sum().item()
                    total_entities += mask.float().sum().item()
                    all_pred_alive.append((pred_state[..., 2] > 0.5)[mask])
                    all_true_alive.append((true_state[..., 2] > 0.5)[mask])
                    all_pred_terminal.append(pred_terminal)
                    all_true_terminal.append(true_term)

                next_full = torch.zeros_like(current_state)
                next_full[:, :, :2] = pred_state[:, :, :2]
                next_full[:, :, 2] = pred_state[:, :, 2]
                gp = pred_state.shape[-1] - 3
                if gp > 0 and D_s > 5:
                    next_full[:, :, 5:5 + gp] = pred_state[:, :, 3:]
                current_state = next_full

        return self._collect_metrics(total_pos_err, total_entities,
                                     all_pred_alive, all_true_alive,
                                     all_pred_terminal, all_true_terminal)

    @torch.no_grad()
    def _evaluate_rollout_stateless(self, horizon: int) -> dict[str, float]:
        """Stateless models: step-by-step predict_rollout with sliding W-frame
        context window and immutable pinning.

        At each step: pass W context frames to predict_rollout (predicting 1
        step), pin immutables to GT, slide window (drop oldest, append prediction).
        This matches training where models see W context frames.
        """
        self.model.eval()
        total_pos_err = 0.0
        total_entities = 0
        all_pred_alive, all_true_alive = [], []
        all_pred_terminal, all_true_terminal = [], []

        use_amp = self.device == "cuda"
        h_start = time.time()
        ctx_W = self.context_len

        for batch in self._iter_batches():
            if self.eval_timeout_per_horizon is not None:
                if time.time() - h_start > self.eval_timeout_per_horizon:
                    break
            B, W, N, D_s = batch["states"].shape
            if ctx_W + horizon > W:
                continue

            mask = self._get_mask(batch)
            mutable = batch["mutable_mask"]
            immutable = ~mutable & batch["entity_mask"]

            # Initialize sliding window with first ctx_W GT frames
            window_states = batch["states"][:, :ctx_W].clone()

            for step in range(horizon):
                # Actions for ctx_W context + 1 prediction step
                act_start = step
                act_end = step + ctx_W + 1
                if act_end > W:
                    break
                step_actions = batch["actions"][:, act_start:act_end]
                step_globals = batch["globals"][:, act_start:act_end]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = self.adapter.predict_rollout(
                        batch["registry"], window_states,
                        step_actions, step_globals,
                        batch["entity_mask"],
                    )

                # Handle (predictions, terminals) tuple or just predictions
                if isinstance(preds, tuple):
                    pred_all, term_all = preds
                else:
                    pred_all = preds
                    term_all = None

                pred_state = pred_all[:, 0]  # single predicted step
                true_state = batch["states"][:, ctx_W + step]
                true_term = batch["terminals"][:, ctx_W + step - 1]

                # Pin immutable entities to ground truth
                imm = immutable.unsqueeze(-1)
                pred_state = torch.where(
                    imm.expand_as(pred_state),
                    true_state[..., :pred_state.shape[-1]],
                    pred_state,
                )

                if step == horizon - 1:
                    err = (pred_state[..., :2] - true_state[..., :2]).abs().sum(-1)
                    total_pos_err += (err * mask.float()).sum().item()
                    total_entities += mask.float().sum().item()
                    all_pred_alive.append((pred_state[..., 2] > 0.5)[mask])
                    all_true_alive.append((true_state[..., 2] > 0.5)[mask])
                    if term_all is not None:
                        all_pred_terminal.append(term_all[:, 0])
                    else:
                        all_pred_terminal.append(torch.zeros(B, device=self.device))
                    all_true_terminal.append(true_term)

                # Build full D_s state for sliding window
                new_frame = torch.zeros(B, 1, N, D_s, device=self.device)
                new_frame[:, 0, :, :2] = pred_state[:, :, :2]
                new_frame[:, 0, :, 2] = pred_state[:, :, 2]
                gp = pred_state.shape[-1] - 3
                if gp > 0 and D_s > 5:
                    new_frame[:, 0, :, 5:5 + gp] = pred_state[:, :, 3:]

                # Slide window: drop oldest, append prediction
                window_states = torch.cat([window_states[:, 1:], new_frame], dim=1)

        return self._collect_metrics(total_pos_err, total_entities,
                                     all_pred_alive, all_true_alive,
                                     all_pred_terminal, all_true_terminal)

    def evaluate_all(self) -> dict:
        """Run all evaluations. Returns dict with h1, h5, etc."""
        t0 = time.time()
        results: dict = {}

        # Check max window size available
        max_W = next(iter(self.val_tensors.values())).shape[1] if "states" in self.val_tensors else 0
        if "states" in self.val_tensors:
            max_W = self.val_tensors["states"].shape[1]

        # Teacher-forced (h=1)
        results["h1"] = self.evaluate_teacher_forced()

        # Open-loop rollouts
        for h in self.horizons:
            if h <= 1:
                continue  # h=1 already covered by teacher-forced
            if h > max_W - 1:
                results[f"h{h}"] = {"skipped": True, "reason": f"window_size={max_W}, need {h+1}"}
                continue
            results[f"h{h}"] = self.evaluate_rollout(h)

        results["eval_time_seconds"] = round(time.time() - t0, 2)
        return results
