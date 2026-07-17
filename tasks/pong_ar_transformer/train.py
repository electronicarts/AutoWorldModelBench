# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""AR-Transformer Baseline — Continuous Autoregressive Transformer for ECS World Modeling.

Predicts continuous next-frame entity positions, alive state, terminal flag,
and gameplay fields via a Transformer conditioned on W frames of history.

Architecture:
  1. Encode entities: Linear(registry || state) -> per-entity embeddings
  2. Temporal context: TemporalContextEncoder (block-causal, W+1 frames)
  3. Cross-attention decoder: target entities attend to encoded context
  4. Output heads: delta-position, alive logit, terminal logit, gameplay fields

Structurally simpler than D3PM/MaskGIT — no noise schedule, no iterative
sampling. Single forward pass produces all predictions.

Self-contained train.py template for the AutoWorldBench task harness.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR.parent / "lib"))        # from templates/
sys.path.insert(0, str(TASK_DIR.parent.parent / "lib"))  # from tasks/game_model/

from temporal import TemporalContextEncoder, CrossAttentionBlock
from trainer import BaselineTrainer
from loader import InMemoryLoader

# ---------------------------------------------------------------------------
# Symlog / symexp transforms (DreamerV3 Appendix B)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


class ARTransformer(nn.Module):
    """Continuous autoregressive Transformer world model.

    Reuses TemporalContextEncoder from temporal.py for block-causal
    encoding of history frames, then decodes per-entity predictions
    via cross-attention.
    """

    def __init__(
        self,
        registry_dim: int = 34,
        state_dim: int = 19,
        action_dim: int = 12,
        global_dim: int = 15,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        max_frames: int = 32,
        num_gameplay_fields: int = 14,
        vel_consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.num_gameplay_fields = num_gameplay_fields
        self.vel_consistency_weight = vel_consistency_weight

        # -- Entity encoder: registry || state -> hidden --
        self.entity_proj = nn.Sequential(
            nn.Linear(registry_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # -- Action + globals projection --
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # -- Temporal position embedding --
        self.temporal_pos_emb = nn.Embedding(max_frames, hidden_dim)

        # -- Role embedding: 0=context, 1=target --
        self.role_emb = nn.Embedding(2, hidden_dim)

        # -- Part 1: Temporal Context Encoder (shared architecture with D3PM/MaskGIT) --
        self.context_encoder = TemporalContextEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        # -- Part 2: Decoder (cross-attention to context) --
        self.decoder = CrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # -- Output heads --
        self.pos_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # Δpos_x, Δpos_y
        )
        self.alive_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # alive logit
        )
        self.terminal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # terminal logit
        )
        self.gameplay_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_gameplay_fields),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_entity_tokens(
        self,
        states: torch.Tensor,
        registry: torch.Tensor,
        actions: torch.Tensor,
        globals_: torch.Tensor,
        frame_indices: torch.Tensor,
        role: int,
    ) -> torch.Tensor:
        """Build entity-level tokens for a sequence of frames.

        Args:
            states: [B, F, N, D_s]
            registry: [B, N, D_r]
            actions: [B, F, D_a]
            globals_: [B, F, D_g]
            frame_indices: [F] long
            role: 0 for context, 1 for target

        Returns:
            [B, F*N, H]
        """
        B, F, N, _ = states.shape
        device = states.device

        # Expand registry across frames: [B, N, D_r] -> [B, F, N, D_r]
        reg_exp = registry.unsqueeze(1).expand(-1, F, -1, -1)

        # Entity projection: [B, F, N, D_r + D_s] -> [B, F, N, H]
        x = self.entity_proj(torch.cat([reg_exp, states], dim=-1))

        # Temporal position: [F, H] -> [1, F, 1, H]
        temp_pos = self.temporal_pos_emb(frame_indices).unsqueeze(0).unsqueeze(2)
        x = x + temp_pos

        # Action + globals: [B, F, D_a + D_g] -> [B, F, H] -> [B, F, 1, H]
        act_emb = self.action_proj(
            torch.cat([actions, globals_], dim=-1)
        ).unsqueeze(2)
        x = x + act_emb

        # Role embedding
        x = x + self.role_emb(torch.tensor(role, device=device))

        return x.reshape(B, F * N, self.hidden_dim)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass over a window of frames.

        batch keys:
          registry:       [B, N, D_r]
          states:         [B, W, N, D_s]
          target_states:  [B, W, N, D_s]
          actions:        [B, W, D_a]
          globals:        [B, W, D_g]
          mutable_mask:   [B, N]
          entity_mask:    [B, N]
          gameplay_mask:  [B, N, G]
          terminals:      [B, W]
        """
        registry = batch["registry"]
        states = batch["states"]
        targets = batch["target_states"]
        actions = batch["actions"]
        globals_ = batch["globals"]
        mutable = batch["mutable_mask"]
        ent_mask = batch["entity_mask"]
        gameplay_mask = batch["gameplay_mask"]
        terminal_targets = batch["terminals"]

        B, W, N, D_s = states.shape
        device = states.device
        G = self.num_gameplay_fields

        # ---- Build W+1 context frames ----
        context_states = torch.cat([
            states, targets[:, W - 1:W],
        ], dim=1)
        context_actions = torch.cat([
            actions, actions[:, W - 1:W],
        ], dim=1)
        context_globals = torch.cat([
            globals_, globals_[:, W - 1:W],
        ], dim=1)

        # ---- Encode context (single pass) ----
        ctx_frame_indices = torch.arange(W + 1, device=device)
        context_tokens = self._build_entity_tokens(
            states=context_states,
            registry=registry,
            actions=context_actions,
            globals_=context_globals,
            frame_indices=ctx_frame_indices,
            role=0,
        )

        context_h = self.context_encoder(
            tokens=context_tokens,
            entity_mask=ent_mask,
            num_frames=W + 1,
            num_entities=N,
        )

        # ---- Predict each target frame ----
        pred_mask = mutable & ent_mask
        pm_float = pred_mask.float()
        n_pred = pm_float.sum().clamp(min=1)

        total_recon = 0.0
        total_terminal = 0.0
        total_gameplay = 0.0
        total_vel_consist = 0.0

        for t in range(W):
            # Build target entity tokens for frame t+1
            target_state = targets[:, t:t + 1]  # [B, 1, N, D_s]
            target_action = actions[:, t:t + 1]
            target_globals = globals_[:, t:t + 1]
            target_frame_idx = torch.tensor([t + 1], device=device)

            target_h = self._build_entity_tokens(
                states=target_state,
                registry=registry,
                actions=target_action,
                globals_=target_globals,
                frame_indices=target_frame_idx,
                role=1,
            )  # [B, N, H]

            # Cross-attention: target attends to context frames 0..t
            ctx_slice = context_h[:, :(t + 1) * N, :]
            ctx_ent_valid = ent_mask.repeat(1, t + 1)
            self_mask = ent_mask.unsqueeze(1).expand(-1, N, -1)
            cross_mask = ctx_ent_valid.unsqueeze(1).expand(-1, N, -1)

            decoded = self.decoder(
                queries=target_h,
                context=ctx_slice,
                cross_mask=cross_mask,
                self_mask=self_mask,
            )
            decoded = self.norm_out(decoded)  # [B, N, H]

            # Predictions
            pred_delta_pos = self.pos_decoder(decoded)              # [B, N, 2]
            pred_alive = self.alive_decoder(decoded).squeeze(-1)    # [B, N]
            pred_gameplay = self.gameplay_decoder(decoded)           # [B, N, G]

            # Terminal (pooled from decoded entities)
            pool = (decoded * ent_mask.unsqueeze(-1).float()).sum(1) / \
                   ent_mask.float().sum(1, keepdim=True).clamp(min=1)
            pred_terminal = self.terminal_head(pool).squeeze(-1)    # [B]

            # Compute targets
            target_pos = targets[:, t, :, :2]
            current_pos = states[:, t, :, :2]
            delta_pos = target_pos - current_pos
            target_alive = targets[:, t, :, 2]

            # Position loss (MSE on deltas, mutable only)
            pos_loss = ((symlog(pred_delta_pos) - symlog(delta_pos)) ** 2 * pm_float.unsqueeze(-1)).sum() / n_pred

            # Alive loss (BCE, mutable only)
            alive_loss = F.binary_cross_entropy_with_logits(
                pred_alive, target_alive, weight=pm_float, reduction="sum"
            ) / n_pred

            # Terminal loss
            terminal_loss = F.binary_cross_entropy_with_logits(
                pred_terminal, terminal_targets[:, t], reduction="mean"
            )

            # Gameplay loss (MSE, masked per entity and per field)
            if D_s > 5 and G > 0:
                target_gp = targets[:, t, :, 5:5 + G]
                gp_mask = gameplay_mask & pred_mask.unsqueeze(-1)
                gp_mask_float = gp_mask.float()
                n_gp = gp_mask_float.sum().clamp(min=1)
                gameplay_loss = ((symlog(pred_gameplay) - symlog(target_gp)) ** 2 * gp_mask_float).sum() / n_gp
            else:
                gameplay_loss = torch.tensor(0.0, device=device)

            # Velocity consistency
            if D_s >= 5 and self.vel_consistency_weight > 0:
                input_vel = states[:, t, :, 3:5]
                vel_active = (input_vel.abs().sum(-1) > 1e-6) & pred_mask
                n_vel = vel_active.float().sum().clamp(min=1)
                vel_err = ((pred_delta_pos - input_vel) ** 2 * vel_active.float().unsqueeze(-1)).sum() / n_vel
                vel_loss = self.vel_consistency_weight * vel_err
            else:
                vel_loss = torch.tensor(0.0, device=device)

            total_recon += pos_loss + alive_loss
            total_terminal += terminal_loss
            total_gameplay += gameplay_loss
            total_vel_consist += vel_loss

        n_steps = max(W, 1)
        loss = (total_recon + total_terminal + total_gameplay + total_vel_consist) / n_steps

        return {
            "loss": loss,
            "recon_loss": (total_recon / n_steps).detach(),
            "terminal_loss": (total_terminal / n_steps).detach(),
            "gameplay_loss": (total_gameplay / n_steps).detach(),
            "vel_consistency": (total_vel_consist / n_steps).detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_step(
        self,
        registry: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step prediction for evaluation.

        Returns:
            next_state: [B, N, 3 + G] — pos(2), alive(1), gameplay(G)
            terminal:   [B] — terminal logit
        """
        B, N, _ = state.shape
        device = state.device
        G = self.num_gameplay_fields

        # Build single-frame context
        ctx_states = state.unsqueeze(1)
        ctx_actions = action.unsqueeze(1)
        ctx_globals = globals_.unsqueeze(1)
        frame_idx = torch.tensor([0], device=device)

        context_tokens = self._build_entity_tokens(
            ctx_states, registry, ctx_actions, ctx_globals, frame_idx, role=0,
        )
        context_h = self.context_encoder(
            context_tokens, entity_mask, num_frames=1, num_entities=N,
        )

        # Build target tokens (use current state as input for target frame)
        target_tokens = self._build_entity_tokens(
            ctx_states, registry, ctx_actions, ctx_globals,
            torch.tensor([1], device=device), role=1,
        )

        # Cross-attention decode
        self_mask = entity_mask.unsqueeze(1).expand(-1, N, -1)
        cross_mask = entity_mask.unsqueeze(1).expand(-1, N, -1)

        decoded = self.norm_out(self.decoder(
            queries=target_tokens,
            context=context_h,
            cross_mask=cross_mask,
            self_mask=self_mask,
        ))

        delta_pos = symexp(self.pos_decoder(decoded))
        alive_logit = self.alive_decoder(decoded).squeeze(-1)
        pred_gameplay = symexp(self.gameplay_decoder(decoded))

        pool = (decoded * entity_mask.unsqueeze(-1).float()).sum(1) / \
               entity_mask.float().sum(1, keepdim=True).clamp(min=1)
        terminal = self.terminal_head(pool).squeeze(-1)

        next_pos = state[:, :, :2] + delta_pos
        next_pos = next_pos.clamp(0.0, 1.0)
        next_alive = (alive_logit > 0).float()

        next_state = torch.cat([
            next_pos,
            next_alive.unsqueeze(-1),
            pred_gameplay,
        ], dim=-1)

        return next_state, terminal

    @torch.no_grad()
    def predict_rollout(
        self,
        registry: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step autoregressive prediction.

        Args:
            registry: [B, N, D_r]
            states: [B, W, N, D_s] — history frames
            actions: [B, W+H, D_a] — actions for history + future
            globals_: [B, W+H, D_g]
            entity_mask: [B, N]

        Returns:
            predictions: [B, H, N, 3 + G]
            terminals:   [B, H] -- terminal logits
        """
        B, W, N, D_s = states.shape
        H = actions.shape[1] - W
        device = states.device
        G = self.num_gameplay_fields

        if H <= 0:
            return (torch.empty(B, 0, N, 3 + G, device=device),
                    torch.empty(B, 0, device=device))

        # Encode initial context
        frame_indices = torch.arange(W, device=device)
        context_tokens = self._build_entity_tokens(
            states, registry, actions[:, :W], globals_[:, :W],
            frame_indices, role=0,
        )
        context_h = self.context_encoder(
            context_tokens, entity_mask, num_frames=W, num_entities=N,
        )
        num_ctx_frames = W

        predictions = []
        terminals_list = []
        last_state = states[:, -1]  # [B, N, D_s]

        for step in range(H):
            abs_frame = W + step
            act_idx = W + step - 1

            # Build target tokens from last known state
            target_tokens = self._build_entity_tokens(
                last_state.unsqueeze(1), registry,
                actions[:, act_idx:act_idx + 1],
                globals_[:, act_idx:act_idx + 1],
                torch.tensor([min(abs_frame, self.max_frames - 1)], device=device),
                role=1,
            )

            # Cross-attention
            ctx_slice = context_h[:, :num_ctx_frames * N]
            self_mask = entity_mask.unsqueeze(1).expand(-1, N, -1)
            ctx_valid = entity_mask.repeat(1, num_ctx_frames)
            cross_mask = ctx_valid.unsqueeze(1).expand(-1, N, -1)

            decoded = self.norm_out(self.decoder(
                queries=target_tokens, context=ctx_slice,
                cross_mask=cross_mask, self_mask=self_mask,
            ))

            pool = (decoded * entity_mask.unsqueeze(-1).float()).sum(1) / \
                   entity_mask.float().sum(1, keepdim=True).clamp(min=1)
            pred_terminal = self.terminal_head(pool).squeeze(-1)
            terminals_list.append(pred_terminal)

            delta_pos = symexp(self.pos_decoder(decoded))
            alive_logit = self.alive_decoder(decoded).squeeze(-1)
            pred_gameplay = symexp(self.gameplay_decoder(decoded))

            next_pos = last_state[:, :, :2] + delta_pos
            next_pos = next_pos.clamp(0.0, 1.0)
            next_alive = (alive_logit > 0).float()

            pred_state = torch.cat([
                next_pos, next_alive.unsqueeze(-1), pred_gameplay,
            ], dim=-1)  # [B, N, 3 + G]
            predictions.append(pred_state)

            # Build full D_s state for next step's context
            new_state_full = torch.zeros(B, N, D_s, device=device)
            new_state_full[:, :, :2] = next_pos
            new_state_full[:, :, 2] = next_alive
            if D_s > 5 and G > 0:
                new_state_full[:, :, 5:5 + G] = pred_gameplay

            # Append to context (approximation: no re-encoding)
            new_ctx = self._build_entity_tokens(
                new_state_full.unsqueeze(1), registry,
                actions[:, W + step:W + step + 1],
                globals_[:, W + step:W + step + 1],
                torch.tensor([min(abs_frame, self.max_frames - 1)], device=device),
                role=0,
            )
            context_h = torch.cat([context_h, new_ctx], dim=1)
            num_ctx_frames += 1
            if num_ctx_frames > W:
                context_h = context_h[:, N:]
                num_ctx_frames -= 1
            last_state = new_state_full

        return torch.stack(predictions, dim=1), torch.stack(terminals_list, dim=1)


# ======================================================================
# Loss function adapter
# ======================================================================

def ar_transformer_loss_fn(model: ARTransformer, batch: dict) -> dict[str, torch.Tensor]:
    """Adapter for the AutoWorldBench training harness."""
    return model(batch)


def build_model(config: dict, train_tensors: dict, device: str = "cuda") -> ARTransformer:
    """Reconstruct an ARTransformer from config and tensor dimensions.

    Used by score.py for independent checkpoint evaluation.
    """
    registry_dim = train_tensors["registry"].shape[-1]
    state_dim = train_tensors["states"].shape[-1]
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]
    model = ARTransformer(
        registry_dim=registry_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 8),
        num_encoder_layers=config.get("num_encoder_layers", 3),
        dropout=config.get("dropout", 0.1),
        max_frames=config.get("max_frames", 32),
        num_gameplay_fields=config.get("num_gameplay_fields", 14),
        vel_consistency_weight=config.get("vel_consistency_weight", 0.1),
    )
    return model.to(device)


# ======================================================================
# Training main
# ======================================================================

def main():
    config = json.loads((TASK_DIR / "config.json").read_text())
    if os.environ.get("MAX_TRAIN_SECONDS"):
        config["max_train_seconds"] = int(os.environ["MAX_TRAIN_SECONDS"])

    cache_dir = Path(config["cache_dir"]) if "cache_dir" in config else TASK_DIR / "_cache"
    train_tensors = torch.load(cache_dir / "train_tensors.pt", weights_only=False)
    val_tensors = torch.load(cache_dir / "val_tensors.pt", weights_only=False)
    eval_path = cache_dir / "val_eval_tensors.pt"
    eval_tensors = torch.load(eval_path, weights_only=False) if eval_path.exists() else None

    batch_size = config.get("batch_size", 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = InMemoryLoader(train_tensors, batch_size, shuffle=True, device=device)
    val_loader = InMemoryLoader(val_tensors, batch_size, shuffle=False, device=device)

    registry_dim = train_tensors["registry"].shape[-1]
    state_dim = train_tensors["states"].shape[-1]
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]

    model = ARTransformer(
        registry_dim=registry_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
    )

    trainer = BaselineTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=ar_transformer_loss_fn,
        lr=config.get("lr", 3e-4),
        max_steps=config.get("max_steps", 10000),
        eval_every=config.get("eval_every", 500),
        patience=config.get("patience", 10),
        output_dir=TASK_DIR / "outputs",
        device=device,
        model_type=config["model_type"],
        val_tensors=val_tensors,
        eval_tensors=eval_tensors,
        max_train_seconds=config.get("max_train_seconds"),
        final_eval_horizons=config.get("final_eval_horizons"),
    )

    result = trainer.train()
    print(f"Best val_loss: {result['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
