# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""DreamerV3-style World Model Baseline for ECS World Modeling.

Continuous delta prediction with discrete categorical latent space.
Based on Hafner et al. (DreamerV3). Operates on Format D tensors:
predicts delta-position, alive, terminal, and gameplay fields for mutable
entities. Conditions on immutable entities via mean-pool. Velocity
consistency serves as auxiliary regularization. Uses symlog transforms
for continuous targets and free-bits KL for the categorical prior.

Self-contained template: copy to tasks/<game>_dreamer/train.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- path setup: works from both templates/ and tasks/game_model/ --
TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR.parent / "lib"))        # from templates/
sys.path.insert(0, str(TASK_DIR.parent.parent / "lib"))  # from tasks/game_model/

from trainer import BaselineTrainer
from loader import InMemoryLoader


# ---------------------------------------------------------------------------
# Symlog / symexp transforms (DreamerV3 Appendix B)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class EntityEncoder(nn.Module):
    """Encode each entity from registry (static) + state (dynamic)."""

    def __init__(self, registry_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(registry_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, registry: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        registry: [B, N, reg_dim]
        state:    [B, N, state_dim]
        Returns:  [B, N, hidden_dim]
        """
        x = torch.cat([registry, state], dim=-1)
        return self.net(x)


class ContextEncoder(nn.Module):
    """Encode action + globals into a context vector."""

    def __init__(self, action_dim: int, global_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, actions: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        """
        actions:  [B, action_dim]
        globals_: [B, global_dim]
        Returns:  [B, hidden_dim]
        """
        return self.net(torch.cat([actions, globals_], dim=-1))


class DreamerWorldModel(nn.Module):
    """DreamerV3-style Recurrent State-Space Model for entity-based world modeling.

    Architecture:
      1. Encode entities -> per-entity embeddings -> mean-pool
      2. Encode action+globals -> context
      3. GRU update: h_t = GRU(h_{t-1}, [entity_pool, context])
      4. Prior:     z_prior ~ Cat(logits_prior(h_t))        [32 categoricals x 32 classes]
      5. Posterior:  z_post ~ Cat(logits_post(h_t, next_enc)) [32 categoricals x 32 classes]
      6. Decode:    per-entity delta-pos, alive logit (mutable only)

    Latent: 32 categorical variables each with 32 classes (straight-through).
    KL: free-bits categorical KL.  Continuous losses: symlog MSE.
    """

    def __init__(
        self,
        registry_dim: int = 34,
        state_dim: int = 19,
        action_dim: int = 12,
        global_dim: int = 15,
        hidden_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
        num_gameplay_fields: int = 14,
        max_entities: int = 64,
        free_nats: float = 1.0,
        vel_consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.latent_dim = num_categoricals * num_classes  # flat size = 1024
        self.free_nats = free_nats
        self.num_gameplay_fields = num_gameplay_fields
        self.vel_consistency_weight = vel_consistency_weight

        # Encoders
        self.entity_encoder = EntityEncoder(registry_dim, state_dim, hidden_dim)
        self.context_encoder = ContextEncoder(action_dim, global_dim, hidden_dim)

        # Recurrent backbone
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)

        # Prior / posterior — output logits for num_categoricals * num_classes
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes),
        )

        decode_in = hidden_dim + self.latent_dim + registry_dim

        # Decoders
        self.pos_decoder = nn.Sequential(
            nn.Linear(decode_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # delta-pos_x, delta-pos_y
        )
        self.alive_decoder = nn.Sequential(
            nn.Linear(decode_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # alive logit
        )
        # Terminal head: global per-frame prediction from hidden state
        self.terminal_head = nn.Sequential(
            nn.Linear(hidden_dim + self.latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),  # terminal logit
        )
        # Gameplay decoder: per-entity continuous predictions, masked per game
        self.gameplay_decoder = nn.Sequential(
            nn.Linear(decode_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_gameplay_fields),
        )

    # ------------------------------------------------------------------
    # Categorical latent helpers
    # ------------------------------------------------------------------

    def _straight_through_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical with straight-through gradient estimator.

        Args:
            logits: [B, num_categoricals * num_classes]
        Returns:
            z_flat: [B, num_categoricals * num_classes]  (one-hot per variable, flat)
        """
        B = logits.shape[0]
        logits_3d = logits.view(B, self.num_categoricals, self.num_classes)

        # Uniform exploration mixture (1%)
        logits_3d = logits_3d * 0.99 + 0.01 / self.num_classes

        # Soft probabilities for gradient path
        probs = F.softmax(logits_3d, dim=-1)  # [B, 32, 32]

        # Hard one-hot sample (no gradient)
        dist = torch.distributions.OneHotCategorical(probs=probs)
        sample = dist.sample()  # [B, 32, 32] one-hot

        # Straight-through: forward uses hard sample, backward uses soft probs
        z = sample + probs - probs.detach()  # [B, 32, 32]
        return z.view(B, -1)  # [B, 1024]

    def _categorical_kl(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Free-bits categorical KL divergence.

        Args:
            post_logits:  [B, num_categoricals * num_classes]
            prior_logits: [B, num_categoricals * num_classes]
        Returns:
            scalar KL loss
        """
        B = post_logits.shape[0]
        post_3d = post_logits.view(B, self.num_categoricals, self.num_classes)
        prior_3d = prior_logits.view(B, self.num_categoricals, self.num_classes)

        post_probs = F.softmax(post_3d, dim=-1)   # [B, 32, 32]
        prior_probs = F.softmax(prior_3d, dim=-1)  # [B, 32, 32]

        # KL per categorical variable
        kl_per_var = (post_probs * (post_probs.log() - prior_probs.log())).sum(-1)  # [B, 32]

        # Free bits: clamp each variable's KL from below
        kl_loss = torch.clamp(kl_per_var, min=self.free_nats).mean()
        return kl_loss

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass over a window of frames.

        batch keys:
          registry:       [B, N, reg_dim]
          states:         [B, W, N, state_dim]   current states
          target_states:  [B, W, N, state_dim]   next states
          actions:        [B, W, action_dim]
          globals:        [B, W, global_dim]
          mutable_mask:   [B, N] bool
          entity_mask:    [B, N] bool
          gameplay_mask:  [B, N, num_stat_fields] bool
          terminals:      [B, W]
        """
        registry = batch["registry"]         # [B, N, D_r]
        states = batch["states"]             # [B, W, N, D_s]
        targets = batch["target_states"]     # [B, W, N, D_s]
        actions = batch["actions"]           # [B, W, D_a]
        globals_ = batch["globals"]          # [B, W, D_g]
        mutable = batch["mutable_mask"]      # [B, N]
        ent_mask = batch["entity_mask"]      # [B, N]
        gameplay_mask = batch["gameplay_mask"]  # [B, N, G]
        terminal_targets = batch["terminals"]  # [B, W]

        B, W, N, D_s = states.shape
        device = states.device
        G = self.num_gameplay_fields

        h = torch.zeros(B, self.hidden_dim, device=device)

        total_recon = 0.0
        total_kl = 0.0
        total_terminal = 0.0
        total_gameplay = 0.0
        total_vel_consist = 0.0
        n_steps = 0

        for t in range(W):
            # Encode current state
            ent_emb = self.entity_encoder(registry, states[:, t])  # [B, N, H]

            # Mask and pool entities (both mutable + immutable provide context)
            mask = ent_mask.unsqueeze(-1).float()  # [B, N, 1]
            entity_pool = (ent_emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, H]

            # Context
            ctx = self.context_encoder(actions[:, t], globals_[:, t])  # [B, H]

            # GRU step
            h = self.gru(torch.cat([entity_pool, ctx], dim=-1), h)  # [B, H]

            # Prior logits
            prior_logits = self.prior_net(h)  # [B, num_cat * num_cls]

            # Posterior (using next state)
            target_emb = self.entity_encoder(registry, targets[:, t])
            target_pool = (target_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
            post_logits = self.posterior_net(torch.cat([h, target_pool], dim=-1))

            # Sample latent via straight-through
            z = self._straight_through_sample(post_logits)  # [B, 1024]

            # Decode per-entity predictions
            hz = torch.cat([h, z], dim=-1)  # [B, H + 1024]
            hz_ent = hz.unsqueeze(1).expand(-1, N, -1)  # [B, N, H+1024]
            decode_input = torch.cat([hz_ent, registry], dim=-1)  # [B, N, H+1024+D_r]

            pred_delta_pos = self.pos_decoder(decode_input)    # [B, N, 2]
            pred_alive = self.alive_decoder(decode_input).squeeze(-1)  # [B, N]
            pred_gameplay = self.gameplay_decoder(decode_input)  # [B, N, G]

            # Terminal prediction (global, from hidden + latent)
            pred_terminal = self.terminal_head(hz).squeeze(-1)  # [B]

            # Compute targets
            target_pos = targets[:, t, :, :2]                     # [B, N, 2]
            current_pos = states[:, t, :, :2]                     # [B, N, 2]
            delta_pos = target_pos - current_pos                  # [B, N, 2]
            target_alive = targets[:, t, :, 2]                    # [B, N]

            # Loss mask: only mutable & entity-present
            pred_mask = mutable & ent_mask  # [B, N]
            pm_float = pred_mask.float()
            n_pred = pm_float.sum().clamp(min=1)

            # Position loss (symlog MSE on deltas, mutable only)
            pos_loss = ((symlog(pred_delta_pos) - symlog(delta_pos)) ** 2 * pm_float.unsqueeze(-1)).sum() / n_pred

            # Alive loss (BCE, mutable only)
            alive_loss = F.binary_cross_entropy_with_logits(
                pred_alive, target_alive, weight=pm_float, reduction="sum"
            ) / n_pred

            # Terminal loss (BCE, per-frame global)
            terminal_loss = F.binary_cross_entropy_with_logits(
                pred_terminal, terminal_targets[:, t], reduction="mean"
            )

            # Gameplay field loss (symlog MSE, masked per entity and per field)
            # gameplay_mask: [B, N, G] -- True where field is meaningful
            # Only compute on mutable + valid entities
            if D_s > 5 and G > 0:
                target_gp = targets[:, t, :, 5:5 + G]  # [B, N, G]
                gp_mask = gameplay_mask & pred_mask.unsqueeze(-1)  # [B, N, G]
                gp_mask_float = gp_mask.float()
                n_gp = gp_mask_float.sum().clamp(min=1)
                gameplay_loss = ((symlog(pred_gameplay) - symlog(target_gp)) ** 2 * gp_mask_float).sum() / n_gp
            else:
                gameplay_loss = torch.tensor(0.0, device=device)

            # Velocity consistency (auxiliary): penalize if predicted delta
            # disagrees with input velocity. Only for entities with nonzero vel.
            if D_s >= 5 and self.vel_consistency_weight > 0:
                input_vel = states[:, t, :, 3:5]  # [B, N, 2]
                vel_active = (input_vel.abs().sum(-1) > 1e-6) & pred_mask  # [B, N]
                n_vel = vel_active.float().sum().clamp(min=1)
                vel_err = ((pred_delta_pos - input_vel) ** 2 * vel_active.float().unsqueeze(-1)).sum() / n_vel
                vel_loss = self.vel_consistency_weight * vel_err
            else:
                vel_loss = torch.tensor(0.0, device=device)

            # Free-bits categorical KL
            kl = self._categorical_kl(post_logits, prior_logits)

            total_recon += pos_loss + alive_loss
            total_terminal += terminal_loss
            total_gameplay += gameplay_loss
            total_vel_consist += vel_loss
            total_kl += kl
            n_steps += 1

        n_steps = max(n_steps, 1)
        loss = (total_recon + total_terminal + total_gameplay + total_vel_consist
                + total_kl) / n_steps

        return {
            "loss": loss,
            "recon_loss": (total_recon / n_steps).detach(),
            "terminal_loss": (total_terminal / n_steps).detach(),
            "gameplay_loss": (total_gameplay / n_steps).detach(),
            "vel_consistency": (total_vel_consist / n_steps).detach(),
            "kl_loss": (total_kl / n_steps).detach(),
        }

    @torch.no_grad()
    def predict_step(
        self,
        registry: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step prediction for evaluation.

        Returns:
            next_state:  [B, N, 3 + G] -- pos(2), alive(1), gameplay(G)
            terminal:    [B] -- terminal logit (sigmoid to get prob)
            updated_h:   [B, H]
        """
        B, N, _ = state.shape
        G = self.num_gameplay_fields
        if h is None:
            h = torch.zeros(B, self.hidden_dim, device=state.device)

        ent_emb = self.entity_encoder(registry, state)
        mask = entity_mask.unsqueeze(-1).float()
        entity_pool = (ent_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        ctx = self.context_encoder(action, globals_)
        h = self.gru(torch.cat([entity_pool, ctx], dim=-1), h)

        # Use prior mode at inference (no access to next state)
        prior_logits = self.prior_net(h)  # [B, num_cat * num_cls]
        logits_3d = prior_logits.view(B, self.num_categoricals, self.num_classes)
        # Mode: argmax one-hot
        indices = logits_3d.argmax(dim=-1)  # [B, 32]
        z_onehot = F.one_hot(indices, self.num_classes).float()  # [B, 32, 32]
        z = z_onehot.view(B, -1)  # [B, 1024]

        hz = torch.cat([h, z], dim=-1)  # [B, H+1024]
        hz_ent = hz.unsqueeze(1).expand(-1, N, -1)  # [B, N, H+1024]
        decode_input = torch.cat([hz_ent, registry], dim=-1)  # [B, N, H+1024+D_r]

        delta_pos_raw = self.pos_decoder(decode_input)         # [B, N, 2]
        alive_logit = self.alive_decoder(decode_input).squeeze(-1)  # [B, N]
        pred_gameplay_raw = self.gameplay_decoder(decode_input)  # [B, N, G]
        pred_terminal = self.terminal_head(hz).squeeze(-1)   # [B]

        # Apply symexp to undo the symlog training target transform
        delta_pos = symexp(delta_pos_raw)
        pred_gameplay = symexp(pred_gameplay_raw)

        next_pos = state[:, :, :2] + delta_pos
        next_pos = next_pos.clamp(0.0, 1.0)
        next_alive = (alive_logit > 0).float()

        next_state = torch.cat([
            next_pos,
            next_alive.unsqueeze(-1),
            pred_gameplay,
        ], dim=-1)  # [B, N, 3 + G]

        return next_state, pred_terminal, h


def dreamer_loss_fn(model: DreamerWorldModel, batch: dict) -> dict[str, torch.Tensor]:
    """Loss function adapter for BaselineTrainer."""
    return model(batch)


def build_model(config: dict, train_tensors: dict, device: str = "cuda") -> DreamerWorldModel:
    """Reconstruct a DreamerWorldModel from config and tensor dimensions.

    Used by score.py for independent checkpoint evaluation.
    """
    registry_dim = train_tensors["registry"].shape[-1]
    state_dim = train_tensors["states"].shape[-1]
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]
    model = DreamerWorldModel(
        registry_dim=registry_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_categoricals=config.get("num_categoricals", 32),
        num_classes=config.get("num_classes", 32),
        num_gameplay_fields=config.get("num_gameplay_fields", 14),
        max_entities=config.get("max_entities", 64),
        free_nats=config.get("free_nats", 1.0),
        vel_consistency_weight=config.get("vel_consistency_weight", 0.1),
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    model = DreamerWorldModel(
        registry_dim=registry_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
    )

    trainer = BaselineTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=dreamer_loss_fn,
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
