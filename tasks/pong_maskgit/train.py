# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""MaskGIT Baseline -- Masked Generative Transformer for ECS World Modeling.

Two-part architecture with W=16 causal temporal context:
  Part 1: TemporalContextEncoder (shared with D3PM) -- block-causal attention
          over (W+1) context frames of clean entity tokens.
  Part 2: CrossAttentionBlock decoder -- masked target tokens attend to context,
          bidirectional self-attention within each target frame.

Training: cosine mask schedule, per-entity masking, CE loss on masked mutables.
Inference: iterative parallel decoding (8 steps) with confidence-based unmasking.
Predicts position, alive, terminal, and per-game gameplay fields.

Based on Chang et al. (MaskGIT), adapted for entity-state prediction.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR.parent / "lib"))        # from templates/
sys.path.insert(0, str(TASK_DIR.parent.parent / "lib"))  # from tasks/game_model/

from tokenizer import EntityTokenizer
from temporal import TemporalContextEncoder, CrossAttentionBlock, make_cross_attn_mask, make_block_causal_mask
from trainer import BaselineTrainer
from loader import InMemoryLoader


class MaskGITWorldModel(nn.Module):
    """Masked Generative World Model with temporal context and iterative decoding.

    Architecture overview:
      1. Build token representations: tok_emb || registry_emb -> input_proj + temporal_pos + action
      2. Context encoder: (W+1) frames of clean tokens, block-causal attention
      3. Masked prediction: sample cosine mask ratio, mask target entities,
         decode via cross-attention to context + bidirectional self-attention
      4. Loss: CE on masked mutable entities only
    """

    def __init__(
        self,
        tokenizer: EntityTokenizer,
        registry_dim: int = 34,
        action_dim: int = 12,
        global_dim: int = 15,
        hidden_dim: int = 256,
        num_heads: int = 8,
        context_layers: int = 3,
        max_context_frames: int = 9,  # W+1 = 9 for W=8
        dropout: float = 0.1,
        vel_consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.max_context_frames = max_context_frames
        self.vel_consistency_weight = vel_consistency_weight
        vocab_sizes = tokenizer.vocab_sizes  # [K_x, K_y, 2, ...gameplay]
        n_fields = len(vocab_sizes)

        # --- Token embeddings (per field, +1 for MASK token) ---
        embed_per_field = max(32, hidden_dim // n_fields)
        self.embed_per_field = embed_per_field
        self.tok_embeddings = nn.ModuleList([
            nn.Embedding(vs + 1, embed_per_field)  # MASK = vs
            for vs in vocab_sizes
        ])
        self.mask_token_ids = [vs for vs in vocab_sizes]

        # --- Registry encoder ---
        self.registry_encoder = nn.Linear(registry_dim, hidden_dim)

        # --- Input projection: concat(tok_emb, registry_emb) -> hidden_dim ---
        tok_dim = embed_per_field * n_fields
        self.input_proj = nn.Linear(tok_dim + hidden_dim, hidden_dim)

        # --- Temporal position embedding (one per frame slot) ---
        self.temporal_pos_emb = nn.Embedding(max_context_frames, hidden_dim)

        # --- Action + globals encoder ---
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Mask ratio conditioning (replaces diffusion time embedding) ---
        self.mask_ratio_embed = nn.Linear(1, hidden_dim)

        # --- Part 1: Shared temporal context encoder ---
        self.context_encoder = TemporalContextEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=context_layers,
            dropout=dropout,
        )

        # --- Part 2: Masked prediction decoder (1 layer) ---
        self.decoder = CrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # --- Output heads (per field, no MASK in output vocab) ---
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vs) for vs in vocab_sizes
        ])

        # --- Terminal head: global per-frame prediction from pooled decoded ---
        self.terminal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token fields and concatenate.

        Args:
            tokens: [..., K] long -- per-entity token ids (may include MASK ids).

        Returns:
            [..., tok_dim] float -- concatenated per-field embeddings.
        """
        parts = [emb(tokens[..., i]) for i, emb in enumerate(self.tok_embeddings)]
        return torch.cat(parts, dim=-1)

    def _build_entity_token(
        self,
        tokens: torch.Tensor,
        registry: torch.Tensor,
        frame_idx: int,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Construct token representation for one frame.

        token = input_proj(tok_emb || registry_emb) + temporal_pos_emb + action_emb

        Args:
            tokens: [B, N, K] long -- entity tokens for one frame.
            registry: [B, N, H] -- pre-encoded registry.
            frame_idx: temporal position index.
            action_emb: [B, H] -- pre-encoded action+globals.

        Returns:
            [B, N, H] -- entity token representations.
        """
        tok_emb = self._embed_tokens(tokens)                              # [B, N, tok_dim]
        x = self.input_proj(torch.cat([tok_emb, registry], dim=-1))       # [B, N, H]
        x = x + self.temporal_pos_emb.weight[frame_idx]                      # + [H] broadcast
        x = x + action_emb.unsqueeze(1)                                   # + [B, 1, H]
        return x

    def _cosine_mask_ratio(self, device: torch.device) -> float:
        """Sample mask ratio from cosine schedule, minimum 10%."""
        u = torch.rand(1, device=device).item()
        ratio = math.cos(math.pi / 2.0 * u)
        return max(0.1, ratio)

    def _apply_entity_mask(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-entity masking: all K fields of an entity are masked together.

        Args:
            tokens: [B, N, K] long -- clean target tokens.
            mask_ratio: fraction of entities to mask.

        Returns:
            masked_tokens: [B, N, K] long -- with MASK ids inserted.
            entity_masked: [B, N] bool -- True for masked entities.
        """
        B, N, K = tokens.shape
        device = tokens.device

        entity_masked = torch.rand(B, N, device=device) < mask_ratio  # [B, N]
        masked = tokens.clone()
        for i, mask_id in enumerate(self.mask_token_ids):
            masked[:, :, i] = torch.where(entity_masked, mask_id, tokens[:, :, i])

        return masked, entity_masked

    def _build_context_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        globals_: torch.Tensor,
        registry: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build context token sequence for (W+1) frames and encode with temporal encoder.

        The context window contains frames [s_0, s_1, ..., s_W] where frame t
        is conditioned on action a_{t-1}. Frame 0 gets a zero action embedding.

        Args:
            states: [B, W+1, N, D_s] -- state observations for W+1 frames.
            actions: [B, W, D_a] -- actions for W transitions.
            globals_: [B, W, D_g] -- global features for W transitions.
            registry: [B, N, H] -- pre-encoded registry (shared across frames).
            entity_mask: [B, N] bool.

        Returns:
            context: [B, (W+1)*N, H] -- encoded context representations.
        """
        B = states.shape[0]
        W_plus_1 = states.shape[1]
        N = states.shape[2]
        device = states.device

        # Pre-encode all actions. Frame 0 gets zero action.
        zero_action = torch.zeros(B, self.hidden_dim, device=device)
        action_embs = [zero_action]  # frame 0
        for t in range(W_plus_1 - 1):
            act_emb = self.action_encoder(
                torch.cat([actions[:, t], globals_[:, t]], dim=-1)
            )  # [B, H]
            action_embs.append(act_emb)

        # Build per-frame token representations
        frame_tokens = []
        for f in range(W_plus_1):
            tok = self.tokenizer.tokenize(states[:, f])  # [B, N, 3]
            h = self._build_entity_token(tok, registry, f, action_embs[f])  # [B, N, H]
            frame_tokens.append(h)

        # Concatenate into [B, (W+1)*N, H] and encode
        ctx_seq = torch.cat(frame_tokens, dim=1)  # [B, (W+1)*N, H]
        ctx_out = self.context_encoder(ctx_seq, entity_mask, W_plus_1, N)
        return ctx_out

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Expected batch keys:
            registry:      [B, N, D_r]
            states:        [B, W, N, D_s]
            target_states: [B, W, N, D_s]
            actions:       [B, W, D_a]
            globals:       [B, W, D_g]
            mutable_mask:  [B, N]
            entity_mask:   [B, N]
            gameplay_mask: [B, N, G]
            terminals:     [B, W]
        """
        registry_raw = batch["registry"]         # [B, N, D_r]
        states = batch["states"]                 # [B, W, N, D_s]
        targets = batch["target_states"]         # [B, W, N, D_s]
        actions = batch["actions"]               # [B, W, D_a]
        globals_ = batch["globals"]              # [B, W, D_g]
        mutable = batch["mutable_mask"]          # [B, N]
        entity_mask = batch["entity_mask"]       # [B, N]
        gameplay_mask = batch["gameplay_mask"]    # [B, N, G]
        terminal_targets = batch["terminals"]    # [B, W]

        B, W, N, D_s = states.shape
        device = states.device
        n_gp = self.tokenizer.num_gameplay_tokens

        # Pre-encode registry
        registry = self.registry_encoder(registry_raw)

        # Build (W+1) context frames
        context_states = torch.cat([
            states[:, 0:1], targets,
        ], dim=1)

        ctx_out = self._build_context_sequence(
            context_states, actions, globals_, registry, entity_mask,
        )

        # Pre-encode actions
        action_embs = []
        for t in range(W):
            act_emb = self.action_encoder(
                torch.cat([actions[:, t], globals_[:, t]], dim=-1)
            )
            action_embs.append(act_emb)

        # Map gameplay_mask to token-level mask for active fields
        gp_token_mask = None
        if n_gp > 0:
            active_indices = [s.index for s in self.tokenizer.config.gameplay_specs]
            gp_token_mask = gameplay_mask[:, :, active_indices]  # [B, N, n_gp]

        total_token_loss = 0.0
        total_terminal_loss = 0.0
        total_vel_consist = 0.0
        n_steps = 0

        for t in range(W):
            target_tok = self.tokenizer.tokenize(targets[:, t])  # [B, N, K]

            mask_ratio = self._cosine_mask_ratio(device)
            masked_target, entity_masked = self._apply_entity_mask(target_tok, mask_ratio)

            target_h = self._build_entity_token(
                masked_target, registry, t + 1, action_embs[t],
            )

            mr_emb = self.mask_ratio_embed(
                torch.tensor([[mask_ratio]], device=device, dtype=torch.float32).expand(B, 1)
            )
            target_h = target_h + mr_emb.unsqueeze(1)

            ctx_slice = ctx_out[:, : (t + 1) * N]
            ctx_ent_valid = entity_mask.repeat(1, t + 1)
            cross_mask = entity_mask.unsqueeze(2) & ctx_ent_valid.unsqueeze(1)
            self_mask = entity_mask.unsqueeze(2) & entity_mask.unsqueeze(1)

            decoded = self.decoder(
                queries=target_h,
                context=ctx_slice,
                cross_mask=cross_mask,
                self_mask=self_mask,
            )
            decoded = self.norm_out(decoded)

            # CE loss on masked mutable entities
            pred_mask = entity_masked & mutable & entity_mask  # [B, N]
            pm = pred_mask.float()
            n_pred = pm.sum().clamp(min=1)

            frame_loss = torch.tensor(0.0, device=device)
            logits_list = []
            for i, (head, vs) in enumerate(zip(self.output_heads, self.tokenizer.vocab_sizes)):
                logits = head(decoded)  # [B, N, V_i]
                logits_list.append(logits)
                ce = F.cross_entropy(
                    logits.reshape(-1, vs),
                    target_tok[:, :, i].reshape(-1),
                    reduction="none",
                ).reshape(B, N)

                if i < 3:
                    frame_loss = frame_loss + (ce * pm).sum() / n_pred
                else:
                    gp_idx = i - 3
                    field_mask = pm * gp_token_mask[:, :, gp_idx].float()
                    n_field = field_mask.sum().clamp(min=1)
                    frame_loss = frame_loss + (ce * field_mask).sum() / n_field

            total_token_loss += frame_loss

            # Terminal loss (pool decoded -> global prediction)
            pool = (decoded * entity_mask.unsqueeze(-1).float()).sum(1) / \
                   entity_mask.float().sum(1, keepdim=True).clamp(min=1)
            term_logit = self.terminal_head(pool).squeeze(-1)
            total_terminal_loss += F.binary_cross_entropy_with_logits(
                term_logit, terminal_targets[:, t], reduction="mean"
            )

            # Velocity consistency (differentiable via soft detokenization)
            if D_s >= 5 and self.vel_consistency_weight > 0:
                mutable_ent_mask = mutable & entity_mask
                exp_pos = self.tokenizer.soft_detokenize_pos(
                    logits_list[0], logits_list[1]
                )
                curr_pos = states[:, t, :, :2]
                pred_delta = exp_pos - curr_pos
                input_vel = states[:, t, :, 3:5]
                vel_active = (input_vel.abs().sum(-1) > 1e-6) & mutable_ent_mask
                n_vel = vel_active.float().sum().clamp(min=1)
                vel_err = ((pred_delta - input_vel) ** 2 * vel_active.float().unsqueeze(-1)).sum() / n_vel
                total_vel_consist += self.vel_consistency_weight * vel_err

            n_steps += 1

        n_steps = max(n_steps, 1)
        loss = (total_token_loss + total_terminal_loss + total_vel_consist) / n_steps
        return {
            "loss": loss,
            "token_loss": (total_token_loss / n_steps).detach(),
            "terminal_loss": (total_terminal_loss / n_steps).detach(),
            "vel_consistency": (total_vel_consist / n_steps).detach() if isinstance(total_vel_consist, torch.Tensor) else torch.tensor(0.0),
        }

    # ------------------------------------------------------------------
    # Inference: iterative parallel decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        context_out: torch.Tensor,
        registry: torch.Tensor,
        action_emb: torch.Tensor,
        entity_mask: torch.Tensor,
        frame_idx: int,
        num_context_frames: int,
        num_iterations: int = 8,
    ) -> torch.Tensor:
        """Iterative parallel decoding for one target frame.

        Start with all target entities masked. At each iteration:
          1. Predict logits for all entities
          2. Commit the most confident entities
          3. Re-mask the rest

        Args:
            context_out: [B, num_context_frames*N, H] -- encoded context.
            registry: [B, N, H] -- pre-encoded registry.
            action_emb: [B, H] -- action embedding for this transition.
            entity_mask: [B, N] bool.
            frame_idx: temporal position of the target frame.
            num_context_frames: how many context frames are available.
            num_iterations: number of decoding iterations (default 8).

        Returns:
            tokens: [B, N, K] long -- predicted target tokens.
        """
        B, N = entity_mask.shape
        device = entity_mask.device

        # Start fully masked
        tokens = torch.stack([
            torch.full((B, N), mid, device=device, dtype=torch.long)
            for mid in self.mask_token_ids
        ], dim=-1)  # [B, N, 3]

        still_masked = torch.ones(B, N, dtype=torch.bool, device=device)

        # Precompute masks
        S_ctx = num_context_frames * N
        ctx_slice = context_out[:, :S_ctx]  # [B, S_ctx, H]
        ctx_ent_valid = entity_mask.repeat(1, num_context_frames)  # [B, S_ctx]
        cross_mask = entity_mask.unsqueeze(2) & ctx_ent_valid.unsqueeze(1)  # [B, N, S_ctx]
        self_mask = entity_mask.unsqueeze(2) & entity_mask.unsqueeze(1)     # [B, N, N]

        pred_tokens = tokens.clone()

        # Precompute mask-ratio embeddings for all iterations
        mask_ratios = [1.0 - (it + 1) / num_iterations for it in range(num_iterations)]
        mr_tensor = torch.tensor(mask_ratios, device=device, dtype=torch.float32).unsqueeze(1)  # [I, 1]
        mr_embs = self.mask_ratio_embed(mr_tensor)  # [I, H]

        for iteration in range(num_iterations):
            mask_ratio = mask_ratios[iteration]

            # Build target representations from current (partially masked) tokens
            target_h = self._build_entity_token(
                tokens, registry, frame_idx, action_emb,
            )  # [B, N, H]

            target_h = target_h + mr_embs[iteration].unsqueeze(0).unsqueeze(0)  # [1, 1, H] broadcast

            # Decode
            decoded = self.decoder(
                queries=target_h,
                context=ctx_slice,
                cross_mask=cross_mask,
                self_mask=self_mask,
            )
            decoded = self.norm_out(decoded)

            # Compute predictions and confidences
            all_logits = [head(decoded) for head in self.output_heads]
            confidences = torch.zeros(B, N, device=device)
            pred_tokens = tokens.clone()

            for i, logits in enumerate(all_logits):
                probs = F.softmax(logits, dim=-1)
                max_prob, max_idx = probs.max(dim=-1)  # [B, N]
                confidences += max_prob
                pred_tokens[:, :, i] = max_idx

            confidences /= len(all_logits)  # mean confidence across 3 fields

            # Determine how many to unmask this iteration
            n_still = still_masked.sum(dim=-1)  # [B]
            n_to_unmask = (n_still.float() * (1.0 - mask_ratio / max(mask_ratio + 1e-8, 1e-8))).clamp(min=1).long()
            # Simpler: unmask a fraction that brings us to mask_ratio remaining
            n_to_keep_masked = max(0, int(mask_ratio * N))
            n_to_unmask_global = max(1, N - n_to_keep_masked)

            # Mask out already-unmasked entities so they don't compete
            confidences[~still_masked] = -1.0
            confidences[~entity_mask] = -1.0

            _, topk_idx = confidences.topk(
                min(n_to_unmask_global, N), dim=-1,
            )  # [B, k]

            # Commit top-k predictions (vectorized)
            unmask = torch.zeros_like(still_masked)
            unmask.scatter_(1, topk_idx, True)
            unmask = unmask & still_masked
            tokens[unmask] = pred_tokens[unmask]
            still_masked[unmask] = False

        # Final pass: commit any remaining masked entities
        if still_masked.any():
            # One last prediction pass with mask_ratio=0
            target_h = self._build_entity_token(tokens, registry, frame_idx, action_emb)
            mr_zero = self.mask_ratio_embed(torch.zeros(1, 1, device=device))  # [1, H]
            target_h = target_h + mr_zero.unsqueeze(0)

            decoded = self.decoder(
                queries=target_h, context=ctx_slice,
                cross_mask=cross_mask, self_mask=self_mask,
            )
            decoded = self.norm_out(decoded)

            for i, head in enumerate(self.output_heads):
                logits = head(decoded)
                pred = logits.argmax(dim=-1)
                tokens[:, :, i] = torch.where(still_masked, pred, tokens[:, :, i])

        return tokens

    @torch.no_grad()
    def predict_step(
        self,
        registry: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step prediction for evaluation (no temporal history).

        Returns:
            next_state: [B, N, 3 + K] -- pos, alive, gameplay (continuous)
            terminal:   [B] -- terminal logit
        """
        B, N, _ = state.shape
        device = state.device

        reg_enc = self.registry_encoder(registry)

        curr_tok = self.tokenizer.tokenize(state)
        zero_action = torch.zeros(B, self.hidden_dim, device=device)
        ctx_h = self._build_entity_token(curr_tok, reg_enc, 0, zero_action)

        ctx_out = self.context_encoder(ctx_h, entity_mask, 1, N)

        act_emb = self.action_encoder(torch.cat([action, globals_], dim=-1))

        pred_tok = self.sample(
            context_out=ctx_out,
            registry=reg_enc,
            action_emb=act_emb,
            entity_mask=entity_mask,
            frame_idx=1,
            num_context_frames=1,
            num_iterations=4,
        )

        # Terminal prediction from context pool
        pool = (ctx_out * entity_mask.unsqueeze(-1).float()).sum(1) / \
               entity_mask.float().sum(1, keepdim=True).clamp(min=1)
        terminal = self.terminal_head(pool).squeeze(-1)

        return self.tokenizer.detokenize(pred_tok), terminal

    @torch.no_grad()
    def predict_rollout(
        self,
        registry: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
        num_iterations: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step prediction with full temporal context.

        Args:
            registry: [B, N, D_r]
            states: [B, W, N, D_s] -- history frames
            actions: [B, W+H, D_a] -- actions for history + future horizon H
            globals_: [B, W+H, D_g]
            entity_mask: [B, N]
            num_iterations: MaskGIT decoding iterations per frame

        Returns:
            predictions: [B, H, N, 3+K] float
            terminals:   [B, H] float (logits)
        """
        B, W, N, D_s = states.shape
        total_actions = actions.shape[1]
        H = total_actions - W
        device = states.device
        out_dim = self.tokenizer.tokens_per_entity  # 3 + K

        if H <= 0:
            return (torch.empty(B, 0, N, out_dim, device=device),
                    torch.empty(B, 0, device=device))

        reg_enc = self.registry_encoder(registry)

        # Build initial W-frame context using _build_context_sequence.
        # _build_context_sequence expects states [B, F, N, D_s], actions [B, F-1, D_a],
        # globals_ [B, F-1, D_g]. Frame 0 gets zero action; frame f>0 gets action[f-1].
        ctx_actions = actions[:, :W - 1] if W > 1 else actions[:, :0]
        ctx_globals = globals_[:, :W - 1] if W > 1 else globals_[:, :0]
        context_h = self._build_context_sequence(
            states, ctx_actions, ctx_globals, reg_enc, entity_mask,
        )
        num_ctx_frames = W

        predictions = []
        terminals = []

        for step in range(H):
            abs_frame = W + step
            act_idx = W + step - 1
            frame_idx = min(abs_frame, self.max_context_frames - 1)

            act_emb = self.action_encoder(
                torch.cat([actions[:, act_idx], globals_[:, act_idx]], dim=-1)
            )

            pred_tok = self.sample(
                context_out=context_h,
                registry=reg_enc,
                action_emb=act_emb,
                entity_mask=entity_mask,
                frame_idx=frame_idx,
                num_context_frames=num_ctx_frames,
                num_iterations=num_iterations,
            )

            pred_state = self.tokenizer.detokenize(pred_tok)
            predictions.append(pred_state)

            # Terminal from pooled context
            S = num_ctx_frames * N
            em_tiled = entity_mask.repeat(1, num_ctx_frames)
            pool = (context_h[:, :S] * em_tiled.unsqueeze(-1).float()).sum(1) / \
                   em_tiled.float().sum(1, keepdim=True).clamp(min=1)
            terminals.append(self.terminal_head(pool).squeeze(-1))

            # Build new context frame token (no re-encoding through context_encoder)
            pad_state = pred_state
            if D_s > out_dim:
                pad = torch.zeros(B, N, D_s - out_dim, device=device)
                pad_state = torch.cat([pred_state, pad], dim=-1)

            new_tok = self.tokenizer.tokenize(pad_state.unsqueeze(1))[:, 0]
            next_act_idx = W + step
            if next_act_idx < total_actions:
                next_act_emb = self.action_encoder(
                    torch.cat([actions[:, next_act_idx], globals_[:, next_act_idx]], dim=-1)
                )
            else:
                next_act_emb = torch.zeros(B, self.hidden_dim, device=device)

            new_frame_h = self._build_entity_token(
                new_tok, reg_enc, min(abs_frame, self.max_context_frames - 1), next_act_emb,
            )

            context_h = torch.cat([context_h, new_frame_h], dim=1)
            num_ctx_frames += 1
            if num_ctx_frames > W:
                context_h = context_h[:, N:]
                num_ctx_frames -= 1

        return torch.stack(predictions, dim=1), torch.stack(terminals, dim=1)


# ----------------------------------------------------------------------
# Loss adapter
# ----------------------------------------------------------------------

def maskgit_loss_fn(model: MaskGITWorldModel, batch: dict) -> dict[str, torch.Tensor]:
    """Loss adapter for the training harness."""
    return model(batch)


def build_model(config: dict, train_tensors: dict, device: str = "cuda") -> MaskGITWorldModel:
    """Reconstruct a MaskGITWorldModel from config and tensor dimensions.

    Used by score.py for independent checkpoint evaluation.
    """
    from tokenizer import EntityTokenizer

    registry_dim = train_tensors["registry"].shape[-1]
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]
    tokenizer = EntityTokenizer(game_id=config["game_id"])
    model = MaskGITWorldModel(
        tokenizer=tokenizer,
        registry_dim=registry_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 8),
        context_layers=config.get("context_layers", 3),
        max_context_frames=config.get("max_context_frames", 9),
        dropout=config.get("dropout", 0.1),
        vel_consistency_weight=config.get("vel_consistency_weight", 0.1),
    )
    return model.to(device)


def main():
    config = json.loads((TASK_DIR / "config.json").read_text())
    if os.environ.get("MAX_TRAIN_SECONDS"):
        config["max_train_seconds"] = int(os.environ["MAX_TRAIN_SECONDS"])
    game_id = config["game_id"]

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
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]
    tokenizer = EntityTokenizer(game_id=game_id)

    model = MaskGITWorldModel(
        tokenizer=tokenizer,
        registry_dim=registry_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
    )

    trainer = BaselineTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=maskgit_loss_fn,
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
