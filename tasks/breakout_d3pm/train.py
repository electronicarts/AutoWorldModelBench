# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""D3PM Baseline -- Discrete Denoising Diffusion for ECS World Modeling.

Predicts next-frame entity states as discrete tokens using D3PM (Austin et al.).
Uses a two-part architecture:
  1. TemporalContextEncoder: block-causal attention over W+1 frames of history
  2. CrossAttentionBlock: denoises target tokens conditioned on temporal context

Supports W=16 causal temporal attention so the model can leverage history frames.
Predicts position, alive, terminal, and per-game gameplay fields.

Single-file model for the AutoWorldBench agent harness.
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
from temporal import TemporalContextEncoder, CrossAttentionBlock, make_block_causal_mask, make_cross_attn_mask
from trainer import BaselineTrainer
from loader import InMemoryLoader


def load_cached_tensors(path: Path, use_mmap: bool = True) -> dict:
    """Load cached tensor dicts with optional memory mapping.

    The cached datasets are extremely large for Breakout. `torch.load(..., mmap=True)`
    keeps storages on-disk and faults pages in on demand, which avoids spending most
    of the run budget on eagerly materializing hundreds of GB before training starts.
    """
    load_kwargs = {"weights_only": False}
    if use_mmap:
        load_kwargs["mmap"] = True
    return torch.load(path, **load_kwargs)


class D3PMWorldModel(nn.Module):
    """Discrete Denoising Diffusion World Model with temporal context.

    Two-part architecture:
      Part 1 -- Temporal Context Encoder:
        Block-causal Transformer over (W+1) frames of entity tokens.
        Within-frame bidirectional, cross-frame causal.

      Part 2 -- Denoising Decoder:
        CrossAttentionBlock that denoises target tokens by attending to
        the context encoder output (causally masked per target frame).

    Training:
      For each of the W target frames, sample a diffusion timestep, corrupt
      target tokens with uniform noise, and decode via cross-attention from
      the appropriate causal context. Loss is CE on mutable entities only.

    Inference:
      Autoregressive over frames: encode context, reverse-diffuse to predict
      the next frame, append prediction to context, repeat.
    """

    def __init__(
        self,
        tokenizer: EntityTokenizer,
        registry_dim: int = 34,
        action_dim: int = 12,
        global_dim: int = 15,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        diffusion_steps: int = 100,
        max_frames: int = 32,
        vel_consistency_weight: float = 0.1,
        rollout_denoise_steps: int = 10,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.T = diffusion_steps
        self.max_frames = max_frames
        self.vel_consistency_weight = vel_consistency_weight
        self.rollout_denoise_steps = rollout_denoise_steps
        vocab_sizes = tokenizer.vocab_sizes  # [K_x, K_y, 2, ...gameplay]
        n_fields = len(vocab_sizes)

        # -- Token embeddings: separate per field, concat to tok_dim --
        embed_per_field = max(32, hidden_dim // n_fields)
        self.embed_per_field = embed_per_field
        self.tok_dim = embed_per_field * n_fields
        self.tok_embeddings = nn.ModuleList([
            nn.Embedding(vs, embed_per_field) for vs in vocab_sizes
        ])

        # -- Registry projection --
        self.registry_proj = nn.Linear(registry_dim, hidden_dim)

        # -- Input projection: tok_emb || registry_emb -> H --
        self.input_proj = nn.Linear(self.tok_dim + hidden_dim, hidden_dim)

        # -- Temporal position embedding (frame index) --
        self.temporal_pos_emb = nn.Embedding(max_frames, hidden_dim)

        # -- Action + globals projection (per-frame, broadcast to entities) --
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # -- Role embedding: 0=context, 1=target --
        self.role_emb = nn.Embedding(2, hidden_dim)

        # -- Diffusion time embedding --
        self.time_emb = nn.Embedding(diffusion_steps + 1, hidden_dim)

        # -- Part 1: Temporal Context Encoder --
        self.context_encoder = TemporalContextEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        # -- Part 2: Denoising Decoder (single CrossAttentionBlock) --
        self.decoder = CrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # -- Output heads: per-field logits --
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vs) for vs in vocab_sizes
        ])

        # -- Terminal head: global per-frame prediction from pooled decoded --
        self.terminal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # -- Noise schedule: linear beta --
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed discrete tokens to continuous space.

        Args:
            tokens: [..., K] long -- K = 3 + num_gameplay_tokens

        Returns:
            [..., tok_dim] float
        """
        parts = [emb(tokens[..., i]) for i, emb in enumerate(self.tok_embeddings)]
        return torch.cat(parts, dim=-1)

    def _build_entity_tokens(
        self,
        states: torch.Tensor,
        registry: torch.Tensor,
        actions: torch.Tensor,
        globals_: torch.Tensor,
        frame_indices: torch.Tensor,
        role: int,
        t_diff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build entity-level tokens for a sequence of frames.

        Args:
            states: [B, F, N, D_s] -- raw entity states for F frames
            registry: [B, N, D_r] -- static entity registry (shared across frames)
            actions: [B, F, D_a] -- per-frame actions
            globals_: [B, F, D_g] -- per-frame globals
            frame_indices: [F] long -- absolute frame index for temporal pos emb
            role: 0 for context, 1 for target
            t_diff: [B] long -- diffusion timestep (only for targets)

        Returns:
            [B, F*N, H] -- entity tokens laid out as [frame_0_ents, ..., frame_{F-1}_ents]
        """
        B, F, N, _ = states.shape
        device = states.device

        # Tokenize all frames at once: [B, F, N, K]
        tokens = self.tokenizer.tokenize(states)
        # Embed: [B, F, N, tok_dim]
        tok_emb = self._embed_tokens(tokens)

        # Registry: [B, N, H] -> [B, 1, N, H] -> [B, F, N, H]
        reg_emb = self.registry_proj(registry).unsqueeze(1).expand(-1, F, -1, -1)

        # Input projection: concat tok_emb and reg_emb along feature dim
        # [B, F, N, tok_dim + H] -> [B, F, N, H]
        x = self.input_proj(torch.cat([tok_emb, reg_emb], dim=-1))

        # Temporal position embedding: [F, H] -> [1, F, 1, H]
        temp_pos = self.temporal_pos_emb(frame_indices).unsqueeze(0).unsqueeze(2)
        x = x + temp_pos

        # Action + globals embedding: [B, F, D_a+D_g] -> [B, F, H] -> [B, F, 1, H]
        act_emb = self.action_proj(
            torch.cat([actions, globals_], dim=-1)
        ).unsqueeze(2)
        x = x + act_emb

        # Role embedding (context=0, target=1)
        x = x + self.role_emb(torch.tensor(role, device=device))

        # Diffusion time embedding (targets only)
        if t_diff is not None:
            # t_diff: [B] -> [B, H] -> [B, 1, 1, H]
            time_e = self.time_emb(t_diff).unsqueeze(1).unsqueeze(2)
            x = x + time_e

        # Reshape to sequence: [B, F*N, H]
        x = x.reshape(B, F * N, self.hidden_dim)
        return x

    def _corrupt_tokens(
        self, clean_tokens: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Apply forward noise: with probability (1 - alpha_bar[t]), replace with uniform.

        Args:
            clean_tokens: [B, N, 3] long
            t: [B] long -- diffusion timestep

        Returns:
            noisy_tokens: [B, N, 3] long
        """
        B, N, K = clean_tokens.shape
        device = clean_tokens.device

        keep_prob = self.alpha_bar[t]  # [B]
        keep_mask = torch.rand(B, N, K, device=device) < keep_prob[:, None, None]

        vocab_sizes = self.tokenizer.vocab_sizes
        random_tokens = torch.stack([
            torch.randint(0, vs, (B, N), device=device) for vs in vocab_sizes
        ], dim=-1)

        return torch.where(keep_mask, clean_tokens, random_tokens)

    def _encode_context(
        self,
        states: torch.Tensor,
        registry: torch.Tensor,
        actions: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a sliding context window with the same path used in training."""
        _, num_frames, num_entities, _ = states.shape
        context_tokens = self._build_entity_tokens(
            states=states,
            registry=registry,
            actions=actions,
            globals_=globals_,
            frame_indices=frame_indices,
            role=0,
        )
        return self.context_encoder(
            tokens=context_tokens,
            entity_mask=entity_mask,
            num_frames=num_frames,
            num_entities=num_entities,
        )

    def _decode_logits(
        self,
        target_h: torch.Tensor,
        context_h: torch.Tensor,
        cross_mask: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Run denoising decoder and produce per-field logits.

        Args:
            target_h: [B, S_tgt, H] -- noised target entity tokens
            context_h: [B, S_ctx, H] -- context encoder output
            cross_mask: [B, S_tgt, S_ctx] bool -- True = can attend
            self_mask: [B, S_tgt, S_tgt] bool -- True = can attend

        Returns:
            list of [B, S_tgt, V_i] logits per token field
        """
        x = self.decoder(
            queries=target_h,
            context=context_h,
            cross_mask=cross_mask,
            self_mask=self_mask,
        )
        x = self.norm_out(x)
        return [head(x) for head in self.output_heads]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass with temporal context.

        Batch layout:
            states:        [B, W, N, D_s]
            target_states: [B, W, N, D_s]
            actions:       [B, W, D_a]
            globals:       [B, W, D_g]
            registry:      [B, N, D_r]
            mutable_mask:  [B, N]
            entity_mask:   [B, N]
            gameplay_mask: [B, N, G]
            terminals:     [B, W]
        """
        registry = batch["registry"]           # [B, N, D_r]
        states = batch["states"]               # [B, W, N, D_s]
        targets = batch["target_states"]       # [B, W, N, D_s]
        actions = batch["actions"]             # [B, W, D_a]
        globals_ = batch["globals"]            # [B, W, D_g]
        mutable = batch["mutable_mask"]        # [B, N]
        ent_mask = batch["entity_mask"]        # [B, N]
        gameplay_mask = batch["gameplay_mask"]  # [B, N, G]
        terminal_targets = batch["terminals"]  # [B, W]

        B, W, N, D_s = states.shape
        device = states.device
        n_gp = self.tokenizer.num_gameplay_tokens

        # ---- Step 1: Build W+1 context frames ----
        context_states = torch.cat([
            states, targets[:, W - 1:W],
        ], dim=1)
        context_actions = torch.cat([
            actions, actions[:, W - 1:W],
        ], dim=1)
        context_globals = torch.cat([
            globals_, globals_[:, W - 1:W],
        ], dim=1)

        # ---- Step 2: Encode context (single pass) ----
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

        # ---- Step 3: Build targets and compute loss ----
        pred_mask = mutable & ent_mask  # [B, N]
        pm = pred_mask.float()
        n_pred = pm.sum().clamp(min=1)

        # Map gameplay_mask [B, N, G] to token-level mask for active fields
        gp_token_mask = None
        if n_gp > 0:
            active_indices = [s.index for s in self.tokenizer.config.gameplay_specs]
            gp_token_mask = gameplay_mask[:, :, active_indices]  # [B, N, n_gp]

        total_token_loss = 0.0
        total_terminal_loss = 0.0
        total_vel_consist = 0.0

        for t in range(W):
            target_tok = self.tokenizer.tokenize(targets[:, t])  # [B, N, K]

            t_diff = torch.randint(0, self.T, (B,), device=device)
            noisy_tok = self._corrupt_tokens(target_tok, t_diff)

            noisy_emb = self._embed_tokens(noisy_tok)
            reg_emb = self.registry_proj(registry)
            target_h = self.input_proj(torch.cat([noisy_emb, reg_emb], dim=-1))

            target_h = target_h + self.temporal_pos_emb(
                torch.tensor(t + 1, device=device)
            )
            target_h = target_h + self.action_proj(
                torch.cat([actions[:, t], globals_[:, t]], dim=-1)
            ).unsqueeze(1)
            target_h = target_h + self.role_emb(torch.tensor(1, device=device))
            target_h = target_h + self.time_emb(t_diff).unsqueeze(1)

            ctx_slice = context_h[:, :(t + 1) * N, :]
            ctx_ent_valid = ent_mask.repeat(1, t + 1)
            self_mask = ent_mask.unsqueeze(1).expand(-1, N, -1)
            cross_mask = ctx_ent_valid.unsqueeze(1).expand(-1, N, -1)

            logits_list = self._decode_logits(
                target_h=target_h,
                context_h=ctx_slice,
                cross_mask=cross_mask,
                self_mask=self_mask,
            )

            # Token CE loss: pos/alive on mutable, gameplay masked per field
            frame_loss = torch.tensor(0.0, device=device)
            for i, (logits, vs) in enumerate(
                zip(logits_list, self.tokenizer.vocab_sizes)
            ):
                ce = F.cross_entropy(
                    logits.reshape(-1, vs),
                    target_tok[:, :, i].reshape(-1),
                    reduction="none",
                ).reshape(B, N)

                if i < 3:
                    # pos_x, pos_y, alive: mask by mutable & entity
                    frame_loss = frame_loss + (ce * pm).sum() / n_pred
                else:
                    # gameplay token: mask by mutable & entity & field-active
                    gp_idx = i - 3
                    field_mask = pm * gp_token_mask[:, :, gp_idx].float()
                    n_field = field_mask.sum().clamp(min=1)
                    frame_loss = frame_loss + (ce * field_mask).sum() / n_field

            total_token_loss = total_token_loss + frame_loss

            # Terminal loss (pool decoded entities -> global prediction)
            decoded_out = self.norm_out(self.decoder(
                queries=target_h, context=ctx_slice,
                cross_mask=cross_mask, self_mask=self_mask,
            ))
            pool = (decoded_out * ent_mask.unsqueeze(-1).float()).sum(1) / \
                   ent_mask.float().sum(1, keepdim=True).clamp(min=1)
            term_logit = self.terminal_head(pool).squeeze(-1)  # [B]
            total_terminal_loss = total_terminal_loss + F.binary_cross_entropy_with_logits(
                term_logit, terminal_targets[:, t], reduction="mean"
            )

            # Velocity consistency (differentiable via soft detokenization)
            if D_s >= 5 and self.vel_consistency_weight > 0:
                exp_pos = self.tokenizer.soft_detokenize_pos(
                    logits_list[0], logits_list[1]
                )  # [B, N, 2]
                curr_pos = states[:, t, :, :2]
                pred_delta = exp_pos - curr_pos
                input_vel = states[:, t, :, 3:5]
                vel_active = (input_vel.abs().sum(-1) > 1e-6) & pred_mask
                n_vel = vel_active.float().sum().clamp(min=1)
                vel_err = ((pred_delta - input_vel) ** 2 * vel_active.float().unsqueeze(-1)).sum() / n_vel
                total_vel_consist = total_vel_consist + self.vel_consistency_weight * vel_err

        loss = (total_token_loss + total_terminal_loss + total_vel_consist) / W
        return {
            "loss": loss,
            "token_loss": (total_token_loss / W).detach(),
            "terminal_loss": (total_terminal_loss / W).detach(),
            "vel_consistency": (total_vel_consist / W).detach() if isinstance(total_vel_consist, torch.Tensor) else torch.tensor(0.0),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        context_h: torch.Tensor,
        context_frames_used: int,
        registry: torch.Tensor,
        action: torch.Tensor,
        globals_: torch.Tensor,
        entity_mask: torch.Tensor,
        target_frame_idx: int,
        num_entities: int,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Reverse diffusion sampling conditioned on temporal context.

        Args:
            context_h: [B, F*N, H] -- encoded context from TemporalContextEncoder
            context_frames_used: how many context frames are in context_h
            registry: [B, N, D_r]
            action: [B, D_a] -- action for this transition
            globals_: [B, D_g] -- globals for this transition
            entity_mask: [B, N]
            target_frame_idx: absolute frame index of the frame being predicted
            num_entities: N
            num_steps: number of denoising steps (strided over T)

        Returns:
            predicted tokens [B, N, K] long (K = tokens_per_entity)
        """
        B = context_h.shape[0]
        N = num_entities
        device = context_h.device
        vocab_sizes = self.tokenizer.vocab_sizes

        if num_steps is None:
            num_steps = self.rollout_denoise_steps

        # Start from uniform noise
        tokens = torch.stack([
            torch.randint(0, vs, (B, N), device=device) for vs in vocab_sizes
        ], dim=-1)  # [B, N, K]

        # Stride through diffusion steps
        step_indices = torch.linspace(self.T - 1, 0, num_steps).long()

        # Slice context to frames 0..target_frame_idx-1
        ctx_tokens = context_frames_used * N
        ctx_slice = context_h[:, :ctx_tokens, :]
        ctx_ent_valid = entity_mask.repeat(1, context_frames_used)
        self_mask = entity_mask.unsqueeze(1).expand(-1, N, -1)
        cross_mask = ctx_ent_valid.unsqueeze(1).expand(-1, N, -1)

        for t_val in step_indices:
            t = torch.full((B,), t_val.item(), device=device, dtype=torch.long)

            # Build target embeddings from current noisy tokens
            noisy_emb = self._embed_tokens(tokens)
            reg_emb = self.registry_proj(registry)
            target_h = self.input_proj(torch.cat([noisy_emb, reg_emb], dim=-1))
            target_h = target_h + self.temporal_pos_emb(
                torch.tensor(target_frame_idx, device=device)
            )
            target_h = target_h + self.action_proj(
                torch.cat([action, globals_], dim=-1)
            ).unsqueeze(1)
            target_h = target_h + self.role_emb(torch.tensor(1, device=device))
            target_h = target_h + self.time_emb(t).unsqueeze(1)

            logits_list = self._decode_logits(
                target_h=target_h,
                context_h=ctx_slice,
                cross_mask=cross_mask,
                self_mask=self_mask,
            )

            # Greedy decode
            for i, logits in enumerate(logits_list):
                tokens[:, :, i] = logits.argmax(dim=-1)

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
        """Single-step prediction for evaluation (no history beyond current frame).

        Args:
            registry: [B, N, D_r]
            state: [B, N, D_s]
            action: [B, D_a]
            globals_: [B, D_g]
            entity_mask: [B, N]

        Returns:
            next_state: [B, N, 3 + K] -- pos, alive, gameplay (continuous)
            terminal:   [B] -- terminal logit
        """
        B, N, _ = state.shape
        device = state.device

        context_states = state.unsqueeze(1)
        context_actions = action.unsqueeze(1)
        context_globals = globals_.unsqueeze(1)
        frame_indices = torch.tensor([0], device=device)

        context_tokens = self._build_entity_tokens(
            states=context_states,
            registry=registry,
            actions=context_actions,
            globals_=context_globals,
            frame_indices=frame_indices,
            role=0,
        )

        context_h = self.context_encoder(
            tokens=context_tokens,
            entity_mask=entity_mask,
            num_frames=1,
            num_entities=N,
        )

        pred_tok = self.sample(
            context_h=context_h,
            context_frames_used=1,
            registry=registry,
            action=action,
            globals_=globals_,
            entity_mask=entity_mask,
            target_frame_idx=1,
            num_entities=N,
            num_steps=self.rollout_denoise_steps,
        )

        # Terminal prediction from context pool
        pool = (context_h * entity_mask.unsqueeze(-1).float()).sum(1) / \
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
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step autoregressive prediction with full temporal context.

        Args:
            registry: [B, N, D_r]
            states: [B, W, N, D_s] -- history frames
            actions: [B, W+H, D_a] -- actions for history + future horizon H
            globals_: [B, W+H, D_g]
            entity_mask: [B, N]
            num_steps: diffusion steps per frame

        Returns:
            predictions: [B, H, N, 3+K] float -- predicted states
            terminals:   [B, H] float -- terminal logits
        """
        B, W, N, D_s = states.shape
        total_actions = actions.shape[1]
        H = total_actions - W
        device = states.device
        out_dim = self.tokenizer.tokens_per_entity  # 3 + K

        if H <= 0:
            return (torch.empty(B, 0, N, out_dim, device=device),
                    torch.empty(B, 0, device=device))

        if num_steps is None:
            num_steps = self.rollout_denoise_steps

        predictions = []
        terminals_list = []
        window_states = states.clone()

        for step in range(H):
            abs_frame = W + step
            act_idx = W + step - 1
            frame_start = max(0, abs_frame - W)
            ctx_frame_indices = torch.arange(frame_start, frame_start + W, device=device)
            ctx_frame_indices = ctx_frame_indices.clamp(max=self.max_frames - 1)
            context_h = self._encode_context(
                states=window_states,
                registry=registry,
                actions=actions[:, step:step + W],
                globals_=globals_[:, step:step + W],
                entity_mask=entity_mask,
                frame_indices=ctx_frame_indices,
            )

            pred_tok = self.sample(
                context_h=context_h,
                context_frames_used=W,
                registry=registry,
                action=actions[:, act_idx],
                globals_=globals_[:, act_idx],
                entity_mask=entity_mask,
                target_frame_idx=min(abs_frame, self.max_frames - 1),
                num_entities=N,
                num_steps=num_steps,
            )

            pred_state = self.tokenizer.detokenize(pred_tok)  # [B, N, 3+K]
            predictions.append(pred_state)

            em_tiled = entity_mask.repeat(1, W)
            S = W * N
            pool = (context_h[:, :S] * em_tiled.unsqueeze(-1).float()).sum(1) / \
                   em_tiled.float().sum(1, keepdim=True).clamp(min=1)
            term = self.terminal_head(pool).squeeze(-1)
            terminals_list.append(term)

            # Pad detokenized state to D_s for context re-encoding
            new_state = pred_state.unsqueeze(1)  # [B, 1, N, 3+K]
            if D_s > out_dim:
                pad = torch.zeros(B, 1, N, D_s - out_dim, device=device)
                new_state = torch.cat([new_state, pad], dim=-1)
            window_states = torch.cat([window_states[:, 1:], new_state], dim=1)

        return torch.stack(predictions, dim=1), torch.stack(terminals_list, dim=1)


# ======================================================================
# Loss function adapter
# ======================================================================

def d3pm_loss_fn(model: D3PMWorldModel, batch: dict) -> dict[str, torch.Tensor]:
    """Adapter for the AutoWorldBench training harness."""
    return model(batch)


def build_model(config: dict, train_tensors: dict, device: str = "cuda") -> D3PMWorldModel:
    """Reconstruct a D3PMWorldModel from config and tensor dimensions.

    Used by score.py for independent checkpoint evaluation.
    """
    from tokenizer import EntityTokenizer

    registry_dim = train_tensors["registry"].shape[-1]
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]
    tokenizer = EntityTokenizer(game_id=config["game_id"])
    model = D3PMWorldModel(
        tokenizer=tokenizer,
        registry_dim=registry_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 8),
        num_encoder_layers=config.get("num_encoder_layers", 3),
        dropout=config.get("dropout", 0.1),
        diffusion_steps=config.get("diffusion_steps", 100),
        max_frames=config.get("max_frames", 32),
        vel_consistency_weight=config.get("vel_consistency_weight", 0.1),
        rollout_denoise_steps=config.get("rollout_denoise_steps", 10),
    )
    return model.to(device)


def main():
    config = json.loads((TASK_DIR / "config.json").read_text())
    if os.environ.get("MAX_TRAIN_SECONDS"):
        config["max_train_seconds"] = int(os.environ["MAX_TRAIN_SECONDS"])
    game_id = config["game_id"]

    cache_dir = Path(config["cache_dir"]) if "cache_dir" in config else TASK_DIR / "_cache"
    use_mmap = config.get("mmap_cache", True)
    train_tensors = load_cached_tensors(cache_dir / "train_tensors.pt", use_mmap=use_mmap)
    val_tensors = load_cached_tensors(cache_dir / "val_tensors.pt", use_mmap=use_mmap)
    eval_path = cache_dir / "val_eval_tensors.pt"
    eval_tensors = (
        load_cached_tensors(eval_path, use_mmap=use_mmap)
        if eval_path.exists()
        else None
    )

    batch_size = config.get("batch_size", 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = InMemoryLoader(train_tensors, batch_size, shuffle=True, device=device)
    val_loader = InMemoryLoader(val_tensors, batch_size, shuffle=False, device=device)

    registry_dim = train_tensors["registry"].shape[-1]
    action_dim = train_tensors["actions"].shape[-1]
    global_dim = train_tensors["globals"].shape[-1]
    tokenizer = EntityTokenizer(game_id=game_id)

    model = D3PMWorldModel(
        tokenizer=tokenizer,
        registry_dim=registry_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        hidden_dim=config.get("hidden_dim", 256),
        rollout_denoise_steps=config.get("rollout_denoise_steps", 10),
    )

    trainer = BaselineTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=d3pm_loss_fn,
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
        eval_batch_size=config.get("eval_batch_size", 32),
        context_len=config.get("window_size", 8),
    )

    result = trainer.train()
    print(f"Best val_loss: {result['best_val_loss']:.6f}")

if __name__ == "__main__":
    main()
