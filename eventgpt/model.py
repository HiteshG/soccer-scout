"""EventGPT backbone — directly adapted from Karpathy's nanoGPT ``model.py``.

Changes from upstream:
  * ``GPTConfig`` defaults tuned for our football event task (smaller vocab,
    fixed block_size = 729 = 29 context + 100 events × 7 sub-tokens,
    dropout=0.1, bias=False by default, explicit ``pad_id``).
  * Cross-entropy loss uses ``ignore_index=config.pad_id`` so padded positions
    in truncated episodes don't contribute to the loss.

Everything else (LayerNorm/bias option, causal self-attention w/ flash path,
weight-tied output head, optimizer grouping, MFU estimation, generate())
is kept verbatim so we can ``diff`` against upstream nanoGPT updates.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 729     # 29 context + 100 events * 7 sub-tokens
    vocab_size: int = 2048    # Unified EventGPT vocab (see configs/tokenizer.yaml)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False
    pad_id: int = 2           # Tokenizer's [PAD] id — loss ignores these positions
    context_len: int = 29     # Context-block length. Paper §2.2: context only conditions
                              # predictions, it is never itself a prediction target. We
                              # mask target indices 0..context_len-2 from the loss.
    # Optional content-based player embedding. Activated when player_metadata is
    # passed at construction time (see PlayerEmbedding).
    player_range_start: int = 181    # vocab range [181, 2048) per configs/tokenizer.yaml
    player_range_end: int = 2048
    use_content_player_emb: bool = False
    n_positions: int = 32     # position archetype vocab (for metadata conditioning)
    n_teams: int = 64         # team vocab (+1 reserved for UNK)
    n_appearance_buckets: int = 5
    n_actions: int = 11       # canonical action-family vocab (style feature)
    n_spatial_bins: int = 16  # 4x4 pitch zone histogram (style feature)
    delta_max: float = 1.0    # bounded-delta norm cap (Let It Go? Not Quite, 2025)
    # Auxiliary embedding objectives. When enabled and use_content_player_emb=True,
    # GPT adds a linear head off the final player embedding (content + bounded δ)
    # for each objective. The heads are computed inside ``compute_aux_losses``
    # and weighted into the total loss by train.py. Each produces a cheap
    # gradient signal that directly shapes the player-embedding manifold:
    #   * role:         CE over the N_POSITIONS-slot primary-position bucket.
    #   * style:        KL against the action_mix histogram (length n_actions).
    #   * contrastive:  triplet margin over cosine distance with a same-position
    #                   positive and a different-position negative. Pulls same-
    #                   role players together in cosine space AND pushes cross-
    #                   role players apart — direct fix for CS2 fine-ranking.
    use_aux_role_loss: bool = False
    use_aux_style_loss: bool = False
    use_aux_contrastive_loss: bool = False
    # Per-head loss weights, in intra-event token order:
    # (team_side, event_type, x, y, delta_t, outcome, rOBV). Default 1.0 across
    # preserves vanilla behaviour. Down-weight a head to free capacity for
    # others; up-weight a head whose eval metric lags (e.g. delta_t MAE).
    tokens_per_event: int = 7
    loss_head_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

class PlayerEmbedding(nn.Module):
    """Content-based player embedding with bounded trainable delta.

    Adapted from *"Let It Go? Not Quite: Addressing Item Cold Start in Sequential
    Recommendations with Content-Based Initialization"* (Konon et al., 2025,
    arXiv:2507.19473).

    Each player's embedding is ``e_p = c_p + δ_p`` where

      * ``c_p`` is a content vector produced by an MLP over hand-engineered
        metadata (primary position, primary team, appearances bucket). It is
        trained, but its gradient path is shared across all players with the
        same metadata category — so cold-start players (present only in val
        because of a mid-season transfer, rookies) inherit a sensible prior
        from the population average for their position/team/playing-time.
      * ``δ_p`` is a per-player learnable residual bounded by
        ``‖δ_p‖₂ ≤ delta_max`` via soft projection in the forward pass. The
        bound keeps the player anchored to their metadata prior while still
        allowing individual style to emerge. Paper used ``δ_max = 0.5`` for
        e-commerce/music; we default to ``1.0`` because football individual
        style matters more than catalogue-item personalisation.

    Metadata is stored as buffers (not parameters) and loaded once at
    construction time from meta.pkl via ``set_metadata``.
    """

    def __init__(
        self,
        n_players: int,
        n_embd: int,
        n_positions: int,
        n_teams: int,
        n_appearance_buckets: int,
        n_actions: int = 11,
        n_spatial_bins: int = 16,
        delta_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_players = int(n_players)
        self.n_actions = int(n_actions)
        self.n_spatial_bins = int(n_spatial_bins)
        self.delta_max = float(delta_max)

        pos_dim, team_dim, app_dim = 32, 32, 16
        action_dim, spatial_dim = 48, 24    # projection dims for style features
        self.pos_emb = nn.Embedding(n_positions, pos_dim)
        self.team_emb = nn.Embedding(n_teams + 1, team_dim)   # +1 for UNK_TEAM
        self.app_emb = nn.Embedding(n_appearance_buckets, app_dim)

        # Style features: action-mix histogram (K=11) and spatial 4x4 histogram.
        # Projected with no activation — they are already unit-simplex inputs
        # (normalised to sum=1 in prepare.py). Entropy is a scalar passed through
        # as-is. A player with zero stats (cold-start) gets all-zero features,
        # which means ``content`` collapses to the pos/team/app prior.
        self.action_proj = nn.Linear(self.n_actions, action_dim)
        self.spatial_proj = nn.Linear(self.n_spatial_bins, spatial_dim)

        in_dim = pos_dim + team_dim + app_dim + action_dim + spatial_dim + 1
        self.content_mlp = nn.Sequential(
            nn.Linear(in_dim, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )

        # Per-player learnable delta residual.
        self.delta = nn.Embedding(n_players, n_embd)
        nn.init.normal_(self.delta.weight, mean=0.0, std=0.02)

        # Metadata buffers. All populated by ``set_metadata``; style/entropy
        # buffers default to zero so legacy meta.pkl files (no new features)
        # load cleanly — the content path just gets zero signal from the new
        # features and reduces to the old (pos, team, apps) prior.
        self.register_buffer("player_positions", torch.zeros(n_players, dtype=torch.long))
        self.register_buffer("player_teams", torch.zeros(n_players, dtype=torch.long))
        self.register_buffer("player_apps", torch.zeros(n_players, dtype=torch.long))
        self.register_buffer(
            "player_action_mix", torch.zeros(n_players, self.n_actions, dtype=torch.float32),
        )
        self.register_buffer(
            "player_spatial_zone", torch.zeros(n_players, self.n_spatial_bins, dtype=torch.float32),
        )
        self.register_buffer("player_pos_entropy", torch.zeros(n_players, dtype=torch.float32))

        # Per-channel z-score stats for action_mix / spatial_zone. Non-persistent
        # (persistent=False) so they are NOT part of the checkpoint state_dict —
        # they are always recomputed from the player metadata at set_metadata
        # time. Default init (mean=0, std=1) is a no-op passthrough, so a model
        # whose set_metadata has not yet been called just sees the raw mix.
        # Zero-centering matters more than variance rescaling: the "Pass" channel
        # sits around 0.57 across all players; without mean-subtract it dominates
        # the content MLP input and washes out rare-but-discriminating channels
        # (Shot, Cross, Tackle). Dividing by std then further amplifies low-
        # variance channels where a small deviation is a large style signal.
        self.register_buffer(
            "action_mix_mean", torch.zeros(self.n_actions, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_mix_std", torch.ones(self.n_actions, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "spatial_zone_mean", torch.zeros(self.n_spatial_bins, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "spatial_zone_std", torch.ones(self.n_spatial_bins, dtype=torch.float32),
            persistent=False,
        )

    def set_metadata(
        self,
        positions: torch.Tensor | list[int],
        teams: torch.Tensor | list[int],
        apps: torch.Tensor | list[int],
        action_mix: torch.Tensor | list | None = None,
        spatial_zone: torch.Tensor | list | None = None,
        pos_entropy: torch.Tensor | list | None = None,
    ) -> None:
        """Populate per-player metadata buffers. New-style feature buffers are
        optional — if a legacy meta.pkl is passed (no action_mix / spatial_zone /
        pos_entropy), those buffers stay zero and the content MLP reduces to the
        pre-style prior (pos + team + apps only). Training with aux_style_loss
        against an all-zero target would be vacuous; train.py is responsible for
        not enabling the style aux head when the feature is missing."""
        self.player_positions.copy_(torch.as_tensor(positions, dtype=torch.long))
        self.player_teams.copy_(torch.as_tensor(teams, dtype=torch.long))
        self.player_apps.copy_(torch.as_tensor(apps, dtype=torch.long))
        if action_mix is not None:
            mix = torch.as_tensor(action_mix, dtype=torch.float32)
            assert mix.shape == self.player_action_mix.shape, (
                f"action_mix shape {tuple(mix.shape)} != expected "
                f"{tuple(self.player_action_mix.shape)}"
            )
            self.player_action_mix.copy_(mix)
            # Per-channel mean/std over players that actually have action data.
            # Cold-start rows (all-zero) would drag the mean toward 0 and
            # distort the z-score; exclude them from stats.
            nonzero = mix.sum(dim=1) > 0
            if nonzero.any():
                valid = mix[nonzero]
                self.action_mix_mean.copy_(valid.mean(dim=0))
                # Floor std at 1e-3 — guards against divide-by-zero for channels
                # where every player emits the same rate (e.g. if a channel is
                # always 0 the std would otherwise be 0).
                self.action_mix_std.copy_(valid.std(dim=0).clamp(min=1e-3))
        if spatial_zone is not None:
            sp = torch.as_tensor(spatial_zone, dtype=torch.float32)
            assert sp.shape == self.player_spatial_zone.shape, (
                f"spatial_zone shape {tuple(sp.shape)} != expected "
                f"{tuple(self.player_spatial_zone.shape)}"
            )
            self.player_spatial_zone.copy_(sp)
            nonzero = sp.sum(dim=1) > 0
            if nonzero.any():
                valid = sp[nonzero]
                self.spatial_zone_mean.copy_(valid.mean(dim=0))
                self.spatial_zone_std.copy_(valid.std(dim=0).clamp(min=1e-3))
        if pos_entropy is not None:
            ent = torch.as_tensor(pos_entropy, dtype=torch.float32).reshape(-1)
            assert ent.shape == self.player_pos_entropy.shape, (
                f"pos_entropy shape {tuple(ent.shape)} != expected "
                f"{tuple(self.player_pos_entropy.shape)}"
            )
            self.player_pos_entropy.copy_(ent)

    def _content(self, player_idx: torch.Tensor) -> torch.Tensor:
        """Content vector (no delta) for ``player_idx``."""
        pos = self.pos_emb(self.player_positions[player_idx])
        tm = self.team_emb(self.player_teams[player_idx])
        ap = self.app_emb(self.player_apps[player_idx])
        # Z-score action_mix and spatial_zone so dominant-mass channels
        # (everyone passes a lot) stop overwhelming the content MLP input.
        # Defaults (mean=0, std=1) = identity for a freshly-constructed module
        # whose set_metadata has not yet run.
        action_z = (
            self.player_action_mix[player_idx] - self.action_mix_mean
        ) / self.action_mix_std
        spatial_z = (
            self.player_spatial_zone[player_idx] - self.spatial_zone_mean
        ) / self.spatial_zone_std
        action = self.action_proj(action_z)
        spatial = self.spatial_proj(spatial_z)
        entropy = self.player_pos_entropy[player_idx].unsqueeze(-1)
        return self.content_mlp(torch.cat([pos, tm, ap, action, spatial, entropy], dim=-1))

    def _delta(self, player_idx: torch.Tensor) -> torch.Tensor:
        """Bounded-norm delta residual for ``player_idx``."""
        delta_raw = self.delta(player_idx)
        norms = delta_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.delta_max / norms, max=1.0)
        return delta_raw * scale

    def forward_components(
        self, player_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(content, delta)`` separately. Used by probes (CS3 ablation,
        archetype discovery) that need to attribute role vs style signal to the
        content MLP vs the per-player delta residual."""
        return self._content(player_idx), self._delta(player_idx)

    def forward(self, player_idx: torch.Tensor) -> torch.Tensor:
        """``player_idx`` is an integer tensor of shape ``(...)`` indexing into
        the player vocab (values in ``[0, n_players)``). Returns embeddings of
        shape ``(..., n_embd)``."""
        content, delta = self.forward_components(player_idx)
        return content + delta


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # Optional content-based player embedding. Takes over ``wte`` for token
        # IDs in the player range when enabled.
        self.player_emb: PlayerEmbedding | None = None
        if config.use_content_player_emb:
            n_players = config.player_range_end - config.player_range_start
            self.player_emb = PlayerEmbedding(
                n_players=n_players,
                n_embd=config.n_embd,
                n_positions=config.n_positions,
                n_teams=config.n_teams,
                n_appearance_buckets=config.n_appearance_buckets,
                n_actions=config.n_actions,
                n_spatial_bins=config.n_spatial_bins,
                delta_max=config.delta_max,
            )

        # Auxiliary heads off the final player embedding. They only make sense
        # when we have a player_emb to supervise; silently skip otherwise.
        # style_head is a 2-layer MLP so it can decode raw action_mix from the
        # z-scored content input. New v5.2+ checkpoints use the MLP form; older
        # v5/v5.1 ckpts carry a single Linear under style_head.weight/bias —
        # load_assets tolerates the mismatch via strict=False (the style head
        # is inference-dead at case-study time anyway).
        self.role_head: nn.Linear | None = None
        self.style_head: nn.Sequential | None = None
        if config.use_content_player_emb:
            if config.use_aux_role_loss:
                self.role_head = nn.Linear(config.n_embd, config.n_positions, bias=True)
            if config.use_aux_style_loss:
                self.style_head = nn.Sequential(
                    nn.Linear(config.n_embd, config.n_embd // 2, bias=True),
                    nn.GELU(),
                    nn.Linear(config.n_embd // 2, config.n_actions, bias=True),
                )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Token embeddings. When content-based player embedding is enabled, route
        # player-vocab positions through PlayerEmbedding (content + bounded delta)
        # instead of the shared ``wte`` table. Non-player tokens still use wte so
        # that weight tying with lm_head remains valid.
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        if self.player_emb is not None:
            p_start = self.config.player_range_start
            p_end = self.config.player_range_end
            player_mask = (idx >= p_start) & (idx < p_end)
            if player_mask.any():
                player_local = (idx - p_start).clamp(
                    min=0, max=self.player_emb.n_players - 1
                )
                player_out = self.player_emb(player_local)
                tok_emb = torch.where(player_mask.unsqueeze(-1), player_out, tok_emb)

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Paper §2.2: loss is only over event-token predictions.
            # targets[i] is the teacher-forced target for position i, which
            # corresponds to predicting the *next* token at original index i+1.
            # Context tokens occupy original indices [0, context_len); the first
            # event token sits at original index context_len. So target index i
            # predicts an event token iff i + 1 >= context_len, i.e.
            # i >= context_len - 1. Mask everything before that AND any padded
            # positions in the event region.
            logits = self.lm_head(x)
            B, T, V = logits.shape
            losses = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1), reduction="none",
            ).view(B, T)

            ctx_cutoff = self.config.context_len - 1
            pad_mask = (targets != self.config.pad_id).float()

            # Per-head weighting: inside the event region (positions
            # ctx_cutoff..T-1), each successive group of tokens_per_event
            # positions corresponds to one full event, and position
            # (i - ctx_cutoff) % tokens_per_event selects which head that
            # token predicts (team_side, event_type, x, y, delta_t, outcome,
            # rOBV in order). Default weights are all 1.0 → identical to the
            # old masked-mean loss.
            weights = torch.ones(T, device=device, dtype=losses.dtype)
            k = self.config.tokens_per_event
            n_event_positions = max(T - ctx_cutoff, 0)
            if n_event_positions > 0:
                head_w = torch.tensor(
                    self.config.loss_head_weights, device=device, dtype=losses.dtype,
                )
                rep = (n_event_positions + k - 1) // k
                expanded = head_w.repeat(rep)[:n_event_positions]
                weights[ctx_cutoff:] = expanded

            # Position gate: zero out context-region weights.
            weights[:ctx_cutoff] = 0.0
            weighted = losses * weights.unsqueeze(0) * pad_mask
            denom = (weights.unsqueeze(0) * pad_mask).sum().clamp(min=1)
            loss = weighted.sum() / denom
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def compute_aux_losses(
        self,
        player_idx: torch.Tensor | None = None,
        contrastive_margin: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Compute the role + style auxiliary losses off the player embedding.

        Returns an empty dict when neither aux head is configured (so callers
        can unconditionally iterate). When ``player_idx`` is None the loss is
        computed over the entire player vocab, which is the right default
        because the aux heads are tiny and we want every player to receive
        gradient on every step regardless of which ids appeared in the batch.

        Role loss: cross-entropy over the N_POSITIONS-slot primary-position
        bucket. Easy-to-fit target — expect it to drop near 0 within ~5k iters
        and then act purely as a pin keeping role identity recoverable from
        the embedding after autoregressive training would otherwise smear it.

        Style loss: KL(target || pred) against the per-player action_mix
        histogram. Harder to fit (11-way soft target), plateaus around 0.05–
        0.15 KL — that plateau IS the style signal being encoded into the
        embedding manifold.
        """
        out: dict[str, torch.Tensor] = {}
        if self.player_emb is None:
            return out
        contrastive_on = getattr(self.config, "use_aux_contrastive_loss", False)
        if self.role_head is None and self.style_head is None and not contrastive_on:
            return out
        if player_idx is None:
            player_idx = torch.arange(self.player_emb.n_players, device=self.player_emb.delta.weight.device)
        player_embs = self.player_emb(player_idx)  # (N, D)
        if self.role_head is not None:
            role_logits = self.role_head(player_embs)
            role_targets = self.player_emb.player_positions[player_idx]
            out["role"] = F.cross_entropy(role_logits, role_targets)
        if self.style_head is not None:
            style_log_probs = F.log_softmax(self.style_head(player_embs), dim=-1)
            style_targets = self.player_emb.player_action_mix[player_idx]
            # KL(P_target || P_pred). batchmean matches the style-head convention.
            out["style"] = F.kl_div(style_log_probs, style_targets, reduction="batchmean")
        if contrastive_on:
            # Vectorised same-position sampling: for each anchor, pick a random
            # other player with the same primary_position as the positive, and
            # a random player with a different position as the negative. The
            # argmax-of-random trick keeps it diff-able-free and O(N·V) memory
            # where V is the player vocab (~1200 for us — trivial).
            device = player_embs.device
            pos_of_anchor = self.player_emb.player_positions[player_idx]  # (N,)
            pos_all = self.player_emb.player_positions                     # (V,)
            N = player_idx.shape[0]
            same_mask = pos_all.unsqueeze(0) == pos_of_anchor.unsqueeze(1)  # (N, V)
            # Exclude the anchor itself from being picked as its own positive.
            # player_idx could be a permutation, not identity — so we have to
            # zero out positions[i]==player_idx[i] per-row.
            anchor_cols = player_idx.unsqueeze(1)  # (N, 1)
            col_range = torch.arange(pos_all.shape[0], device=device).unsqueeze(0)  # (1, V)
            self_mask = col_range == anchor_cols
            same_mask = same_mask & ~self_mask
            diff_mask = ~(pos_all.unsqueeze(0) == pos_of_anchor.unsqueeze(1))
            # Random score, -inf on masked-out positions; argmax picks a random valid.
            rand_pos = torch.rand(N, pos_all.shape[0], device=device)
            rand_pos = rand_pos.masked_fill(~same_mask, -1.0)
            pos_idx = rand_pos.argmax(dim=-1)
            rand_neg = torch.rand(N, pos_all.shape[0], device=device)
            rand_neg = rand_neg.masked_fill(~diff_mask, -1.0)
            neg_idx = rand_neg.argmax(dim=-1)
            # Rows whose positions group has size 1 (no same-position peer)
            # degenerate to self as the positive; guard by checking rand_pos
            # max was -1 → skip that row from the loss.
            valid_pos = rand_pos.max(dim=-1).values > -0.5
            valid_neg = rand_neg.max(dim=-1).values > -0.5
            valid = valid_pos & valid_neg
            if valid.any():
                anchor = player_embs[valid]
                pos = self.player_emb(pos_idx[valid])
                neg = self.player_emb(neg_idx[valid])
                anchor_n = F.normalize(anchor, dim=-1)
                pos_n = F.normalize(pos, dim=-1)
                neg_n = F.normalize(neg, dim=-1)
                d_pos = 1.0 - (anchor_n * pos_n).sum(dim=-1)
                d_neg = 1.0 - (anchor_n * neg_n).sum(dim=-1)
                out["contrastive"] = F.relu(d_pos - d_neg + contrastive_margin).mean()
            else:
                out["contrastive"] = torch.zeros((), device=device, requires_grad=True)
        return out

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
