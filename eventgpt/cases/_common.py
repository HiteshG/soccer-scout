"""Shared helpers for CS1–CS4 counterfactual case studies.

Each study follows the paper's substitution recipe (§2.4 / §4):
  1. Collect episodes involving a target player in a given tactical context.
  2. Substitute the target player's ID in the context block with a candidate.
  3. Re-run the model on the modified episode (teacher-forced on the original
     event sequence) and record the predicted rOBV sub-token at each step.
  4. Aggregate across episodes:
       * attackers → truncated mean of top 25 % (paper Table 3 footnote);
       * other roles → simple mean.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from eventgpt.model import GPT, GPTConfig
from eventgpt.tokenizer import Tokenizer, TokenizerConfig


ATTACKER_FAMILIES = {"CF", "W", "AM"}


@dataclass
class Assets:
    model: GPT
    tokenizer: Tokenizer
    meta: dict
    device: str


def load_assets(ckpt: Path, meta_path: Path, cfg_path: Path, device: str = "cpu") -> Assets:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    tokenizer = Tokenizer.load_meta(meta_path, cfg_path=cfg_path)

    ck = torch.load(ckpt, map_location=device)
    model_cfg_kwargs = dict(ck["model_cfg"])
    # `pad_id` was not part of original nanoGPT config. Allow missing key.
    model = GPT(GPTConfig(**model_cfg_kwargs))
    # strict=False so checkpoints from before the style_head became a 2-layer
    # MLP (v5 / v5.1 single-linear style_head.weight/bias) still load for
    # inference — aux heads are not used at case-study time. Warn if anything
    # other than style_head is missing so real schema drift doesn't slip past.
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    unexpected_nontrivial = [k for k in unexpected if not k.startswith("style_head")]
    missing_nontrivial = [k for k in missing if not k.startswith("style_head")]
    if unexpected_nontrivial or missing_nontrivial:
        print(f"[load_assets] WARN state_dict drift: missing={missing_nontrivial} "
              f"unexpected={unexpected_nontrivial}")
    model.to(device).eval()
    return Assets(model=model, tokenizer=tokenizer, meta=meta, device=device)


def player_name_lookup(versa_root: Path) -> pl.DataFrame:
    """Build (player_id → common player_name) map from VERSA events."""
    frames = []
    for season_dir in sorted((versa_root / "events").glob("season=*")):
        frames.append(
            pl.scan_parquet(str(season_dir / "events.parquet"))
            .select([pl.col("player_id"), pl.col("player_name")])
            .drop_nulls()
            .collect()
        )
    cat = pl.concat(frames).group_by("player_id").agg(
        pl.col("player_name").mode().first().alias("player_name")
    )
    return cat


def find_player_id(names: pl.DataFrame, query: str) -> int | None:
    row = names.filter(pl.col("player_name").str.contains(query, literal=False)).head(1)
    if row.is_empty():
        return None
    return int(row["player_id"][0])


def episodes_for_player(
    versa_root: Path,
    season: str,
    player_id: int,
) -> pl.DataFrame:
    """Return a dataframe of (match_id, episode_id) pairs where the player
    appears (either as actor or on-pitch in the context)."""
    ctx_path = versa_root / "episode_context" / f"season={season}" / "episode_context.parquet"
    lf = pl.scan_parquet(str(ctx_path))
    cols = lf.collect_schema().names()
    on_pitch_home = pl.col("on_pitch_home") if "on_pitch_home" in cols else pl.lit([])
    on_pitch_away = pl.col("on_pitch_away") if "on_pitch_away" in cols else pl.lit([])
    mid_col = "matchId" if "matchId" in cols else "match_id"
    result = (
        lf.filter(
            on_pitch_home.list.contains(player_id) | on_pitch_away.list.contains(player_id)
        )
        .select([pl.col(mid_col).alias("match_id"), pl.col("episode_id")])
        .collect()
    )
    return result


def truncated_top_mean(values: np.ndarray, q: float = 0.75) -> float:
    if len(values) == 0:
        return 0.0
    threshold = np.quantile(values, q)
    top = values[values >= threshold]
    if len(top) == 0:
        return float(np.mean(values))
    return float(np.mean(top))


def paired_bootstrap_delta(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap the mean of (b - a) paired per-episode.

    Returns dict with keys:
      * mean_delta   — mean of the per-episode deltas
      * lo, hi       — (1-alpha) percentile bootstrap CI
      * frac_neg     — fraction of pairs where b < a (sub drops below orig)
      * n            — number of pairs used
      * significant  — True iff the CI excludes 0

    CS1/CS4 spreads were buried in rOBV MAE (~0.03) when aggregated as
    independent means. Pairing within episode cancels most per-episode
    variance because both conditions share the same context/events.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = min(len(a_arr), len(b_arr))
    if n == 0:
        return {"mean_delta": 0.0, "lo": 0.0, "hi": 0.0,
                "frac_neg": 0.0, "n": 0, "significant": False}
    a_arr = a_arr[:n]
    b_arr = b_arr[:n]
    deltas = b_arr - a_arr
    mean_delta = float(deltas.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = deltas[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    frac_neg = float((deltas < 0).mean())
    significant = (lo > 0) or (hi < 0)
    return {
        "mean_delta": mean_delta,
        "lo": lo, "hi": hi,
        "frac_neg": frac_neg,
        "n": int(n),
        "significant": bool(significant),
    }


@torch.no_grad()
def score_episode_rOBV(
    assets: Assets,
    episode_tokens: np.ndarray,
    swap_player: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run the model on an episode and return per-event predicted rOBV values.

    ``episode_tokens`` is a uint16 array of length ``block_size``.

    ``swap_player`` = (orig_token_id, new_token_id). Wherever the original
    token appears in the sequence we replace it with the new token before
    running the model — this is the counterfactual substitution mechanic.
    """
    tok = assets.tokenizer
    cfg = tok.cfg
    block = cfg.block_size
    assert episode_tokens.shape == (block,)

    seq = episode_tokens.astype(np.int64).copy()
    if swap_player is not None:
        orig, new = swap_player
        seq[seq == orig] = new

    x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(assets.device)
    # GPT.forward collapses to (B, 1, V) when targets is None (training-loop
    # speedup). We need per-position logits here, so pass a dummy `y` to take
    # the full-sequence branch. Loss is ignored.
    y = torch.from_numpy(seq[1:]).unsqueeze(0).to(assets.device)
    logits, _ = assets.model(x, y)

    # rOBV occupies the 7th sub-token of each event.
    pad_id = cfg.specials["PAD"]
    ctx_len = cfg.context_len
    tok_per_ev = cfg.tokens_per_event
    r_range = cfg.ranges["rOBV_bin"]
    r_edges = np.asarray(assets.meta["rOBV_edges"])
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Top-K truncated expected-value decoding for rOBV: the quantile bins have
    # heavy tails (width ~0.3 at extremes) so argmax-bin-centre quantises
    # predictions into a narrow ±0.06 band, collapsing CS2/CS4 rankings into
    # noise. Using top-5 EV on the head's probability distribution gives
    # continuous rOBV predictions and widens the discriminating spread.
    rOBV_probs = torch.softmax(logits[0, :, r_range.start:r_range.end], dim=-1).cpu().numpy()
    top_k = 5

    out: list[float] = []
    for pos in range(ctx_len, block - 1, tok_per_ev):
        # Predict label at pos = r_subtoken (position index is (pos-1) in preds since y is shifted).
        target_idx = pos + (tok_per_ev - 1) - 1
        if target_idx < 0 or target_idx >= rOBV_probs.shape[0]:
            break
        if seq[target_idx + 1] == pad_id:
            break
        p = rOBV_probs[target_idx]                           # (n_bins,)
        if top_k < len(p):
            keep = np.argsort(p)[-top_k:]
            mask = np.zeros_like(p)
            mask[keep] = 1.0
            p = p * mask
        p = p / max(p.sum(), 1e-9)
        out.append(float((p * r_centers).sum()))
    return np.asarray(out, dtype=np.float64)


def encode_episode_with_swap(
    tokenizer: Tokenizer,
    context: dict,
    events: list[dict],
    swap: tuple[int, int] | None = None,
) -> np.ndarray:
    """Encode one episode with optional context-level player swap. The events
    themselves are NOT substituted — only the on-pitch context block, per the
    paper's counterfactual recipe."""
    ctx = dict(context)
    if swap is not None:
        orig, new = swap
        ctx["on_pitch_ids"] = [new if p == orig else p for p in ctx["on_pitch_ids"]]
    return tokenizer.encode_episode(ctx, events)
