"""What-if counterfactual swap — generalises ``cases/cs4_role_transfer.py``.

Given an incumbent player + a candidate replacement (or, if no candidate is
provided, the incumbent's top-K style peers), score a sample of episodes
involving the incumbent and report the per-episode rOBV delta with paired
bootstrap CI. Returns a structure the UI ``swap_impact_card`` consumes
directly to render the green/amber/red verdict + plain-English summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from eventgpt.cases._common import (
    Assets, encode_episode_with_swap, episodes_for_player,
    paired_bootstrap_delta, player_name_lookup,
    score_episode_rOBV, truncated_top_mean,
)
from eventgpt.data.episodes import iter_episodes_in_season


def _style_peers(
    assets: Assets, query_local_idx: int, n_peers: int,
) -> list[tuple[int, float]]:
    """Top-N (local_idx, cosine) peers by player-embedding cosine. Inlined
    from cases/cs4_role_transfer.py so the deploy is standalone."""
    assert assets.model.player_emb is not None, (
        "Style-peer mode requires a model with use_content_player_emb=True."
    )
    with torch.no_grad():
        all_local = torch.arange(
            assets.model.player_emb.n_players, device=assets.device,
        )
        vecs = assets.model.player_emb(all_local).detach().cpu().numpy()
    q = vecs[query_local_idx]
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    cos = vecs @ q_norm / (np.linalg.norm(vecs, axis=1) + 1e-9)
    cos[query_local_idx] = -np.inf
    top = np.argsort(cos)[-n_peers:][::-1]
    return [(int(i), float(cos[i])) for i in top]


def swap_impact(
    assets: Assets,
    versa_root: Path,
    *,
    incumbent_player_id: int,
    candidate_player_id: int | None = None,
    n_peers: int = 5,
    season: str = "23-24",
    max_episodes: int = 80,
) -> dict:
    """Run a CS4-style swap and return the structured impact summary.

    If ``candidate_player_id`` is None, uses the top-N style peers of the
    incumbent (mean of their per-peer episode predictions). If it's set,
    uses just that single candidate.

    Returns ``{incumbent, candidate_or_peers, season, n_episodes,
       mean_delta, delta_ci_lo, delta_ci_hi, frac_drop, significant,
       orig_mean, sub_mean, peers_used: [{name, cosine}]}``.
    """
    pidx = assets.tokenizer._player_index
    if incumbent_player_id not in pidx:
        raise KeyError(f"incumbent {incumbent_player_id} not in vocab")
    pl_range = assets.tokenizer.cfg.ranges["players"]
    inc_local = pidx[incumbent_player_id]
    inc_tok = pl_range.start + inc_local
    names_df = player_name_lookup(versa_root)
    name_lut = {int(r["player_id"]): str(r["player_name"])
                for r in names_df.iter_rows(named=True)}

    if candidate_player_id is None:
        peers = _style_peers(assets, inc_local, n_peers=n_peers)
        peer_locals = [int(i) for i, _cos in peers]
        peer_cos = [float(c) for _i, c in peers]
        pid_by_local = {v: k for k, v in pidx.items()}
        peers_used = [
            {"player_id": int(pid_by_local.get(l, -1)),
             "name": name_lut.get(int(pid_by_local.get(l, -1)), str(l)),
             "cosine": peer_cos[i]}
            for i, l in enumerate(peer_locals)
        ]
        sub_label = "style-peer-mean"
    else:
        if candidate_player_id not in pidx:
            raise KeyError(f"candidate {candidate_player_id} not in vocab")
        cand_local = pidx[candidate_player_id]
        peer_locals = [cand_local]
        # cosine from incumbent to candidate for transparency
        pe = assets.model.player_emb
        with torch.no_grad():
            idx = torch.arange(pe.n_players, device=assets.device)
            vecs = pe(idx).detach().cpu().numpy()
        a = vecs[inc_local] / (np.linalg.norm(vecs[inc_local]) + 1e-9)
        b = vecs[cand_local] / (np.linalg.norm(vecs[cand_local]) + 1e-9)
        peers_used = [{
            "player_id": int(candidate_player_id),
            "name": name_lut.get(int(candidate_player_id), str(candidate_player_id)),
            "cosine": float(np.dot(a, b)),
        }]
        sub_label = "single-candidate"

    peer_toks = [pl_range.start + l for l in peer_locals]

    # Pull episodes featuring the incumbent in the chosen season.
    pairs = episodes_for_player(versa_root, season=season, player_id=incumbent_player_id)
    if pairs.is_empty():
        return {
            "incumbent_player_id": int(incumbent_player_id),
            "incumbent_name": name_lut.get(int(incumbent_player_id), str(incumbent_player_id)),
            "season": season,
            "sub_label": sub_label,
            "peers_used": peers_used,
            "n_episodes": 0,
            "mean_delta": 0.0,
            "delta_ci_lo": 0.0, "delta_ci_hi": 0.0,
            "frac_drop": 0.0, "significant": False,
            "orig_mean": 0.0, "sub_mean": 0.0,
            "warning": f"No episodes for player {incumbent_player_id} in {season}",
        }
    pairs = pairs.sample(min(max_episodes, pairs.height), seed=42)
    want = set(zip(pairs["match_id"].to_list(), pairs["episode_id"].to_list()))
    cached = []
    for epi in iter_episodes_in_season(
        versa_root, season, match_filter=set(pairs["match_id"].to_list())
    ):
        if (epi.match_id, epi.episode_id) not in want:
            continue
        if incumbent_player_id not in epi.context["on_pitch_ids"]:
            continue
        cached.append(epi)
        if len(cached) >= max_episodes:
            break

    orig_vals: list[float] = []
    sub_vals: list[float] = []
    for epi in cached:
        tokens = encode_episode_with_swap(assets.tokenizer, epi.context, epi.events)
        orig = score_episode_rOBV(assets, tokens, swap_player=None)
        if not orig.size:
            continue
        peer_means = []
        for tok in peer_toks:
            sub = score_episode_rOBV(assets, tokens, swap_player=(inc_tok, tok))
            if sub.size:
                peer_means.append(truncated_top_mean(sub))
        if not peer_means:
            continue
        orig_vals.append(truncated_top_mean(orig))
        sub_vals.append(float(np.mean(peer_means)))

    if not orig_vals:
        return {
            "incumbent_player_id": int(incumbent_player_id),
            "incumbent_name": name_lut.get(int(incumbent_player_id), str(incumbent_player_id)),
            "season": season,
            "sub_label": sub_label,
            "peers_used": peers_used,
            "n_episodes": 0,
            "mean_delta": 0.0,
            "delta_ci_lo": 0.0, "delta_ci_hi": 0.0,
            "frac_drop": 0.0, "significant": False,
            "orig_mean": 0.0, "sub_mean": 0.0,
            "warning": "No scorable episodes",
        }
    boot = paired_bootstrap_delta(orig_vals, sub_vals)
    return {
        "incumbent_player_id": int(incumbent_player_id),
        "incumbent_name": name_lut.get(int(incumbent_player_id), str(incumbent_player_id)),
        "candidate_player_id": int(candidate_player_id) if candidate_player_id else None,
        "candidate_name": (name_lut.get(int(candidate_player_id), str(candidate_player_id))
                           if candidate_player_id else None),
        "season": season,
        "sub_label": sub_label,
        "peers_used": peers_used,
        "n_episodes": int(boot["n"]),
        "mean_delta": boot["mean_delta"],
        "delta_ci_lo": boot["lo"],
        "delta_ci_hi": boot["hi"],
        "frac_drop": boot["frac_neg"],
        "significant": bool(boot["significant"]),
        "orig_mean": float(np.mean(orig_vals)),
        "sub_mean": float(np.mean(sub_vals)),
    }
