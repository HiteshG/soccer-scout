"""Team-style aggregation — refactored core of ``probes/team_style.py``.

Three queries the UI needs:
  * ``team_vector(team_id)`` — minutes-weighted mean of the team's player
    embeddings (event counts proxy for minutes).
  * ``team_fit(candidate_id, team_id)`` — cosine of candidate vs team vector,
    plus its rank vs the team's existing players + an action-mix delta vs
    the team's average style. The UI consumes the deltas as bullet points.
  * ``list_teams()`` — top-N teams by roster coverage with surname-trio labels.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import torch

from eventgpt.cases._common import Assets, player_name_lookup
from eventgpt.web.lib.players import (
    POS_FAMILY, _scan_team_labels, _scan_versa_player_facts,
)


@lru_cache(maxsize=1)
def _team_player_events(versa_root_str: str) -> pl.DataFrame:
    versa_root = Path(versa_root_str)
    frames = []
    for season_dir in sorted((versa_root / "events").glob("season=*")):
        cols = pl.scan_parquet(str(season_dir / "events.parquet")).collect_schema().names()
        if "team_id" not in cols:
            continue
        frames.append(
            pl.scan_parquet(str(season_dir / "events.parquet"))
            .select([pl.col("team_id"), pl.col("player_id")])
            .drop_nulls()
            .collect()
        )
    if not frames:
        return pl.DataFrame(schema={"team_id": pl.Int64, "player_id": pl.Int64, "n_events": pl.Int64})
    return (
        pl.concat(frames)
        .group_by(["team_id", "player_id"])
        .agg(pl.len().alias("n_events"))
    )


def _full_embeddings(assets: Assets) -> np.ndarray:
    pe = assets.model.player_emb
    if pe is None:
        raise RuntimeError("This checkpoint lacks PlayerEmbedding.")
    with torch.no_grad():
        idx = torch.arange(pe.n_players, device=assets.device)
        return pe(idx).detach().cpu().numpy()


def _team_vectors(
    assets: Assets, versa_root: Path, min_team_events: int,
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, str], dict[int, np.ndarray]]:
    """Return ``(team_vec, n_players_per_team, team_label, team_action_mix)``
    for every team with >= min_team_events events.
    """
    versa_root_str = str(versa_root)
    tp = _team_player_events(versa_root_str)
    team_labels = _scan_team_labels(versa_root_str)
    vecs = _full_embeddings(assets)
    pidx = assets.tokenizer._player_index
    pe = assets.model.player_emb
    action_mix_all = pe.player_action_mix.detach().cpu().numpy()

    totals = (
        tp.group_by("team_id").agg(pl.col("n_events").sum().alias("total"))
    )
    total_by_team = {int(r["team_id"]): int(r["total"]) for r in totals.iter_rows(named=True)}

    team_vec: dict[int, np.ndarray] = {}
    team_size: dict[int, int] = {}
    team_action: dict[int, np.ndarray] = {}
    label_out: dict[int, str] = {}
    for tid in sorted(total_by_team):
        if total_by_team[tid] < min_team_events:
            continue
        rows = tp.filter(pl.col("team_id") == tid)
        agg = np.zeros_like(vecs[0])
        agg_act = np.zeros(action_mix_all.shape[1], dtype=np.float64)
        total_w = 0.0
        count = 0
        for r in rows.iter_rows(named=True):
            pid = int(r["player_id"])
            if pid not in pidx:
                continue
            local = pidx[pid]
            w = float(r["n_events"])
            agg = agg + w * vecs[local]
            agg_act = agg_act + w * action_mix_all[local]
            total_w += w
            count += 1
        if total_w > 0:
            team_vec[tid] = agg / total_w
            team_action[tid] = (agg_act / total_w).astype(np.float32)
            team_size[tid] = count
            label_out[tid] = team_labels.get(tid, f"team_{tid}")
    return team_vec, team_size, label_out, team_action


def list_teams(assets: Assets, versa_root: Path, *, min_team_events: int = 5000) -> dict:
    """Top teams by roster coverage with similarity matrix among them."""
    team_vec, team_size, team_label, _ = _team_vectors(assets, versa_root, min_team_events)
    ids = sorted(team_size, key=lambda t: -team_size[t])
    norms = {t: float(np.linalg.norm(team_vec[t])) + 1e-9 for t in ids}
    mat = np.zeros((len(ids), len(ids)))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            mat[i, j] = float(np.dot(team_vec[a], team_vec[b]) / (norms[a] * norms[b]))
    return {
        "min_team_events": min_team_events,
        "n_teams": len(ids),
        "teams": [
            {"team_id": int(t), "label": team_label[t], "roster_size": team_size[t]}
            for t in ids
        ],
        "similarity_matrix": mat.tolist(),
    }


def team_fit(
    assets: Assets,
    versa_root: Path,
    *,
    candidate_player_id: int,
    team_id: int,
    min_team_events: int = 5000,
) -> dict:
    """Score how stylistically a candidate fits the target team.

    Returns:
      ``{candidate_player_id, candidate_name, team_id, team_label,
         fit_score (cosine, 0..1), team_size, candidate_team_rank,
         current_top_player {name, fit}, action_diff_vs_team {fam: float},
         spatial_diff_vs_team [16-vec], peers_in_team [{name, cosine}]}``
    """
    pidx = assets.tokenizer._player_index
    if candidate_player_id not in pidx:
        raise KeyError(f"player {candidate_player_id} not in vocab")
    versa_root_str = str(versa_root)
    team_vec, team_size, team_label, team_action = _team_vectors(
        assets, versa_root, min_team_events,
    )
    if team_id not in team_vec:
        raise KeyError(f"team {team_id} below min_team_events or unknown")

    vecs = _full_embeddings(assets)
    pe = assets.model.player_emb
    action_mix_all = pe.player_action_mix.detach().cpu().numpy()
    spatial_all = pe.player_spatial_zone.detach().cpu().numpy()
    names = player_name_lookup(versa_root)
    name_lut = {int(r["player_id"]): str(r["player_name"])
                for r in names.iter_rows(named=True)}

    cand_local = pidx[candidate_player_id]
    cand_vec = vecs[cand_local]
    tvec = team_vec[team_id]
    cand_n = cand_vec / (np.linalg.norm(cand_vec) + 1e-9)
    tvec_n = tvec / (np.linalg.norm(tvec) + 1e-9)
    fit = float(np.dot(cand_n, tvec_n))

    # Rank vs team's existing players.
    tp = _team_player_events(versa_root_str)
    member_rows = tp.filter(pl.col("team_id") == team_id)
    member_pids = [int(r["player_id"]) for r in member_rows.iter_rows(named=True)
                   if int(r["player_id"]) in pidx]
    member_fits = []
    for pid in member_pids:
        v = vecs[pidx[pid]]
        c = float(np.dot(v / (np.linalg.norm(v) + 1e-9), tvec_n))
        member_fits.append((pid, c, name_lut.get(pid, str(pid))))
    member_fits.sort(key=lambda r: -r[1])
    rank = 1 + sum(1 for _pid, c, _n in member_fits if c > fit)

    # Closest peers IN the team to the candidate (by cosine to candidate).
    peer_cos: list[tuple[str, float]] = []
    for pid in member_pids:
        v = vecs[pidx[pid]]
        c = float(np.dot(cand_n, v / (np.linalg.norm(v) + 1e-9)))
        peer_cos.append((name_lut.get(pid, str(pid)), c))
    peer_cos.sort(key=lambda r: -r[1])

    # Action-mix delta: candidate vs team weighted action mix.
    action_families = list(
        assets.meta.get("player_metadata", {}).get("action_families", []) or
        [f"action_{i}" for i in range(action_mix_all.shape[1])]
    )
    action_diff = (action_mix_all[cand_local] - team_action[team_id]).tolist()
    spatial_diff = (spatial_all[cand_local] -
                    np.zeros_like(spatial_all[cand_local])).tolist()  # placeholder until team-spatial caching is added

    return {
        "candidate_player_id": int(candidate_player_id),
        "candidate_name": name_lut.get(int(candidate_player_id), str(candidate_player_id)),
        "team_id": int(team_id),
        "team_label": team_label[team_id],
        "team_size": team_size[team_id],
        "fit_score": fit,
        "candidate_team_rank": int(rank),
        "current_top_player": {
            "player_id": int(member_fits[0][0]) if member_fits else None,
            "name": member_fits[0][2] if member_fits else None,
            "fit": float(member_fits[0][1]) if member_fits else None,
        },
        "action_families": action_families,
        "action_diff_vs_team": {fam: float(v) for fam, v in zip(action_families, action_diff)},
        "spatial_diff_vs_team": [float(v) for v in spatial_diff],
        "peers_in_team": [{"name": n, "cosine": float(c)} for n, c in peer_cos[:5]],
    }
