"""Replacement search — refactored core of ``probes/replacement_search.py``.

Pure-function entry point used by both the Modal HTTP endpoint and the typer
CLI. Returns a JSON-shaped dict the UI can render directly, including the
*action-mix difference vs the query player* per-row so the Streamlit table can
explain "more direct", "less aerial work" without a second round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from eventgpt.cases._common import Assets, player_name_lookup
from eventgpt.web.lib.players import (
    POS_FAMILY, _scan_team_labels, _scan_versa_player_facts,
)


def _embeddings(assets: Assets, mode: str) -> np.ndarray:
    """Return the (n_players, n_embd) array for the requested view."""
    pe = assets.model.player_emb
    if pe is None:
        raise RuntimeError("This checkpoint lacks PlayerEmbedding (use_content_player_emb=False).")
    with torch.no_grad():
        idx = torch.arange(pe.n_players, device=assets.device)
        if mode == "full":
            content, delta = pe.forward_components(idx)
            vecs = (content + delta).detach().cpu().numpy()
        elif mode == "content":
            content, _ = pe.forward_components(idx)
            vecs = content.detach().cpu().numpy()
        elif mode == "delta":
            _, delta = pe.forward_components(idx)
            vecs = delta.detach().cpu().numpy()
        else:
            raise ValueError(f"mode must be full|content|delta, got {mode!r}")
    return vecs


def search_replacements(
    assets: Assets,
    versa_root: Path,
    *,
    query_player_id: int,
    top_k: int = 20,
    mode: str = "full",
    same_family: bool = False,
    same_position: bool = False,
    in_team_id: int | None = None,
    not_in_team_id: int | None = None,
    min_events: int = 500,
) -> dict:
    """Top-K cosine-nearest players to ``query_player_id`` under filters.

    Returns ``{query, query_player_id, query_position, query_family,
    query_team_id, query_team_label, mode, filters, n_candidates,
    results: [{rank, player_id, player_name, cosine, position, family,
                team_id, team_label, n_events,
                action_diff: {fam: signed_diff}}]}``.

    ``action_diff`` is the per-channel difference between the candidate's
    action_mix and the query's; the UI / explainer can convert positive
    values into "More X" / "Less X" badges without recomputing.
    """
    if query_player_id not in assets.tokenizer._player_index:
        raise KeyError(f"query_player_id {query_player_id} not in vocab")
    pidx = assets.tokenizer._player_index
    q_local = pidx[query_player_id]
    versa_root_str = str(versa_root)
    pos_map, team_map, count_map = _scan_versa_player_facts(versa_root_str)
    team_labels = _scan_team_labels(versa_root_str)
    names = player_name_lookup(versa_root)
    name_lut = {int(r["player_id"]): str(r["player_name"])
                for r in names.iter_rows(named=True)}

    vecs = _embeddings(assets, mode)
    q_vec = vecs[q_local]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    cos = vecs @ q_norm / (np.linalg.norm(vecs, axis=1) + 1e-9)

    pe = assets.model.player_emb
    action_mix_all = pe.player_action_mix.detach().cpu().numpy()
    action_families = list(
        assets.meta.get("player_metadata", {}).get("action_families", []) or
        [f"action_{i}" for i in range(action_mix_all.shape[1])]
    )
    q_mix = action_mix_all[q_local]

    q_pos = pos_map.get(int(query_player_id))
    q_family = POS_FAMILY.get(q_pos or "", "?")
    q_team_id = team_map.get(int(query_player_id))

    rows: list[dict] = []
    for pid, local in pidx.items():
        if pid == query_player_id:
            continue
        if count_map.get(int(pid), 0) < min_events:
            continue
        pos = pos_map.get(int(pid))
        family = POS_FAMILY.get(pos or "", "?")
        team_id = team_map.get(int(pid))
        if same_family and family != q_family:
            continue
        if same_position and pos != q_pos:
            continue
        if in_team_id is not None and team_id != in_team_id:
            continue
        if not_in_team_id is not None and team_id == not_in_team_id:
            continue
        diff = (action_mix_all[local] - q_mix).tolist()
        rows.append({
            "player_id": int(pid),
            "player_name": name_lut.get(int(pid), str(pid)),
            "cosine": float(cos[local]),
            "position": pos,
            "family": family,
            "team_id": int(team_id) if team_id is not None else None,
            "team_label": team_labels.get(team_id) if team_id is not None else None,
            "n_events": int(count_map.get(int(pid), 0)),
            "action_diff": {fam: float(v) for fam, v in zip(action_families, diff)},
        })

    rows.sort(key=lambda r: r["cosine"], reverse=True)
    top = rows[:top_k]
    for i, row in enumerate(top):
        row["rank"] = i + 1

    return {
        "query": name_lut.get(int(query_player_id), str(query_player_id)),
        "query_player_id": int(query_player_id),
        "query_position": q_pos,
        "query_family": q_family,
        "query_team_id": int(q_team_id) if q_team_id is not None else None,
        "query_team_label": team_labels.get(q_team_id) if q_team_id is not None else None,
        "mode": mode,
        "filters": {
            "same_family": same_family, "same_position": same_position,
            "in_team_id": in_team_id, "not_in_team_id": not_in_team_id,
            "min_events": min_events,
        },
        "n_candidates": len(rows),
        "action_families": action_families,
        "results": top,
    }
