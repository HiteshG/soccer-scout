"""Per-family archetype clustering — refactored core of
``probes/archetype_discovery.py`` for the Strategy-Archetype-map view.

The Modal endpoint caches the result on first call: clustering is expensive
(~30 s for HDBSCAN over 800 players) but the output never changes for a
fixed checkpoint. ``compute_archetypes`` is the underlying pure function;
the endpoint wraps it in a per-process cache.
"""

from __future__ import annotations

import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import torch

from eventgpt.cases._common import Assets, player_name_lookup
from eventgpt.web.lib.players import POS_FAMILY


def _cluster(vecs: np.ndarray, min_cluster_size: int):
    try:
        import hdbscan
        c = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(5, min_cluster_size // 2),
            metric="euclidean", cluster_selection_method="eom",
        )
        return c.fit_predict(vecs), "hdbscan"
    except ImportError:
        from sklearn.cluster import KMeans
        k = max(6, min(30, len(vecs) // 25))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vecs)
        return km.labels_, "kmeans"


def _project_2d(vecs: np.ndarray, seed: int = 42) -> np.ndarray:
    """Best-effort 2D projection for the map. UMAP if available, else t-SNE."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(vecs)
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=seed,
                    init="pca").fit_transform(vecs)


def compute_archetypes(
    assets: Assets,
    versa_root: Path,
    *,
    mode: str = "full",
    min_events: int = 1000,
    min_cluster_size: int = 5,
    per_family: bool = True,
) -> dict:
    """Cluster per-family + return a 2D projection for the map view.

    ``mode``: full | content | delta.
    Output: ``{algorithm, n_clusters, n_noise, mean_family_purity,
              clusters: [{cluster_id, dominant_family, family_purity,
                          size, top_actions, exemplars, position_histogram}],
              players_xy: [{player_id, name, x, y, cluster_id, family,
                            position}]}``.
    """
    pe = assets.model.player_emb
    if pe is None:
        raise RuntimeError("Checkpoint lacks PlayerEmbedding.")

    # Load minimum-event filter via VERSA.
    frames = []
    for season_dir in sorted((versa_root / "events").glob("season=*")):
        frames.append(
            pl.scan_parquet(str(season_dir / "events.parquet"))
            .select([pl.col("player_id"), pl.col("pos_t")])
            .drop_nulls().collect()
        )
    ev_all = pl.concat(frames) if frames else pl.DataFrame()
    counts = (
        ev_all.group_by("player_id")
        .agg([pl.len().alias("n"),
              pl.col("pos_t").mode().first().alias("modal_pos")])
        .filter(pl.col("n") >= min_events)
    )
    keep = {int(r["player_id"]): (str(r["modal_pos"]) if r["modal_pos"] else None,
                                  int(r["n"]))
            for r in counts.iter_rows(named=True)}
    pidx = assets.tokenizer._player_index
    kept_ids: list[int] = []
    kept_local: list[int] = []
    for pid, local in pidx.items():
        if pid in keep:
            kept_ids.append(int(pid))
            kept_local.append(local)

    with torch.no_grad():
        idx = torch.arange(pe.n_players, device=assets.device)
        if mode == "full":
            content, delta = pe.forward_components(idx)
            vecs_all = (content + delta).detach().cpu().numpy()
        elif mode == "content":
            vecs_all = pe.forward_components(idx)[0].detach().cpu().numpy()
        elif mode == "delta":
            vecs_all = pe.forward_components(idx)[1].detach().cpu().numpy()
        else:
            raise ValueError(mode)
    action_mix_all = pe.player_action_mix.detach().cpu().numpy()

    kept_arr = np.asarray(kept_local)
    vecs = vecs_all[kept_arr]
    action_mix = action_mix_all[kept_arr]

    if per_family:
        fam_of = np.array(
            [POS_FAMILY.get(keep[pid][0] or "", "?") for pid in kept_ids]
        )
        labels = np.full(len(kept_ids), -1, dtype=int)
        next_id = 0
        algo = "hdbscan-per-family"
        for fam in sorted(set(fam_of)):
            if fam == "?":
                continue
            fam_mask = fam_of == fam
            fam_idx = np.where(fam_mask)[0]
            if len(fam_idx) < min_cluster_size:
                continue
            sub_labels, _algo = _cluster(vecs[fam_idx], min_cluster_size=min_cluster_size)
            for raw in sorted(set(sub_labels)):
                if raw == -1:
                    continue
                mask = sub_labels == raw
                labels[fam_idx[mask]] = next_id
                next_id += 1
    else:
        labels, algo = _cluster(vecs, min_cluster_size=min_cluster_size)

    # 2D projection for the map (over ALL kept players, not just clustered).
    proj = _project_2d(vecs)

    # Per-cluster description.
    names_df = player_name_lookup(versa_root)
    name_lut = {int(r["player_id"]): str(r["player_name"])
                for r in names_df.iter_rows(named=True)}
    action_families = list(
        assets.meta.get("player_metadata", {}).get("action_families", []) or
        [f"action_{i}" for i in range(action_mix_all.shape[1])]
    )
    clusters: list[dict] = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        mask = labels == c
        member_pids = [kept_ids[i] for i in np.where(mask)[0]]
        positions = [keep[pid][0] for pid in member_pids]
        families = [POS_FAMILY.get(p or "", "?") for p in positions]
        fam_counts = Counter(families)
        dom_family, dom_n = fam_counts.most_common(1)[0]
        purity = float(dom_n / len(member_pids))
        centroid = vecs[mask].mean(axis=0)
        action_centroid = action_mix[mask].mean(axis=0)
        top_idx = np.argsort(action_centroid)[-3:][::-1]
        top_actions = [
            {"family": action_families[i], "share": float(action_centroid[i])}
            for i in top_idx
        ]
        # Exemplars closest to centroid.
        member_idx = np.where(mask)[0]
        dists = np.linalg.norm(vecs[member_idx] - centroid, axis=1)
        order = np.argsort(dists)[:5]
        exemplars = []
        for i in order:
            pid = member_pids[i]
            exemplars.append({
                "player_id": int(pid),
                "name": name_lut.get(pid, str(pid)),
                "position": positions[i],
            })
        clusters.append({
            "cluster_id": int(c),
            "size": int(len(member_pids)),
            "dominant_family": dom_family,
            "family_purity": purity,
            "position_histogram": dict(Counter([p for p in positions if p])),
            "top_actions": top_actions,
            "exemplars": exemplars,
        })

    n_noise = int((labels == -1).sum())
    mean_purity = (
        float(np.mean([c["family_purity"] for c in clusters])) if clusters else 0.0
    )

    players_xy = []
    for i, pid in enumerate(kept_ids):
        players_xy.append({
            "player_id": int(pid),
            "name": name_lut.get(pid, str(pid)),
            "x": float(proj[i, 0]),
            "y": float(proj[i, 1]),
            "cluster_id": int(labels[i]),
            "family": POS_FAMILY.get(keep[pid][0] or "", "?"),
            "position": keep[pid][0],
        })

    return {
        "mode": mode,
        "per_family": per_family,
        "algorithm": algo,
        "min_cluster_size": min_cluster_size,
        "min_events": min_events,
        "n_players_clustered": len(kept_ids),
        "n_noise": n_noise,
        "n_clusters": len(clusters),
        "mean_family_purity": mean_purity,
        "clusters": clusters,
        "players_xy": players_xy,
    }
