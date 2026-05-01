"""Player roster + per-player metadata, served as the web app's ``/players``
endpoint and used by all other probes for autocomplete + display.

The output rows are the smallest shape the UI needs: enough to populate a
search dropdown, render a player card, and preselect filters. Heavy fields
(action_mix, spatial_zone) are *not* included here — they're served by
``/player_profile`` so listing 1200 players stays cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

from eventgpt.cases._common import Assets, player_name_lookup


POS_FAMILY = {
    "GOALKEEPER": "GK",
    "CENTRAL_DEFENDER": "DEF",
    "LEFT_WINGBACK_DEFENDER": "DEF",
    "RIGHT_WINGBACK_DEFENDER": "DEF",
    "DEFENSE_MIDFIELD": "MID",
    "CENTRAL_MIDFIELD": "MID",
    "ATTACKING_MIDFIELD": "ATT",
    "LEFT_WINGER": "ATT",
    "RIGHT_WINGER": "ATT",
    "CENTER_FORWARD": "ATT",
}


# Curated mapping VERSA team_id -> human team name. VERSA does not ship team
# names anywhere in its parquet schema (only integer team_ids), so this is
# derived once by inspecting each team's top-3 most-recent-season players
# and labelling them by hand. Any team_id not in this map falls back to a
# surname-trio heuristic at the end of ``_scan_team_labels``.
TEAM_ID_TO_NAME: dict[int, str] = {
    233: "Arsenal",
    238: "Tottenham Hotspur",
    240: "Manchester City",
    250: "Leicester City",
    252: "Manchester United",
    293: "Sunderland",
    294: "Burnley",
    295: "Chelsea",
    296: "Watford",
    297: "Liverpool",
    298: "Southampton",
    299: "Everton",
    300: "Crystal Palace",
    302: "Bournemouth",
    303: "West Ham United",
    867: "Ipswich Town",
    872: "Newcastle United",
    874: "Nottingham Forest",
    883: "Brentford",
    888: "Wolverhampton",
    889: "Leeds United",
    894: "Sheffield United",
    912: "Aston Villa",
    917: "Fulham",
    928: "Norwich City",
    932: "Luton Town",
    961: "Brighton & Hove Albion",
}


@lru_cache(maxsize=1)
def _scan_versa_player_facts(versa_root_str: str) -> tuple:
    """One-pass scan over VERSA event parquets returning per-player:
    modal position, **most-recent-season team_id**, total event count.

    Uses *most-recent* season team rather than lifetime-modal team — a player
    who spent 22-25 at Leicester and moved to Bournemouth in 25-26 should be
    labelled at Bournemouth, not Leicester. Lifetime-modal otherwise stamps
    relegated/promoted teams' lineups onto every transferred player.
    """
    versa_root = Path(versa_root_str)
    frames = []
    for season_dir in sorted((versa_root / "events").glob("season=*")):
        season = season_dir.name.removeprefix("season=")
        cols = pl.scan_parquet(str(season_dir / "events.parquet")).collect_schema().names()
        select = [pl.col("player_id"), pl.col("pos_t").alias("pos")]
        if "team_id" in cols:
            select.append(pl.col("team_id"))
        else:
            select.append(pl.lit(None).alias("team_id"))
        df = (pl.scan_parquet(str(season_dir / "events.parquet"))
              .select(select).drop_nulls(subset=["player_id"]).collect()
              .with_columns(pl.lit(season).alias("season")))
        frames.append(df)
    if not frames:
        return {}, {}, {}
    full = pl.concat(frames)

    pos = (
        full.drop_nulls(subset=["pos"])
        .group_by("player_id").agg(pl.col("pos").mode().first().alias("pos"))
    )
    counts = full.group_by("player_id").agg(pl.len().alias("n_events"))

    # Most-recent season team_id: per player, take the modal team_id of the
    # latest season they appear in (handles loanees with cameo events too).
    with_team = full.drop_nulls(subset=["team_id"])
    latest = (with_team.group_by("player_id")
              .agg(pl.col("season").max().alias("latest_season")))
    recent = with_team.join(latest, on="player_id").filter(
        pl.col("season") == pl.col("latest_season"),
    )
    team = (recent.group_by("player_id")
            .agg(pl.col("team_id").mode().first().alias("team_id")))

    pos_map = {int(r["player_id"]): str(r["pos"]) if r["pos"] else None
               for r in pos.iter_rows(named=True)}
    team_map = {int(r["player_id"]): int(r["team_id"]) if r["team_id"] is not None else None
                for r in team.iter_rows(named=True)}
    count_map = {int(r["player_id"]): int(r["n_events"]) for r in counts.iter_rows(named=True)}
    return pos_map, team_map, count_map


@lru_cache(maxsize=1)
def _scan_team_labels(versa_root_str: str) -> dict[int, str]:
    """team_id → human team name (e.g. "Arsenal", "Manchester City").

    Primary source: the curated ``TEAM_ID_TO_NAME`` map at the top of this
    module. Any team_id without a curated entry falls back to the surname-
    trio of its three most-frequent players in the most-recent season they
    played for that team — better than nothing for unrecognised IDs.
    """
    versa_root = Path(versa_root_str)
    out: dict[int, str] = dict(TEAM_ID_TO_NAME)

    pos_map, team_map, count_map = _scan_versa_player_facts(versa_root_str)
    unknown = {tid for tid in set(team_map.values()) if tid is not None and tid not in out}
    if not unknown:
        return out

    names = player_name_lookup(versa_root)
    name_lut = {int(r["player_id"]): str(r["player_name"])
                for r in names.iter_rows(named=True)}
    by_team: dict[int, list[tuple[int, int]]] = {}
    for pid, tid in team_map.items():
        if tid in unknown:
            by_team.setdefault(tid, []).append((pid, count_map.get(pid, 0)))
    for tid, members in by_team.items():
        members.sort(key=lambda t: -t[1])
        surnames: list[str] = []
        for pid, _n in members[:3]:
            full = name_lut.get(pid)
            if full:
                surnames.append(full.split()[-1])
        out[tid] = "/".join(surnames) if surnames else f"team_{tid}"
    return out


def list_players(assets: Assets, versa_root: Path) -> dict:
    """Return roster summary for the UI dropdown.

    Each entry has just enough for the autocomplete + active-player card:
    ``{player_id, name, position, family, team_id, team_label, n_events}``.
    Players outside the tokenizer vocab (cold-start tail) are skipped.
    """
    versa_root_str = str(versa_root)
    names = player_name_lookup(versa_root)
    name_lut = {int(r["player_id"]): str(r["player_name"])
                for r in names.iter_rows(named=True)}
    pos_map, team_map, count_map = _scan_versa_player_facts(versa_root_str)
    team_labels = _scan_team_labels(versa_root_str)

    rows: list[dict] = []
    for pid, _local in assets.tokenizer._player_index.items():
        pid_i = int(pid)
        pos = pos_map.get(pid_i)
        team_id = team_map.get(pid_i)
        rows.append({
            "player_id": pid_i,
            "name": name_lut.get(pid_i, str(pid_i)),
            "position": pos,
            "family": POS_FAMILY.get(pos or "", "?"),
            "team_id": team_id,
            "team_label": team_labels.get(team_id) if team_id is not None else None,
            "n_events": count_map.get(pid_i, 0),
        })
    rows.sort(key=lambda r: -r["n_events"])
    return {"n_players": len(rows), "players": rows, "teams": [
        {"team_id": int(tid), "label": label}
        for tid, label in sorted(team_labels.items(), key=lambda kv: kv[1])
    ]}


@lru_cache(maxsize=1)
def baselines_by_family(versa_root_str: str, assets_id: int) -> dict:
    """Per-positional-family mean action_mix and mean spatial_zone.

    The radar in the UI compares a player to OTHERS IN THE SAME ROLE FAMILY
    (CBs vs CBs, wingers vs wingers) — comparing a CB's aerial share to the
    league-wide average is misleading because CBs do far more aerial work
    than the average player. ``assets_id`` is just an LRU cache key so the
    function rebuilds when the loaded checkpoint changes.

    Output: ``{family: {action_families, action_mix_mean: [...],
    spatial_zone_mean: [...], n_players}}``.
    """
    # We need access to the assets (the action_mix lives on PlayerEmbedding
    # buffers), so this function is wrapped at endpoint time with a closure
    # that injects assets. Implementation lives in ``_compute_baselines`` below.
    raise NotImplementedError("call _compute_baselines via the endpoint")


def _compute_baselines(assets: Assets, versa_root: Path) -> dict:
    """Actual baseline computation. Cached at the endpoint layer."""
    versa_root_str = str(versa_root)
    pos_map, _team_map, count_map = _scan_versa_player_facts(versa_root_str)
    pe = assets.model.player_emb
    if pe is None:
        return {}
    action_mix = pe.player_action_mix.detach().cpu().numpy()
    spatial = pe.player_spatial_zone.detach().cpu().numpy()
    action_families = list(
        assets.meta.get("player_metadata", {}).get("action_families", []) or
        [f"action_{i}" for i in range(action_mix.shape[1])]
    )
    by_family: dict[str, list[int]] = {}
    pidx = assets.tokenizer._player_index
    for pid, local in pidx.items():
        fam = POS_FAMILY.get(pos_map.get(int(pid)) or "", "?")
        if fam == "?":
            continue
        # Skip cold-start players whose action_mix sums to 0.
        if float(action_mix[local].sum()) < 0.5:
            continue
        # Drop very-low-events tail to keep the baseline robust.
        if count_map.get(int(pid), 0) < 500:
            continue
        by_family.setdefault(fam, []).append(local)
    out: dict[str, dict] = {}
    for fam, locs in by_family.items():
        if not locs:
            continue
        amix = action_mix[locs].mean(axis=0)
        amix_std = action_mix[locs].std(axis=0)
        spz = spatial[locs].mean(axis=0)
        out[fam] = {
            "action_families": action_families,
            "action_mix_mean": [float(v) for v in amix.tolist()],
            "action_mix_std": [float(v) for v in amix_std.tolist()],
            "spatial_zone_mean": [float(v) for v in spz.tolist()],
            "n_players": int(len(locs)),
        }
    return out


def player_profile(assets: Assets, versa_root: Path, player_id: int) -> dict:
    """Per-player profile for the Scout-Profile tab. Returns the structured
    style fingerprint the UI components consume directly:
    ``{player_id, name, position, family, team_id, team_label, n_events,
       action_mix: {family: pct}, spatial_zone: 16-vec, pos_entropy}``.
    """
    if player_id not in assets.tokenizer._player_index:
        raise KeyError(f"player_id {player_id} not in tokenizer vocab")
    local = assets.tokenizer._player_index[player_id]
    versa_root_str = str(versa_root)
    pos_map, team_map, count_map = _scan_versa_player_facts(versa_root_str)
    team_labels = _scan_team_labels(versa_root_str)
    names = player_name_lookup(versa_root)
    name_row = names.filter(pl.col("player_id") == player_id)["player_name"]
    name = str(name_row[0]) if len(name_row) else str(player_id)
    pos = pos_map.get(player_id)
    team_id = team_map.get(player_id)
    pe = assets.model.player_emb
    if pe is None:
        raise RuntimeError("This checkpoint has no PlayerEmbedding — UI requires use_content_player_emb=True.")

    action_families = list(
        assets.meta.get("player_metadata", {}).get("action_families", []) or
        ["Pass", "Carry", "Cross", "TakeOn", "Shot", "Tackle",
         "Interception", "Clearance", "Aerial", "Duel", "Other"]
    )
    action_mix = pe.player_action_mix[local].detach().cpu().tolist()
    spatial = pe.player_spatial_zone[local].detach().cpu().tolist()
    entropy = float(pe.player_pos_entropy[local].detach().cpu())
    return {
        "player_id": int(player_id),
        "name": name,
        "position": pos,
        "family": POS_FAMILY.get(pos or "", "?"),
        "team_id": int(team_id) if team_id is not None else None,
        "team_label": team_labels.get(team_id) if team_id is not None else None,
        "n_events": int(count_map.get(player_id, 0)),
        "action_families": action_families,
        "action_mix": {fam: float(v) for fam, v in zip(action_families, action_mix)},
        "spatial_zone": [float(v) for v in spatial],
        "pos_entropy": entropy,
    }
