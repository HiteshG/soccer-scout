"""Load VERSA Parquet into per-episode (context, events) tuples.

Drives both ``prepare.py`` (fit quantile edges + dump bin files) and the
counterfactual case-study scripts (substitute a player id, re-encode the same
episode, rerun the model).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl


VERSA_EVENT_COLUMNS: tuple[str, ...] = (
    "match_id", "event_id", "episode_id",
    "period", "minute", "gameTimeInSec", "season",
    "acting_team_side", "team_id", "player_id", "player_name",
    "action_type", "outcome",
    "x_start", "y_start", "delta_t",
    "rOBV_off", "rOBV_def",
    "gs_flag", "gc_flag",
)

# episode_context sidecar was emitted by versa_etl before the public rename,
# so it still uses ``matchId``. Events parquet uses ``match_id``. Normalise at
# read time.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "matchId", "episode_id",
    "start_period", "start_gameTimeInSec",
    "home_goals_at_start", "away_goals_at_start",
    "home_yellow_at_start", "away_yellow_at_start",
    "home_red_at_start", "away_red_at_start",
    "on_pitch_home", "on_pitch_away",
    "formation_home", "formation_away",
)


@dataclass
class EpisodeSample:
    match_id: int
    episode_id: int
    season: str
    context: dict
    events: list[dict]   # each has h_t, e_t, x, y, delta_t, o_t, rOBV, player_id


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def versa_events_path(root: str | Path, season: str) -> Path:
    return Path(root) / "events" / f"season={season}" / "events.parquet"


def versa_episode_ctx_path(root: str | Path, season: str) -> Path:
    return Path(root) / "episode_context" / f"season={season}" / "episode_context.parquet"


# ---------------------------------------------------------------------------
# Episode iterator
# ---------------------------------------------------------------------------


def _total_minute(start_period: int, start_game_sec: float) -> int:
    """Convert Impect's offset-encoded gameTimeInSec to total match minute.

    Offsets: H1=0, H2=10000, ET1=20000, ET2=30000 — seconds since period
    kickoff. Each period runs 45 min (mild inaccuracy during stoppage time).
    """
    offsets = {1: 0.0, 2: 10000.0, 3: 20000.0, 4: 30000.0}
    p = int(start_period or 1)
    secs = float(start_game_sec or 0.0) - offsets.get(p, 0.0)
    in_period_min = max(0, int(secs // 60))
    total = (p - 1) * 45 + in_period_min
    return min(total, 120)


def iter_episodes_in_season(
    versa_root: str | Path,
    season: str,
    match_filter: Iterable[int] | None = None,
) -> Iterator[EpisodeSample]:
    """Yield one ``EpisodeSample`` per (match_id, episode_id) in the season.

    Joins events.parquet with episode_context.parquet on (match_id, episode_id),
    sorts by ``eventNumber`` within an episode, and caps at
    ``max_events_per_episode`` downstream (tokenizer handles the cap).
    """
    events_path = versa_events_path(versa_root, season)
    ctx_path = versa_episode_ctx_path(versa_root, season)
    if not events_path.exists() or not ctx_path.exists():
        raise FileNotFoundError(f"Missing VERSA outputs for season {season}")

    ctx = (
        pl.scan_parquet(str(ctx_path))
        .select(list(CONTEXT_COLUMNS))
        .rename({"matchId": "match_id"})
        .collect()
    )
    events_cols = list(VERSA_EVENT_COLUMNS) + ["eventNumber"]
    events = (
        pl.scan_parquet(str(events_path))
        .select([c for c in events_cols if c])
        .collect()
    )
    if match_filter is not None:
        keep = list(int(m) for m in match_filter)
        ctx = ctx.filter(pl.col("match_id").is_in(keep))
        events = events.filter(pl.col("match_id").is_in(keep))

    # Group events by (match_id, episode_id) once.
    ev_by_key: dict[tuple[int, int], list[dict]] = {}
    ev_sorted = events.sort(["match_id", "episode_id", "eventNumber"])
    for row in ev_sorted.iter_rows(named=True):
        key = (int(row["match_id"]), int(row["episode_id"]))
        ev_by_key.setdefault(key, []).append(row)

    for ctx_row in ctx.iter_rows(named=True):
        mid, eid = int(ctx_row["match_id"]), int(ctx_row["episode_id"])
        events_in_episode = ev_by_key.get((mid, eid), [])
        if not events_in_episode:
            continue
        on_pitch = list(ctx_row.get("on_pitch_home") or []) + list(ctx_row.get("on_pitch_away") or [])
        # Compose structured context.
        context = {
            "on_pitch_ids": on_pitch,
            "minute": _total_minute(ctx_row["start_period"], ctx_row["start_gameTimeInSec"]),
            "h_g": int(ctx_row["home_goals_at_start"] or 0),
            "a_g": int(ctx_row["away_goals_at_start"] or 0),
            "h_r": int(ctx_row["home_red_at_start"] or 0),
            "a_r": int(ctx_row["away_red_at_start"] or 0),
            "h_y": int(ctx_row["home_yellow_at_start"] or 0),
            "a_y": int(ctx_row["away_yellow_at_start"] or 0),
            "formation_home": ctx_row.get("formation_home"),
            "formation_away": ctx_row.get("formation_away"),
        }
        # Compose structured events.
        ev_list: list[dict] = []
        for e in events_in_episode:
            robv_off = e.get("rOBV_off") or 0.0
            robv_def = e.get("rOBV_def") or 0.0
            # Paper uses a single rOBV signal per event. We fuse offensive and
            # (negative) defensive into one scalar so attackers and defenders
            # both have a meaningful target.
            rOBV = float(robv_off) - float(robv_def)
            ev_list.append({
                "h_t": int(e.get("acting_team_side") or 0),
                "e_t": str(e.get("action_type") or "UNMAPPED"),
                "x": float(e.get("x_start") or 0.0),
                "y": float(e.get("y_start") or 0.0),
                "delta_t": float(e.get("delta_t") or 0.0),
                "o_t": e.get("outcome"),
                "rOBV": rOBV,
                "player_id": e.get("player_id"),
                "event_id": int(e.get("event_id") or 0),
                "team_id": e.get("team_id"),
            })

        yield EpisodeSample(
            match_id=mid,
            episode_id=eid,
            season=str(ctx_row.get("match_id") and season),
            context=context,
            events=ev_list,
        )


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def split_matches_by_chrono(
    versa_root: str | Path,
    seasons: list[str],
    train_full_seasons: list[str],
    half_split_seasons: list[str],
    val_tail_pct: float = 0.0,
    val_train_pct: float = 0.05,
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
    """Chronological split aligned with EventGPT §3.1 + ScoutGPT §4.1.

    - ``train`` = every match in ``train_full_seasons`` + first half (chrono)
      of every season in ``half_split_seasons``.
    - ``val`` = the second half of the **first** ``half_split_season``
      (chronologically adjacent to train — the cleanest held-out block the
      paper describes).
    - ``test`` = the second halves of **all subsequent** half_split seasons
      (truly future data, saved for the end-of-project evaluation).
    - ``val_train`` = random ``val_train_pct`` of train matches removed from
      train and tracked separately. An in-distribution sibling of ``val`` used
      to disambiguate overfit (val_train loss rises) from distribution shift
      (val_train stays low, val rises).

    The legacy ``val_tail_pct`` behaviour (carve val out of the tail of train)
    is preserved behind the kwarg, but defaults to 0 now because it caused a
    severe recency-bias artifact — the tail was mostly first-half-of-25-26,
    i.e. the newest matches, so val was always maximally distribution-shifted
    vs. train. If a caller sets ``val_tail_pct > 0`` it takes precedence and
    the per-season val assignment below is skipped.
    """
    split: dict[str, list[tuple[str, int]]] = {
        "train": [], "val": [], "val_train": [], "test": [],
    }

    for season in seasons:
        ev_path = versa_events_path(versa_root, season)
        if not ev_path.exists():
            continue
        lf = pl.scan_parquet(str(ev_path))
        schema_cols = lf.collect_schema().names()
        mid_col = "match_id" if "match_id" in schema_cols else "matchId"
        dt_col = "dateTime" if "dateTime" in schema_cols else "gameTimeInSec"
        # Sort matches by earliest timestamp = first event of season chronology.
        match_order = (
            lf.group_by(mid_col)
            .agg(pl.col(dt_col).min().alias("first_dt"))
            .sort("first_dt")
            .select(pl.col(mid_col).alias("match_id"))
            .collect()["match_id"]
            .to_list()
        )
        n = len(match_order)
        if season in train_full_seasons:
            split["train"].extend((season, int(m)) for m in match_order)
        elif season in half_split_seasons:
            half = n // 2
            split["train"].extend((season, int(m)) for m in match_order[:half])
            # Second halves fan out across val/test based on position in
            # half_split_seasons: earliest held-out season → val (closest in
            # time to train), later ones → test.
            second_half = match_order[half:]
            idx_in_split = half_split_seasons.index(season)
            if idx_in_split == 0 and val_tail_pct <= 0:
                split["val"].extend((season, int(m)) for m in second_half)
            else:
                split["test"].extend((season, int(m)) for m in second_half)

    # Legacy tail-carve path (only if explicitly requested).
    if val_tail_pct > 0 and split["train"]:
        n_val = max(1, int(len(split["train"]) * val_tail_pct))
        split["val"] = split["train"][-n_val:]
        split["train"] = split["train"][:-n_val]

    # Held-out train slice: randomly pull ``val_train_pct`` of train matches
    # into a separate split. This is our overfit-vs-drift oracle at eval time.
    if val_train_pct > 0 and split["train"]:
        n_valtrain = max(1, int(len(split["train"]) * val_train_pct))
        rng = random.Random(seed)
        idxs = list(range(len(split["train"])))
        rng.shuffle(idxs)
        held = set(idxs[:n_valtrain])
        split["val_train"] = [m for i, m in enumerate(split["train"]) if i in held]
        split["train"] = [m for i, m in enumerate(split["train"]) if i not in held]

    return split
