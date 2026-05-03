"""Thin httpx client for the Modal HTTP endpoints.

Modal exposes each ``@modal.fastapi_endpoint`` as a separate URL of the form
``<MODAL_URL>-<endpoint-name>.modal.run``. We assume ``MODAL_URL`` is the
common stem (everything up to the endpoint suffix) and append ``-{name}`` per
call. If your deployment uses a single mounted ASGI app instead, set
``MODAL_URL`` to its base URL and ``MODAL_ENDPOINT_STYLE=path`` and we'll
join with ``/{name}`` instead.

Light caching:
  * ``GET /players``  is invariant for the session — cached in-process.
  * ``GET /teams``    same.
  * ``GET /archetypes`` same — also expensive server-side, so the cache helps.
  * ``GET /player_profile`` per-id LRU.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import httpx

from app.config import load_config


def _endpoint_url(name: str) -> str:
    """Build the per-endpoint URL.

    ``MODAL_URL`` should be the *stem* — i.e. the part shared by every
    endpoint (e.g. ``https://hiteshgautam026--eventgpt-web-webapi``). Each
    endpoint name is converted to kebab-case (Modal's convention: method
    names with underscores are hyphenated in the URL) and appended.
    """
    cfg = load_config()
    if not cfg.has_modal:
        raise RuntimeError(
            "MODAL_URL is not set. Run `modal deploy eventgpt/web/modal_endpoints.py` "
            "in this repo and put the printed URL stem into .env."
        )
    kebab = name.replace("_", "-")
    style = os.environ.get("MODAL_ENDPOINT_STYLE", "subdomain").lower()
    base = cfg.modal_url
    # `modal serve` URLs have a `-dev` suffix between the endpoint name and
    # `.modal.run`. Set MODAL_DEV=1 in .env when pointing at a serve session.
    dev_suffix = "-dev" if os.environ.get("MODAL_DEV", "").lower() in ("1", "true", "yes") else ""
    if style == "path":
        return f"{base}/{kebab}"
    if base.endswith(".modal.run") or ".modal.run/" in base:
        return base.replace(".modal.run", f"-{kebab}{dev_suffix}.modal.run", 1)
    return f"{base}-{kebab}{dev_suffix}.modal.run"


def _client() -> httpx.Client:
    cfg = load_config()
    return httpx.Client(timeout=cfg.request_timeout_s)


def _get(name: str, **params: Any) -> dict:
    url = _endpoint_url(name)
    with _client() as client:
        r = client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def _post(name: str, payload: dict) -> dict:
    url = _endpoint_url(name)
    with _client() as client:
        r = client.post(url, json=payload)
    r.raise_for_status()
    return r.json()


# --- Cached read endpoints -------------------------------------------------


@lru_cache(maxsize=1)
def list_players() -> dict:
    return _get("players")


@lru_cache(maxsize=1)
def list_teams() -> dict:
    return _get("teams")


@lru_cache(maxsize=1)
def archetypes() -> dict:
    return _get("archetypes")


@lru_cache(maxsize=1)
def baselines() -> dict:
    return _get("baselines")


@lru_cache(maxsize=64)
def player_profile(player_id: int) -> dict:
    return _get("player_profile", player_id=player_id)


# --- Mutating / non-cached endpoints --------------------------------------


def search_replacements(
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
    return _post("search_replacements", {
        "query_player_id": query_player_id,
        "top_k": top_k,
        "mode": mode,
        "same_family": same_family,
        "same_position": same_position,
        "in_team_id": in_team_id,
        "not_in_team_id": not_in_team_id,
        "min_events": min_events,
    })


def team_fit(*, candidate_player_id: int, team_id: int) -> dict:
    return _post("team_fit", {
        "candidate_player_id": candidate_player_id,
        "team_id": team_id,
    })


def swap_impact(
    *,
    incumbent_player_id: int,
    candidate_player_id: int | None = None,
    n_peers: int = 5,
    season: str = "23-24",
    max_episodes: int = 80,
) -> dict:
    return _post("swap_impact", {
        "incumbent_player_id": incumbent_player_id,
        "candidate_player_id": candidate_player_id,
        "n_peers": n_peers,
        "season": season,
        "max_episodes": max_episodes,
    })


def clear_caches() -> None:
    """Useful when switching MODAL_URL during development."""
    list_players.cache_clear()
    list_teams.cache_clear()
    archetypes.cache_clear()
    baselines.cache_clear()
    player_profile.cache_clear()
