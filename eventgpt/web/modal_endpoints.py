"""Modal HTTP endpoints for the eventGPT scouting web app.

Single ``@app.cls`` container (A10G + ``keep_warm=1``) loads the v5.2
checkpoint once on cold start, then services HTTP requests for the
Streamlit SPA. Each endpoint is a thin wrapper around the corresponding
``web.lib`` function.

Deploy:
    modal deploy src/eventgpt/web/modal_endpoints.py

The deployed URL is shown in the deploy output and can be passed to the
Streamlit app via the ``MODAL_URL`` env var. ``modal serve`` (live-reload)
also works for local-tunnel development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal


PYTHON_VERSION = "3.11"

# REPO_ROOT is /Users/hitesh/racer/deploy/ at deploy time, /app inside the
# Modal container. The eventgpt source tree sits at deploy/eventgpt/, the
# tokenizer config at deploy/configs/.
# parents: [0]=web, [1]=eventgpt, [2]=deploy.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    REPO_ROOT = Path("/app")

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy>=1.24", "polars>=1.0", "pyarrow>=15", "tqdm>=4.65",
        "pyyaml>=6", "pydantic>=2", "typer>=0.9", "structlog>=24.1",
        "scikit-learn>=1.4", "pandas>=2.0", "hdbscan>=0.8.33",
        "fastapi>=0.110", "umap-learn>=0.5",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(str(REPO_ROOT / "eventgpt"), remote_path="/app/eventgpt")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/app/configs")
)

app = modal.App("eventgpt-web", image=image)

data_volume = modal.Volume.from_name("eventgpt-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("eventgpt-ckpt", create_if_missing=True)
versa_volume = modal.Volume.from_name("eventgpt-versa", create_if_missing=True)


CKPT_PATH = "/ckpt/small-v5.2-1777047678/best.pt"
META_PATH = "/data/bin/meta.pkl"
CFG_PATH = "/app/configs/tokenizer.yaml"
VERSA_ROOT = "/versa/versa"


@app.cls(
    gpu="A10G",
    image=image,
    volumes={
        "/data": data_volume,
        "/ckpt": ckpt_volume,
        "/versa": versa_volume,
    },
    timeout=60 * 30,
    min_containers=1,            # keep one container hot for snappy first-byte
    scaledown_window=900,        # 15 min idle window before scale-down
)
@modal.concurrent(max_inputs=4)
class WebApi:
    """Endpoints share the loaded ``Assets`` across requests in the same
    container. Modal autoscales additional containers under load; each one
    pays the cold-start once."""

    @modal.enter()
    def load(self) -> None:
        import sys
        sys.path.insert(0, "/app")
        from eventgpt.cases._common import load_assets
        from pathlib import Path as _P
        self.assets = load_assets(
            ckpt=_P(CKPT_PATH),
            meta_path=_P(META_PATH),
            cfg_path=_P(CFG_PATH),
            device="cuda",
        )
        self.versa_root = _P(VERSA_ROOT)
        # Warm the per-process caches for autocomplete + team lookup.
        from eventgpt.web.lib.players import _scan_versa_player_facts, _scan_team_labels
        _scan_versa_player_facts(str(self.versa_root))
        _scan_team_labels(str(self.versa_root))
        print("[web] Assets loaded, caches warmed")

    @modal.fastapi_endpoint(method="GET")
    def players(self) -> dict:
        from eventgpt.web.lib.players import list_players
        return list_players(self.assets, self.versa_root)

    @modal.fastapi_endpoint(method="GET")
    def player_profile(self, player_id: int) -> dict:
        from eventgpt.web.lib.players import player_profile
        return player_profile(self.assets, self.versa_root, int(player_id))

    _baselines_cache: dict | None = None

    @modal.fastapi_endpoint(method="GET")
    def baselines(self) -> dict:
        if self._baselines_cache is not None:
            return self._baselines_cache
        from eventgpt.web.lib.players import _compute_baselines
        self._baselines_cache = _compute_baselines(self.assets, self.versa_root)
        return self._baselines_cache

    @modal.fastapi_endpoint(method="POST")
    def search_replacements(self, payload: dict) -> dict:
        from eventgpt.web.lib.search import search_replacements
        return search_replacements(
            self.assets, self.versa_root,
            query_player_id=int(payload["query_player_id"]),
            top_k=int(payload.get("top_k", 20)),
            mode=str(payload.get("mode", "full")),
            same_family=bool(payload.get("same_family", False)),
            same_position=bool(payload.get("same_position", False)),
            in_team_id=payload.get("in_team_id"),
            not_in_team_id=payload.get("not_in_team_id"),
            min_events=int(payload.get("min_events", 500)),
        )

    @modal.fastapi_endpoint(method="GET")
    def teams(self) -> dict:
        from eventgpt.web.lib.team import list_teams
        return list_teams(self.assets, self.versa_root)

    @modal.fastapi_endpoint(method="POST")
    def team_fit(self, payload: dict) -> dict:
        from eventgpt.web.lib.team import team_fit as _team_fit
        return _team_fit(
            self.assets, self.versa_root,
            candidate_player_id=int(payload["candidate_player_id"]),
            team_id=int(payload["team_id"]),
            min_team_events=int(payload.get("min_team_events", 5000)),
        )

    # Cache the archetype payload at the class level — clustering is the
    # single most expensive call here (~30 s) and the result is invariant
    # for a fixed checkpoint.
    _archetype_cache: Optional[dict] = None

    @modal.fastapi_endpoint(method="GET")
    def archetypes(self) -> dict:
        if self._archetype_cache is not None:
            return self._archetype_cache
        from eventgpt.web.lib.archetype import compute_archetypes
        self._archetype_cache = compute_archetypes(
            self.assets, self.versa_root,
            mode="full", min_events=1000, min_cluster_size=5, per_family=True,
        )
        return self._archetype_cache

    @modal.fastapi_endpoint(method="POST")
    def swap_impact(self, payload: dict) -> dict:
        from eventgpt.web.lib.swap import swap_impact as _swap
        return _swap(
            self.assets, self.versa_root,
            incumbent_player_id=int(payload["incumbent_player_id"]),
            candidate_player_id=(int(payload["candidate_player_id"])
                                 if payload.get("candidate_player_id") is not None else None),
            n_peers=int(payload.get("n_peers", 5)),
            season=str(payload.get("season", "23-24")),
            max_episodes=int(payload.get("max_episodes", 80)),
        )
