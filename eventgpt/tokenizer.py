"""Unified-vocab tokenizer for EventGPT.

Every attribute type (team side, event type, spatial bin, ΔT bin, outcome, rOBV
bin, minute, cumulative score/card counts, player id) occupies a distinct
contiguous ID range in one shared vocabulary of size 2048. A single
``nn.Embedding`` then spans the entire input, and the output head is
weight-tied to it.

Bin layout and ranges live in ``configs/tokenizer.yaml``. Quantile edges for
ΔT and rOBV are fitted at prepare time and written to ``meta.pkl`` alongside
the binary episode streams. Pitch bins and categorical vocabularies are static
(known at import time).

Per paper §2.1:
  * context c = (pID_{1:22}, minute, h_g', a_g', h_r', a_r', h_y', a_y') → 29 tokens
  * each event v_t = (h_t, e_t, x_t, y_t, δ_t, o_t, rOBV_t) → 7 tokens
  * sequence layout per episode: [ctx 29] + [event 7] * up to 100 = ≤ 729 tokens
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "tokenizer.yaml"


# ---------------------------------------------------------------------------
# Static data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Range:
    start: int
    end: int  # exclusive

    def size(self) -> int:
        return self.end - self.start

    def token_at(self, offset: int) -> int:
        if not (0 <= offset < self.size()):
            raise ValueError(f"offset {offset} outside range [0, {self.size()})")
        return self.start + offset

    def offset_of(self, token: int) -> int:
        if not (self.start <= token < self.end):
            raise ValueError(f"token {token} outside range [{self.start}, {self.end})")
        return token - self.start


@dataclass
class TokenizerConfig:
    vocab_size: int
    block_size: int
    context_len: int
    max_events_per_episode: int
    tokens_per_event: int
    ranges: dict[str, Range]
    specials: dict[str, int]
    event_types: list[str]
    outcomes: list[str]
    x_bins: int
    x_range: tuple[float, float]
    y_bins: int
    y_range: tuple[float, float]
    delta_t_bins: int
    rOBV_bins: int
    minute_bin_width: int
    caps: dict[str, int]
    score_diff_cap: int = 3            # clamp range [-cap, +cap] → 2*cap+1 bins
    time_remaining_bins: int = 20      # quantile — fitted at prepare time
    formations: list[str] = field(default_factory=list)  # v4: formation vocabulary

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> "TokenizerConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        ranges = {k: Range(v[0], v[1]) for k, v in d["ranges"].items()}
        return cls(
            vocab_size=int(d["vocab_size"]),
            block_size=int(d["block_size"]),
            context_len=int(d["context_len"]),
            max_events_per_episode=int(d["max_events_per_episode"]),
            tokens_per_event=int(d["tokens_per_event"]),
            ranges=ranges,
            specials={k: int(v) for k, v in d["specials"].items()},
            event_types=list(d["event_types"]),
            outcomes=list(d["outcomes"]),
            x_bins=int(d["x_bins"]),
            x_range=tuple(d["x_range"]),
            y_bins=int(d["y_bins"]),
            y_range=tuple(d["y_range"]),
            delta_t_bins=int(d["delta_t_bins"]),
            rOBV_bins=int(d["rOBV_bins"]),
            minute_bin_width=int(d["minute_bin_width"]),
            caps={k: int(v) for k, v in d["cap"].items()},
            score_diff_cap=int(d.get("score_diff_cap", 3)),
            time_remaining_bins=int(d.get("time_remaining_bins", 20)),
            formations=list(d.get("formations", [])),
        )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class Tokenizer:
    """Encode/decode EventGPT episodes to/from a flat uint16 token stream."""

    def __init__(
        self,
        cfg: TokenizerConfig,
        player_ids: list[int] | None = None,
        delta_t_edges: np.ndarray | None = None,
        rOBV_edges: np.ndarray | None = None,
        time_remaining_edges: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.player_ids: list[int] = list(player_ids or [])
        self._player_index: dict[int, int] = {
            pid: i for i, pid in enumerate(self.player_ids)
        }
        self.delta_t_edges = np.asarray(delta_t_edges) if delta_t_edges is not None else None
        self.rOBV_edges = np.asarray(rOBV_edges) if rOBV_edges is not None else None
        self.time_remaining_edges = (
            np.asarray(time_remaining_edges) if time_remaining_edges is not None else None
        )

        # Static bin edges for pitch coords — equal-width.
        self._x_edges = np.linspace(cfg.x_range[0], cfg.x_range[1], cfg.x_bins + 1)
        self._y_edges = np.linspace(cfg.y_range[0], cfg.y_range[1], cfg.y_bins + 1)

    # ---- Setters used by prepare.py after fitting on training data ----

    def set_player_vocab(self, player_ids: Iterable[int]) -> None:
        seen: list[int] = []
        seen_set: set[int] = set()
        for pid in player_ids:
            if pid is None or pid in seen_set:
                continue
            seen.append(int(pid))
            seen_set.add(int(pid))
        cap = self.cfg.ranges["players"].size()
        if len(seen) > cap:
            raise ValueError(
                f"Player vocab ({len(seen)}) exceeds reserved capacity ({cap}). "
                "Raise vocab_size or shrink an earlier range."
            )
        self.player_ids = seen
        self._player_index = {pid: i for i, pid in enumerate(self.player_ids)}

    def set_delta_t_edges(self, edges: np.ndarray) -> None:
        assert len(edges) == self.cfg.delta_t_bins + 1, (
            f"delta_t_edges must have {self.cfg.delta_t_bins + 1} entries"
        )
        self.delta_t_edges = np.asarray(edges, dtype=np.float64)

    def set_rOBV_edges(self, edges: np.ndarray) -> None:
        assert len(edges) == self.cfg.rOBV_bins + 1, (
            f"rOBV_edges must have {self.cfg.rOBV_bins + 1} entries"
        )
        self.rOBV_edges = np.asarray(edges, dtype=np.float64)

    def set_time_remaining_edges(self, edges: np.ndarray) -> None:
        assert len(edges) == self.cfg.time_remaining_bins + 1, (
            f"time_remaining_edges must have {self.cfg.time_remaining_bins + 1} entries"
        )
        self.time_remaining_edges = np.asarray(edges, dtype=np.float64)

    # ---- Token encoders (atomic) ----

    def _digitize(self, value: float, edges: np.ndarray, n_bins: int) -> int:
        """Clip and bucket a continuous value into [0, n_bins)."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        idx = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, n_bins - 1))
        return idx

    def _team_side_token(self, h_t: int | bool) -> int:
        return self.cfg.ranges["team_side"].token_at(1 if int(h_t) else 0)

    def _event_type_token(self, e_t: str) -> int:
        try:
            idx = self.cfg.event_types.index(e_t)
        except ValueError:
            idx = self.cfg.event_types.index("UNMAPPED")
        return self.cfg.ranges["event_type"].token_at(idx)

    def _x_token(self, x: float) -> int:
        idx = self._digitize(x, self._x_edges, self.cfg.x_bins)
        return self.cfg.ranges["x_bin"].token_at(idx)

    def _y_token(self, y: float) -> int:
        idx = self._digitize(y, self._y_edges, self.cfg.y_bins)
        return self.cfg.ranges["y_bin"].token_at(idx)

    def _delta_t_token(self, delta_t: float) -> int:
        if self.delta_t_edges is None:
            raise RuntimeError("delta_t_edges not fitted")
        idx = self._digitize(delta_t, self.delta_t_edges, self.cfg.delta_t_bins)
        return self.cfg.ranges["delta_t_bin"].token_at(idx)

    def _outcome_token(self, o_t: str | None) -> int:
        if o_t is None or o_t == "":
            idx = self.cfg.outcomes.index("NA")
        else:
            try:
                idx = self.cfg.outcomes.index(o_t)
            except ValueError:
                idx = self.cfg.outcomes.index("NA")
        return self.cfg.ranges["outcome"].token_at(idx)

    def _rOBV_token(self, rOBV: float) -> int:
        if self.rOBV_edges is None:
            raise RuntimeError("rOBV_edges not fitted")
        idx = self._digitize(rOBV, self.rOBV_edges, self.cfg.rOBV_bins)
        return self.cfg.ranges["rOBV_bin"].token_at(idx)

    def _minute_token(self, minute: int | float) -> int:
        m = int(minute or 0)
        idx = min(m // self.cfg.minute_bin_width, 24)
        return self.cfg.ranges["minute_bin"].token_at(idx)

    def _cap_token(self, range_key: str, value: int | None, cap_key: str) -> int:
        v = int(value or 0)
        v = max(0, min(v, self.cfg.caps[cap_key]))
        return self.cfg.ranges[range_key].token_at(v)

    def _player_token(self, pid: int | None) -> int:
        if pid is None:
            return self.cfg.specials["UNK_PLAYER"]
        idx = self._player_index.get(int(pid))
        if idx is None:
            return self.cfg.specials["UNK_PLAYER"]
        return self.cfg.ranges["players"].token_at(idx)

    def _score_diff_token(self, h_g: int | None, a_g: int | None) -> int:
        """Clamp (h_g - a_g) into [-cap, +cap], map to range offset [0, 2*cap]."""
        diff = int((h_g or 0) - (a_g or 0))
        cap = self.cfg.score_diff_cap
        diff = max(-cap, min(diff, cap))
        return self.cfg.ranges["score_diff"].token_at(diff + cap)

    def _time_remaining_token(self, minute: int | float | None) -> int:
        if self.time_remaining_edges is None:
            raise RuntimeError("time_remaining_edges not fitted")
        # Use 120-minute as feature; clamp negatives (extra time) to zero.
        t_rem = max(0.0, 120.0 - float(minute or 0))
        idx = self._digitize(t_rem, self.time_remaining_edges, self.cfg.time_remaining_bins)
        return self.cfg.ranges["time_remaining_bin"].token_at(idx)

    def _formation_token(self, formation: str | None) -> int:
        """Map formation string (e.g. '4-3-3') to a token. Unknown/missing
        → UNK_FORMATION slot (last formation vocab entry)."""
        if formation is None or formation == "" or not self.cfg.formations:
            # Fall back to UNK_FORMATION (conventionally last).
            idx = max(0, len(self.cfg.formations) - 1)
        else:
            try:
                idx = self.cfg.formations.index(str(formation))
            except ValueError:
                idx = max(0, len(self.cfg.formations) - 1)
        return self.cfg.ranges["formation"].token_at(idx)

    # ---- Encoders (public) ----

    def encode_context(
        self,
        on_pitch_ids: list[int | None],
        minute: int,
        h_g: int, a_g: int,
        h_r: int, a_r: int,
        h_y: int, a_y: int,
        formation_home: str | None = None,
        formation_away: str | None = None,
    ) -> np.ndarray:
        """Return ``context_len`` tokens for one episode.

        Layout (v4, context_len=33):
          0..21   22 on-pitch player IDs
          22      minute bin (5-min buckets)
          23..28  h_g, a_g, h_r, a_r, h_y, a_y (cumulative match state, capped)
          29      score_diff                  (v2+)
          30      time_remaining_bin          (v2+)
          31      formation_home              (v4+)
          32      formation_away              (v4+)
        """
        if len(on_pitch_ids) != 22:
            # Pad or truncate to 22 (paper assumes full pitch; we tolerate red-card gaps).
            on_pitch_ids = list(on_pitch_ids) + [None] * 22
            on_pitch_ids = on_pitch_ids[:22]
        tokens: list[int] = []
        tokens.extend(self._player_token(p) for p in on_pitch_ids)
        tokens.append(self._minute_token(minute))
        tokens.append(self._cap_token("h_g", h_g, "h_g"))
        tokens.append(self._cap_token("a_g", a_g, "a_g"))
        tokens.append(self._cap_token("h_r", h_r, "h_r"))
        tokens.append(self._cap_token("a_r", a_r, "a_r"))
        tokens.append(self._cap_token("h_y", h_y, "h_y"))
        tokens.append(self._cap_token("a_y", a_y, "a_y"))
        # v2 additions (guarded so v1 meta.pkl loads in 29-token mode).
        if "score_diff" in self.cfg.ranges:
            tokens.append(self._score_diff_token(h_g, a_g))
        if "time_remaining_bin" in self.cfg.ranges:
            tokens.append(self._time_remaining_token(minute))
        # v4 additions (guarded so v2/v3 meta.pkl load in 31-token mode).
        if "formation" in self.cfg.ranges:
            tokens.append(self._formation_token(formation_home))
            tokens.append(self._formation_token(formation_away))
        assert len(tokens) == self.cfg.context_len, (len(tokens), self.cfg.context_len)
        return np.asarray(tokens, dtype=np.uint16)

    def encode_event(
        self,
        h_t: int,
        e_t: str,
        x: float, y: float,
        delta_t: float,
        o_t: str | None,
        rOBV: float,
    ) -> np.ndarray:
        return np.asarray([
            self._team_side_token(h_t),
            self._event_type_token(e_t),
            self._x_token(x),
            self._y_token(y),
            self._delta_t_token(delta_t),
            self._outcome_token(o_t),
            self._rOBV_token(rOBV),
        ], dtype=np.uint16)

    def encode_episode(
        self,
        context: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> np.ndarray:
        """Pack one episode into a fixed-width ``block_size`` token array.

        Truncates to the most-recent ``max_events_per_episode`` events; pads to
        block_size with ``[PAD]``.
        """
        events = events[-self.cfg.max_events_per_episode:]
        ctx = self.encode_context(
            on_pitch_ids=list(context.get("on_pitch_ids") or []),
            minute=int(context.get("minute") or 0),
            h_g=int(context.get("h_g") or 0),
            a_g=int(context.get("a_g") or 0),
            h_r=int(context.get("h_r") or 0),
            a_r=int(context.get("a_r") or 0),
            h_y=int(context.get("h_y") or 0),
            a_y=int(context.get("a_y") or 0),
            formation_home=context.get("formation_home"),
            formation_away=context.get("formation_away"),
        )
        ev_tokens = [
            self.encode_event(
                h_t=int(ev.get("h_t") or 0),
                e_t=str(ev.get("e_t")),
                x=float(ev.get("x") or 0.0),
                y=float(ev.get("y") or 0.0),
                delta_t=float(ev.get("delta_t") or 0.0),
                o_t=ev.get("o_t"),
                rOBV=float(ev.get("rOBV") or 0.0),
            )
            for ev in events
        ]
        if ev_tokens:
            ev_arr = np.concatenate(ev_tokens)
        else:
            ev_arr = np.zeros((0,), dtype=np.uint16)
        tokens = np.concatenate([ctx, ev_arr]).astype(np.uint16)
        pad = self.cfg.specials["PAD"]
        if tokens.shape[0] < self.cfg.block_size:
            tokens = np.concatenate([
                tokens,
                np.full(self.cfg.block_size - tokens.shape[0], pad, dtype=np.uint16),
            ])
        else:
            tokens = tokens[: self.cfg.block_size]
        return tokens

    # ---- Decoders (for eval) ----

    def decode_event(self, tokens: np.ndarray) -> dict[str, Any]:
        """Inverse of encode_event. Continuous values recovered at bin centres."""
        assert len(tokens) == self.cfg.tokens_per_event
        t_h, t_e, t_x, t_y, t_dt, t_o, t_r = (int(t) for t in tokens)
        h_t = self.cfg.ranges["team_side"].offset_of(t_h)
        e_idx = self.cfg.ranges["event_type"].offset_of(t_e)
        x_idx = self.cfg.ranges["x_bin"].offset_of(t_x)
        y_idx = self.cfg.ranges["y_bin"].offset_of(t_y)
        dt_idx = self.cfg.ranges["delta_t_bin"].offset_of(t_dt)
        o_idx = self.cfg.ranges["outcome"].offset_of(t_o)
        r_idx = self.cfg.ranges["rOBV_bin"].offset_of(t_r)

        e_name = self.cfg.event_types[e_idx] if e_idx < len(self.cfg.event_types) else "UNMAPPED"
        o_name = self.cfg.outcomes[o_idx] if o_idx < len(self.cfg.outcomes) else "NA"

        x_center = 0.5 * (self._x_edges[x_idx] + self._x_edges[x_idx + 1])
        y_center = 0.5 * (self._y_edges[y_idx] + self._y_edges[y_idx + 1])
        dt_center = self._bin_center(self.delta_t_edges, dt_idx) if self.delta_t_edges is not None else 0.0
        r_center = self._bin_center(self.rOBV_edges, r_idx) if self.rOBV_edges is not None else 0.0

        return {
            "h_t": h_t,
            "e_t": e_name,
            "x": float(x_center),
            "y": float(y_center),
            "delta_t": float(dt_center),
            "o_t": o_name,
            "rOBV": float(r_center),
        }

    @staticmethod
    def _bin_center(edges: np.ndarray, idx: int) -> float:
        lo = edges[idx]
        hi = edges[min(idx + 1, len(edges) - 1)]
        # If edges are monotonic-non-strict (possible from quantile on ties), fall back to lo.
        if hi <= lo:
            return float(lo)
        return float(0.5 * (lo + hi))

    # ---- Meta persistence ----

    def save_meta(
        self,
        path: str | Path,
        player_metadata: dict | None = None,
    ) -> None:
        """Dump player vocab + bin edges + cfg path to a pickle alongside bin
        files. ``player_metadata`` is an optional dict (produced by
        ``prepare._derive_player_metadata``) used to seed the content-based
        PlayerEmbedding at train time; its keys are persisted verbatim.
        """
        payload = {
            "vocab_size": self.cfg.vocab_size,
            "block_size": self.cfg.block_size,
            "context_len": self.cfg.context_len,
            "tokens_per_event": self.cfg.tokens_per_event,
            "max_events_per_episode": self.cfg.max_events_per_episode,
            "ranges": {k: (r.start, r.end) for k, r in self.cfg.ranges.items()},
            "specials": dict(self.cfg.specials),
            "event_types": list(self.cfg.event_types),
            "outcomes": list(self.cfg.outcomes),
            "x_edges": self._x_edges.tolist(),
            "y_edges": self._y_edges.tolist(),
            "delta_t_edges": None if self.delta_t_edges is None else self.delta_t_edges.tolist(),
            "rOBV_edges": None if self.rOBV_edges is None else self.rOBV_edges.tolist(),
            "time_remaining_edges": (
                None if self.time_remaining_edges is None else self.time_remaining_edges.tolist()
            ),
            "formations": list(self.cfg.formations),
            "player_ids": list(self.player_ids),
        }
        if player_metadata is not None:
            payload["player_metadata"] = player_metadata
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_meta(cls, meta_path: str | Path, cfg_path: str | Path = DEFAULT_CONFIG) -> "Tokenizer":
        """Reconstruct a tokenizer matching the checkpoint that was trained on
        this meta.pkl. meta.pkl's stored layout (vocab_size, context_len, ranges,
        tokens_per_event, ...) takes precedence over configs/tokenizer.yaml,
        so v1 meta + v1 checkpoint keep working after the YAML has evolved.
        """
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        cfg = TokenizerConfig.from_yaml(cfg_path)
        # meta.pkl is authoritative for the layout that the checkpoint was
        # trained against. Only override fields that are actually present so
        # older meta.pkl versions missing some keys still work.
        if "vocab_size" in meta:
            cfg.vocab_size = int(meta["vocab_size"])
        if "block_size" in meta:
            cfg.block_size = int(meta["block_size"])
        if "context_len" in meta:
            cfg.context_len = int(meta["context_len"])
        if "tokens_per_event" in meta:
            cfg.tokens_per_event = int(meta["tokens_per_event"])
        if "max_events_per_episode" in meta:
            cfg.max_events_per_episode = int(meta["max_events_per_episode"])
        if "ranges" in meta:
            cfg.ranges = {k: Range(int(v[0]), int(v[1])) for k, v in meta["ranges"].items()}
        if "specials" in meta:
            cfg.specials = {k: int(v) for k, v in meta["specials"].items()}
        if "event_types" in meta:
            cfg.event_types = list(meta["event_types"])
        if "outcomes" in meta:
            cfg.outcomes = list(meta["outcomes"])
        if "formations" in meta:
            cfg.formations = list(meta["formations"])
        tok = cls(
            cfg=cfg,
            player_ids=meta.get("player_ids", []),
            delta_t_edges=np.asarray(meta["delta_t_edges"]) if meta.get("delta_t_edges") else None,
            rOBV_edges=np.asarray(meta["rOBV_edges"]) if meta.get("rOBV_edges") else None,
            time_remaining_edges=(
                np.asarray(meta["time_remaining_edges"]) if meta.get("time_remaining_edges") else None
            ),
        )
        return tok


def fit_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute ``n_bins + 1`` quantile edges clipped to the observed value range."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.linspace(-1.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, qs)
    # Break ties so searchsorted produces distinct buckets.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def fit_logscale_dt_edges(
    n_bins: int, min_val: float = 0.05, max_val: float = 120.0,
) -> np.ndarray:
    """Log-spaced edges for the Δt head. Replaces quantile binning because Δt
    is heavy-tailed (>60% of events at Δt < 3s, long tail to 90s+) and quantile
    binning wastes ~half the bins on rare tail values where a ±5 s prediction
    is meaningless. Log-scale allocates fine resolution in the dense 0–3 s
    region — that's where paper parity (MAE 1.11 s) lives — and wider bins in
    the tail where accuracy cost per second is low.

    Returns ``n_bins + 1`` edges starting at 0 with the last edge at +inf so
    out-of-range high Δt lands in the last bucket cleanly.

    With n_bins=40 and min_val=0.05, max_val=120 the resolution is:
      * first bin 0–0.05 s        (~0.05 s width)
      * 10th bin ~0.5 s window    (±0.25 s MAE floor)
      * 20th bin ~3 s window      (±1.5 s MAE floor)
      * 30th bin ~20 s window     (wide tail)
      * 40th bin 120 s → +inf     (open-ended)
    """
    if n_bins < 2:
        return np.linspace(0.0, max_val, n_bins + 1)
    inner = np.logspace(np.log10(min_val), np.log10(max_val), n_bins - 1)
    # Last edge 1.5 × max_val (finite) so the last bin's center stays in range
    # for _bin_center decode. Out-of-range high Δt values get clamped to this
    # last bin by searchsorted; accuracy cost is minimal because Δt > 120 s is
    # vanishingly rare (late-episode idle gap) and any prediction there barely
    # moves MAE.
    tail_cap = max_val * 1.5
    return np.concatenate([[0.0], inner, [tail_cap]])
