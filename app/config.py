"""Runtime configuration for the Streamlit app.

Reads from environment variables (.env via python-dotenv) so the app can
run end-to-end on a developer laptop with one ``modal deploy`` and one
``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # idempotent; safe to call from any module that imports config


@dataclass(frozen=True)
class AppConfig:
    modal_url: str
    openai_api_key: str | None
    openai_model: str          # heavy model for the headline reports
    openai_fast_model: str     # cheaper model for per-row peer phrases etc.
    request_timeout_s: float

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_modal(self) -> bool:
        return bool(self.modal_url)


def load_config() -> AppConfig:
    return AppConfig(
        modal_url=os.environ.get("MODAL_URL", "").rstrip("/"),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        openai_fast_model=os.environ.get("OPENAI_FAST_MODEL", "gpt-5.4-mini"),
        request_timeout_s=float(os.environ.get("MODAL_TIMEOUT_S", "120")),
    )


APP_TITLE = "Scouting Engine"

# Role-family accent colours, tuned for the dark forest-green theme so
# header bars sit comfortably on the deep background.
ROLE_ACCENTS: dict[str, dict[str, str]] = {
    "DEF": {"bg": "#103A2A", "fg": "#E8F0EC", "line": "#2EA37A"},
    "MID": {"bg": "#10362F", "fg": "#E8F0EC", "line": "#2E9F94"},
    "ATT": {"bg": "#1A3A2A", "fg": "#FFE8C7", "line": "#D6A658"},
    "GK":  {"bg": "#0E2E2C", "fg": "#CFEEEA", "line": "#3FB8B0"},
    "?":   {"bg": "#0F2C25", "fg": "#E8F0EC", "line": "#2EA37A"},
}

# Twelve-style brand palette — surface tints used by the section header
# bars, summary card, and strengths/weaknesses bars.
BRAND = {
    "surface":      "#0F2C25",   # card surface
    "surface_dim":  "#0A1F1A",   # page background
    "header_bar":   "#103A2A",   # section header band
    "accent":       "#2EA37A",   # active progress / primary line
    "accent_soft":  "#1F6F58",   # muted bar fill
    "warn":         "#D6A658",   # weakness amber
    "text":         "#E8F0EC",
    "text_dim":     "rgba(232,240,236,0.65)",
    "border":       "rgba(232,240,236,0.10)",
    "twelve_orange": "#FF5C2E",  # Twelve's signature accent on top-right wordmark
}


def role_accent(family: str | None) -> dict[str, str]:
    return ROLE_ACCENTS.get((family or "?").upper(), ROLE_ACCENTS["?"])
APP_TAGLINE = "Style-aware player search, fit scoring, and transfer reasoning."
APP_DESCRIPTION = (
    "A scouting and recruitment companion built on top of an event-level "
    "language model trained on five Premier-League seasons. The model learns "
    "what each player **does** on the pitch — passing, carrying, pressing, "
    "shooting, where they touch the ball — and uses that to find stylistic "
    "peers, score how a candidate would fit a target club, and answer "
    "what-if questions about transfers."
)
