"""EventGPT Scouting Engine — single-page Streamlit app.

Layout:
  ┌─ Sidebar (always visible) ─────────────────────────────────────────┐
  │  Mode toggle: Scout / Strategy                                      │
  │  Player search (autocomplete from /players)                         │
  │  Active player card (sticky)                                        │
  │  Service status footer                                              │
  └────────────────────────────────────────────────────────────────────┘
  ┌─ Main panel ───────────────────────────────────────────────────────┐
  │  Mode-specific tabs (see app/views/scout.py and strategy.py)        │
  └────────────────────────────────────────────────────────────────────┘

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.config import APP_TAGLINE, APP_TITLE, load_config
from app.services import modal_client
from app.views import scout as scout_view


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Session-state defaults -----------------------------------------

st.session_state.setdefault("active_player_id", None)
st.session_state.setdefault("mode", "Scout")


# ---------- Header ---------------------------------------------------------

cfg = load_config()
st.title(APP_TITLE)
st.caption(APP_TAGLINE)


# ---------- Sidebar --------------------------------------------------------

with st.sidebar:
    st.header("Player selection")
    st.session_state["mode"] = "Scout"

    if not cfg.has_modal:
        st.error(
            "MODAL_URL is not set. Deploy the model service first:\n\n"
            "```\nmodal deploy eventgpt/web/modal_endpoints.py\n```\n\n"
            "Then put the URL into `deploy/.env`."
        )
        st.stop()

    # Player search.
    with st.spinner("Loading the player pool…"):
        try:
            players = modal_client.list_players()
        except Exception as e:
            st.error(f"Couldn't reach Modal: {e}")
            st.stop()
    roster = players.get("players", [])
    if not roster:
        st.warning("No players returned from /players.")
        st.stop()

    # Build label → id map (limit to top-N events for the dropdown to stay snappy).
    top = sorted(roster, key=lambda r: -r.get("n_events", 0))[:600]
    label_to_id = {
        f"{r['name']} — {(r.get('team_label') or '—')} ({(r.get('position') or '?').replace('_',' ').title()})": r["player_id"]
        for r in top
    }
    options = [""] + list(label_to_id.keys())
    # Default selection: keep current active player if still in list.
    current_label = next(
        (k for k, v in label_to_id.items() if v == st.session_state["active_player_id"]),
        "",
    )
    selected = st.selectbox(
        "Search by name", options,
        index=options.index(current_label) if current_label in options else 0,
        help="Top 600 players by minutes on the ball. Start typing to filter.",
    )
    if selected and selected in label_to_id:
        st.session_state["active_player_id"] = label_to_id[selected]

    st.divider()
    if st.session_state["active_player_id"]:
        try:
            profile = modal_client.player_profile(st.session_state["active_player_id"])
            st.markdown(f"**Now scouting:** {profile['name']}")
            st.caption(
                f"{(profile.get('position') or '?').replace('_',' ').title()}"
                f" · {profile.get('team_label') or '—'}"
            )
        except Exception as e:
            st.warning(f"Profile load failed: {e}")
    else:
        st.caption("Choose a player to start scouting.")

    st.divider()
    st.markdown(
        "Presented by **[Hitesh](https://www.linkedin.com/in/hiteshgautam026/)**.  \n"
        "Designing a proprietary transformer-driven scouting system powered by "
        "event-level data from the 2020–2025 Premier League seasons.",
        unsafe_allow_html=False,
    )


# ---------- Main panel -----------------------------------------------------

scout_view.render(st.session_state["active_player_id"])
