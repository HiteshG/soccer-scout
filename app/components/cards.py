"""Card / verdict / chip components — visual summaries that go in the
headers of tabs and below charts.
"""

from __future__ import annotations

import httpx
import streamlit as st

from app.services.explainer import (
    Verdict, archetype_to_label, fit_to_verdict,
)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def _wiki_thumbnail(name: str) -> str | None:
    """Look up a player headshot from Wikipedia. Returns a thumbnail URL or
    None. Cached for a week per name. Best-effort — failures are silent and
    the card falls back to a text-only header.
    """
    if not name:
        return None
    try:
        with httpx.Client(timeout=4.0, follow_redirects=True) as c:
            r = c.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "format": "json",
                    "prop": "pageimages", "piprop": "thumbnail",
                    "pithumbsize": "200", "redirects": "1",
                    "titles": name,
                },
                headers={"User-Agent": "scouting-engine/0.1 (demo)"},
            )
            r.raise_for_status()
            pages = (r.json().get("query") or {}).get("pages") or {}
            for p in pages.values():
                thumb = (p.get("thumbnail") or {}).get("source")
                if thumb:
                    return thumb
    except Exception:
        return None
    return None


_TONE_BG = {
    "positive": "#E8F5E9",   # mint surface, dark text
    "neutral":  "#F3EDF7",   # matches secondaryBackgroundColor
    "warn":     "#FFF3E0",   # warm peach
    "negative": "#FCE4EC",   # soft rose
}
_TONE_FG = {
    "positive": "#1B5E20",
    "neutral":  "#1C1B1F",
    "warn":     "#7A4F01",
    "negative": "#7A1535",
}


def player_card(
    profile: dict,
    *,
    title: str | None = None,
    headline: str | None = None,
) -> None:
    """Hero player card: tinted role-coloured band with photo + name +
    optional one-line trait headline (e.g. 'Elite at long passing for a CB').
    """
    from app.config import role_accent
    accent = role_accent(profile.get("family"))

    name = title or profile["name"]
    role = (profile.get("position") or "").replace("_", " ").title()
    fam = profile.get("family") or ""
    team = profile.get("team_label") or ""
    n_events = profile.get("n_events") or 0

    photo = _wiki_thumbnail(profile.get("name", ""))
    photo_html = (
        f'<img src="{photo}" style="width:100%;max-width:140px;border-radius:10px;'
        f'object-fit:cover;border:2px solid {accent["line"]};" alt="">'
        if photo else ""
    )
    photo_col_html = (
        f'<div style="flex:0 0 140px;margin-right:18px;">{photo_html}</div>'
        if photo else ""
    )
    role_chip = (
        f'<span style="background:{accent["line"]};color:white;padding:2px 10px;'
        f'border-radius:999px;font-size:0.78rem;font-weight:600;letter-spacing:0.3px;">'
        f'{fam}</span>' if fam else ""
    )
    role_line = (
        f'<div style="margin-top:4px;color:{accent["fg"]};opacity:0.85;'
        f'font-size:0.95rem;">{role}'
        + (f' · <em>Most recently with</em> <strong>{team}</strong>' if team else "")
        + '</div>'
    )
    headline_html = (
        f'<div style="margin-top:10px;font-size:0.98rem;color:{accent["fg"]};">'
        f'{headline}</div>' if headline else ""
    )
    events_html = (
        f'<div style="margin-top:6px;color:{accent["fg"]};opacity:0.65;'
        f'font-size:0.82rem;">{n_events:,} on-ball events analysed</div>'
        if n_events else ""
    )

    html = (
        f'<div style="background:{accent["bg"]};border-left:6px solid {accent["line"]};'
        f'padding:18px 20px;border-radius:12px;margin-bottom:10px;'
        f'display:flex;align-items:center;">'
        f'{photo_col_html}'
        f'<div style="flex:1;">'
        f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">'
        f'<span style="font-size:1.45rem;font-weight:700;color:{accent["fg"]};">{name}</span>'
        f'{role_chip}'
        f'</div>'
        f'{role_line}'
        f'{headline_html}'
        f'{events_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def stat_strip(items: list[tuple[str, str]]) -> None:
    """Compact KPI strip — pairs of (label, value). Used under the hero
    card on the Profile tab."""
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)


def section_header(title: str, *, sub: str | None = None) -> None:
    """Twelve-style section header — uppercase label, accent underline,
    optional one-line subtitle. Use between top-level sections of the
    Profile tab so the page reads like a report instead of a wall of
    markdown."""
    sub_html = (
        f'<div style="font-size:0.92rem;color:#4A5A52;margin-top:4px;">{sub}</div>'
        if sub else ""
    )
    html = (
        f'<div style="margin:24px 0 12px 0;">'
        f'<div style="font-size:0.78rem;font-weight:700;color:#1B5E20;'
        f'letter-spacing:1.5px;text-transform:uppercase;">{title}</div>'
        f'<div style="height:3px;width:48px;background:#1B5E20;'
        f'border-radius:2px;margin-top:6px;"></div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def page_footer(text: str = "Generated by EventGPT Scouting Engine · presented by Hitesh") -> None:
    """Small report-style footer rendered at the bottom of a tab."""
    st.markdown(
        f'<div style="text-align:center;margin-top:36px;padding-top:14px;'
        f'border-top:1px solid rgba(28,27,31,0.10);'
        f'color:rgba(28,27,31,0.55);font-size:0.82rem;">{text}</div>',
        unsafe_allow_html=True,
    )


def caveat_block(items: list[str]) -> None:
    """Inline-styled caveat strip — softer than a full warning, more
    visible than a plain caption."""
    if not items:
        return
    body = "".join(
        f'<div style="margin:4px 0;color:#5C4400;">• {c}</div>' for c in items
    )
    st.markdown(
        f'<div style="background:#FFF8E1;border-left:4px solid #C99A2E;'
        f'padding:10px 14px;border-radius:8px;margin:10px 0;'
        f'font-size:0.9rem;">'
        f'<div style="font-weight:600;color:#7A4F01;margin-bottom:4px;">'
        f'Data caveats</div>{body}</div>',
        unsafe_allow_html=True,
    )


def now_scouting_badge(profile: dict) -> None:
    """Sticky 'Now scouting' chip rendered above the tab strip so the
    active player stays in eye-line as you flip between tabs."""
    from app.config import role_accent
    accent = role_accent(profile.get("family"))
    role = (profile.get("position") or "").replace("_", " ").title()
    team = profile.get("team_label") or "—"
    html = (
        f'<div style="background:{accent["bg"]};color:{accent["fg"]};'
        f'border-left:4px solid {accent["line"]};padding:6px 12px;'
        f'border-radius:8px;margin-bottom:10px;font-size:0.88rem;">'
        f'<span style="opacity:0.7;">Now scouting</span> · '
        f'<strong>{profile["name"]}</strong> · {role} · {team}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def empty_state(suggestions: list[str]) -> None:
    """Friendly empty-state card with clickable suggested players."""
    st.markdown(
        '<div style="background:#F3EDF7;border:1px dashed rgba(28,27,31,0.2);'
        'padding:24px;border-radius:12px;text-align:center;">'
        '<div style="font-size:1.1rem;font-weight:600;color:#1C1B1F;">'
        'Pick a player from the sidebar to start scouting.</div>'
        '<div style="margin-top:8px;color:rgba(28,27,31,0.7);font-size:0.92rem;">'
        'Every tab on this page works off whoever you select. '
        'Type a name in the sidebar, or try one of these:</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if suggestions:
        chips = " ".join(
            f'<span style="background:#EADDFF;color:#21005D;padding:6px 14px;'
            f'border-radius:999px;margin:4px 6px 0 0;display:inline-block;'
            f'font-size:0.88rem;">{s}</span>'
            for s in suggestions
        )
        st.markdown(
            f'<div style="text-align:center;margin-top:10px;">{chips}</div>',
            unsafe_allow_html=True,
        )


def archetype_chip(cluster: dict | None, action_families: list[str] | None = None) -> None:
    """Render the data-driven archetype label in a high-contrast panel
    so it reads cleanly against any background."""
    if cluster is None:
        st.info("No archetype assigned (likely a low-events / cold-start player).")
        return
    label = archetype_to_label(cluster, action_families=action_families)
    exemplars = cluster.get("exemplars") or []
    exemplar_names = ", ".join(
        e.get("name", str(e.get("player_id"))) for e in exemplars[:5]
    )
    exemplar_html = (
        f'<div style="margin-top:8px;font-size:0.88rem;color:#1B5E20;">'
        f'<strong>Plays like:</strong> {exemplar_names}</div>'
        if exemplar_names else ""
    )
    html = (
        f'<div style="background:#E8F5E9;border-left:5px solid #1B5E20;'
        f'padding:14px 18px;border-radius:10px;margin:10px 0 16px 0;'
        f'color:#0B2611;line-height:1.5;">'
        f'<div style="font-size:0.78rem;font-weight:700;color:#1B5E20;'
        f'letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">'
        f'Archetype</div>'
        f'<div style="font-size:1rem;font-weight:600;">{label}</div>'
        f'{exemplar_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _verdict_block(verdict: Verdict, *, sub: str | None = None) -> None:
    bg = _TONE_BG.get(verdict.tone, "#F3EDF7")
    fg = _TONE_FG.get(verdict.tone, "#1C1B1F")
    sub_html = (
        f'<div style="color:{fg};opacity:0.75;font-size:0.9rem;margin-top:6px;">{sub}</div>'
        if sub else ""
    )
    html = (
        f'<div style="background:{bg};color:{fg};padding:14px 18px;'
        f'border-radius:12px;margin-bottom:8px;border:1px solid rgba(28,27,31,0.06);">'
        f'<div style="font-size:1.35rem;font-weight:700;">{verdict.label}</div>'
        f'<div style="opacity:0.9;margin-top:4px;">{verdict.hint}</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def swap_impact_card(payload: dict, swap_verdict: Verdict, bullets: list[str]) -> None:
    """Verdict block + 3 plain-English bullets + sample-size footer."""
    _verdict_block(swap_verdict)
    if bullets:
        for b in bullets:
            st.markdown(f"- {b}")
    n = payload.get("n_episodes", 0)
    if n:
        st.caption(f"Based on {n} matched episodes featuring {payload.get('incumbent_name', 'the incumbent')}.")
    else:
        warning = payload.get("warning") or "Not enough sample episodes to estimate impact."
        st.warning(warning)


def team_fit_gauge(payload: dict, bullets: list[str]) -> None:
    """0–100 fit gauge band + reasoning bullets."""
    verdict = fit_to_verdict(payload.get("fit_score", 0.0))
    rank = payload.get("candidate_team_rank")
    team_size = payload.get("team_size") or 0
    sub = (f"Would rank #{rank} of {team_size} in this club's current roster by stylistic fit."
           if rank else None)
    _verdict_block(verdict, sub=sub)
    for b in bullets:
        st.markdown(f"- {b}")
