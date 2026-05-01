"""Plotly visuals: radar, heatmap, similarity table, archetype map.

All take dicts/lists pre-flattened by the lib endpoints (or ``modal_client``)
and return plotly figures rendered via ``st.plotly_chart``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.services.explainer import (
    action_diff_bullets, cosine_to_pct, cosine_to_verdict,
)


# ---------- 1. Action radar -----------------------------------------------


_FALLBACK_MEAN = {
    "Pass": 0.60, "Carry": 0.22, "Cross": 0.04, "TakeOn": 0.04, "Shot": 0.04,
    "Tackle": 0.04, "Interception": 0.04, "Clearance": 0.04, "Aerial": 0.04,
    "Duel": 0.04, "Other": 0.10,
}


# Twelve-style metric categories — each gets its own mini-radar with its
# sub-metrics. We map our action families to the closest equivalent of
# what Twelve shows (Involvement, Defence, Progression, Effectiveness,
# Box threat). Sub-metrics here are existing action families because
# that's the data we already have on /player_profile; richer sub-metrics
# (xGBuildup, xT, deep completions) need upstream pipeline work.
METRIC_CATEGORIES: list[dict] = [
    {
        "key": "involvement",
        "label": "Involvement",
        "subs": ["Pass", "Carry", "TakeOn", "Duel"],
        "blurb": "How often the player is on the ball at all.",
    },
    {
        "key": "defence",
        "label": "Defence",
        "subs": ["Tackle", "Interception", "Clearance", "Aerial"],
        "blurb": "Stopping play — duels, ball-winning, aerial work.",
    },
    {
        "key": "progression",
        "label": "Progression",
        "subs": ["Pass", "Carry", "TakeOn"],
        "blurb": "Moving the ball forward — through the lines.",
    },
    {
        "key": "creation",
        "label": "Creation & box threat",
        "subs": ["Cross", "Shot", "TakeOn", "Carry"],
        "blurb": "Final-third output — chances created, attempts on goal.",
    },
]


def category_mini_radar(
    cat: dict,
    action_mix: dict[str, float],
    baseline: dict[str, float] | None,
    baseline_std: dict[str, float] | None,
    *,
    role_label: str = "role peers",
) -> None:
    """Single-category mini radar — z-scores for the 3-4 sub-metrics in
    that category, on a 0-100 scale where 50 = role-typical, 100 = elite.
    Mirrors Twelve's per-section radars (Active Defence, Progression,
    etc.). Plotted as a polygon over a peer-mean reference circle.
    """
    subs = [s for s in cat["subs"] if s in action_mix]
    if not subs:
        return
    base_mean = baseline or _FALLBACK_MEAN
    base_std = baseline_std or {f: max(0.05, base_mean.get(f, 0.05) * 0.5) for f in subs}
    # Convert z to a 0-100 score: 50 = peer mean, +2σ = 100, -2σ = 0.
    def z_to_score(z: float) -> float:
        return max(0.0, min(100.0, 50 + 25 * z))
    z_vals = [_zscore(action_mix[s], base_mean.get(s, 0.05), base_std.get(s, 0.05))
              for s in subs]
    scores = [z_to_score(z) for z in z_vals]
    pretty = {"Pass": "Passing", "Carry": "Carrying", "Cross": "Crossing",
              "TakeOn": "Take-ons", "Shot": "Shooting", "Tackle": "Tackling",
              "Interception": "Interceptions", "Clearance": "Clearances",
              "Aerial": "Aerials", "Duel": "Duels", "Other": "Other"}
    labels = [pretty.get(s, s) for s in subs]

    # Section heading goes in Streamlit (more reliable than a plotly title
    # on a small radar — keeps the chart breathing and labels readable).
    st.markdown(
        f'<div style="font-size:1.05rem;font-weight:600;color:#1C1B1F;'
        f'margin:6px 0 2px 0;">{cat["label"]}</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    # Peer reference ring at 50.
    fig.add_trace(go.Scatterpolar(
        r=[50] * (len(subs) + 1),
        theta=labels + [labels[0]],
        line=dict(color="rgba(28,27,31,0.45)", dash="dot", width=1.5),
        name=f"Avg {role_label}",
        hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(46,163,122,0.45)",
        line=dict(color="#1B5E20", width=3),
        name="This player",
        customdata=[_zscore_to_phrase(z) for z in z_vals] + [_zscore_to_phrase(z_vals[0])],
        hovertemplate="%{theta}: %{customdata}<extra></extra>",
        showlegend=True,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[25, 50, 75], showticklabels=False,
                gridcolor="rgba(28,27,31,0.18)",
                linecolor="rgba(28,27,31,0.25)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#1C1B1F", size=13, family="sans-serif"),
                gridcolor="rgba(28,27,31,0.18)",
                linecolor="rgba(28,27,31,0.25)",
            ),
        ),
        height=360, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1C1B1F"),
        margin=dict(l=50, r=50, t=20, b=60),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(color="#1C1B1F", size=11),
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"_{cat['blurb']}_ Compared to other {role_label}.")


def category_grid(
    action_mix: dict[str, float],
    baseline: dict[str, float] | None,
    baseline_std: dict[str, float] | None,
    *,
    role_label: str = "role peers",
) -> None:
    """Render all four category mini-radars in a 2×2 grid. This is the
    Twelve pattern — instead of one big multi-axis radar, each metric
    family gets its own focused chart so the scout can read them
    independently."""
    cats = METRIC_CATEGORIES
    rows = [cats[:2], cats[2:]]
    for row in rows:
        cols = st.columns(len(row))
        for col, cat in zip(cols, row):
            with col:
                category_mini_radar(cat, action_mix, baseline, baseline_std,
                                    role_label=role_label)


def _zscore(value: float, mean: float, std: float) -> float:
    """z = (value − mean) / std, with a floor on std so tiny-variance
    actions don't blow up to ±10σ."""
    return (value - mean) / max(std, 0.01)


def _zscore_to_phrase(z: float) -> str:
    if z >= 1.75: return "Elite for the role"
    if z >= 0.85: return "Above role peers"
    if z >= 0.35: return "Slightly above"
    if z >  -0.35: return "In line with peers"
    if z >  -0.85: return "Slightly below"
    if z >  -1.75: return "Below role peers"
    return "Light for the role"


def _trait_summary(families: list[str], z_values: list[float]) -> str:
    """Turn the strongest +/− z-scores into a one-line scouting sentence."""
    pairs = sorted(zip(families, z_values), key=lambda p: -abs(p[1]))
    above = [f for f, z in pairs if z >= 0.85][:3]
    below = [f for f, z in pairs if z <= -0.85][:2]
    parts: list[str] = []
    if above:
        parts.append(f"**Stands out for:** {', '.join(above)}")
    if below:
        parts.append(f"**Light for the role:** {', '.join(below)}")
    if not parts:
        return "Sits close to the role-typical profile across the board — no extreme traits."
    return " · ".join(parts)


def action_radar(
    action_mix: dict[str, float],
    *,
    baseline: dict[str, float] | None = None,
    baseline_std: dict[str, float] | None = None,
    baseline_label: str = "Role-typical baseline",
    title: str = "How this player compares to others in the same role",
) -> None:
    """Diverging horizontal bar chart of action-mix z-scores.

    For each action family we compute ``z = (player − family_mean) /
    family_std``. Bars to the right of 0 = this player does that action
    *more* than role peers; bars to the left = less. Reference lines at
    ±1σ and ±2σ make the magnitude legible at a glance — something a
    radar chart obscures.
    """
    if not action_mix:
        st.info("No action data available for this player.")
        return
    base_mean = baseline or _FALLBACK_MEAN
    base_std = baseline_std or {f: max(0.05, base_mean.get(f, 0.05) * 0.5)
                                 for f in action_mix}
    families = list(action_mix.keys())
    z = [_zscore(action_mix[f], base_mean.get(f, 0.05), base_std.get(f, 0.05))
         for f in families]
    # Sort families by absolute z-score so the most distinctive traits
    # land at the top.
    order = sorted(range(len(families)), key=lambda i: -abs(z[i]))
    families_sorted = [families[i] for i in order]
    z_sorted = [max(-3.0, min(3.0, z[i])) for i in order]
    colors = ["#1B5E20" if v >= 0 else "#B3261E" for v in z_sorted]

    # Plain-English headline above the chart.
    st.markdown(_trait_summary(families_sorted, z_sorted))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=z_sorted, y=families_sorted, orientation="h",
        marker=dict(color=colors),
        customdata=[_zscore_to_phrase(v) for v in z_sorted],
        hovertemplate="%{y}: %{customdata}<extra></extra>",
        showlegend=False,
    ))
    # Reference lines, labelled in football language (no σ on screen).
    for x_ref, dash in [(-2, "dash"), (-1, "dot"), (0, "solid"),
                        (1, "dot"), (2, "dash")]:
        fig.add_shape(type="line", x0=x_ref, x1=x_ref, y0=-0.5,
                      y1=len(families_sorted) - 0.5,
                      line=dict(color="rgba(28,27,31,0.35)", width=1, dash=dash))
    fig.update_layout(
        title=title, height=max(360, 36 * len(families_sorted) + 80),
        paper_bgcolor="#FFFBFF", plot_bgcolor="#FFFBFF",
        font=dict(color="#1C1B1F"),
        xaxis=dict(
            title="",
            range=[-2.6, 2.6],
            tickmode="array",
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["Light for role", "Below peers",
                      "Role-typical", "Above peers", "Elite for role"],
            gridcolor="rgba(28,27,31,0.08)", zeroline=False,
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=60, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Compared to other {baseline_label.lower().replace('average ', '').replace(' player', 's')} "
        "in the league. Bars to the right = does this **more** than typical role peers; "
        "left = does it **less**. The further out the bar, the rarer the trait. "
        "Rows are sorted by how distinctive each trait is."
    )


def action_radar_compare(
    a_name: str, a_mix: dict[str, float],
    b_name: str, b_mix: dict[str, float],
    *,
    baseline: dict[str, float] | None = None,
    baseline_std: dict[str, float] | None = None,
    title: str = "How they compare, side by side, against role peers",
) -> None:
    """Grouped horizontal z-score bars for two players on the same axes.

    Both players are compared against the *same* role baseline, so the
    bars line up axis-for-axis: same families, same x-scale, two colours.
    """
    if not a_mix or not b_mix:
        st.info("Need both players to have action data.")
        return
    base_mean = baseline or _FALLBACK_MEAN
    base_std = baseline_std or {f: max(0.05, base_mean.get(f, 0.05) * 0.5)
                                 for f in a_mix}
    families = list(a_mix.keys())
    a_z = [_zscore(a_mix[f], base_mean.get(f, 0.05), base_std.get(f, 0.05))
           for f in families]
    b_z = [_zscore(b_mix.get(f, 0.0), base_mean.get(f, 0.05), base_std.get(f, 0.05))
           for f in families]
    # Sort by combined magnitude so the largest divergences sit at the top.
    order = sorted(range(len(families)),
                   key=lambda i: -(abs(a_z[i]) + abs(b_z[i])))
    families_sorted = [families[i] for i in order]
    a_sorted = [max(-3.0, min(3.0, a_z[i])) for i in order]
    b_sorted = [max(-3.0, min(3.0, b_z[i])) for i in order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=a_sorted, y=families_sorted, orientation="h",
        marker=dict(color="#6750A4"), name=a_name,
        customdata=[_zscore_to_phrase(v) for v in a_sorted],
        hovertemplate=f"{a_name} — %{{y}}: %{{customdata}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=b_sorted, y=families_sorted, orientation="h",
        marker=dict(color="#D46D52"), name=b_name,
        customdata=[_zscore_to_phrase(v) for v in b_sorted],
        hovertemplate=f"{b_name} — %{{y}}: %{{customdata}}<extra></extra>",
    ))
    for x_ref, dash in [(-2, "dash"), (-1, "dot"), (0, "solid"),
                        (1, "dot"), (2, "dash")]:
        fig.add_shape(type="line", x0=x_ref, x1=x_ref,
                      y0=-0.5, y1=len(families_sorted) - 0.5,
                      line=dict(color="rgba(28,27,31,0.35)", width=1, dash=dash))
    fig.update_layout(
        title=title, height=max(420, 44 * len(families_sorted) + 80),
        barmode="group",
        paper_bgcolor="#FFFBFF", plot_bgcolor="#FFFBFF",
        font=dict(color="#1C1B1F"),
        xaxis=dict(
            title="",
            range=[-2.6, 2.6],
            tickmode="array",
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["Light for role", "Below peers",
                      "Role-typical", "Above peers", "Elite for role"],
            gridcolor="rgba(28,27,31,0.08)", zeroline=False,
        ),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    font=dict(color="#1C1B1F")),
        margin=dict(l=20, r=20, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Both players measured against the same group of role peers. Where "
        "the two bars sit on opposite sides of the centre line is where "
        "their styles genuinely diverge — that's the conversation to have."
    )


def strengths_weaknesses(
    action_mix: dict[str, float],
    baseline: dict[str, float] | None,
    baseline_std: dict[str, float] | None,
    n_peers: int | None = None,
    *,
    role_label: str = "role peers",
) -> None:
    """Twelve-style Strengths / Weaknesses panel — top 3 above-peer
    actions and top 3 below-peer actions, each with an approximate
    rank-of-N label derived from the z-score (normal CDF approximation).
    Mirrors the 'Run Quality 2/118' format from the Doku report.
    """
    import math
    if not action_mix or not baseline or not baseline_std:
        return
    n = n_peers or 100
    pretty = {"Pass": "Passing", "Carry": "Carrying", "Cross": "Crossing",
              "TakeOn": "Take-ons", "Shot": "Shooting", "Tackle": "Tackling",
              "Interception": "Interceptions", "Clearance": "Clearances",
              "Aerial": "Aerial duels", "Duel": "Physical duels", "Other": "Set-piece work"}
    rows = []
    for fam, share in action_mix.items():
        std = max(baseline_std.get(fam, 0.05), 0.01)
        z = (share - baseline.get(fam, 0.05)) / std
        # Approx percentile from z. For positive z, percentile = 1 - cdf;
        # rank ≈ percentile * N (rank 1 = best).
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        rank_above = max(1, int(round((1 - cdf) * n)))
        rank_below = max(1, int(round(cdf * n)))
        rows.append({"fam": fam, "z": z, "rank_above": rank_above,
                     "rank_below": rank_below})
    rows.sort(key=lambda r: -r["z"])
    strengths = [r for r in rows if r["z"] >= 0.85][:3]
    weaknesses = [r for r in rows if r["z"] <= -0.85][-3:]

    def render_section(title: str, items: list[dict], rank_key: str,
                       fill: str) -> None:
        st.markdown(f"**{title}**")
        if not items:
            st.caption(f"No standout {title.lower()} vs {role_label}.")
            return
        for r in items:
            label = pretty.get(r["fam"], r["fam"])
            rank = r[rank_key]
            # Convert z to a 0-100% bar fill (capped at ±2.5σ).
            pct = max(5, min(95, int(50 + 18 * abs(r["z"]))))
            html = (
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.92rem;margin-bottom:3px;">'
                f'<span>{label}</span>'
                f'<span style="opacity:0.7;">{rank}/{n}</span>'
                f'</div>'
                f'<div style="background:rgba(232,240,236,0.10);height:8px;'
                f'border-radius:4px;overflow:hidden;">'
                f'<div style="background:{fill};height:100%;width:{pct}%;"></div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    col_s, col_w = st.columns(2)
    with col_s:
        render_section("Strengths", strengths, "rank_above", "#2EA37A")
    with col_w:
        render_section("Weaknesses", weaknesses, "rank_below", "#D6A658")


def phase_bars(
    phase_a: dict[str, float], a_name: str,
    phase_b: dict[str, float] | None = None, b_name: str | None = None,
    *, title: str = "Where their work happens",
) -> None:
    """Stacked-by-phase comparison. One row per player; phases coloured."""
    phases = ["build_up", "progression", "creation", "finishing", "defense", "set_pieces", "transition"]
    labels = {
        "build_up":    "Build-up",
        "progression": "Progression",
        "creation":    "Creation",
        "finishing":   "Finishing",
        "defense":     "Defending",
        "set_pieces":  "Set pieces",
        "transition":  "Transition",
    }
    colors = {
        "build_up":    "#9DBEDC",
        "progression": "#6750A4",
        "creation":    "#D46D52",
        "finishing":   "#B3261E",
        "defense":     "#1B5E20",
        "set_pieces":  "#7A4F01",
        "transition":  "#7A1535",
    }
    fig = go.Figure()
    rows = [(a_name, phase_a)]
    if phase_b is not None and b_name:
        rows.append((b_name, phase_b))
    for phase in phases:
        fig.add_trace(go.Bar(
            x=[(prof.get(phase, 0.0) or 0.0) * 100 for _, prof in rows],
            y=[name for name, _ in rows],
            orientation="h", name=labels[phase],
            marker=dict(color=colors[phase]),
            hovertemplate="%{y} — " + labels[phase] + ": %{x:.0f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", title=title, height=140 + 60 * len(rows),
        paper_bgcolor="#FFFBFF", plot_bgcolor="#FFFBFF",
        font=dict(color="#1C1B1F"),
        xaxis=dict(title="% of activity", range=[0, 100],
                   gridcolor="rgba(28,27,31,0.08)"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.3, x=0, font=dict(color="#1C1B1F")),
        margin=dict(l=10, r=10, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- 2. Pitch heatmap ----------------------------------------------


def qualitative_pitch_grid(
    spatial_zone: list[float],
    *,
    role_baseline_zone: list[float] | None = None,
    title: str = "Performance by pitch zone",
    role_label: str = "role peers",
) -> None:
    """Twelve-style qualitative pitch grid. Splits the pitch into a 4×4
    grid and labels each zone with one of six quality words —
    *Outstanding · Excellent · Good · Average · Below average · Poor* —
    derived from how this player's touch share compares to role-peer
    mean for that zone. Tells the scout *where the player performs*,
    not just where the touches land.
    """
    if not spatial_zone or len(spatial_zone) != 16:
        st.info("No spatial data for this player.")
        return
    grid = np.asarray(spatial_zone, dtype=np.float32).reshape(4, 4)
    base = (np.asarray(role_baseline_zone, dtype=np.float32).reshape(4, 4)
            if role_baseline_zone and len(role_baseline_zone) == 16
            else np.full((4, 4), 1.0 / 16))
    # Ratio of player share to peer share, per zone.
    ratio = grid / np.maximum(base, 1e-3)

    def label(r: float) -> tuple[str, str]:
        if r >= 1.75: return ("Outstanding", "#1B5E20")
        if r >= 1.35: return ("Excellent",   "#2E7D32")
        if r >= 1.05: return ("Good",        "#388E3C")
        if r >= 0.80: return ("Average",     "#0F2C25")
        if r >= 0.50: return ("Below avg",   "#7A1535")
        return ("Poor", "#5D1A1A")

    PITCH_L, PITCH_W = 105, 68
    cell_w, cell_h = PITCH_L / 4, PITCH_W / 4

    fig = go.Figure()
    line = dict(color="rgba(232,240,236,0.55)", width=2)

    for r in range(4):
        for c in range(4):
            txt, fill = label(float(ratio[r, c]))
            x0, x1 = c * cell_w, (c + 1) * cell_w
            y0, y1 = r * cell_h, (r + 1) * cell_h
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          line=dict(color="rgba(232,240,236,0.20)", width=1),
                          fillcolor=fill, layer="below")
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                text=txt, showarrow=False,
                font=dict(color="#E8F0EC", size=11, family="sans-serif"),
            )
            # Invisible hover marker carrying the percentage.
            share_pct = f"{grid[r, c] * 100:.1f}% of touches · {txt} vs {role_label}"
            fig.add_trace(go.Scatter(
                x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers",
                marker=dict(size=cell_w, opacity=0),
                hovertemplate=share_pct + "<extra></extra>",
                showlegend=False,
            ))

    # Pitch furniture — outer box, halfway, centre circle, penalty boxes.
    fig.add_shape(type="rect", x0=0, x1=PITCH_L, y0=0, y1=PITCH_W, line=line)
    fig.add_shape(type="line", x0=PITCH_L/2, x1=PITCH_L/2, y0=0, y1=PITCH_W, line=line)
    fig.add_shape(type="circle", x0=PITCH_L/2-9.15, x1=PITCH_L/2+9.15,
                  y0=PITCH_W/2-9.15, y1=PITCH_W/2+9.15, line=line)
    for x_goal in (0, PITCH_L):
        sign = 1 if x_goal == 0 else -1
        fig.add_shape(type="rect", x0=x_goal, x1=x_goal + sign * 16.5,
                      y0=(PITCH_W - 40.32) / 2, y1=(PITCH_W + 40.32) / 2, line=line)
        fig.add_shape(type="rect", x0=x_goal, x1=x_goal + sign * 5.5,
                      y0=(PITCH_W - 18.32) / 2, y1=(PITCH_W + 18.32) / 2, line=line)

    fig.update_layout(
        title=dict(text=title, y=0.96, font=dict(color="#E8F0EC", size=15)),
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=70, b=50),
        font=dict(color="#E8F0EC"),
        xaxis=dict(range=[-2, PITCH_L + 2], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-2, PITCH_W + 2], visible=False),
        annotations=[
            *fig.layout.annotations,
            dict(x=PITCH_L/4, y=-3, xref="x", yref="y", showarrow=False,
                 text="Defensive third", font=dict(color="rgba(232,240,236,0.7)", size=11)),
            dict(x=PITCH_L*3/4, y=-3, xref="x", yref="y", showarrow=False,
                 text="Attacking third", font=dict(color="rgba(232,240,236,0.7)", size=11)),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each zone labelled by how the player's touch share there compares "
        f"to other {role_label}: **Outstanding** is much above peers, "
        "**Average** is in line, **Poor** is well below. Use it with the "
        "raw heatmap below — quality vs. quantity."
    )


def pitch_heatmap(spatial_zone: list[float], *, title: str = "Where they touch the ball") -> None:
    """Real pitch shape with the 4×4 zone histogram overlaid as tinted
    rectangles. Pitch furniture (centre circle, penalty boxes, six-yard
    boxes, goals) is drawn so the orientation is unambiguous.

    Coordinate mapping (matches ``prepare.py:_spatial_bin``):
      * cols (0..3) = pitch length, col 0 = own third, col 3 = att third
      * rows (0..3) = pitch width
    """
    if not spatial_zone or len(spatial_zone) != 16:
        st.info("No spatial data available for this player.")
        return
    grid = np.asarray(spatial_zone, dtype=np.float32).reshape(4, 4)
    vmax = max(0.15, float(grid.max()))

    PITCH_L, PITCH_W = 105, 68  # FIFA standard
    cell_w, cell_h = PITCH_L / 4, PITCH_W / 4

    def shade(v: float) -> str:
        """Map share to a transparent purple. 0 → invisible, vmax → opaque."""
        a = max(0.0, min(1.0, v / vmax))
        return f"rgba(103,80,164,{0.05 + 0.7 * a:.3f})"

    fig = go.Figure()

    # Tinted zone rectangles (the data layer).
    for r in range(4):
        for c in range(4):
            v = float(grid[r, c])
            x0 = c * cell_w
            x1 = (c + 1) * cell_w
            y0 = r * cell_h
            y1 = (r + 1) * cell_h
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(color="rgba(28,27,31,0.10)", width=1),
                fillcolor=shade(v), layer="below",
            )
            # Invisible scatter for hover.
            fig.add_trace(go.Scatter(
                x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
                mode="markers",
                marker=dict(size=cell_w * 1.5, opacity=0,
                            color=shade(v)),
                hovertemplate=f"{v:.1%} of touches<extra></extra>",
                showlegend=False,
            ))

    # Pitch furniture — drawn over the heatmap.
    line = dict(color="rgba(28,27,31,0.6)", width=2)
    # Outer boundary
    fig.add_shape(type="rect", x0=0, x1=PITCH_L, y0=0, y1=PITCH_W, line=line)
    # Halfway line
    fig.add_shape(type="line", x0=PITCH_L/2, x1=PITCH_L/2, y0=0, y1=PITCH_W, line=line)
    # Centre circle + spot
    fig.add_shape(type="circle", x0=PITCH_L/2-9.15, x1=PITCH_L/2+9.15,
                  y0=PITCH_W/2-9.15, y1=PITCH_W/2+9.15, line=line)
    fig.add_shape(type="circle", x0=PITCH_L/2-0.4, x1=PITCH_L/2+0.4,
                  y0=PITCH_W/2-0.4, y1=PITCH_W/2+0.4,
                  fillcolor="rgba(28,27,31,0.6)", line=line)
    # Penalty + 6-yard boxes (both ends).
    for x_goal in (0, PITCH_L):
        sign = 1 if x_goal == 0 else -1
        # Penalty box (16.5 × 40.32)
        fig.add_shape(type="rect",
                      x0=x_goal, x1=x_goal + sign * 16.5,
                      y0=(PITCH_W - 40.32) / 2, y1=(PITCH_W + 40.32) / 2, line=line)
        # 6-yard box (5.5 × 18.32)
        fig.add_shape(type="rect",
                      x0=x_goal, x1=x_goal + sign * 5.5,
                      y0=(PITCH_W - 18.32) / 2, y1=(PITCH_W + 18.32) / 2, line=line)
        # Goal line tick
        fig.add_shape(type="rect",
                      x0=x_goal, x1=x_goal - sign * 1.5,
                      y0=(PITCH_W - 7.32) / 2, y1=(PITCH_W + 7.32) / 2,
                      line=dict(color="rgba(28,27,31,0.8)", width=2))
        # Penalty spot
        fig.add_shape(type="circle",
                      x0=x_goal + sign * 11 - 0.4, x1=x_goal + sign * 11 + 0.4,
                      y0=PITCH_W/2 - 0.4, y1=PITCH_W/2 + 0.4,
                      fillcolor="rgba(28,27,31,0.6)", line=line)

    fig.update_layout(
        title=dict(text=title, y=0.95),
        height=420, paper_bgcolor="#FFFBFF", plot_bgcolor="#E9F3E9",
        margin=dict(l=20, r=20, t=80, b=50),
        font=dict(color="#1C1B1F"),
        xaxis=dict(range=[-2, PITCH_L + 2], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-2, PITCH_W + 2], visible=False),
        annotations=[
            dict(x=PITCH_L/2, y=-1, xref="x", yref="y", showarrow=False,
                 text="Own goal  ←——  Direction of attack  ——→  Opponent goal",
                 font=dict(color="#1C1B1F", size=11)),
        ],
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Heat = share of this player's on-ball touches in each pitch zone. "
        "Darker purple = more time on the ball there. Use this with the role "
        "card above: a wide forward should glow in the attacking-third channels; "
        "a deep-lying playmaker should pile up touches in their own half."
    )


# ---------- 3. Similar-players table --------------------------------------


def similar_players_table(payload: dict) -> int | None:
    """Render the KNN result as a styled DataFrame. Returns the player_id of
    the row the user clicks (via Streamlit's row selection), or None."""
    results = payload.get("results", []) or []
    if not results:
        st.info("No similar players match those filters.")
        return None
    rows = []
    for r in results:
        v = cosine_to_verdict(r["cosine"])
        bullets = action_diff_bullets(r.get("action_diff") or {}, top_n=2, threshold=0.02)
        rows.append({
            "Rank": r["rank"],
            "Player": r["player_name"],
            "Position": (r.get("position") or "—").replace("_", " ").title(),
            "Team": r.get("team_label") or "—",
            "Style match": f"{v.label} ({v.hint})",
            "Key differences": "; ".join(bullets) if bullets else "—",
            "_player_id": r["player_id"],
        })
    df = pd.DataFrame(rows)
    selection = st.dataframe(
        df.drop(columns=["_player_id"]),
        use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )
    sel_rows = (selection.selection.rows if hasattr(selection, "selection") else
                selection.get("selection", {}).get("rows", []))
    if sel_rows:
        return int(df.iloc[sel_rows[0]]["_player_id"])
    return None


# ---------- 4. Archetype map -----------------------------------------------


def archetype_map(payload: dict, *, height: int = 600) -> None:
    """Plotly scatter of (x, y) per player, coloured by cluster_id."""
    rows = payload.get("players_xy") or []
    if not rows:
        st.info("Archetype map is empty for this checkpoint.")
        return
    df = pd.DataFrame(rows)
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["display"] = df.apply(
        lambda r: f"{r['name']} ({(r['position'] or '?').replace('_', ' ').title()})",
        axis=1,
    )
    fig = px.scatter(
        df, x="x", y="y", color="cluster_id", hover_name="display",
        hover_data={"family": True, "x": False, "y": False, "cluster_id": True},
        height=height, opacity=0.85,
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        legend_title="Archetype cluster",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- 5. Team-style similarity heatmap -------------------------------


def team_similarity_heatmap(teams_payload: dict, *, top_n: int = 25) -> None:
    """Compact heatmap of inter-team style similarity. Useful in Strategy
    mode for context: "Arsenal play closer to Man City than to Burnley"."""
    teams = teams_payload.get("teams") or []
    mat = teams_payload.get("similarity_matrix") or []
    if not teams or not mat:
        st.info("No team-style data available.")
        return
    n = min(top_n, len(teams))
    sub_labels = [t["label"] for t in teams[:n]]
    sub_mat = np.asarray(mat)[:n, :n]
    fig = px.imshow(
        sub_mat, x=sub_labels, y=sub_labels,
        color_continuous_scale="Blues", zmin=0.7, zmax=1.0,
        labels={"color": "Style similarity"},
    )
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Darker = the two teams play more alike (similar action mix and pitch zones). "
        "Labels are top-3 most-active player surnames per team."
    )
