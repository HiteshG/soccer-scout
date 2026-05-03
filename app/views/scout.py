"""Scout mode — per-player drilldown.

Four sub-tabs:
  1. Profile        — player card + style fingerprint + archetype
  2. Find similar   — replacement search with filters
  3. What-if swap   — counterfactual: replace this player with a candidate
  4. AI report      — OpenAI scouting one-pager
"""

from __future__ import annotations

import streamlit as st

from app.components import (
    action_radar, action_radar_compare, archetype_chip, category_grid,
    caveat_block, empty_state, now_scouting_badge, page_footer, phase_bars,
    pitch_heatmap, player_card, qualitative_pitch_grid, section_header,
    similar_players_table, stat_strip, strengths_weaknesses, swap_impact_card,
)
from app.config import load_config
from app.services import modal_client
from app.services.explainer import (
    action_diff_bullets, archetype_to_label, caveats as caveats_for_player,
    cosine_to_pct, cosine_to_verdict, defensive_workload,
    delta_robv_to_verdict, partnership_requirements, peer_differentiators,
    phase_profile, phase_profile_phrases, risk_profile,
    system_fit_hypothesis, trait_headline,
)
from app.services.openai_client import (
    _action_emphasis_phrases, _spatial_phrase, _top_action_phrases,
    head_to_head, peer_difference_phrase, scouting_pager, swap_narrative,
)


def _archetype_for(player_id: int) -> dict | None:
    """Look up the archetype cluster a player belongs to. Cached at the
    modal_client layer."""
    arch = modal_client.archetypes()
    cluster_by_id: dict[int, dict] = {c["cluster_id"]: c for c in arch.get("clusters", [])}
    for row in arch.get("players_xy", []):
        if row["player_id"] == player_id:
            cid = row.get("cluster_id")
            if cid is not None and cid >= 0:
                return cluster_by_id.get(int(cid))
            return None
    return None


def render(active_player_id: int | None) -> None:
    if not active_player_id:
        empty_state(["Bukayo Saka", "Declan Rice", "Virgil van Dijk",
                     "Cole Palmer", "Rodri"])
        return

    profile = modal_client.player_profile(active_player_id)
    arch_cluster = _archetype_for(active_player_id)
    now_scouting_badge(profile)

    tab_profile, tab_similar, tab_compare, tab_swap, tab_ai = st.tabs([
        "Player profile",
        "Stylistic peers",
        "Head-to-head",
        "Transfer what-if",
        "AI scouting report",
    ])

    # ---------- 1. Profile ----------
    # Pull baselines up here so the hero card can render with a trait headline.
    try:
        base_payload = modal_client.baselines() or {}
    except Exception:
        base_payload = {}
    fam = profile.get("family") or "?"
    fam_baseline_data = base_payload.get(fam) or {}
    baseline_families = fam_baseline_data.get("action_families") or []
    baseline_means = fam_baseline_data.get("action_mix_mean") or []
    baseline_stds = fam_baseline_data.get("action_mix_std") or []
    baseline_dict = (
        {f: float(v) for f, v in zip(baseline_families, baseline_means)}
        if baseline_families and baseline_means else None
    )
    baseline_std_dict = (
        {f: float(v) for f, v in zip(baseline_families, baseline_stds)}
        if baseline_families and baseline_stds else None
    )
    headline_text = trait_headline(
        profile.get("action_mix", {}) or {},
        baseline_dict, baseline_std_dict, family=fam,
    )

    with tab_profile:
        player_card(profile, headline=headline_text)

        # KPI stat strip directly under the hero.
        ph_full = phase_profile(
            profile.get("action_mix", {}) or {},
            profile.get("spatial_zone", []) or [],
        )
        top_phase = (
            max(ph_full.items(), key=lambda kv: kv[1])[0].replace("_", " ").title()
            if ph_full else "—"
        )
        # Three short KPIs in the strip; the full archetype label is
        # rendered by ``archetype_chip`` below where it has room to wrap.
        stat_strip([
            ("Events analysed", f"{(profile.get('n_events') or 0):,}"),
            ("Primary phase", top_phase),
            ("Role family", fam),
        ])

        archetype_chip(arch_cluster, action_families=profile.get("action_families"))
        n_peers = fam_baseline_data.get("n_players")
        baseline_label = (
            f"Average {fam} player" + (f" (n={n_peers})" if n_peers else "")
            if baseline_dict else "League-typical baseline"
        )
        role_label = (f"{fam} peers" if fam and fam != "?" else "role peers")

        # Strengths & Weaknesses, ranked vs role peers — the "2/118"
        # framing scouts read first.
        section_header(
            "Strengths & weaknesses",
            sub="Where this player ranks against others in the same role.",
        )
        if n_peers:
            st.caption(
                f"Each line shows where this player ranks among **{n_peers} other "
                f"{fam} players** with comparable sample size (lower = better). "
                f"For example, *14/{n_peers}* means top-14 in this trait among "
                f"all {fam}s in the dataset."
            )
        strengths_weaknesses(
            profile.get("action_mix", {}) or {},
            baseline_dict, baseline_std_dict, n_peers=n_peers,
            role_label=role_label,
        )

        # Category mini-radars (Involvement / Defence / Progression / Creation)
        # so each metric family is read independently.
        section_header(
            "Performance by category",
            sub="Each radar isolates one slice of the game so the spokes "
                "can be read on their own.",
        )
        category_grid(
            profile.get("action_mix", {}) or {},
            baseline_dict, baseline_std_dict,
            role_label=role_label,
        )

        # Pitch view: the qualitative grid (where the player *performs*)
        # next to the raw heatmap (where the player *touches*).
        section_header(
            "Pitch presence",
            sub="Quality vs. quantity — performance per zone alongside the "
                "raw touch heatmap.",
        )
        col_qual, col_heat = st.columns([1, 1])
        baseline_zone = fam_baseline_data.get("spatial_zone_mean")
        with col_qual:
            qualitative_pitch_grid(
                profile.get("spatial_zone", []) or [],
                role_baseline_zone=baseline_zone,
                role_label=role_label,
            )
        with col_heat:
            pitch_heatmap(profile.get("spatial_zone", []) or [])

        # The full 11-axis style fingerprint — kept as a deeper drill-down
        # below the category grid for analysts who want every spoke.
        with st.expander("Full style fingerprint (all action families)", expanded=False):
            action_radar(
                profile.get("action_mix", {}),
                baseline=baseline_dict,
                baseline_std=baseline_std_dict,
                baseline_label=baseline_label,
                title=f"Style vs {fam} peers" if baseline_dict else "Style fingerprint",
            )

        # Tactical depth panel — phase mix, risk, defensive workload,
        # partnership needs, system fit. All deterministic, all in
        # plain football English.
        ph = phase_profile(
            profile.get("action_mix", {}) or {},
            profile.get("spatial_zone", []) or [],
        )
        ph_phrases = phase_profile_phrases(ph)
        risk = risk_profile(profile.get("action_mix", {}) or {})
        d_load = defensive_workload(
            profile.get("action_mix", {}) or {},
            family_baseline=baseline_dict, family=fam,
        )
        partners = partnership_requirements(
            profile.get("action_mix", {}) or {},
            profile.get("spatial_zone", []) or [],
            family=fam,
        )
        system_fit = system_fit_hypothesis(ph, risk, fam)

        section_header(
            "Tactical depth",
            sub="How the player's work breaks down by phase, and what the "
                "rest of the XI has to provide around them.",
        )
        if ph:
            phase_bars(ph, profile["name"])

        def _depth_card(title: str, items: list[str]) -> None:
            if not items:
                return
            body = "".join(f'<li style="margin:3px 0;">{i}</li>' for i in items)
            html = (
                f'<div style="background:#F7FAF8;border:1px solid '
                f'rgba(28,27,31,0.08);border-radius:10px;padding:14px 18px;'
                f'height:100%;margin-bottom:12px;">'
                f'<div style="font-size:0.78rem;font-weight:700;color:#1B5E20;'
                f'letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">'
                f'{title}</div>'
                f'<ul style="margin:0;padding-left:18px;color:#1C1B1F;'
                f'font-size:0.92rem;line-height:1.5;">{body}</ul>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

        depth_cols = st.columns(3)
        with depth_cols[0]:
            _depth_card("Phase emphasis",
                       ph_phrases + ([risk] if risk else []))
        with depth_cols[1]:
            _depth_card("Defensive workload",
                       ([d_load] if d_load else []) + (partners or []))
        with depth_cols[2]:
            _depth_card("Best-fit systems", system_fit or [])

        # Caveats — styled inline strip so the scout sees the data limits
        # without it feeling like a warning dialog.
        cv = caveats_for_player(profile, family_baseline_n=n_peers)
        if cv:
            caveat_block(cv)

        # Glossary — every metric the scout sees, in a clean two-column
        # definition layout.
        with st.expander("Glossary — what each metric means", expanded=False):
            terms = [
                ("Involvement",
                 "How often the player is on the ball (passes + carries + "
                 "take-ons + duels combined). A hub-of-build-up vs. specialist signal."),
                ("Defence",
                 "Stopping play: tackles, interceptions, clearances, aerial duels."),
                ("Progression",
                 "Moving the ball forward through the lines, primarily via "
                 "passing and carrying."),
                ("Creation & box threat",
                 "Final-third output: crosses, take-ons in advanced areas, "
                 "shots, attacking carries."),
                ("Strengths / Weaknesses",
                 "Top 3 above- and below-peer actions, ranked approximately "
                 "within the player's role family. <em>n / N</em> is the "
                 "player's position out of N role peers with enough events."),
                ("Style fingerprint",
                 "Every action family compared to the role-typical mean. "
                 "Bars right of centre = does the action <em>more</em> than role peers."),
                ("Performance by pitch zone",
                 "Touch share in each of 16 pitch zones, labelled "
                 "<em>Outstanding · Excellent · Good · Average · Below average · "
                 "Poor</em> based on the player's share vs. role peers."),
                ("Tactical depth",
                 "Phase-of-play emphasis: build-up, progression, creation, "
                 "finishing, defending, set pieces, transition. Sums to 100% of work."),
                ("Archetype",
                 "Data-driven cluster of players with a similar style "
                 "fingerprint. Exemplars listed for sanity-check by name recognition."),
            ]
            rows = "".join(
                f'<div style="display:grid;grid-template-columns:200px 1fr;'
                f'gap:14px;padding:8px 0;border-bottom:1px solid '
                f'rgba(28,27,31,0.06);">'
                f'<div style="font-weight:600;color:#1B5E20;font-size:0.92rem;">{term}</div>'
                f'<div style="color:#1C1B1F;font-size:0.92rem;line-height:1.5;">{desc}</div>'
                f'</div>'
                for term, desc in terms
            )
            st.markdown(
                f'<div style="background:#F7FAF8;border-radius:10px;'
                f'padding:6px 18px;">{rows}</div>',
                unsafe_allow_html=True,
            )

        page_footer()

    # ---------- 2. Find similar ----------
    with tab_similar:
        st.markdown("#### Players who play like this one")
        st.caption(
            "Ranked by stylistic cosine — same shape of actions, same areas of the pitch. "
            "Tighten the filters to constrain by role, position, or sample size."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            same_family = st.checkbox("Match role family", value=True,
                                      help="Restrict to the same broad role (DEF / MID / ATT / GK). On by default for honest comparisons.")
        with col2:
            same_position = st.checkbox("Match exact position", value=False,
                                        help="Stricter: only players with the same primary position label.")
        with col3:
            min_events = st.slider("Minimum events on record", 200, 5000, 800, step=100,
                                   help="Filter out small-sample players whose style estimate is noisy.")
        top_k = st.slider("Candidates to return", 5, 50, 15)

        with st.spinner("Searching for similar players…"):
            payload = modal_client.search_replacements(
                query_player_id=active_player_id,
                top_k=top_k, same_family=same_family,
                same_position=same_position, min_events=min_events,
            )
        st.caption(
            f"Found {payload.get('n_candidates', 0)} candidates matching the filters; "
            f"showing the top {len(payload.get('results', []))}."
        )
        cfg = load_config()
        ai_richness = (
            st.checkbox(
                "Sharpen 'Key differences' with AI phrasing",
                value=False,
                help="Sends one short OpenAI call per row (cached per session) for "
                     "more specific tactical language than the deterministic fallback.",
            ) if cfg.has_openai else False
        )
        # Pre-translate cosine + diff bullets for the table.
        diff_cache: dict = st.session_state.setdefault("scout_peer_diff_cache", {})
        for r in payload.get("results", []):
            v = cosine_to_verdict(r["cosine"])
            r["match_phrase"] = f"{v.label} (~{cosine_to_pct(r['cosine'])}% style match)"
            differentiators = peer_differentiators(r.get("action_diff") or {}, top_n=3)
            r["differentiator_specs"] = differentiators
            cache_key = (active_player_id, r["player_id"], ai_richness)
            if ai_richness and differentiators:
                if cache_key not in diff_cache:
                    try:
                        diff_cache[cache_key] = peer_difference_phrase(
                            query_name=profile["name"],
                            peer_name=r["player_name"],
                            peer_team=r.get("team_label"),
                            differentiators=differentiators,
                            position=r.get("position"),
                        )
                    except Exception:
                        diff_cache[cache_key] = ""
                phrase = diff_cache.get(cache_key) or ""
                r["key_differences"] = (
                    [phrase] if phrase else
                    [d["phrase"] for d in differentiators[:2]]
                )
            else:
                r["key_differences"] = [d["phrase"] for d in differentiators[:2]]
        clicked = similar_players_table(payload)
        if clicked is not None:
            st.session_state["active_player_id"] = clicked
            st.rerun()
        st.session_state["scout_last_search"] = payload

    # ---------- 3. Compare two players ----------
    with tab_compare:
        st.markdown("#### Side-by-side, on the same axes")
        st.caption(
            "Pick a second player from the same role family. "
            "We overlay their action mix and phase profile, surface where "
            "each one does *more*, and — if AI is enabled — draft a scout "
            "brief explaining which player suits which system."
        )
        try:
            roster = modal_client.list_players().get("players", [])
        except Exception as e:
            st.error(f"Couldn't reach Modal: {e}")
            roster = []
        # Restrict candidate dropdown to the same role family for honest
        # apples-to-apples comparison.
        active_family = profile.get("family")
        candidates = [r for r in roster if r["family"] == active_family
                      and r["player_id"] != active_player_id]
        candidates.sort(key=lambda r: -r.get("n_events", 0))
        candidate_label_to_id = {
            f"{r['name']} — {(r.get('team_label') or '—')}": r["player_id"]
            for r in candidates[:300]
        }
        choice = st.selectbox(
            "Compare against", [""] + list(candidate_label_to_id.keys()),
            help=f"Other {active_family or 'role-matched'} players, top 300 by event volume.",
        )
        if choice:
            cmp_id = candidate_label_to_id[choice]
            try:
                cmp_profile = modal_client.player_profile(cmp_id)
            except Exception as e:
                st.error(f"Could not load comparison player: {e}")
                cmp_profile = None
            if cmp_profile:
                # Side-by-side header.
                head_cols = st.columns(2)
                with head_cols[0]:
                    player_card(profile, title=profile["name"])
                with head_cols[1]:
                    player_card(cmp_profile, title=cmp_profile["name"])

                # Stylistic proximity headline.
                # Compute cosine quick on-the-fly via search endpoint result
                # if it's been run, else from the profile vectors directly.
                # We don't have raw vectors here, so call replacement search.
                try:
                    sr = modal_client.search_replacements(
                        query_player_id=active_player_id,
                        top_k=300, same_family=True, min_events=0,
                    )
                    cmp_row = next(
                        (r for r in sr.get("results", []) if r["player_id"] == cmp_id),
                        None,
                    )
                    cosine = float(cmp_row["cosine"]) if cmp_row else 0.0
                except Exception:
                    cosine = 0.0
                v = cosine_to_verdict(cosine)
                st.markdown(f"**Stylistic proximity:** {v.label} ({v.hint})")

                # Overlay radar.
                action_radar_compare(
                    profile["name"], profile.get("action_mix", {}) or {},
                    cmp_profile["name"], cmp_profile.get("action_mix", {}) or {},
                    baseline=baseline_dict,
                    baseline_std=baseline_std_dict,
                    title="Style overlay (z-scores vs role peers)",
                )

                # Phase comparison bars.
                ph_a = phase_profile(profile.get("action_mix", {}) or {},
                                     profile.get("spatial_zone", []) or [])
                ph_b = phase_profile(cmp_profile.get("action_mix", {}) or {},
                                     cmp_profile.get("spatial_zone", []) or [])
                phase_bars(ph_a, profile["name"], ph_b, cmp_profile["name"])

                # Differentiator bullets.
                a_mix = profile.get("action_mix", {}) or {}
                b_mix = cmp_profile.get("action_mix", {}) or {}
                action_diff = {k: a_mix.get(k, 0.0) - b_mix.get(k, 0.0) for k in a_mix}
                diff_a_more = peer_differentiators(
                    {k: v for k, v in action_diff.items() if v > 0}, top_n=3,
                )
                diff_a_less = peer_differentiators(
                    {k: -v for k, v in action_diff.items() if v < 0}, top_n=3,
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**{profile['name']} does more of**")
                    for d in diff_a_more:
                        st.markdown(f"- {d['phrase']}")
                    if not diff_a_more:
                        st.caption("No standout 'more' direction.")
                with col_b:
                    st.markdown(f"**{cmp_profile['name']} does more of**")
                    for d in diff_a_less:
                        st.markdown(f"- {d['phrase']}")
                    if not diff_a_less:
                        st.caption("No standout 'more' direction.")

                # AI head-to-head brief.
                cfg = load_config()
                if cfg.has_openai:
                    if st.button("Draft AI head-to-head brief", key="cmp_ai"):
                        with st.spinner("Drafting head-to-head…"):
                            md = head_to_head(
                                player_a=profile, player_b=cmp_profile,
                                cosine=cosine,
                                differentiators_a_vs_b=peer_differentiators(action_diff, top_n=5),
                                phase_a=ph_a, phase_b=ph_b,
                                risk_a=risk_profile(a_mix), risk_b=risk_profile(b_mix),
                                partnership_a=partnership_requirements(
                                    a_mix, profile.get("spatial_zone", []) or [], fam),
                                partnership_b=partnership_requirements(
                                    b_mix, cmp_profile.get("spatial_zone", []) or [], fam),
                                system_fit_a=system_fit_hypothesis(
                                    ph_a, risk_profile(a_mix), fam),
                                system_fit_b=system_fit_hypothesis(
                                    ph_b, risk_profile(b_mix), fam),
                            )
                        st.session_state["scout_compare_brief"] = md
                    if st.session_state.get("scout_compare_brief"):
                        st.markdown("---")
                        st.markdown(st.session_state["scout_compare_brief"])
                else:
                    st.caption("Set OPENAI_API_KEY in deploy/.env to unlock the AI head-to-head brief.")

    # ---------- 4. What-if swap ----------
    with tab_swap:
        st.markdown("#### What if this player were replaced?")
        st.caption(
            "Pick a candidate (or let the model use the closest stylistic peers). "
            "We replay the same matches with that player in the slot and report "
            "the change in attacking output, in plain English."
        )
        # Candidate picker — two sources:
        #   1. "Top stylistic peers" — from the last Stylistic-peers search,
        #      ordered by style match. Empty if that tab hasn't been run.
        #   2. "Browse the full roster" — every player in the same role
        #      family, ordered by event volume. The fall-back when the
        #      scout knows the name they want.
        source = st.radio(
            "Pick from",
            ["Top stylistic peers", "Browse the full roster (same role)"],
            horizontal=True,
            help=(
                "**Top stylistic peers** uses the ranked output of the "
                "Stylistic-peers tab — closest style first. "
                "**Browse the full roster** lists every player of the same "
                "role family (DEF / MID / ATT / GK) sorted by event volume."
            ),
        )

        candidate_id: int | None = None
        if source == "Top stylistic peers":
            last_search = st.session_state.get("scout_last_search") or {}
            peers = last_search.get("results") or []
            if not peers:
                st.info(
                    "No peers cached yet — open the **Stylistic peers** tab and "
                    "run a search first, or switch to **Browse the full roster**."
                )
            opts = {
                "(let the model use the closest peers)": None,
                **{f"{p['player_name']} — {p.get('team_label') or '—'}":
                       p["player_id"] for p in peers},
            }
            st.caption(
                f"Source: the last Stylistic-peers search "
                f"({len(peers)} candidates ranked by style match). "
                "Increase the *Candidates to return* slider on that tab to "
                "see more here."
            )
            choice = st.selectbox("Replacement candidate", list(opts.keys()))
            candidate_id = opts[choice]
        else:
            try:
                full_roster = modal_client.list_players().get("players", [])
            except Exception as e:
                st.error(f"Couldn't reach Modal: {e}")
                full_roster = []
            same_fam = [
                r for r in full_roster
                if r.get("family") == fam and r["player_id"] != active_player_id
            ]
            same_fam.sort(key=lambda r: -r.get("n_events", 0))
            opts = {
                "(let the model use the closest peers)": None,
                **{f"{r['name']} — {r.get('team_label') or '—'} "
                   f"({(r.get('position') or '?').replace('_',' ').title()})":
                       r["player_id"] for r in same_fam[:600]},
            }
            st.caption(
                f"Source: every {fam} player in the dataset, sorted by "
                f"event volume. Showing the top {min(600, len(same_fam))} "
                f"of {len(same_fam)} players in this role family. "
                "Type to filter."
            )
            choice = st.selectbox("Replacement candidate", list(opts.keys()))
            candidate_id = opts[choice]
        col_a, col_b = st.columns(2)
        with col_a:
            season = st.selectbox("Sampling season", ["23-24", "22-23", "24-25"], index=0)
        with col_b:
            n_episodes = st.slider("Match scenarios to score", 30, 200, 80, step=10)

        if st.button("Run the swap", type="primary"):
            with st.spinner("Scoring sample episodes — this calls the GPU model…"):
                impact = modal_client.swap_impact(
                    incumbent_player_id=active_player_id,
                    candidate_player_id=candidate_id,
                    n_peers=5, season=season, max_episodes=n_episodes,
                )
            st.session_state["scout_last_swap"] = impact

        impact = st.session_state.get("scout_last_swap")
        if impact:
            verdict = delta_robv_to_verdict(
                impact.get("mean_delta", 0.0),
                impact.get("delta_ci_lo", 0.0),
                impact.get("delta_ci_hi", 0.0),
                bool(impact.get("significant", False)),
                frac_drop=float(impact.get("frac_drop", 0.5)),
            )
            # Plain-English bullets — combine WHO with HOW the play would change.
            peer_names = [p.get("name") for p in (impact.get("peers_used") or [])]
            bullets: list[str] = []
            if impact.get("candidate_name"):
                bullets.append(f"Candidate: **{impact['candidate_name']}**")
            elif peer_names:
                bullets.append("Stylistic peers used: " + ", ".join(peer_names[:5]))

            # Style-differentiation: incumbent profile is already loaded above
            # as ``profile``. Pull the candidate's profile to compute the
            # action-mix delta — this is what makes every swap card actually
            # *different* even when the rOBV impact is small.
            cand_diff_bullets: list[str] = []
            try:
                cand_id = impact.get("candidate_player_id")
                cand_mix: dict[str, float] = {}
                inc_mix = profile.get("action_mix", {}) or {}
                if cand_id is not None:
                    cand_profile = modal_client.player_profile(int(cand_id))
                    cand_mix = cand_profile.get("action_mix", {}) or {}
                elif peer_names:
                    # Average the action mix across the style peers.
                    peer_ids = [int(p["player_id"]) for p in (impact.get("peers_used") or [])
                                if p.get("player_id")]
                    mix_acc: dict[str, float] = {}
                    for pid in peer_ids:
                        try:
                            pp = modal_client.player_profile(pid)
                        except Exception:
                            continue
                        for k, v in (pp.get("action_mix") or {}).items():
                            mix_acc[k] = mix_acc.get(k, 0.0) + v
                    if peer_ids:
                        cand_mix = {k: v / len(peer_ids) for k, v in mix_acc.items()}
                if cand_mix and inc_mix:
                    diff = {k: cand_mix.get(k, 0.0) - inc_mix.get(k, 0.0)
                            for k in inc_mix}
                    cand_diff_bullets = action_diff_bullets(
                        diff, top_n=3, threshold=0.015,
                    )
            except Exception:
                pass

            for b in cand_diff_bullets:
                bullets.append(f"_{b}_")

            # Directional signal — only emit ONE coherent sentence using
            # the SAME direction as the verdict. ``frac_drop`` is the share
            # of paired episodes where the candidate produced LESS attacking
            # output, so when frac_drop < 0.5 the candidate is BETTER.
            fd = impact.get("frac_drop")
            if fd is not None:
                worse_pct = int(round(fd * 100))
                better_pct = 100 - worse_pct
                if worse_pct < 45:
                    bullets.append(
                        f"In **{better_pct}%** of matched scenarios the candidate "
                        f"produced *more* attacking output — the directional signal "
                        f"is positive but the gap per match is small."
                    )
                elif worse_pct > 55:
                    bullets.append(
                        f"In **{worse_pct}%** of matched scenarios the candidate "
                        f"produced *less* attacking output — the slot would lose "
                        f"some on-ball production."
                    )
                else:
                    bullets.append(
                        f"Roughly a coin flip across **{impact.get('n_episodes', 0)}** "
                        f"matched scenarios — the candidate finishes above the incumbent "
                        f"about as often as below."
                    )
            swap_impact_card(impact, verdict, bullets)

            # AI tactical brief — the prompt receives the action-diff bullets
            # and frac_drop directly so it can write something specific.
            cfg = load_config()
            if cfg.has_openai:
                if st.button("Draft AI tactical brief", key="swap_ai_btn"):
                    cand_top = []
                    try:
                        cid = impact.get("candidate_player_id")
                        if cid is not None:
                            cp = modal_client.player_profile(int(cid))
                            cand_top = _top_action_phrases(cp.get("action_mix", {}) or {}, top_n=4)
                    except Exception:
                        pass
                    fd = impact.get("frac_drop")
                    fd_pct = int(round(fd * 100)) if fd is not None else None
                    with st.spinner("Drafting tactical brief…"):
                        narrative = swap_narrative(
                            incumbent_name=impact.get("incumbent_name", profile["name"]),
                            incumbent_team=profile.get("team_label"),
                            incumbent_position=profile.get("position"),
                            candidate_or_peers=(
                                [{"name": impact.get("candidate_name"),
                                  "team_label": None}]
                                if impact.get("candidate_name")
                                else (impact.get("peers_used") or [])
                            ),
                            impact_verdict_label=verdict.label,
                            impact_hint=verdict.hint,
                            action_diff_bullets=cand_diff_bullets,
                            frac_drop_pct=fd_pct,
                            n_episodes=int(impact.get("n_episodes", 0) or 0),
                            incumbent_action_top=_top_action_phrases(
                                profile.get("action_mix", {}) or {}, top_n=4,
                            ),
                            candidate_action_top=cand_top,
                        )
                    st.session_state["scout_swap_narrative"] = narrative
                if st.session_state.get("scout_swap_narrative"):
                    st.markdown("---")
                    st.markdown(st.session_state["scout_swap_narrative"])
            else:
                st.caption("Set OPENAI_API_KEY in deploy/.env to unlock the AI tactical brief.")
        else:
            st.caption("Click 'Run the swap' to score the replacement against the incumbent.")

    # ---------- 4. AI report ----------
    with tab_ai:
        cfg = load_config()
        if not cfg.has_openai:
            st.warning(
                "OPENAI_API_KEY is not set — AI report generation is disabled. "
                "Add it to deploy/.env and reload."
            )
        else:
            st.caption(
                "A one-page scouting report combining the style fingerprint, "
                "closest peers, tactical depth read, and caveats into prose you "
                "can paste straight into a recruitment doc."
            )
            if st.button("Draft scouting one-pager", type="primary"):
                last_search = st.session_state.get("scout_last_search") or {}
                peers = []
                diffs_by_name: dict[str, list] = {}
                for r in (last_search.get("results") or [])[:5]:
                    peer_name = r["player_name"]
                    peers.append({
                        "player_name": peer_name,
                        "team_label": r.get("team_label"),
                        "match_phrase": r.get("match_phrase"),
                        "key_differences": r.get("key_differences", []),
                    })
                    diffs_by_name[peer_name] = peer_differentiators(
                        r.get("action_diff") or {}, top_n=3,
                    )
                arch_label = (
                    archetype_to_label(arch_cluster, action_families=profile.get("action_families"))
                    if arch_cluster else "Role archetype not assigned"
                )
                # Recompute the deep signals so the prompt is self-contained
                # (the Profile tab already showed them but we don't want to
                # depend on tab order of execution).
                amix = profile.get("action_mix", {}) or {}
                sz = profile.get("spatial_zone", []) or []
                ph = phase_profile(amix, sz)
                with st.spinner("Drafting the scouting report…"):
                    md = scouting_pager(
                        profile=profile, peers=peers,
                        archetype_label=arch_label,
                        team_fit_summary=None,
                        phase_profile=ph,
                        risk_profile=risk_profile(amix),
                        defensive_workload=defensive_workload(
                            amix, family_baseline=baseline_dict, family=fam,
                        ),
                        partnership_requirements=partnership_requirements(amix, sz, fam),
                        system_fit=system_fit_hypothesis(ph, risk_profile(amix), fam),
                        caveats=caveats_for_player(profile, family_baseline_n=n_peers),
                        peer_differentiators_by_name=diffs_by_name,
                    )
                st.session_state["scout_last_report"] = md
            md = st.session_state.get("scout_last_report")
            if md:
                # Split on section boundaries to inject real charts
                overview_split = md.split("\n## Involvement", 1)
                st.markdown(overview_split[0])
                action_radar(
                    profile.get("action_mix", {}),
                    baseline=baseline_dict,
                    baseline_std=baseline_std_dict,
                    baseline_label=baseline_label,
                    title=f"Style fingerprint vs {fam} peers" if baseline_dict else "Style fingerprint",
                    key="ai_report_radar",
                )
                if len(overview_split) > 1:
                    progression_split = (
                        "\n## Involvement" + overview_split[1]
                    ).split("\n## Stylistic peers", 1)
                    st.markdown(progression_split[0])
                    pitch_heatmap(profile.get("spatial_zone", []) or [], key="ai_report_heatmap")
                    if len(progression_split) > 1:
                        st.markdown("\n## Stylistic peers" + progression_split[1])
                st.download_button(
                    "Download report (.md)", data=md,
                    file_name=f"scouting_{profile['name'].replace(' ', '_')}.md",
                    mime="text/markdown",
                )
