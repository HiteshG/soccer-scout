"""OpenAI prompt templates for the AI report tabs.

Three template entry points:
  * ``scouting_pager(...)`` — Scout-mode 1-page profile of a target player,
    with peers + best-fit-team commentary + suggested role.
  * ``swap_narrative(...)`` — 3-bullet impact story for the what-if swap
    panel (more conversational than the verdict copy in ``explainer.py``).
  * ``board_memo(...)`` — Strategy-mode shortlist memo comparing 3-5 players.

The system prompt is the most important part: it forbids ML jargon and
forces the model to lead with conclusions, back claims with names + plain-
English action descriptions, and use what-if framing.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import load_config


SYSTEM_PROMPT = """\
You are a senior football scout writing for a Premier-League club's
recruitment department. Your reader is a chief scout or sporting director
who has watched thousands of matches but does NOT speak data-science.

You DESCRIBE players. You do not recommend Sign / Pursue / Pass — the
recruitment committee makes that call. Your job is to make the player
legible: what they do, where they do it, and how they compare to others
in the same role.

Style anchor — your prose should feel like Twelve Football's reports:
short, descriptive paragraphs (3–4 sentences), every paragraph ending
with an *implication* ("...indicating room for improvement", "...marking
him as a standout performer", "...limiting his impact on the game"),
NEVER a recommendation. Each section is paired with a chart and the
prose names the metrics on that chart.

Hard rules — every output you produce must obey all of these:

1. Forbidden vocabulary. Never use: cosine, embedding, vector, KL,
   divergence, rOBV, manifold, latent, neural, dataset, sample size,
   confidence interval, p-value, model, prediction, inference. Translate
   them into football language ("style match", "attacking output",
   "build-up volume", "matched scenarios").

2. Be a scout, not a chatbot. Speak like someone who has watched the
   player live: tactical phases ("high press", "low block", "second
   phase"), zones ("right half-space", "between the lines",
   "ten-yards-from-goal"), partnerships ("good fit beside a destroyer
   #6", "needs an overlapping FB outside him"), and known peers ("plays
   like a younger Mahrez but with Saka's discipline off the ball").

3. Lead each section with a SUB-HEADLINE — a short noun-phrase label
   (e.g. "Highly active in attack but struggles defensively",
   "Standout performer with high recoveries"). Then 3–4 sentences of
   prose describing the chart for that section. Avoid generic adjectives
   without an action behind them ("technical" is empty; "carries the
   ball 30+ yards before releasing" is gold).

4. Frame what-ifs concretely. When discussing transfers or swaps, write
   "If you replaced [name] with [candidate], expect [team] to lose
   [specific quality] and gain [specific quality]". Don't hedge with
   "could possibly". State the directional change confidently when the
   data supports it.

5. Reference the supplied numbers when they help, but ALWAYS in football
   units: "involved in 60% of build-up sequences", "wins 7-in-10 aerial
   duels", "the candidate produced more attacking output in 64% of
   matched scenarios". Never use raw decimals.

6. NO Sign / Pursue / Pass verdict. The committee decides; you describe.
   End each section with an *implication sentence*, not a recommendation
   — e.g. "...marking him as a standout performer among wingers" or
   "...indicating room for improvement in those positions".

7. NO trailing "Limitations" footer. Caveats only appear when the data
   is genuinely thin (small sample, unassigned archetype) — and then
   only as a single inline sentence, not a separate section.

8. Format: clean Markdown. No emoji. No filler ("In conclusion…", "It
   is worth noting that…"). Short paragraphs of 2–4 sentences.

9. Charts. When asked to include charts, embed them as Markdown image
   placeholders with descriptive alt-text the host app will substitute,
   e.g. `![Style fingerprint vs role peers](style-radar)` or
   `![Pitch heatmap of touches](pitch-heatmap)`. Reference each chart
   by name once in the prose so the layout has a chart caption to
   anchor.

10. Top-3 stylistic peers MUST be named explicitly with their current
    club in parentheses. Do this once, in its own labelled section.
    Format each line as: `**Name** (Current Club) — one-sentence
    on the shared profile and the standout difference`.
"""


def _client() -> OpenAI:
    cfg = load_config()
    if not cfg.has_openai:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to deploy/.env."
        )
    return OpenAI(api_key=cfg.openai_api_key)


def _complete(
    user_prompt: str, *, max_output_tokens: int = 900, fast: bool = False,
    extra_system: str | None = None,
) -> str:
    """Call the Responses API (the modern path for gpt-5.x).

    ``fast=True`` swaps in ``OPENAI_FAST_MODEL`` (default gpt-5.4-mini) for
    less-critical output: per-row peer phrases, per-card swap narratives,
    head-to-head briefs. Headline outputs (full scouting one-pager,
    board-style memos) keep the heavier model.

    ``extra_system`` lets a caller append a per-call instruction to the
    system prompt (used for low-sample caveat injection: "this player
    has only N events, soften every claim and add an inline caveat").
    """
    cfg = load_config()
    client = _client()
    model = cfg.openai_fast_model if fast else cfg.openai_model
    system = SYSTEM_PROMPT + ("\n\n" + extra_system if extra_system else "")
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text or ""


def _extract_claims(structured_payload: dict) -> list[str]:
    """Stage-1 of two-stage prompting: ask the fast model to list the
    5–8 *atomic* football claims grounded in the structured context.
    Each claim must cite the field it came from. Stage 2 prose then
    has to incorporate at least 4 of these. Cuts hallucination at the
    cost of one extra fast call.
    """
    user = (
        "Read the structured scouting context below and return 5–8 atomic, "
        "verifiable football claims about this player. Each line MUST be "
        "one sentence, MUST be in football language (no analytics jargon), "
        "MUST end with `[from: field_name]` indicating which payload field "
        "it came from. Output only the bulleted list, no preamble.\n\n"
        f"{json.dumps(structured_payload, indent=2)}"
    )
    raw = _complete(user, max_output_tokens=400, fast=True)
    lines = [
        ln.strip(" -*•").strip() for ln in raw.splitlines()
        if ln.strip() and "[from:" in ln
    ]
    return lines[:8]


def _self_critique(markdown: str, claims: list[str]) -> str:
    """Stage-3 pass — find sentences in ``markdown`` that don't have a
    number, named peer, or specific zone behind them, and rewrite them.
    Returns the revised Markdown. If the call fails, returns the original."""
    if not markdown:
        return markdown
    user = (
        "Below is a scouting report and the list of grounded claims it was "
        "supposed to cite. Identify any sentence in the report that asserts "
        "something *without* a number, a named peer, or a specific pitch "
        "zone behind it — rewrite each one to add that grounding from the "
        "claims list. Return the FULL revised report in Markdown. Do not "
        "change the section headings or order.\n\n"
        f"## Grounded claims\n" + "\n".join(f"- {c}" for c in claims) +
        f"\n\n## Report\n{markdown}"
    )
    try:
        return _complete(user, max_output_tokens=1500, fast=True) or markdown
    except Exception:
        return markdown


# ----- Templates -----------------------------------------------------------


def scouting_pager(
    *,
    profile: dict,
    peers: list[dict],
    archetype_label: str,
    team_fit_summary: dict | None = None,
    phase_profile: dict[str, float] | None = None,
    risk_profile: str | None = None,
    defensive_workload: str | None = None,
    partnership_requirements: list[str] | None = None,
    system_fit: list[str] | None = None,
    caveats: list[str] | None = None,
    peer_differentiators_by_name: dict[str, list[dict]] | None = None,
) -> str:
    """One-page scouting profile in Markdown — the headline product output.

    ``profile`` and ``peers`` are structured outputs from the modal
    endpoints. Anything number-shaped is converted into football units before
    being passed to the LLM (action shares → "involved in N%", peer cosines
    → "very similar style").
    """
    action_mix = profile.get("action_mix", {}) or {}
    diff_lookup = peer_differentiators_by_name or {}
    payload = {
        "player": {
            "name": profile["name"],
            "primary_position": (profile.get("position") or "").replace("_", " ").title() or None,
            "role_family": profile.get("family"),
            "current_team": profile.get("team_label"),
            "events_in_dataset": profile.get("n_events"),
            "positional_flexibility": (
                "highly flexible — rotates between multiple roles match-to-match"
                if profile.get("pos_entropy", 0) > 0.5
                else "moderately flexible — occasionally drops into adjacent roles"
                if profile.get("pos_entropy", 0) > 0.25
                else "specialised — plays a single role consistently"
            ),
        },
        "data_driven_archetype": archetype_label,
        "style_fingerprint": {
            "headline_actions": _top_action_phrases(action_mix, top_n=4),
            "action_shares_vs_role_baseline": _action_emphasis_phrases(action_mix),
            "spatial_dominance": _spatial_phrase(
                profile.get("spatial_zone", []),
                position=profile.get("position"),
            ),
        },
        # The deep tactical signals — these are what take the report from
        # "looks like a winger" to "fits a possession 4-3-3 with an inverted
        # 8 alongside, doesn't track back so needs an aggressive RB behind".
        "phase_of_play_emphasis": phase_profile,
        "risk_profile": risk_profile,
        "defensive_workload_for_role": defensive_workload,
        "partnership_requirements": partnership_requirements or [],
        "tactical_systems_this_player_fits": system_fit or [],
        "five_closest_stylistic_peers": [
            {
                "name": p["player_name"],
                "current_team": p.get("team_label"),
                "match_phrase": p.get("match_phrase"),
                "ranked_differentiators_vs_target":
                    diff_lookup.get(p.get("player_name", "")) or
                    p.get("key_differences", []),
            }
            for p in peers[:5]
        ],
        "team_fit_summary": team_fit_summary,
        "data_caveats": caveats or [],
    }
    # Stage 1 — extract atomic claims so the prose has citations to lean on.
    claims = _extract_claims(payload)

    # Caveat injection — small samples force softer language inline.
    n_events = profile.get("n_events") or 0
    extra_system = None
    if n_events and n_events < 1000:
        extra_system = (
            f"DATA CAVEAT: This player has only {n_events:,} events on record "
            "— a small sample. Soften every claim, prefer 'limited evidence' "
            "framing, and add a single inline sentence at the end of the "
            "Overview section noting the small sample."
        )

    fam_label = (profile.get("family") or "role").lower() + "s"
    user = f"""Write a SCOUTING REPORT on **{profile['name']}** in the style
of Twelve Football's Earpiece reports.

The report DESCRIBES the player. It does NOT recommend Sign / Pursue /
Pass. It does NOT have a "Limitations" footer. Each section pairs prose
with a chart and the prose names the chart's metrics directly.

Hard requirements:
- Reference at least TWO of the supplied phase percentages (e.g. "44% of
  his work happens in the final third") and at least ONE partnership
  requirement, in football language.
- Name at least TWO Premier-League comparables (use the supplied peer
  list — pick the ones whose ranked differentiators are most useful to
  contrast).
- Every section opens with a SUB-HEADLINE (a short noun-phrase, no verb
  needed): e.g. "Highly active in attack but struggles defensively",
  "Standout performer with high recoveries". Then 3–4 sentences.
- Every section ends with an *implication sentence*, not a recommendation.
- Incorporate at least FOUR of the grounded claims below.
- "Compared to other {fam_label}" is the consistent peer frame.

## Grounded claims (cite at least 4 of these in the prose)
{chr(10).join(f"- {c}" for c in claims) if claims else "(none extracted — write conservatively from the structured context only)"}

Markdown structure — use these exact headings:

# {profile['name']}
*One-line PAGE HEADLINE describing the player's defining trait, in
present tense. Examples of the right shape: "Dynamic 23-Year-Old Winger
at Manchester City Excels in Dribbling and Playmaking" or
"Standout Performer with High Recoveries and Defensive Intensity".
No filler.*

## Overview
A 4–6 sentence paragraph. Cover, in order: who they are (age + position +
role family + current club if known), what their dominant phase emphasis
is (cite a percentage), the 1–2 traits that make them stand out vs role
peers, and the 1–2 traits where they sit below peers. End with the
*implication sentence* — what kind of side this player is most useful to.

![Style fingerprint vs role peers](style-radar)

## Involvement
**Sub-headline** — one bold line, noun-phrase only.

3–4 sentences describing how often the player is on the ball, where in
the pitch, and how that compares to other {fam_label}. Reference the
involvement-related phases and zones from the structured context. End
with an implication sentence.

## Defence
**Sub-headline** — one bold line.

3–4 sentences describing the player's defensive contributions: tackling,
interceptions, aerial work. Compare to other {fam_label}. End with an
implication sentence.

## Progression & creation
**Sub-headline** — one bold line.

3–4 sentences describing how the player moves the ball forward and
creates chances — passes into the final third, ball carries through
lines, take-ons, crosses. Reference the partnership requirement they
impose. End with an implication.

![Where they touch the ball](pitch-heatmap)

## Stylistic peers
A short prose paragraph naming 2–3 supplied peers (with current clubs in
parentheses) and the SINGLE biggest tactical difference for each. NOT a
bullet list — flowing sentences. Each peer must be discussed with a
contrastive word (more, less, instead, whereas, but, fewer, deeper).

## Best tactical fit
One paragraph. Pick ONE system from the supplied list and explain why in
football terms. Mention which 1–2 PL clubs deploy that system. Add ONE
system this player would NOT fit and why. End with an implication.

Structured scouting context (turn into prose; do NOT dump verbatim, do
NOT echo phase percentages as decimals — convert to integer percent):
{json.dumps(payload, indent=2)}
"""
    draft = _complete(user, max_output_tokens=1500, extra_system=extra_system)
    # Stage 3 — self-critique pass to tighten ungrounded sentences.
    return _self_critique(draft, claims)


def swap_narrative(
    *,
    incumbent_name: str,
    incumbent_team: str | None,
    incumbent_position: str | None,
    candidate_or_peers: list[dict],
    impact_verdict_label: str,
    impact_hint: str,
    action_diff_bullets: list[str],
    frac_drop_pct: int | None,
    n_episodes: int,
    incumbent_action_top: list[str] | None = None,
    candidate_action_top: list[str] | None = None,
) -> str:
    """Tactical what-if narrative for the swap-card panel.

    Returns Markdown — used inline below the verdict box. Designed to give
    the scout enough to know whether to go further, not the whole story.
    """
    payload = {
        "incumbent": {
            "name": incumbent_name,
            "current_team": incumbent_team,
            "primary_position": (incumbent_position or "").replace("_", " ").title() or None,
            "what_they_do_most": incumbent_action_top or [],
        },
        "replacement_or_peers": [
            {"name": c.get("name") or c.get("player_name"),
             "current_team": c.get("team_label") or c.get("team")}
            for c in candidate_or_peers
        ],
        "candidate_top_actions": candidate_action_top or [],
        "directional_signal": {
            "verdict_label": impact_verdict_label,
            "verdict_hint": impact_hint,
            "share_of_scenarios_candidate_was_worse_pct": frac_drop_pct,
            "scenarios_compared": n_episodes,
        },
        "stylistic_differences_in_play": action_diff_bullets,
    }
    user = f"""Produce a tactical WHAT-IF report (Markdown) on replacing
**{incumbent_name}** with the candidate(s) below in their current slot.

The reader is a chief scout. Be specific, decisive, football-native.

Use exactly this structure:

### Headline
One sentence — the SINGLE biggest tactical change in concrete terms,
phrased as: "If you replaced {incumbent_name} with [candidate], expect …".
Descriptive, not prescriptive.

### What changes on the pitch
2–3 bullets. Each bullet names a specific tactical phase or zone and the
direction of change. Examples of the right shape:
- "**Build-up under pressure** — less progressive carrying out of the
  back; the team would need a deeper-lying playmaker to compensate."
- "**Final-third entries** — more crosses from the right channel,
  fewer cut-backs across the box."

### Knock-on effects on team-mates
One paragraph (3–4 sentences). Who has to do MORE work or change role?
Who BENEFITS from the new profile? End with an *implication sentence*
about the kind of system this swap would tilt the team toward — NOT a
recommendation.

Structured context (turn into prose, do NOT echo):
{json.dumps(payload, indent=2)}
"""
    return _complete(user, max_output_tokens=700, fast=True)


def peer_difference_phrase(
    *, query_name: str, peer_name: str, peer_team: str | None,
    differentiators: list[dict], position: str | None,
) -> str:
    """Generate a single short, specific tactical phrase describing how the
    peer differs from the query player. Used in the Find-similar table's
    'Key differences' column. Output is one line, ~12-22 words, written as
    a scout would talk: action + zone or action + tactical role.

    Examples of the right shape:
      "More direct ball-carrier — drives through the half-space rather than
       holding width like Cucurella."
      "Higher cross volume from the byline; lower second-phase passing —
       a width-and-finish full-back, not a possession-anchor."
    """
    payload = {
        "query_player": query_name,
        "peer": {"name": peer_name, "team": peer_team, "position": position},
        "ranked_differentiators": differentiators,
    }
    user = f"""Write ONE short tactical sentence (12–22 words) describing how
**{peer_name}** differs from **{query_name}** in football terms.

Hard rules:
- Lead with the standout difference, then add a tactical-role consequence.
- Use ACTIONS and ZONES, not numbers ("more crosses from the byline",
  "carries through the half-space", "anchors the back-post").
- The sentence MUST contain at least one explicitly contrastive word —
  one of: more, less, instead, whereas, but, fewer, deeper, higher,
  rather than. This is non-negotiable. Without contrast, the line is
  useless to a scout.
- No filler. No "compared to". No analytics jargon.
- One sentence only. No bullets. No list.

Context (use the strongest differentiator as the hook):
{json.dumps(payload, indent=2)}"""
    return _complete(user, max_output_tokens=90, fast=True).strip()


def head_to_head(
    *,
    player_a: dict,
    player_b: dict,
    cosine: float,
    differentiators_a_vs_b: list[dict],
    phase_a: dict[str, float],
    phase_b: dict[str, float],
    risk_a: str,
    risk_b: str,
    partnership_a: list[str],
    partnership_b: list[str],
    system_fit_a: list[str],
    system_fit_b: list[str],
) -> str:
    """Head-to-head scout brief comparing two players on the same role.

    Renders as Markdown for the Compare tab. Forces explicit "which fits
    which system" reasoning rather than abstract similarity.
    """
    payload = {
        "player_a": {
            "name": player_a["name"],
            "current_team": player_a.get("team_label"),
            "primary_position": (player_a.get("position") or "").replace("_", " ").title() or None,
            "phase_emphasis_pct": {k: int(round(v * 100)) for k, v in (phase_a or {}).items()},
            "risk_profile": risk_a,
            "partnership_requirements": partnership_a,
            "tactical_systems_fit": system_fit_a,
        },
        "player_b": {
            "name": player_b["name"],
            "current_team": player_b.get("team_label"),
            "primary_position": (player_b.get("position") or "").replace("_", " ").title() or None,
            "phase_emphasis_pct": {k: int(round(v * 100)) for k, v in (phase_b or {}).items()},
            "risk_profile": risk_b,
            "partnership_requirements": partnership_b,
            "tactical_systems_fit": system_fit_b,
        },
        "stylistic_proximity_phrase": (
            "almost identical" if cosine >= 0.85 else
            "very similar" if cosine >= 0.75 else
            "broadly similar" if cosine >= 0.6 else
            "different style" if cosine >= 0.45 else
            "very different"
        ),
        "ranked_differentiators_a_does_more_or_less": differentiators_a_vs_b,
    }
    user = f"""Compare **{player_a['name']}** and **{player_b['name']}**
side-by-side as a senior scout writing for a chief scout. Keep it tight —
this is the document that decides which one to recruit.

Markdown structure:

# {player_a['name']} vs {player_b['name']}
*One-line headline — overall stylistic relationship + one decisive
differentiator.*

## Shared profile — what unites them
2–3 sentences. The shared profile — phases they emphasise together,
shared on-pitch behaviours.

## Where their styles split
3 bullets. Each names ONE concrete on-pitch difference, with direction
and (if useful) magnitude. Use the ranked differentiators payload.

## Which system gets the most out of each
One paragraph (4–5 sentences). Pick the SYSTEM each player fits BEST.
If they fit the same system, name the partner who would extract more
from each. Be decisive.

## The partner each one needs
Two bullet rows (one per player) listing the partnership requirement
each player imposes on the rest of the XI.

## Best-fit recruitment context
One short paragraph (3–4 sentences) describing which kind of side
extracts the most from each player. Frame it as system fit, not a
recommendation: "[name] suits [system] because [reason]; [other name]
is better paired with [system] because [reason]." End with an implication
about the partner profile each one needs around them.

Structured scouting context (turn into prose; do NOT echo verbatim):
{json.dumps(payload, indent=2)}
"""
    return _complete(user, max_output_tokens=1100, fast=True)


def board_memo(
    *,
    target_team_label: str,
    shortlist: list[dict],
    archetype_summary: list[dict] | None = None,
) -> str:
    """1-page board memo comparing a shortlist for a target club."""
    payload = {
        "target_team": target_team_label,
        "shortlist": [
            {
                "name": p["candidate_name"],
                "fit_label": p.get("fit_verdict_label"),
                "fit_hint": p.get("fit_hint"),
                "reasoning_bullets": p.get("reasoning_bullets", []),
                "closest_existing_peer": p.get("current_top_player", {}).get("name"),
            }
            for p in shortlist
        ],
        "archetype_summary": archetype_summary or [],
    }
    user = f"""Write a 1-page board memo for {target_team_label}'s recruitment committee.

Structure (Markdown):
# Recruitment shortlist — {target_team_label}
## Recommended order
1. [name] — [one-sentence why]
2. ...
## Per-candidate breakdown
For each: 60-word paragraph in football language. Cover style, fit, what they replace or complement, risk.
## Reading of the shortlist (2 sentences, descriptive — which candidate fits which kind of side, no Sign/Pursue/Pass)

Structured context:
{json.dumps(payload, indent=2)}
"""
    return _complete(user, max_output_tokens=1200)


# ----- helpers -------------------------------------------------------------


_ACTION_LABEL_MAP = {
    "Pass":         "passing involvement",
    "Carry":        "ball-carrying",
    "Cross":        "wide deliveries",
    "TakeOn":       "take-ons",
    "Shot":         "shooting",
    "Tackle":       "tackling",
    "Interception": "interceptions",
    "Clearance":    "clearing",
    "Aerial":       "aerial duels",
    "Duel":         "physical duels",
    "Other":        "set-piece work",
}

# League-typical baseline shares (must match the radar baseline).
_TYPICAL = {
    "Pass": 0.60, "Carry": 0.22, "Cross": 0.04, "TakeOn": 0.04, "Shot": 0.04,
    "Tackle": 0.04, "Interception": 0.04, "Clearance": 0.04, "Aerial": 0.04,
    "Duel": 0.04, "Other": 0.10,
}


def _top_action_phrases(action_mix: dict[str, float], top_n: int = 3) -> list[str]:
    """Top action families as phrases for the prompt context."""
    if not action_mix:
        return []
    items = sorted(action_mix.items(), key=lambda kv: -kv[1])
    return [_ACTION_LABEL_MAP.get(k, k) for k, _v in items[:top_n]]


def _action_emphasis_phrases(action_mix: dict[str, float]) -> list[str]:
    """Phrases describing which actions a player does MORE / LESS than the
    league baseline. Helps the LLM ground its style summary in concrete
    deviations rather than absolute shares.

    Returns up to ~4 standout phrases with directional language:
      "does ~2.5× more take-ons than typical for their role"
      "noticeably less aerial work than typical"
    """
    if not action_mix:
        return []
    deltas: list[tuple[str, float]] = []
    for fam, share in action_mix.items():
        baseline = _TYPICAL.get(fam, 0.05)
        if baseline <= 0:
            continue
        ratio = share / baseline
        deltas.append((fam, ratio))
    deltas.sort(key=lambda kv: -abs(kv[1] - 1.0))
    out: list[str] = []
    for fam, ratio in deltas:
        label = _ACTION_LABEL_MAP.get(fam, fam.lower())
        if ratio >= 1.5:
            out.append(f"does ~{ratio:.1f}× more {label} than typical for their role")
        elif ratio >= 1.2:
            out.append(f"noticeably more {label} than typical")
        elif ratio <= 0.5:
            out.append(f"does ~{ratio:.1f}× the typical {label} rate (well below average)")
        elif ratio <= 0.8:
            out.append(f"notably less {label} than typical")
        if len(out) >= 4:
            break
    return out


_ZONE_LABELS = [
    # Row 0 (defensive third), row 1 (def-mid third), row 2 (att-mid), row 3 (att third)
    "own-box-left",   "own-half-left-half-space",   "own-half-right-half-space",   "own-box-right",
    "left-channel-deep",     "left-half-space",      "right-half-space",     "right-channel-deep",
    "left-channel-mid",      "left-zone-14",         "right-zone-14",        "right-channel-mid",
    "left-byline",           "left-edge-of-box",     "right-edge-of-box",    "right-byline",
]


def _spatial_phrase(spatial_zone: list[float], position: str | None = None) -> str:
    """Translate the 4×4 zone histogram into a one-line tactical phrase
    describing where the player operates most. Returns "" if no data.
    """
    if not spatial_zone or len(spatial_zone) != 16 or sum(spatial_zone) <= 0:
        return ""
    arr = list(enumerate(spatial_zone))
    arr.sort(key=lambda kv: -kv[1])
    top = [_ZONE_LABELS[i] for i, v in arr[:3] if v > 0.03]
    if not top:
        return "no dominant zone — touches scattered across the pitch"
    return "operates most in: " + ", ".join(top)
