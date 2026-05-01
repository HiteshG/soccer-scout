"""Deterministic metric → plain English translator.

The product rule: scouts and sporting directors should never see the words
*cosine*, *embedding*, *KL*, *rOBV*, *vector*. Every numeric output gets
mapped through this module before it reaches the UI.

Each function takes a metric and returns one of:
  * a short verdict word ("Very similar", "Different style"),
  * a sentence ("Operates higher up the pitch"),
  * a structured ``{label, tone, hint}`` dict for badges/gauges.

Tone band conventions used by the UI components:
  * ``positive`` — green tint (good fit, strong upgrade)
  * ``neutral``  — grey
  * ``warn``     — amber (uncertain / mild downside)
  * ``negative`` — red (clear downside)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Verdict:
    label: str
    tone: str   # "positive" | "neutral" | "warn" | "negative"
    hint: str = ""


_ACTION_PRETTY = {
    "Pass": "passing", "Carry": "ball-carrying", "Cross": "crossing",
    "TakeOn": "take-ons", "Shot": "shooting", "Tackle": "tackling",
    "Interception": "interceptions", "Clearance": "clearing",
    "Aerial": "aerial duels", "Duel": "physical duels",
    "Other": "set-piece work",
}


def trait_headline(
    action_mix: dict[str, float],
    baseline_mean: dict[str, float] | None,
    baseline_std: dict[str, float] | None,
    family: str | None = None,
) -> str:
    """Convert z-scores into a one-line scouting headline rendered in the
    hero card. Picks the single biggest standout trait and frames it
    role-relative ('Elite for a centre-back at long passing')."""
    if not action_mix or not baseline_mean or not baseline_std:
        return ""
    z_pairs = []
    for fam, share in action_mix.items():
        std = max(baseline_std.get(fam, 0.05), 0.01)
        z = (share - baseline_mean.get(fam, 0.05)) / std
        z_pairs.append((fam, z))
    z_pairs.sort(key=lambda p: -abs(p[1]))
    if not z_pairs or abs(z_pairs[0][1]) < 0.85:
        return f"Plays a role-typical profile for a {family or 'player'} — no extreme traits."
    fam, z = z_pairs[0]
    pretty = _ACTION_PRETTY.get(fam, fam.lower())
    role_word = (family or "player").lower()
    if z >= 1.75:
        return f"Elite for a {role_word} at {pretty}."
    if z >= 0.85:
        return f"Above peers for a {role_word} at {pretty}."
    if z <= -1.75:
        return f"Notably light for a {role_word} on {pretty}."
    return f"Below role peers for a {role_word} on {pretty}."


# --------- similarity / fit -----------------------------------------------

def cosine_to_phrase(cos: float) -> str:
    """Translate cosine ∈ [-1, 1] to a plain-English similarity phrase."""
    if cos >= 0.85:
        return "Almost identical style"
    if cos >= 0.75:
        return "Very similar style"
    if cos >= 0.6:
        return "Broadly similar style"
    if cos >= 0.45:
        return "Some overlap"
    if cos >= 0.25:
        return "Different style"
    return "Very different style"


def cosine_to_pct(cos: float) -> int:
    """Cosine → 'style match %' for headline display.

    Maps the realistic 0.4–1.0 cosine range to 0–100 so a 0.79 reads as
    ~80% match (intuitive for non-technical readers) without compressing
    the high end."""
    pct = (cos - 0.4) / 0.6 * 100
    return max(0, min(100, int(round(pct))))


def cosine_to_verdict(cos: float) -> Verdict:
    pct = cosine_to_pct(cos)
    label = cosine_to_phrase(cos)
    if cos >= 0.75:
        tone = "positive"
    elif cos >= 0.6:
        tone = "neutral"
    elif cos >= 0.45:
        tone = "warn"
    else:
        tone = "negative"
    return Verdict(label=label, tone=tone, hint=f"~{pct}% style match")


def fit_to_verdict(fit_score: float) -> Verdict:
    """For the team-fit gauge. Same scale as cosine but worded as fit."""
    pct = cosine_to_pct(fit_score)
    if fit_score >= 0.8:
        return Verdict("Strong fit", "positive", f"~{pct}% match to club style")
    if fit_score >= 0.65:
        return Verdict("Good fit", "positive", f"~{pct}% match")
    if fit_score >= 0.5:
        return Verdict("Workable fit", "neutral", f"~{pct}% match — adapts but not natural")
    if fit_score >= 0.35:
        return Verdict("Stretched fit", "warn", f"~{pct}% match — would change how the team plays")
    return Verdict("Mismatch", "negative", f"~{pct}% match — clashes with current style")


# --------- swap impact -----------------------------------------------------

def delta_robv_to_verdict(
    mean_delta: float,
    ci_lo: float,
    ci_hi: float,
    significant: bool,
    *,
    frac_drop: float = 0.5,
) -> Verdict:
    """Δ rOBV + frac_drop → 'effect on attacking output' verdict.

    rOBV deltas are tiny in absolute terms (paired-within-episode bootstrap
    CIs around 0 are normal). Significance alone is a noisy gate — for an
    80-episode sample, almost no swap is "significant" even when the
    candidate is consistently better/worse. We lead with ``frac_drop`` (the
    directional signal: share of paired episodes where the candidate
    produced *less* rOBV than the incumbent), and use ``mean_delta`` magnitude
    plus significance to upgrade the strength of the wording.

    ``frac_drop`` semantics:
      * 0.50 → swap is a coin flip; truly indifferent.
      * 0.60 → candidate is worse in ~60% of matched episodes.
      * 0.40 → candidate is better in ~60% of matched episodes.
    """
    direction = frac_drop - 0.5   # >0 → candidate worse on average
    abs_dir = abs(direction)
    big_mag = abs(mean_delta) >= 0.0008
    mid_mag = abs(mean_delta) >= 0.0003

    # Truly flat: directional signal weak AND magnitude tiny.
    if abs_dir < 0.07 and not mid_mag:
        return Verdict(
            "Essentially unchanged",
            "neutral",
            "The two players produce roughly the same attacking output in these matches.",
        )

    if direction > 0:
        # Candidate produces LESS rOBV → downgrade.
        if (significant and big_mag) or abs_dir > 0.20:
            return Verdict(
                "Clear downgrade",
                "negative",
                "Attacking output drops noticeably — the candidate produces less than the incumbent in most of these episodes.",
            )
        return Verdict(
            "Small drop",
            "warn",
            "Attacking output drifts down slightly — the candidate is below the incumbent more often than not.",
        )
    else:
        if (significant and big_mag) or abs_dir > 0.20:
            return Verdict(
                "Clear upgrade",
                "positive",
                "Attacking output rises noticeably — the candidate produces more than the incumbent in most of these episodes.",
            )
        return Verdict(
            "Small uplift",
            "positive",
            "Attacking output drifts up slightly — the candidate is above the incumbent more often than not.",
        )


# --------- archetype labelling --------------------------------------------

ARCHETYPE_TEMPLATES: dict[str, str] = {
    "GK":  "Goalkeeper — distributes from the back, claims aerial balls, anchors the build-up.",
    "DEF": "Defender — covers the back line, contests duels and aerials, recycles possession.",
    "MID": "Midfielder — connects defence and attack, breaks up play, sets tempo.",
    "ATT": "Attacker — creates and finishes chances, threatens in the final third.",
}


def archetype_to_label(
    cluster: dict, action_families: list[str] | None = None,
) -> str:
    """One-line plain-English label for an archetype cluster.

    Uses the dominant family + top-action centroid to compose phrases like
    "Press-resistant midfielder — heavy passing, low aerial work" or
    "Inverted wide forward — heavy carrying, frequent take-ons".
    """
    fam = cluster.get("dominant_family", "?")
    top_actions = cluster.get("top_actions") or []
    base = ARCHETYPE_TEMPLATES.get(fam, "Player archetype")
    if not top_actions:
        return base
    descriptor_map = {
        "Pass":         "heavy passing involvement",
        "Carry":        "heavy ball-carrying",
        "Cross":        "wide deliveries",
        "TakeOn":       "frequent take-ons",
        "Shot":         "shooting volume",
        "Tackle":       "ball-winning tackles",
        "Interception": "reading and intercepting",
        "Clearance":    "clearing under pressure",
        "Aerial":       "aerial duels",
        "Duel":         "physical contests",
        "Other":        "set-piece involvement",
    }
    parts = []
    for spec in top_actions[:2]:
        fam_action = spec.get("family") if isinstance(spec, dict) else None
        if fam_action and fam_action in descriptor_map:
            parts.append(descriptor_map[fam_action])
    detail = "; ".join(parts) if parts else ""
    return f"{base} ({detail})" if detail else base


# --------- action-mix difference ------------------------------------------

ACTION_DIFF_PHRASES_POS = {
    "Pass":         "More involved in build-up",
    "Carry":        "Carries the ball further",
    "Cross":        "Plays more crosses",
    "TakeOn":       "Takes opponents on more often",
    "Shot":         "Shoots more",
    "Tackle":       "Wins more tackles",
    "Interception": "Reads play and intercepts more",
    "Clearance":    "Clears more often",
    "Aerial":       "Wins more aerial duels",
    "Duel":         "Engages in more physical duels",
    "Other":        "Takes more set-pieces",
}
ACTION_DIFF_PHRASES_NEG = {
    "Pass":         "Less involved in build-up",
    "Carry":        "Carries the ball less",
    "Cross":        "Crosses less often",
    "TakeOn":       "Takes on opponents less",
    "Shot":         "Shoots less",
    "Tackle":       "Tackles less",
    "Interception": "Intercepts less",
    "Clearance":    "Clears less often",
    "Aerial":       "Less aerial work",
    "Duel":         "Less physical involvement",
    "Other":        "Fewer set-pieces",
}


def action_diff_bullets(
    action_diff: dict[str, float], top_n: int = 3, threshold: float = 0.02,
) -> list[str]:
    """Convert a per-action mix delta into 2–3 plain-English bullets.

    Drops the channel from the report if |delta| < threshold (2 percentage
    points of action share is the usual noise floor). Returns at most ``top_n``
    bullets, sorted by absolute magnitude — the loudest stylistic differences
    first.
    """
    if not action_diff:
        return []
    items = sorted(action_diff.items(), key=lambda kv: -abs(kv[1]))
    out: list[str] = []
    for fam, diff in items:
        if abs(diff) < threshold:
            continue
        if diff > 0:
            phrase = ACTION_DIFF_PHRASES_POS.get(fam, f"More {fam.lower()}")
        else:
            phrase = ACTION_DIFF_PHRASES_NEG.get(fam, f"Less {fam.lower()}")
        out.append(phrase)
        if len(out) >= top_n:
            break
    return out


def fit_reasoning_bullets(
    action_diff_vs_team: dict[str, float],
    peers_in_team: list[dict],
    top_action_diffs: int = 3,
) -> list[str]:
    """Reasoning bullets for the Strategy team-fit panel.

    Combines the candidate-vs-team action delta and the candidate's nearest
    peer inside the team into 3–4 sentences a sporting director can read at
    a glance.
    """
    bullets = action_diff_bullets(action_diff_vs_team, top_n=top_action_diffs, threshold=0.015)
    if peers_in_team:
        top_peer = peers_in_team[0]
        bullets.append(
            f"Closest peer in the squad: {top_peer['name']}"
            + (" (likely overlap)" if top_peer.get("cosine", 0) > 0.8 else " (complement)")
        )
    return bullets[:4]


# =========================================================================
# Deeper, scout-language signals — derived deterministically from the
# action_mix and spatial_zone marginals so they're cheap, stable, and
# reviewable without an LLM round-trip.
# =========================================================================


def phase_profile(
    action_mix: dict[str, float],
    spatial_zone: list[float],
) -> dict[str, float]:
    """Translate action_mix × spatial_zone into the six phases scouts think
    in. Approximate (action and zone are marginals, not joint), but signal-
    strong: the answer reads as "X% build-up, Y% creation" rather than
    "0.6 Pass, 0.04 Shot".

    Phases:
      * build_up      — passing in own third (CB/FB/DM territory)
      * progression   — passing/carrying in middle third (CM/AM)
      * creation      — passes/crosses/take-ons in final third (W/AM)
      * finishing     — shots + final-third presence
      * defense       — tackles/interceptions/clearances/aerials weighted
                        toward own + middle third
      * set_pieces    — set-piece (Other) volume

    Output sums to ~1.0; any tail is captured as ``transition``.
    """
    if not action_mix or not spatial_zone or len(spatial_zone) != 16:
        return {}
    sz = list(spatial_zone)
    own_third = sum(sz[i] for i in range(16) if (i % 4) == 0)
    def_mid = sum(sz[i] for i in range(16) if (i % 4) == 1)
    att_mid = sum(sz[i] for i in range(16) if (i % 4) == 2)
    att_third = sum(sz[i] for i in range(16) if (i % 4) == 3)
    s = own_third + def_mid + att_mid + att_third
    if s <= 0:
        own_third = def_mid = att_mid = att_third = 0.25
    else:
        own_third /= s; def_mid /= s; att_mid /= s; att_third /= s

    pass_share = action_mix.get("Pass", 0.0)
    carry_share = action_mix.get("Carry", 0.0)
    cross_share = action_mix.get("Cross", 0.0)
    take_share = action_mix.get("TakeOn", 0.0)
    shot_share = action_mix.get("Shot", 0.0)
    tackle = action_mix.get("Tackle", 0.0)
    inter = action_mix.get("Interception", 0.0)
    clear_ = action_mix.get("Clearance", 0.0)
    aerial = action_mix.get("Aerial", 0.0)
    other = action_mix.get("Other", 0.0)

    build_up = pass_share * own_third + 0.5 * carry_share * own_third
    progression = pass_share * def_mid + carry_share * (def_mid + att_mid) * 0.7
    creation = (pass_share * att_third * 0.6 + cross_share * att_third
                + take_share * (att_mid + att_third) * 0.5)
    finishing = shot_share + att_third * 0.05
    defense = (tackle + inter + clear_ + 0.5 * aerial) * (own_third + def_mid)
    set_pieces = other

    raw = dict(build_up=build_up, progression=progression, creation=creation,
               finishing=finishing, defense=defense, set_pieces=set_pieces)
    total = sum(raw.values())
    if total <= 0:
        return raw
    norm = {k: v / total for k, v in raw.items()}
    norm["transition"] = max(0.0, 1.0 - sum(norm.values()))
    return norm


def phase_profile_phrases(profile: dict[str, float]) -> list[str]:
    """One-line tactical phrases describing the player's phase emphasis."""
    if not profile:
        return []
    items = sorted(profile.items(), key=lambda kv: -kv[1])
    label_map = {
        "build_up":    "build-up specialist (heavy own-third passing)",
        "progression": "progressive carrier (mid-third ball-mover)",
        "creation":    "final-third creator (deliveries, cut-backs, take-ons)",
        "finishing":   "finisher (high shot involvement)",
        "defense":     "defensive workhorse (tackles, interceptions, clearances)",
        "set_pieces":  "set-piece taker (corners / free-kicks volume)",
        "transition":  "transition runner",
    }
    out: list[str] = []
    for phase, share in items:
        if share < 0.10:
            continue
        out.append(f"{int(round(share * 100))}% {label_map.get(phase, phase)}")
        if len(out) >= 3:
            break
    return out


def risk_profile(action_mix: dict[str, float]) -> str:
    """Aggressive ↔ conservative axis. Take-Ons + Crosses are both risky
    (low completion rate); Pass without Take-On is the safer end."""
    if not action_mix:
        return "unknown"
    risky = action_mix.get("TakeOn", 0.0) + action_mix.get("Cross", 0.0)
    safe = action_mix.get("Pass", 0.0)
    if risky >= 0.10 and safe < 0.55:
        return "Chaos creator — high take-on / cross volume, low passing share"
    if risky >= 0.07:
        return "Calculated risk — meaningful take-on or crossing volume on top of normal passing"
    if safe >= 0.62:
        return "Conservative ball-circulator — low individual risk, high passing share"
    return "Balanced — mixes safe passes with occasional individual actions"


def defensive_workload(
    action_mix: dict[str, float],
    family_baseline: dict[str, float] | None,
    family: str,
) -> str:
    """How much defensive work this player does *for their role* (vs the
    family baseline). A winger doing as much defensive work as a CM is a big
    tactical signal."""
    if not action_mix or not family_baseline:
        return "unknown"
    def_actions = ["Tackle", "Interception", "Clearance", "Aerial", "Duel"]
    p = sum(action_mix.get(a, 0.0) for a in def_actions)
    b = sum(family_baseline.get(a, 0.0) for a in def_actions)
    if b <= 0:
        return "unknown"
    ratio = p / b
    if ratio >= 1.30:
        return f"Heavy defensive workload for a {family} ({ratio:.1f}× the role baseline) — tracks back, contests duels"
    if ratio >= 1.10:
        return f"Above-average defensive contribution for a {family} ({ratio:.1f}×)"
    if ratio <= 0.70:
        return f"Low defensive workload for a {family} ({ratio:.1f}×) — likely a rest-defender or specialist"
    return f"Typical defensive workload for a {family}"


def partnership_requirements(
    action_mix: dict[str, float],
    spatial_zone: list[float],
    family: str,
) -> list[str]:
    """What does this player NEED around them? Concrete partnership
    archetypes scouts trade in: 'needs a destroyer #6 behind him', 'needs
    overlapping FB outside him', 'needs target-man to occupy CBs'."""
    out: list[str] = []
    if not action_mix:
        return out
    take = action_mix.get("TakeOn", 0.0)
    cross = action_mix.get("Cross", 0.0)
    pass_share = action_mix.get("Pass", 0.0)
    shot = action_mix.get("Shot", 0.0)
    aerial = action_mix.get("Aerial", 0.0)
    defensive = (action_mix.get("Tackle", 0.0) + action_mix.get("Interception", 0.0))

    if family == "ATT":
        if take >= 0.07 and cross < 0.05:
            out.append("Inverted-style winger — needs an overlapping FB to provide the touchline")
        if cross >= 0.05 and take >= 0.05:
            out.append("Width-creator — fits a system that runs the wide channels and arrives late in the box")
        if shot >= 0.06 and aerial < 0.05:
            out.append("Penalty-area finisher — needs a creator who feeds the box via cut-backs or low crosses")
        if aerial >= 0.06:
            out.append("Aerial threat — pairs with a winger who delivers from deep")
    elif family == "MID":
        if defensive >= 0.20 and pass_share < 0.55:
            out.append("Ball-winning mid — needs a deep-lying playmaker beside him to circulate possession")
        if pass_share >= 0.62 and defensive < 0.15:
            out.append("Deep-lying playmaker — needs a destroyer #6 to do the dirty work")
        if take >= 0.04:
            out.append("Press-resistant — useful in possession-heavy systems with high pressing trigger")
    elif family == "DEF":
        if pass_share >= 0.60 and aerial < 0.05:
            out.append("Ball-playing defender — best in a high-line possession side")
        if aerial >= 0.07:
            out.append("Aerial-dominant CB — anchors a low-block or back-three")
        if take >= 0.03:
            out.append("Inverted-FB profile — needs a winger holding width outside him")
    elif family == "GK":
        if pass_share >= 0.50:
            out.append("Sweeper-keeper — fits a possession side with a high defensive line")
        else:
            out.append("Traditional keeper — better behind a deep block with aerial-strong CBs")
    return out


def caveats(profile: dict, family_baseline_n: int | None = None) -> list[str]:
    """Sample-size / flexibility / data caveats for the report footer."""
    out: list[str] = []
    n = int(profile.get("n_events", 0) or 0)
    if n < 1500:
        out.append(
            f"Limited sample ({n:,} events) — confidence on rare actions (shots, set-pieces) is lower."
        )
    elif n < 4000:
        out.append(f"Modest sample ({n:,} events) — robust on volume actions, less so on rare ones.")
    flex = profile.get("pos_entropy", 0.0)
    if flex >= 0.55:
        out.append(
            "Plays multiple roles (high positional flexibility) — the profile is an average across "
            "those roles and may underweight any single one. Cross-check against video for the role you're recruiting for."
        )
    if family_baseline_n is not None and family_baseline_n < 60:
        out.append(
            f"Role peer-group is small (n={family_baseline_n}) — comparisons against this baseline carry more noise than usual."
        )
    return out


def peer_differentiators(
    action_diff: dict[str, float], top_n: int = 3,
) -> list[dict]:
    """Top-N strongest stylistic differentiators between candidate and query,
    each as a structured ``{action, direction, magnitude, phrase}``. The LLM
    consumes this directly to build "Madueke crosses 1.8× more, shoots from
    deeper, wins 2× more aerials"-style sentences.
    """
    if not action_diff:
        return []
    items = sorted(action_diff.items(), key=lambda kv: -abs(kv[1]))
    out: list[dict] = []
    for fam, diff in items:
        if abs(diff) < 0.015:
            continue
        direction = "more" if diff > 0 else "less"
        # Express as "1.8× more" / "0.6× less" only when the change is large.
        if abs(diff) >= 0.05:
            magnitude_phrase = f"considerably {direction}"
        elif abs(diff) >= 0.025:
            magnitude_phrase = f"noticeably {direction}"
        else:
            magnitude_phrase = f"slightly {direction}"
        action_phrase = {
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
        }.get(fam, fam.lower())
        out.append({
            "action": fam,
            "direction": direction,
            "magnitude": abs(diff),
            "phrase": f"{magnitude_phrase} {action_phrase}",
        })
        if len(out) >= top_n:
            break
    return out


def system_fit_hypothesis(
    phase: dict[str, float],
    risk: str,
    family: str,
) -> list[str]:
    """Plain-English mapping from phase profile + risk + family to the
    tactical SYSTEMS this player fits. Up to 3 systems, ordered by fit."""
    out: list[str] = []
    if not phase:
        return out
    creation = phase.get("creation", 0.0)
    build_up = phase.get("build_up", 0.0)
    defense = phase.get("defense", 0.0)
    finishing = phase.get("finishing", 0.0)

    if family == "ATT":
        if creation >= 0.20 and "Chaos" in risk:
            out.append("Possession-heavy 4-3-3 with inverted wide forwards (Arsenal / Man City template)")
        elif finishing >= 0.10:
            out.append("Direct 4-2-3-1 with a #10 supporter and a single counter-press (Newcastle template)")
        if creation >= 0.15 and defense < 0.10:
            out.append("Asymmetric back-three with one inverted FB (Brighton / Chelsea template)")
    elif family == "MID":
        if build_up >= 0.25 and "Conservative" in risk:
            out.append("Possession-control 4-3-3 anchored by a regista (Liverpool template)")
        if defense >= 0.20:
            out.append("Mid-block 4-4-2 with two ball-winning #8s (Brentford / Forest template)")
        if "Press-resistant" in risk:
            out.append("High-press 3-4-3 with double-pivot under pressure (Bayer Leverkusen-esque)")
    elif family == "DEF":
        if build_up >= 0.30:
            out.append("High-line possession 4-3-3 with progressive CBs (Man City template)")
        if defense >= 0.30:
            out.append("Low-block 5-4-1 / 4-5-1 with deep CBs and minimal possession (Burnley template)")
    elif family == "GK":
        if build_up >= 0.20:
            out.append("Possession side with a sweeper-keeper (Man City / Brighton template)")
        else:
            out.append("Counter-attacking side with a shot-stopping #1 (Forest / Burnley template)")
    return out[:3]
