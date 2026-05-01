"""Reusable Streamlit components.

Each component is a function that calls Streamlit + plotly primitives. They
take pre-translated, structured data (the explainer has already turned
numbers into phrases) and never accept raw cosines / vectors — that
contract keeps the UI consistent with the no-jargon product rule.
"""

from app.components.cards import (
    player_card, archetype_chip, swap_impact_card, team_fit_gauge,
    stat_strip, now_scouting_badge, empty_state,
    section_header, page_footer, caveat_block,
)
from app.components.charts import (
    action_radar, action_radar_compare, archetype_map, category_grid,
    category_mini_radar, phase_bars, pitch_heatmap, qualitative_pitch_grid,
    similar_players_table, strengths_weaknesses, team_similarity_heatmap,
)

__all__ = [
    "player_card", "archetype_chip", "swap_impact_card", "team_fit_gauge",
    "stat_strip", "now_scouting_badge", "empty_state",
    "section_header", "page_footer", "caveat_block",
    "action_radar", "action_radar_compare", "phase_bars",
    "pitch_heatmap", "similar_players_table", "archetype_map",
    "team_similarity_heatmap", "category_grid", "category_mini_radar",
    "qualitative_pitch_grid", "strengths_weaknesses",
]
