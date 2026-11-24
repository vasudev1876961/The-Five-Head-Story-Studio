"""Hamilton d10 Visualizer component."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from five_head_nlp.ui.explanation_service import ExplanationService
from five_head_nlp.ui.session_manager import SessionManager


def render_hamilton_visualizer(session_mgr: SessionManager) -> None:
    """Render the pedagogical Hamilton d10 apportionment breakdown and visual die budget."""
    cfg = session_mgr.get_config()
    weights = cfg.get("weights", {
        "Grammar Scout": 0.40,
        "Character Keeper": 0.15,
        "Setting Painter": 0.15,
        "Action Booster": 0.10,
        "Repeat Blocker": 0.20,
    })

    st.markdown("### 🎲 Hamilton d10 Apportionment Visualizer")
    st.markdown(
        "<p style='color:#8b949e;'>Hamilton's largest-remainder method converts continuous "
        "head weights into an exact 10-point budget for unplugged tabletop and classroom dice gameplay.</p>",
        unsafe_allow_html=True,
    )

    breakdown = ExplanationService.compute_hamilton_breakdown(weights, total_points=10)
    alloc = breakdown["final_allocation"]

    # Visual d10 Die Representation
    st.markdown("#### 10-Sided Die (d10) Face Assignment")
    colors = {
        "Grammar Scout": "#58a6ff",
        "Character Keeper": "#f778ba",
        "Setting Painter": "#7ee787",
        "Action Booster": "#ffa657",
        "Repeat Blocker": "#d29922",
    }

    bar_html = "<div class='d10-bar'>"
    for k in ExplanationService.HEAD_NAMES:
        pts = alloc.get(k, 0)
        if pts > 0:
            pct = pts * 10
            bar_html += f"<div class='d10-slot' style='width:{pct}%; background:{colors[k]};'>{pts} pts</div>"
    bar_html += "</div>"
    st.markdown(bar_html, unsafe_allow_html=True)

    # Summary Badges
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    cols = [b_col1, b_col2, b_col3, b_col4, b_col5]
    for i, k in enumerate(ExplanationService.HEAD_NAMES):
        pts = alloc.get(k, 0)
        cols[i].metric(label=k, value=f"{pts} / 10 faces")

    st.divider()

    # Mathematical Step-by-Step Table
    st.markdown("#### Mathematical Step-by-Step Apportionment Table")
    df = pd.DataFrame(breakdown["table"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px;">
                <b>Apportionment Invariants:</b><br>
                • Total Integer Budget: <code>K = {breakdown['total_budget']}</code><br>
                • Base Integer Sum (⌊q⌋): <code>{breakdown['base_sum']}</code><br>
                • Surplus Points Awarded: <code>+{breakdown['surplus_points']}</code><br>
                • Final Allocated Sum: <code>{sum(alloc.values())} = 10</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.info(breakdown["pedagogical_note"])
