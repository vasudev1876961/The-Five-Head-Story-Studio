"""Candidate explorer component for token-level inspection and explanation."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from five_head_nlp.ui.explanation_service import ExplanationService
from five_head_nlp.ui.session_manager import SessionManager


def render_candidate_explorer(session_mgr: SessionManager) -> None:
    """Render token-level candidate exploration and scoring breakdowns."""
    story = session_mgr.get_current_story()
    steps = session_mgr.get_current_steps()

    if not story or not steps:
        st.info("ℹ️ Generate a story first to inspect candidate token decisions.")
        return

    st.markdown("### 🔬 Candidate Token Explorer")
    st.caption("Inspect the five transparent scoring heads for any token position in the generated narrative.")

    # Step selector slider
    total_steps = len(steps)
    curr_idx = min(session_mgr.get_selected_token_idx(), total_steps - 1)

    selected_step_num = st.slider(
        "Token Generation Step",
        min_value=1,
        max_value=total_steps,
        value=curr_idx + 1,
        help="Select which token prediction event to inspect.",
    )
    step_idx = selected_step_num - 1
    session_mgr.set_selected_token_idx(step_idx)
    step = steps[step_idx]

    # Context and Selected Token Banner
    context_str = " ".join(step.get("context", []))
    selected_tok = step.get("token", "")
    mode = step.get("mode", "five_head")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Preceding Context:** `{context_str or '[START]'}`")
    with col2:
        st.markdown(f"**Selected Token:** <span style='font-size:1.2rem; font-weight:700; color:#58a6ff;'>`{selected_tok}`</span>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"**Latency:** `{step.get('latency_ms', 0.0):.2f} ms`")

    st.divider()

    # Candidate Breakdown Table
    table_data = ExplanationService.get_candidate_breakdown_table(step, top_n=8)
    if table_data:
        df = pd.DataFrame(table_data)
        st.markdown("#### Candidate Ranking & Multi-Head Breakdown")
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Explanation Cards for Selected Token
    if mode == "five_head":
        st.markdown("#### 💬 Transparent Head Explanations")
        explanations = ExplanationService.get_token_explanations(step, selected_tok)

        if explanations:
            e_col1, e_col2 = st.columns(2)
            heads = list(explanations.keys())

            for i, h_name in enumerate(heads):
                h_info = explanations[h_name]
                target_col = e_col1 if i % 2 == 0 else e_col2
                with target_col:
                    norm_score = h_info.get("normalized_score", 0.0)
                    st.markdown(
                        f"""
                        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:700; color:#58a6ff;">{h_name}</span>
                                <span style="font-weight:700; color:#7ee787;">Score: {norm_score:.3f}</span>
                            </div>
                            <div style="font-size:0.85rem; color:#8b949e; margin-top:4px;">
                                {h_info.get('explanation', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No individual head explanations available for this step.")
