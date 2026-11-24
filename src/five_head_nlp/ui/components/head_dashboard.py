"""Head dashboard component for adjusting weights and previewing d10 budget allocations."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.explanation_service import ExplanationService
from five_head_nlp.ui.session_manager import SessionManager


def render_head_dashboard(session_mgr: SessionManager) -> None:
    """Render interactive head weight sliders, presets, and live d10 allocation."""
    cfg = session_mgr.get_config()
    weights = cfg.get("weights", {
        "Grammar Scout": 0.40,
        "Character Keeper": 0.15,
        "Setting Painter": 0.15,
        "Action Booster": 0.10,
        "Repeat Blocker": 0.20,
    })
    enabled = cfg.get("enabled_heads", {k: True for k in weights})

    st.markdown("### 🎛️ Head Weights & d10 Apportionment")
    st.caption("Adjust the relative weights of the five transparent scoring heads. Weights automatically normalize to sum to 1.0.")

    # Preset profiles
    preset_cols = st.columns(5)
    presets = {
        "Balanced (Default)": {"Grammar Scout": 0.40, "Character Keeper": 0.15, "Setting Painter": 0.15, "Action Booster": 0.10, "Repeat Blocker": 0.20},
        "Character Focus": {"Grammar Scout": 0.30, "Character Keeper": 0.40, "Setting Painter": 0.10, "Action Booster": 0.10, "Repeat Blocker": 0.10},
        "Action Pacing": {"Grammar Scout": 0.30, "Character Keeper": 0.10, "Setting Painter": 0.10, "Action Booster": 0.40, "Repeat Blocker": 0.10},
        "World Building": {"Grammar Scout": 0.30, "Character Keeper": 0.10, "Setting Painter": 0.40, "Action Booster": 0.10, "Repeat Blocker": 0.10},
        "Pure Grammar": {"Grammar Scout": 1.00, "Character Keeper": 0.00, "Setting Painter": 0.00, "Action Booster": 0.00, "Repeat Blocker": 0.00},
    }

    for idx, (p_name, p_w) in enumerate(presets.items()):
        if preset_cols[idx].button(p_name, use_container_width=True):
            weights = dict(p_w)
            cfg["weights"] = weights
            session_mgr.update_config(cfg)
            st.rerun()

    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Weight Controls")
        w_grammar = st.slider("📘 Grammar Scout (Statistical Plausibility)", 0.0, 1.0, float(weights.get("Grammar Scout", 0.40)), step=0.05)
        w_char = st.slider("👤 Character Keeper (Protagonist Continuity)", 0.0, 1.0, float(weights.get("Character Keeper", 0.15)), step=0.05)
        w_setting = st.slider("🏰 Setting Painter (World Consistency)", 0.0, 1.0, float(weights.get("Setting Painter", 0.15)), step=0.05)
        w_action = st.slider("⚡ Action Booster (Plot Momentum)", 0.0, 1.0, float(weights.get("Action Booster", 0.10)), step=0.05)
        w_repeat = st.slider("🛡️ Repeat Blocker (Loop Suppression)", 0.0, 1.0, float(weights.get("Repeat Blocker", 0.20)), step=0.05)

        raw_weights = {
            "Grammar Scout": w_grammar,
            "Character Keeper": w_char,
            "Setting Painter": w_setting,
            "Action Booster": w_action,
            "Repeat Blocker": w_repeat,
        }
        tot = sum(raw_weights.values())
        norm_weights = {k: (v / tot if tot > 0 else 0.2) for k, v in raw_weights.items()}
        cfg["weights"] = norm_weights
        session_mgr.update_config(cfg)

    with col2:
        st.markdown("#### Live Hamilton d10 Allocation")
        h_breakdown = ExplanationService.compute_hamilton_breakdown(norm_weights, total_points=10)
        alloc = h_breakdown["final_allocation"]

        st.markdown(
            f"""
            <div style="background:#21262d; border:1px solid #30363d; border-radius:8px; padding:16px;">
                <div style="font-weight:700; color:#58a6ff; margin-bottom:8px;">10-Sided Die (d10) Point Budget</div>
                <div style="font-size:0.9rem; line-height:1.7;">
                    <span class="pill pill-grammar">Grammar</span> : <b>{alloc.get('Grammar Scout', 0)}</b> faces / 10<br>
                    <span class="pill pill-character">Character</span> : <b>{alloc.get('Character Keeper', 0)}</b> faces / 10<br>
                    <span class="pill pill-setting">Setting</span> : <b>{alloc.get('Setting Painter', 0)}</b> faces / 10<br>
                    <span class="pill pill-action">Action</span> : <b>{alloc.get('Action Booster', 0)}</b> faces / 10<br>
                    <span class="pill pill-repeat">Repeat</span> : <b>{alloc.get('Repeat Blocker', 0)}</b> faces / 10
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Color-coded d10 bar
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
                bar_html += f"<div class='d10-slot' style='width:{pct}%; background:{colors[k]};'>{pts}</div>"
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)
