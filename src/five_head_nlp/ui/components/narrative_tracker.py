"""Narrative tracker component visualizing characters, settings, actions, and loop blockers."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.session_manager import SessionManager


def render_narrative_tracker(session_mgr: SessionManager) -> None:
    """Render character tracking, setting consistency, dynamic action rates, and repetition analysis."""
    story = session_mgr.get_current_story()
    if not story:
        st.info("ℹ️ Generate a story first to view narrative tracking analytics.")
        return

    st.markdown("### 🗺️ Narrative State Tracker")
    st.caption("Deep-dive tracking data across the 4 thematic heads and the repetition suppression engine.")

    metrics = story.get("metrics", {})
    char_m = metrics.get("character", {})
    set_m = metrics.get("setting", {})
    act_m = metrics.get("action", {})
    rep_m = metrics.get("repetition", {})

    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Character Keeper",
        "🏰 Setting Painter",
        "⚡ Action Booster",
        "🛡️ Repeat Blocker",
    ])

    with tab1:
        st.markdown("#### Character Continuity & Entity Mentions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Character Mentions", char_m.get("character_mentions", 0))
        c2.metric("Unique Characters", char_m.get("unique_characters", 0))
        c3.metric("Continuity Ratio", f"{char_m.get('character_continuity_ratio', 0.0):.3f}")
        c4.metric("Primary Protagonist", str(char_m.get("primary_character") or "None").capitalize())

        found_chars = char_m.get("characters_found", [])
        if found_chars:
            st.markdown(f"**Identified Characters:** {', '.join([c.capitalize() for c in found_chars])}")
        else:
            st.caption("No character lexicon tokens detected in narrative.")

    with tab2:
        st.markdown("#### Setting Consistency & Transitions")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Setting Mentions", set_m.get("setting_mentions", 0))
        s2.metric("Unique Landmarks", set_m.get("unique_settings", 0))
        s3.metric("Consistency Ratio", f"{set_m.get('setting_consistency_ratio', 0.0):.3f}")
        s4.metric("Dominant Setting", str(set_m.get("dominant_setting") or "None").capitalize())

        found_settings = set_m.get("settings_found", [])
        if found_settings:
            st.markdown(f"**Identified Settings:** {', '.join([s.capitalize() for s in found_settings])}")

    with tab3:
        st.markdown("#### Action Verbs & Plot Momentum")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Dynamic Verbs", act_m.get("dynamic_action_count", 0))
        a2.metric("Stative Verbs", act_m.get("stative_action_count", 0))
        a3.metric("Total Actions", act_m.get("total_action_count", 0))
        a4.metric("Dynamic Ratio", f"{act_m.get('dynamic_action_ratio', 0.0)*100:.1f}%")

    with tab4:
        st.markdown("#### Repetition Suppression & Loop Breaking")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Repeated Bigrams", rep_m.get("repeated_bigram_count", 0), f"{rep_m.get('repeated_bigram_rate', 0.0)*100:.1f}%")
        r2.metric("Repeated Trigrams", rep_m.get("repeated_trigram_count", 0), f"{rep_m.get('repeated_trigram_rate', 0.0)*100:.1f}%")
        r3.metric("Immediate Repeats", rep_m.get("immediate_token_repeat_count", 0))
        r4.metric("Window (w=8) Repeat", f"{rep_m.get('window_repetition_rate', 0.0)*100:.1f}%")

        has_event = rep_m.get("has_repetition_event", False)
        if has_event:
            st.warning("⚠️ Repetition event detected within the generated narrative.")
        else:
            st.success("✅ Clean narrative: No repetitive attractor loops detected!")
