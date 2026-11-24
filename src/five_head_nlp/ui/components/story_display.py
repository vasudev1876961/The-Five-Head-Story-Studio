"""Story display component presenting generated narrative and performance badges."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from five_head_nlp.ui.export_service import ExportService


def render_story_display(story: Optional[Dict[str, Any]]) -> None:
    """Render the generated story card, diagnostic badges, and export buttons."""
    if not story:
        st.info("👋 No story generated yet. Click **Generate Story** above to create a new narrative!")
        return

    st.markdown("### 📖 Generated Narrative")

    # Metrics badges row
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-badge">
                <div class="metric-label">Tokens</div>
                <div class="metric-value">{story.get('token_count', 0)}</div>
            </div>
            <div class="metric-badge">
                <div class="metric-label">Sentences</div>
                <div class="metric-value">{story.get('sentence_count', 0)}</div>
            </div>
            <div class="metric-badge">
                <div class="metric-label">Latency</div>
                <div class="metric-value">{story.get('mean_token_latency_ms', 0.0):.2f} <span style="font-size:0.7rem; font-weight:normal;">ms/tok</span></div>
            </div>
            <div class="metric-badge">
                <div class="metric-label">Throughput</div>
                <div class="metric-value">{story.get('tokens_per_second', 0.0):.1f} <span style="font-size:0.7rem; font-weight:normal;">tok/s</span></div>
            </div>
            <div class="metric-badge">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{story.get('generation_time_sec', 0.0):.2f} <span style="font-size:0.7rem; font-weight:normal;">sec</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Narrative Card
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-text">{story.get('text', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Termination status
    st.caption(f"🏁 Termination Reason: **{story.get('termination_reason', 'Completed')}** • Generated at {story.get('timestamp')}")

    # Export download options
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    with col1:
        txt_data = ExportService.export_plain_text(story)
        st.download_button("📥 Text (.txt)", txt_data, file_name=f"{story.get('story_id')}.txt", mime="text/plain", use_container_width=True)
    with col2:
        md_data = ExportService.export_markdown(story)
        st.download_button("📥 Markdown (.md)", md_data, file_name=f"{story.get('story_id')}.md", mime="text/markdown", use_container_width=True)
    with col3:
        json_data = ExportService.export_story_json(story)
        st.download_button("📥 JSON (.json)", json_data, file_name=f"{story.get('story_id')}.json", mime="application/json", use_container_width=True)
