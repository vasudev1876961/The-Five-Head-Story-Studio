"""Main Streamlit application coordination module for The Five-Head Story Studio."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.components import (
    apply_custom_styles,
    render_analytics_dashboard,
    render_candidate_explorer,
    render_comparison_panel,
    render_generation_panel,
    render_hamilton_visualizer,
    render_head_dashboard,
    render_narrative_tracker,
    render_sidebar,
    render_story_display,
    render_story_history,
)
from five_head_nlp.ui.generation_service import GenerationService
from five_head_nlp.ui.session_manager import SessionManager


def render_model_info_page() -> None:
    """Render technical architecture documentation, pipeline diagram, and limitations."""
    st.markdown("### 🏛️ System Architecture & Model Pipeline")
    st.caption("Technical specification and component breakdown of the Five-Head Story-Generation System.")

    st.markdown(
        """
        ```text
        Story Corpus (34 sentences, 415 tokens, 120 vocabulary)
             │
             ▼
        CorpusLoader & StoryTokenizer (Word-level punctuation isolation)
             │
             ▼
        N-Gram Language Models (Unigram, Bigram, Trigram)
             │
             ▼
        Probability Smoothing (Laplace, Add-k, Good-Turing, Backoff, Interpolation)
             │
             ▼
        NextTokenPredictor (Continuous Candidate Generation)
             │
             ▼
        Five Transparent Scoring Heads:
        ├── Head 1: Grammar Scout      G(w | h) = P(w | h)
        ├── Head 2: Character Keeper   C(w) = Continuity bonus & tracking
        ├── Head 3: Setting Painter    S(w) = World consistency & shifts
        ├── Head 4: Action Booster     A(w) = Dynamic action acceleration
        └── Head 5: Repeat Blocker     R(w) = Exponential frequency suppression
             │
             ▼
        Linear Score Fusion: S(w) = Σ α_k * H_k(w)
             │
             ▼
        Hamilton Largest-Remainder Apportionment (10-Point d10 Game Mapping)
             │
             ▼
        Decoding Engine (Greedy Argmax / Top-K / Temperature Softmax)
             │
             ▼
        Narrative State Update & Autoregressive Story Rollout
        ```
        """
    )

    st.markdown("#### The Five Scoring Heads Specification")
    st.markdown(
        """
        | Head | Name | Target Function | Theoretical Role |
        | :--- | :--- | :--- | :--- |
        | **Head 1** | Grammar Scout | $G(w \\mid h) = P(w \\mid h)$ | Statistical language plausibility using smoothed N-gram transition counts. |
        | **Head 2** | Character Keeper | $C(w) \\in [0.10, 0.85]$ | Maintains focal protagonist continuity and tracks character introductions. |
        | **Head 3** | Setting Painter | $S(w) \\in [0.20, 0.85]$ | Anchors narrative in recognizable spatial environments (castle, cave, forest, garden). |
        | **Head 4** | Action Booster | $A(w) \\in [0.20, 0.90]$ | Rewards dynamic verbs (flew, ran, opened) over stative verbs to drive momentum. |
        | **Head 5** | Repeat Blocker | $R(w) = \\exp(-\\gamma \\cdot c)$ | Penalizes immediate self-loops and 2-token attractor cycles, exempting function words. |
        """
    )

    st.markdown("#### Glass-Box Design & Explicit Limitations")
    st.info(
        "**Glass-Box Guarantee**: Every prediction step can be inspected with complete transparency. "
        "No black-box neural networks, embeddings, or external LLM APIs are used. "
        "\n\n**Known System Boundaries**:\n"
        "• **Closed Lexical Bank**: Micro-vocabulary of 120 tokens reconstructed from the canonical 34-sentence children's corpus.\n"
        "• **Heuristic Narrative Scoring**: Thematic guidance relies on corpus-derived lexicons rather than general-purpose world knowledge.\n"
        "• **Deterministic Tie-Breaking**: All candidate ties are broken deterministically by alphabetical token ordering."
    )


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="The Five-Head Story Studio",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()

    # Header Banner
    st.markdown(
        """
        <div class="studio-header">
            <h1 class="studio-title">🎲 The Five-Head Story Studio</h1>
            <p class="studio-subtitle">
                Transparent Multi-Head Statistical Story Generation & Empirical Research Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_mgr = SessionManager()
    gen_service = GenerationService()

    active_page = render_sidebar(session_mgr)

    if active_page == "Story Studio":
        render_generation_panel(session_mgr, gen_service)
        st.divider()
        render_story_display(session_mgr.get_current_story())

    elif active_page == "Five-Head Inspector":
        render_head_dashboard(session_mgr)
        st.divider()
        render_candidate_explorer(session_mgr)

    elif active_page == "Hamilton d10 Visualizer":
        render_hamilton_visualizer(session_mgr)

    elif active_page == "Narrative Tracker":
        render_narrative_tracker(session_mgr)

    elif active_page == "Experimental Analytics":
        render_analytics_dashboard()

    elif active_page == "Story History & Export":
        render_story_history(session_mgr)
        st.divider()
        render_comparison_panel(session_mgr)

    elif active_page == "Model Architecture Info":
        render_model_info_page()


if __name__ == "__main__":
    main()
