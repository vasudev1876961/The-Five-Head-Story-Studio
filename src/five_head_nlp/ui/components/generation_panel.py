"""Generation panel component providing story generation controls."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.generation_service import GenerationService
from five_head_nlp.ui.model_service import ModelService
from five_head_nlp.ui.session_manager import SessionManager


def render_generation_panel(
    session_mgr: SessionManager,
    gen_service: GenerationService,
) -> None:
    """Render story generation controls and trigger story rollouts."""
    cfg = session_mgr.get_config()

    st.markdown("### ✍️ Generation Controls")

    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    with col1:
        mode_label = st.radio(
            "Architecture Mode",
            ("Five-Head Multi-Head Scoring", "Statistical Baseline N-gram"),
            index=0 if cfg.get("mode") == "five_head" else 1,
            help="Choose between transparent 5-head candidate scoring or pure statistical N-gram decoding.",
        )
        mode = "five_head" if "Five-Head" in mode_label else "baseline"
        cfg["mode"] = mode

        smoothers = list(ModelService.get_supported_smoothers().keys())
        curr_sm = cfg.get("smoother_name", smoothers[0])
        smoother = st.selectbox(
            "Probability Smoothing",
            smoothers,
            index=smoothers.index(curr_sm) if curr_sm in smoothers else 0,
            help="Select probability smoothing model to resolve unseen N-gram sparsity.",
        )
        cfg["smoother_name"] = smoother

    with col2:
        decoders = ModelService.DECODER_CHOICES
        curr_dec = cfg.get("decoder_name", decoders[0])
        decoder = st.selectbox(
            "Decoding Strategy",
            decoders,
            index=decoders.index(curr_dec) if curr_dec in decoders else 0,
            help="Strategy for selecting the next token from ranked candidates.",
        )
        cfg["decoder_name"] = decoder

        # Dynamic decoder parameters
        d_params = cfg.get("decoder_params", {})
        if decoder == "Top-k":
            k_val = st.slider("Top-k (Candidates Pool)", min_value=2, max_value=15, value=int(d_params.get("k", 5)))
            d_params["k"] = k_val
        elif decoder == "Temperature":
            t_val = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=float(d_params.get("temperature", 0.8)), step=0.05)
            d_params["temperature"] = t_val
        cfg["decoder_params"] = d_params

    with col3:
        seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=999999,
            value=int(cfg.get("seed", 42)),
            help="Isolated deterministic seed for exact reproducibility.",
        )
        cfg["seed"] = seed

        max_tokens = st.slider(
            "Target Tokens",
            min_value=50,
            max_value=150,
            value=int(cfg.get("max_tokens", 100)),
            step=5,
            help="Target token length for the multi-sentence narrative.",
        )
        cfg["max_tokens"] = max_tokens

    # Starter prompt selection
    starters = [
        "the clever cat milo lived near the crystal castle",
        "pip lived near the dark cave and flew to the bright forest",
        "zara lived near the secret garden and liked the crystal castle",
        "a sleepy owl rested in the old tree near the dark cave",
        "milo found the silver key hidden under the bright green moss",
        "Custom starter prompt...",
    ]

    selected_starter = st.selectbox("Narrative Starter Prompt", starters, index=0)
    if selected_starter == "Custom starter prompt...":
        starter_text = st.text_input("Enter starting prompt tokens:", value="milo found the silver key")
    else:
        starter_text = selected_starter
    cfg["starter_prompt"] = starter_text

    session_mgr.update_config(cfg)

    st.markdown("")
    gen_col1, gen_col2 = st.columns([1, 4])
    with gen_col1:
        generate_clicked = st.button("🚀 Generate Story", type="primary", use_container_width=True)

    if generate_clicked:
        with st.spinner("Generating narrative via five_head_nlp backend..."):
            story_record, steps = gen_service.generate_story(cfg)
            session_mgr.set_current_story(story_record, steps)
            st.success(f"Generated {story_record['token_count']} tokens in {story_record['generation_time_sec']:.2f}s!")
            st.rerun()
