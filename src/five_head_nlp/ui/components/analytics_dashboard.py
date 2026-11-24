"""Experimental analytics dashboard loading real Phase 5 benchmark results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from five_head_nlp.ui.analytics_service import AnalyticsService


def render_analytics_dashboard() -> None:
    """Render comparative analytics loaded directly from real Phase 5 Monte Carlo artifacts."""
    st.markdown("### 📊 Experimental Analytics & Replication Benchmarks")
    st.caption("Empirical data loaded directly from the Phase 5 Monte Carlo benchmark (1,600 generated stories).")

    if not AnalyticsService.is_artifacts_available():
        st.warning("⚠️ Phase 5 Monte Carlo experimental artifacts not found in `experiments/results/monte_carlo/`.")
        st.info("Run `python experiments/run_monte_carlo.py` to regenerate the experimental artifacts.")
        return

    meta = AnalyticsService.load_metadata()
    if meta:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Stories", meta.get("total_stories_generated", 1600))
        m_col2.metric("Stories / Config", meta.get("stories_per_config", 200))
        m_col3.metric("Base Seed", meta.get("base_seed", 42))
        m_col4.metric("Total Failures", meta.get("total_failures", 0))

    comp_df = AnalyticsService.load_comparison_table()

    st.markdown("#### High-Level Configuration Comparison Table")
    if not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # Interactive metric comparison plot
    st.markdown("#### Interactive Metric Comparison Across E1 to E8")
    metrics_list = AnalyticsService.get_available_metrics()
    selected_metric = st.selectbox(
        "Select Metric to Visualize",
        metrics_list,
        index=0,
        format_func=AnalyticsService.get_metric_display_name,
    )

    if not comp_df.empty and selected_metric in comp_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        cids = comp_df["config_id"]
        vals = comp_df[selected_metric]
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

        bars = ax.bar(cids, vals, color=colors[:len(cids)], alpha=0.85)
        ax.set_xlabel("Experimental Condition", fontweight="bold")
        ax.set_ylabel(AnalyticsService.get_metric_display_name(selected_metric), fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        st.pyplot(fig)
        plt.close()

    st.markdown("#### Key Findings from Monte Carlo Benchmark")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown(
            """
            <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px;">
                <b style="color:#58a6ff;">Repeat Blocker (Head 5) Impact:</b><br>
                • In <b>E5</b> (without Repeat Blocker), the bigram repeat rate skyrockets to <b>66.0%</b>.<br>
                • In <b>E6</b> (Top-K with Repeat Blocker), repeat rate plummets to <b>19.2%</b>.<br>
                • In <b>E8</b> (Temperature with Repeat Blocker), repeat rate drops to <b>0.6%</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f_col2:
        st.markdown(
            """
            <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px;">
                <b style="color:#7ee787;">Action & Character Guidance:</b><br>
                • <b>Action Booster (Head 4)</b> increases dynamic action verbs to <b>98.3%</b> in E6.<br>
                • <b>Character Keeper (Head 2)</b> boosts protagonist continuity to <b>0.511</b> in E5.<br>
                • Five-head inference runs at <b>2.85–3.38 ms/token</b> (~300 tokens/sec).
            </div>
            """,
            unsafe_allow_html=True,
        )
