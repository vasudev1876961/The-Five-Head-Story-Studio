"""Custom CSS styling for The Five-Head Story Studio."""

CUSTOM_CSS = """
<style>
/* Main Background and Typography */
.main {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Header Styling */
.studio-header {
    background: linear-gradient(135deg, #1f242d 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.studio-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.studio-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Story Card */
.story-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 22px 26px;
    margin: 16px 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.story-text {
    font-family: 'Georgia', serif;
    font-size: 1.15rem;
    line-height: 1.8;
    color: #f0f6fc;
}

/* Metric Cards */
.metric-container {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}

.metric-badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 120px;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #8b949e;
    font-weight: 600;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #58a6ff;
    margin-top: 2px;
}

/* Badge Pills */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
}

.pill-grammar { background: rgba(88, 166, 255, 0.2); color: #58a6ff; border: 1px solid #58a6ff; }
.pill-character { background: rgba(247, 120, 186, 0.2); color: #f778ba; border: 1px solid #f778ba; }
.pill-setting { background: rgba(126, 231, 135, 0.2); color: #7ee787; border: 1px solid #7ee787; }
.pill-action { background: rgba(255, 166, 87, 0.2); color: #ffa657; border: 1px solid #ffa657; }
.pill-repeat { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #d29922; }

/* Hamilton d10 Die Slot */
.d10-bar {
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    height: 32px;
    margin: 12px 0 20px 0;
    border: 1px solid #30363d;
}

.d10-slot {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 700;
    font-size: 0.85rem;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom styles into Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
