# Phase 6 — The Five-Head Story Studio Application Guide

## 1. Application Overview

**The Five-Head Story Studio** is an interactive web application built with Streamlit for the Five-Head Story-Generation System. It provides an intuitive, glass-box interface for primary students, educators, and NLP researchers to explore multi-signal autoregressive text generation.

The application allows users to:
1. Generate complete multi-sentence narratives using either pure statistical N-gram baselines or the multi-head candidate scoring architecture.
2. Interactively configure the five scoring heads and preview the corresponding 10-sided die (d10) budget allocations.
3. Inspect token-by-token candidate rankings, individual head scores, and transparent human-readable explanations.
4. Track narrative state (characters, settings, dynamic verbs, and repetition suppression) across the story rollout.
5. Explore empirical analytics and visualizations loaded directly from the 1,600-story Phase 5 Monte Carlo benchmark.
6. Export stories and diagnostic scoring logs as TXT, Markdown, CSV, and JSON.

---

## 2. Installation and Execution

### 2.1 Prerequisites
- Python 3.11 or Python 3.12
- Operating System: Windows, macOS, or Linux
- Standard virtual environment or conda environment

### 2.2 Dependencies
Install the required packages from the repository root:

```powershell
pip install -r requirements.txt
```

### 2.3 Running the Application
Launch the application from the repository root via Windows PowerShell:

```powershell
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

---

## 3. Application Architecture

The UI architecture strictly separates presentation from core business logic:

```text
NLP-GAME-PROJECT/
├── app.py                         # Root Streamlit entry point
├── src/five_head_nlp/ui/
│   ├── __init__.py                # Package exports
│   ├── application.py             # Streamlit page layout and coordinator
│   ├── session_manager.py         # Session state container and history tracker
│   ├── model_service.py           # Model caching and component factory
│   ├── generation_service.py      # Autoregressive generation & diagnostics
│   ├── explanation_service.py     # Candidate breakdowns & Hamilton allocation
│   ├── analytics_service.py       # Phase 5 Monte Carlo artifact queries
│   ├── export_service.py          # Plain text, Markdown, CSV, and JSON exporters
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py             # Navigation radio and active configuration badge
│       ├── generation_panel.py    # Mode, smoother, decoder, seed, prompt controls
│       ├── story_display.py       # Formatted narrative card and metric badges
│       ├── head_dashboard.py      # Head weight sliders, presets, live d10 preview
│       ├── candidate_explorer.py  # Token step selector, candidate table, explanations
│       ├── hamilton_visualizer.py # Quota math, remainders, d10 die face slots
│       ├── narrative_tracker.py   # Characters, settings, actions, repeat tabs
│       ├── analytics_dashboard.py # Interactive charts from Phase 5 benchmark
│       ├── comparison_panel.py    # Side-by-side story comparator
│       ├── story_history.py       # Session history viewer & bulk downloads
│       └── styles.py              # Dark AI research studio CSS styling
```

---

## 4. Page Walkthrough

### 4.1 Page 1 — Story Studio
The primary generation workspace:
- **Architecture Mode**: Toggle between **Five-Head Multi-Head Scoring** and **Statistical Baseline N-gram**.
- **Probability Smoothing**: Choose between *Trigram Backoff (Recommended)*, *Laplace (Add-1)*, *Add-k*, *Good-Turing*, *Linear Interpolation*, or *Unsmoothed MLE*.
- **Decoding Strategy**: Select *Top-k* (with pool size $k$), *Greedy* (deterministic argmax), or *Temperature* (creativity parameter $T$).
- **Generation Parameters**: Random seed, target token length (50–150 tokens), sentence constraints, and narrative starting prompt.
- **Story Output**: Renders readable narrative text alongside real-time metrics: token count, completed sentences, per-token latency (ms), throughput (tokens/sec), duration, and termination reason.

### 4.2 Page 2 — Five-Head Inspector
The core research and transparency demonstration:
- **Head Controls**: Interactive sliders for the five head weights ($w_1 \dots w_5$) with automatic normalization ($\sum w_i = 1.0$) and preset buttons (*Balanced*, *Character Focus*, *Action Pacing*, *World Building*, *Pure Grammar*).
- **Token Position Slider**: Scrub through token generation events ($1 \dots N$).
- **Candidate Breakdown Table**: Displays the top candidate tokens at the selected position with rank, fused score / probability, and individual normalized scores from Grammar Scout, Character Keeper, Setting Painter, Action Booster, and Repeat Blocker.
- **Transparent Explanations**: Human-readable explanations detailing why the winning token received its specific score from each head.

### 4.3 Page 3 — Hamilton d10 Visualizer
Explains the mathematical connection between continuous scoring and unplugged classroom game mechanics:
- **Mathematical Step-by-Step Table**: Displays the exact normalized weights, quotas ($q_i = 10 \cdot w_i$), base integer allocations ($\lfloor q_i \rfloor$), fractional remainders ($r_i$), and surplus points distributed.
- **d10 Die Face Assignment**: A visual 10-slot die face budget showing the integer slots assigned to each head ($d_1 + d_2 + d_3 + d_4 + d_5 = 10$).
- **Pedagogical Note**: Clarifies that continuous weights govern digital score fusion, while Hamilton integer allocation provides a tangible glass-box game representation.

### 4.4 Page 4 — Narrative Tracker
Tracks narrative state updates committed after token selection:
- **Character Keeper**: Mentions count, unique characters, primary protagonist focus, and character continuity ratio over time.
- **Setting Painter**: Landmark presence (castle, cave, forest, garden), dominant setting, and spatial consistency.
- **Action Booster**: Counts of dynamic verbs (flew, ran, found, opened) vs stative verbs, plus dynamic action percentage.
- **Repeat Blocker**: Immediate repetition counts, repeated bigram rate, repeated trigram rate, and 8-token rolling window repetition alerts.

### 4.5 Page 5 — Experimental Analytics
Loads and analyzes real empirical data from the Phase 5 Monte Carlo benchmark (1,600 stories across E1 to E8):
- **Summary Metrics**: Total stories generated ($1,600$), stories per config ($200$), zero failures.
- **High-Level Comparison Table**: Tabular view comparing all 8 conditions on length, repetition rates, lexical diversity (TTR), character continuity, and latency.
- **Interactive Metric Selector**: Select any metric to generate an interactive Matplotlib comparison chart across E1 through E8.
- **Key Empirical Takeaways**: Highlights the drastic repetition reduction in E6/E8 vs E5.

### 4.6 Page 6 — Story History & Export
Session-level memory and export utilities:
- **History Viewer**: Collapsible list of all stories generated during the active browser session.
- **Side-by-Side Story Comparison**: Select any two stories from history to inspect differences in text, length, and repetition rates side-by-side.
- **Downloadable Formats**:
  - Plain Text (`.txt`)
  - GitHub-flavored Markdown (`.md`)
  - Token Scoring CSV (`.csv`)
  - Complete Story JSON (`.json`)

### 4.7 Page 7 — Model Architecture Info
Comprehensive documentation of the system:
- Complete ASCII pipeline diagram from raw corpus to final narrative.
- Mathematical target functions for all five scoring heads.
- System limitations and glass-box design principles.

---

## 5. Automated Verification

Run the complete automated test suite (316 tests):

```powershell
pytest tests
```

To run only the Phase 6 UI tests:

```powershell
pytest tests/test_ui
```

All 316 unit and integration tests pass in under 15 seconds.
