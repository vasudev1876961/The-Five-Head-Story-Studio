# The Five-Head Story-Generation System

> **Teaching Foundational Large Language Model Concepts to Grades 4–5 Through a Transparent Five-Head Story-Generation System**

---

## 📖 Overview

**The Five-Head Story-Generation System** is an educational Natural Language Processing (NLP) framework designed to demystify autoregressive text generation for primary school learners (Grades 4–5, ages 9–11) and provide researchers with an inspectable testbed for multi-signal decoding.

Rather than treating modern generative AI as an opaque black box, this project deconstructs next-token prediction into five observable, independently parameterized scoring components:
1. **Head 1 — Grammar Scout**: Predicts statistically probable continuations using smoothed n-gram frequency distributions (Trigram with backoff).
2. **Head 2 — Character Keeper**: Enforces character continuity by scoring candidate associations with the currently active protagonist.
3. **Head 3 — Setting Painter**: Preserves environmental and scenic consistency across sentences.
4. **Head 4 — Action Booster**: Injects narrative momentum and lexical diversity through descriptive action verbs.
5. **Head 5 — Repeat Blocker**: Detects and suppresses immediate lexical and bigram repetition loops.

---

## 🎲 The Five-Head Story Studio (Streamlit Web Application)

Phase 6 introduces **The Five-Head Story Studio**, a full-featured, dark-themed AI research and classroom application.

### Key Features
- **Story Studio**: Autoregressively generates multi-sentence stories (80–120 tokens) with configurable smoothers (Trigram Backoff, Laplace, Add-k, Good-Turing, Interpolation, MLE) and decoders (Top-$k$, Greedy, Temperature).
- **Five-Head Inspector**: Token-by-token candidate ranking explorer displaying raw head scores, weights, fused scores, and human-readable explanations.
- **Hamilton d10 Visualizer**: Interactive pedagogical tool mapping continuous head weights into an exact 10-point budget for 10-sided dice ($d10$) tabletop gameplay using Hamilton's largest-remainder method.
- **Narrative Tracker**: Live analytics for character continuity, setting consistency, dynamic action rates, and repetition suppression.
- **Experimental Analytics**: Interactive exploration of real Phase 5 Monte Carlo benchmark results (1,600 stories across 8 conditions).
- **Story History & Export**: Side-by-side story comparison and export to `.txt`, `.md`, `.csv`, and `.json`.

---

## 🏛️ System Architecture

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

---

## ⚙️ Technical Requirements & Installation

- **Python Compatibility**: `requires-python >= 3.10` (`Python 3.12.8` runtime-verified locally on Windows; Python `3.10` and `3.11` statically assessed and configured alongside `ubuntu-latest`/`macos-latest` in CI, pending remote GitHub Actions execution)
- **Zero Black-Box Dependencies**: Strictly no external LLM APIs (OpenAI, Anthropic, Gemini) and no opaque pretrained neural networks (BERT, GPT, HuggingFace transformers).
- **Mathematical Integrity**: Weighted linear score fusion (`ScoreFusion`) with continuous scoring; Hamilton `d10` integer apportionment (`HamiltonApportionment`) serves as an explanatory classroom representation.

### Standard Installation

Install the package in editable mode with development and testing dependencies:
```bash
pip install -e ".[dev]"
```

Or install core dependencies only:
```bash
pip install -e .
```

---

## 🚀 Command-Line Interface (CLI)

The package installs four cross-platform console scripts that function both inside the repository and from any external working directory:

### 1. The Five-Head Story Studio (Web UI)
```bash
five-head-studio
```
Launches the interactive Streamlit application on `http://localhost:8501`. Supports native Streamlit flags (e.g. `--server.port=8598`, `--server.headless=true`).

### 2. Monte Carlo Simulation Engine
```bash
five-head-monte-carlo --stories-per-config 200 --seed 42
```
Generates 1,600 benchmark stories across 8 model conditions with aggregation and metrics.

### 3. Five-Head Scoring Demonstration
```bash
five-head-scoring --output-dir experiments/results/five_head_scoring
```
Executes the Phase 4 scoring head evaluator, generating candidate breakdown plots and tables.

### 4. Next-Token Prediction Demonstration
```bash
five-head-prediction --output-dir experiments/results/next_token_prediction
```
Evaluates Unigram, Bigram, and Trigram language models across 5 smoothing techniques and 3 decoding strategies.

*Note: All CLI commands support side-effect-free `--help` output with zero computation.*

---

## 🧪 Quality Assurance & Verification

### Run Automated Full Test Suite (351 Tests)
```bash
python -m pytest tests -v
```
*Full suite (automatically discovered): **351 passing tests** (`319` unit & integration tests, `8` Streamlit process lifecycle scenarios, and `24` reproducibility harness tests) with 0 failures, 0 errors, and 0 skips on Python 3.12.8.*

### Run Streamlit Lifecycle Verification (8 Scenarios)
```bash
python -m pytest tests/test_ui/test_studio_lifecycle.py -v
```
*Verifies clean startup, `/_stcore/health` HTTP 200 with body `"ok"`, 3.0s observation, graceful signal shutdown, port release, and fallback termination.*

### Verify Coverage Baseline & Reproducibility Harness
```bash
python -m pytest tests --cov=src/five_head_nlp --cov-branch --cov-report=term-missing --cov-report=xml:reports/coverage.xml --cov-fail-under=70
python -m five_head_nlp.experiments.reproduce_all --dry-run
```
*Current measured coverage across the 88 discovered modules: `76.10%` statement (gate `72%`), `66.88%` branch (gate `63%`), and `74.20%` combined (gate `70%`). Four unimported files under `src/five_head_nlp/experiments/` remain excluded due to the absence of `__init__.py` (see [Developer Handbook](docs/developer_handbook.md)).*

### Validate Scratch 3.0 Educational Project
```bash
python phase_7/scratch/validate_sb3.py
```
*Validates ZIP archive integrity, AST JSON schema, all 12 SVG MD5 checksums, and weight/Hamilton summation checks structurally (note: structural `.sb3` validation is not equivalent to manual execution in the Scratch editor).*

---

## 🎮 Scratch 3.0 Educational Suite & Classroom Unplugged Guide

Phase 7 introduces **The Five-Head Story Machine** for K-12 and introductory AI classrooms, demystifying language generation through visual blocks and physical roleplaying:

- **Scratch 3.0 Project (`five_head_story_machine.sb3`)**: An interactive, inspectable simulation running on vanilla Scratch 3.0. Features 12 custom vector SVGs, 23 global variables, 4 lists, live score displays, profile toggles, and Hamilton d10 roll animations.
- **Classroom Unplugged Activity**: A 60-minute kinesthetic simulation where students play the roles of the five scoring heads, calculating weighted scores on paper and rolling a 10-sided die ($d10$) to co-author a story.
- **Teacher & Student Resources**: Printable student worksheets, full teacher lesson guide with assessment rubrics, and worked answer keys with step-by-step arithmetic.
- **Documentation Suite**: Architectural specifications, mathematical bridges from N-gram MLE to linear score fusion, and automated verification reports.

---

## 📁 Repository Structure

```text
NLP-GAME-PROJECT/
├── .github/workflows/ci.yml       # Cross-platform CI matrix (Windows, Ubuntu, macOS; Py 3.10–3.12)
├── pyproject.toml                 # Modern PEP 518/621 packaging & coverage configuration
├── requirements.txt               # Pinned project dependencies
├── pytest.ini                     # Pytest configuration
├── LICENSE                        # MIT Open Source License
├── CONTRIBUTING.md                # Contributor & testing guidelines
├── CITATION.cff                   # Citation metadata (CFF v1.2.0)
├── app.py                         # Root Streamlit application entry point
├── archive/                       # Preserved historical materials
│   └── legacy_phase_1_scaffolds/  # Preserved Phase 1 exploration & prototypes
├── data/
│   ├── raw/story_corpus.txt       # Canonical 34-sentence children's story corpus
│   └── processed/                 # Tokenized corpora, vocabulary mappings
├── docs/                          # Technical specifications & documentation
│   ├── academic_defense_runbook.md# Academic defense runbook & viva Q&A
│   ├── architecture.md            # System architecture
│   ├── developer_handbook.md      # Developer onboarding, CLI, CI & troubleshooting
│   ├── implementation_log.md      # Engineering milestones & changelog
│   ├── mathematical_formulation.md# Probabilistic & mathematical specifications
│   ├── phase_5_experimental_methodology.md # Phase 5 benchmark replication guide
│   ├── phase_6_application_guide.md # Phase 6 Streamlit application guide
│   └── release_notes_v1.0.md      # Official v1.0.0 Release Notes
├── experiments/                   # Monte Carlo harness & benchmark artifacts
│   ├── run_monte_carlo.py         # 1,600-story generation CLI script
│   ├── run_five_head_scoring.py   # Phase 4 scoring heads demo
│   ├── run_next_token_prediction.py # Phase 3 prediction demo
│   └── results/monte_carlo/       # JSONL stories, CSV metrics, and plots
├── phase_7/                       # Scratch 3.0 & Classroom Unplugged Suite
│   ├── scratch/                   # Scratch 3.0 project, SVGs, builder, validator
│   │   ├── five_head_story_machine.sb3 # Production Scratch 3.0 archive
│   │   ├── build_scratch_project.py    # Automated SVG & project builder
│   │   ├── validate_sb3.py             # Archive & asset MD5 validator
│   │   ├── assets/                     # 12 scalable vector graphics (SVGs)
│   │   └── source/project.json         # Human-readable Scratch AST
│   ├── classroom/                 # Unplugged curriculum & printable materials
│   │   ├── unplugged_activity.md       # 60-min kinesthetic simulation guide
│   │   ├── teacher_guide.md            # Lesson plan, pedagogy, and grading rubric
│   │   ├── student_worksheet.md        # Printable 6-part student activity worksheet
│   │   └── answer_key.md               # Teacher answer key with worked solutions
│   └── documentation/             # Engineering & pedagogical documentation
├── src/five_head_nlp/             # Core Python package
│   ├── tokenization/              # Word-level tokenizer and corpus loader
│   ├── models/                    # Unigram, Bigram, Trigram language models
│   ├── probability/               # N-gram MLE probability engine
│   ├── smoothing/                 # Laplace, Add-k, Good-Turing, Backoff, Interpolation
│   ├── heads/                     # The Five Transparent Scoring Heads (H1 to H5)
│   ├── prediction/                # Decoders (Greedy, Top-k, Temperature) & predictor
│   ├── decoding/                  # Compatibility facade for decoding strategies
│   ├── experiments/               # CLI entry points, configs, metrics, generation & reproduce_all harness
│   └── ui/                        # Streamlit web application & CLI wrapper
└── tests/                         # Automated test suite (351 tests across all sub-suites)
    ├── test_tokenization/
    ├── test_models/
    ├── test_probability/
    ├── test_smoothing/
    ├── test_heads/
    ├── test_prediction/
    ├── test_decoding/             # Facade re-export regression tests
    ├── test_experiments/          # Metrics, harness & reproducibility tests
    └── test_ui/                   # UI services & 8-scenario lifecycle harness (test_studio_lifecycle.py)
```

---

## 📚 Handover & Reference Documentation

- [Developer Handbook](docs/developer_handbook.md) — Environment setup, exact CLI & test commands, package architecture, coverage discovery details, CI matrix, and troubleshooting.
- [Academic Defense Runbook](docs/academic_defense_runbook.md) — Mathematical formulations, empirical smoothing & Monte Carlo results, Phase 9 reproducibility statistics, limitations, and viva Q&A.
- [Release Notes (v1.0.0)](docs/release_notes_v1.0.md) — Verified release summary and known platform containment boundaries.

---

## 🚦 Project Milestone Status

- [x] **Phase 1: Project Scaffolding & Architecture Design** (Completed)
- [x] **Phase 2: Corpus Engineering, Tokenization & EDA** (Completed)
- [x] **Phase 3: N-Gram Modeling & Probability Smoothing** (Completed)
- [x] **Phase 4: Five Transparent Scoring Heads & Hamilton Allocation** (Completed)
- [x] **Phase 5: Experimental Suite & 1,600-Story Benchmark Replication** (Completed)
- [x] **Phase 6: Interactive Streamlit Glass-Box Web Application** (Completed)
- [x] **Phase 7: Scratch 3.0 Project Assets & Classroom Unplugged Guide** (Completed)
- [x] **Phase 8: Final Integration, Quality Assurance & Release Preparation (v1.0.0)** (Completed)
- [x] **Phase 9: Reproducibility Harness, Coverage Gates, CI Matrix & Handover (Local Verification Complete; Remote CI Pending)** (Completed)

---

## 📄 License & Citation

- **License:** Released under the [MIT License](LICENSE).
- **Citation:** See [CITATION.cff](CITATION.cff) for academic citation formatting.
