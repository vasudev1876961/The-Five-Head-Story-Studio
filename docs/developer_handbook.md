# Developer Handbook — The Five-Head Story-Generation System

**Release Baseline:** `v1.0.0` (Phase 8 Release + Phase 9 Handover, CI, and Reproducibility Additions)  
**Package:** `five-head-nlp`  
**License:** MIT ([LICENSE](../LICENSE))

---

## 1. Project Overview, Purpose, Scope, and Repository Structure

**The Five-Head Story-Generation System** is a glass-box Natural Language Processing (NLP) research and educational software package built without deep neural networks or external LLM APIs. Every next-token decision is decomposed into statistical N-gram probabilities, smoothing, five transparent scoring heads, linear weighted score fusion, and deterministic/seeded decoding.

### Repository Layout

```text
NLP-GAME-PROJECT/
├── .github/workflows/ci.yml                 # 9-job cross-platform CI workflow (Ubuntu, Windows, macOS x Py 3.10–3.12)
├── pyproject.toml                           # PEP 517/621 build, CLI scripts, [dev] extras, coverage settings
├── pytest.ini                               # Pytest discovery root (testpaths = tests, pythonpath = src)
├── app.py                                   # Root Streamlit shim importing five_head_nlp.ui.application
├── CITATION.cff                             # Citation metadata (v1.0.0)
├── CONTRIBUTING.md                          # Contributor workflow and guidelines
├── LICENSE                                  # MIT License
├── README.md                                # Top-level project overview and quickstart
├── archive/legacy_phase_1_scaffolds/        # Quarantined Phase 1 exploration artifacts
├── data/
│   ├── raw/story_corpus.txt                 # Immutable 34-sentence corpus (415 tokens, 120 vocab)
│   └── processed/                           # Exported tokenized corpus & vocabulary reference JSON/CSV
├── docs/                                    # Architecture, math, methodology, app guide, handbooks, release notes
│   ├── architecture.md
│   ├── corpus_statistics.md
│   ├── developer_handbook.md                # This developer handover guide
│   ├── academic_defense_runbook.md          # Mathematical and empirical academic defense guide
│   ├── implementation_log.md
│   ├── mathematical_formulation.md
│   ├── phase_5_experimental_methodology.md
│   ├── phase_6_application_guide.md
│   └── release_notes_v1.0.md
├── experiments/
│   ├── run_monte_carlo.py                   # Phase 5 CLI script driving Monte Carlo generation
│   ├── run_five_head_scoring.py             # Phase 4 scoring demonstration script
│   ├── run_next_token_prediction.py         # Phase 3 next-token prediction demonstration script
│   ├── models/                              # Tracked baseline serialized Unigram/Bigram/Trigram JSONs
│   └── results/                             # Protected canonical Phase 2–5 artifacts (CSV, JSON, JSONL, PNG)
├── phase_7/
│   ├── scratch/                             # Scratch 3.0 .sb3 archive, builder, AST JSON, validate_sb3.py, 12 SVGs
│   ├── classroom/                           # Teacher guide, student worksheet, unplugged activity, answer key
│   └── documentation/                       # Phase 7 architecture & structural testing documentation
├── reports/
│   ├── coverage.xml                         # Gitignored Cobertura XML report (generated during coverage runs)
│   └── reproducibility/run_b42_s1042_n200/  # Isolated WP9.4 reproducibility run outputs
├── src/five_head_nlp/                       # Installable Python package
│   ├── tokenization/                        # StoryTokenizer, CorpusLoader, Vocabulary, artifact export
│   ├── probability/                         # NGramProbabilityEngine (MLE, intra-sentence counts)
│   ├── ngrams/                              # Compatibility re-exports of probability N-gram utilities
│   ├── models/                              # UnigramModel, BigramModel, TrigramModel, JSON load/save
│   ├── smoothing/                           # Laplace, Add-k, Good-Turing, Backoff, Interpolation, evaluators/plots
│   ├── prediction/                          # CandidateRanker, NextTokenPredictor, Greedy/Top-K/Temperature decoders
│   ├── decoding/                            # Compatibility facade re-exporting decoding classes & strategy enum
│   ├── heads/                               # Five scoring heads, ScoreFusion, HamiltonApportionment, orchestrator
│   ├── experiments/                         # Experiment configs, StoryGenerator, GenerationHarness, metrics, reproduce_all
│   ├── ui/                                  # Streamlit app, CLI wrapper, services, session manager, components
│   └── utils/                               # Corpus validator utility
└── tests/                                   # Automated test suite (351 tests collected under default discovery)
```

---

## 2. System Requirements and Supported Python Versions

`pyproject.toml` specifies `requires-python = ">=3.10"` with core runtime dependencies:
- `numpy>=1.24.0,<2.0.0`
- `pandas>=2.0.0,<3.0.0`
- `streamlit>=1.30.0,<2.0.0`
- `matplotlib>=3.7.0,<4.0.0`

Development extras (`[dev]`):
- `pytest>=7.4.0,<9.0.0`
- `pytest-cov>=4.1.0,<6.0.0`
- `psutil>=5.9.0,<7.0.0`

### Empirical vs. Static Verification Status
- **Runtime-Verified Locally:** **Python `3.12.8` on Windows 11** (`351/351` tests passing, CLI `--help` checks passing, Streamlit lifecycle suite passing, `1,600`-story WP9.4 reproducibility run verified).
- **Statically Assessed & Configured in CI:** **Python `3.10` and `3.11`** syntax (PEP 604 union annotations compatible with 3.10+) and dependency constraints are assessed statically and configured in `.github/workflows/ci.yml`, along with **Ubuntu (`ubuntu-latest`)** and **macOS (`macos-latest`)** runner targets. **Remote GitHub-hosted CI execution remains pending** because no remote repository is configured and no push has been authorized.

---

## 3. Clean Environment Creation and Editable Installation

From the repository root:

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows PowerShell / Unix bash)
.\.venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate

# 3. Upgrade packaging tools and install package with dev dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For runtime use without test tooling (`pytest`, `pytest-cov`, `psutil`):

```bash
python -m pip install -e .
```

> **Note:** `pyproject.toml` defines only the `[dev]` optional dependency group (`streamlit` and `matplotlib` are included in core `dependencies`). Do not pass unsupported extras such as `[ui]`.

---

## 4. Exact Operational Commands

### 4.1 Test Suite Commands
```bash
# Collect all tests via default recursive discovery (expected: 351 tests)
python -m pytest tests --collect-only -q

# Execute full automated suite (unit, integration, lifecycle, and reproducibility tests)
python -m pytest tests -v

# Execute fast compatibility subset excluding subprocess server lifecycle harness (343 tests)
python -m pytest tests --ignore=tests/test_ui/test_studio_lifecycle.py -q

# Execute Streamlit process lifecycle suite explicitly (8 scenarios)
python -m pytest tests/test_ui/test_studio_lifecycle.py -v

# Execute Phase 5 reproducibility harness test suite explicitly (24 tests)
python -m pytest tests/test_experiments/test_reproduce_all.py -v
```

### 4.2 Installed CLI Entry Points
All four entry points registered in [pyproject.toml](../pyproject.toml) operate from inside or outside the repository root:

```bash
# 1. Launch Streamlit Story Studio UI (passes remaining CLI flags to `streamlit run`)
five-head-studio --help
five-head-studio --server.headless=true --server.port=8501

# 2. Run Phase 5 Monte Carlo generation CLI (supports side-effect-free --dry-run and --help)
five-head-monte-carlo --help
five-head-monte-carlo --dry-run --stories-per-config 200 --seed 42

# 3. Run Phase 4 scoring demonstration CLI
five-head-scoring --help
five-head-scoring --dry-run --output-dir experiments/results/five_head_scoring

# 4. Run Phase 3 next-token prediction demonstration CLI
five-head-prediction --help
five-head-prediction --dry-run --output-dir experiments/results/next_token_prediction
```

Alternatively, launch the root UI directly during local development:

```bash
streamlit run app.py
```

### 4.3 Phase 7 Scratch `.sb3` Structural Validator
```bash
python phase_7/scratch/validate_sb3.py
```
Reads [five_head_story_machine.sb3](../phase_7/scratch/five_head_story_machine.sb3) in read-only mode to verify ZIP structure, `project.json` schema (`12` targets, `4` monitors, `23` variables, `4` lists), all `12` SVG asset MD5 checksums, and weight/Hamilton summation checks.

### 4.4 Phase 5 Reproducibility Harness (`reproduce_all`)
```bash
# Validate canonical audit, E6 configuration, and path-isolation guards without generating stories
python -m five_head_nlp.experiments.reproduce_all --dry-run

# Execute full 1,600-story Phase 5 reproduction into an isolated directory under reports/reproducibility/
python -m five_head_nlp.experiments.reproduce_all

# Execute a subset or custom seed with explicit run identifier
python -m five_head_nlp.experiments.reproduce_all --configs E1 E6 E7 --stories-per-config 200 --base-seed 42 --stochastic-seed 1042 --run-id custom_check
```
Exact supported flags (`python -m five_head_nlp.experiments.reproduce_all --help`):
`--base-seed`, `--stochastic-seed`, `--stories-per-config`, `--configs`, `--run-id`, `--output-root`, `--dry-run`.

---

## 5. Package Architecture and Module Responsibilities

| Subpackage / Module | Primary Classes / Entry Points | Responsibility |
|---|---|---|
| `five_head_nlp.tokenization` | `StoryTokenizer`, `CorpusLoader`, `Vocabulary`, `export_processed_artifacts` | Whitespace/punctuation tokenization preserving `.` sentence boundaries, deterministic frequency-then-alphabetical vocab IDs (`V = 120`). |
| `five_head_nlp.probability` | `NGramProbabilityEngine`, `extract_ngrams` | Intra-sentence unigram/bigram/trigram frequency tables and unsmoothed conditional MLE probabilities. |
| `five_head_nlp.ngrams` | Re-exports from `probability.mle` | Backward-compatible namespace re-exporting N-gram extraction and MLE engine classes. |
| `five_head_nlp.models` | `BaseNGramModel`, `UnigramModel`, `BigramModel`, `TrigramModel` | Stateful fitted N-gram models with schema-validated JSON `save()`/`load()` persistence. |
| `five_head_nlp.smoothing` | `BaseSmoother`, `LaplaceSmoother`, `AddKSmoother`, `GoodTuringSmoother`, `BackoffSmoother`, `InterpolationSmoother` | Resolves unseen N-gram zero-probabilities while enforcing simplex sum $= 1.0$; perplexity evaluation across 24/5/5 leak-free split. |
| `five_head_nlp.prediction` | `CandidateRanker`, `NextTokenPredictor`, `GenerationState`, `GreedyDecoder`, `TopKDecoder`, `TemperatureDecoder` | Ranked candidate retrieval with $\epsilon$-consistent `(-prob, token)` alphabetical tie-breaking and decoding rollout. |
| `five_head_nlp.decoding` | `DecodingStrategy`, decoder classes | Phase 8 compatibility facade re-exporting decoder abstractions alongside `DecodingStrategy` enum. |
| `five_head_nlp.heads` | `GrammarScout`, `CharacterKeeper`, `SettingPainter`, `ActionBooster`, `RepeatBlocker`, `ScoreFusion`, `HamiltonApportionment`, `HeadOrchestrator`, `ScoringConfig` | Five modular scoring heads, linear fusion, Hamilton `d10` classroom apportionment, and ablation `HeadEvaluator`. |
| `five_head_nlp.experiments` | `ExperimentConfig`, `EXPERIMENT_REGISTRY`, `StoryGenerator`, `GenerationHarness`, `aggregation`, `reproduce_all`, `metrics/*` | Defines `E1–E8`, Monte Carlo rollout, 8 metric families, statistical aggregation, and safe reproducibility verification. *(Note: lacks `__init__.py`; acts as an implicit namespace directory.)* |
| `five_head_nlp.evaluation` | `CorpusStatistics`, plotting helpers | Exploratory corpus metrics (`34` sentences, `415` tokens, `259` unique bigrams, pivot transitions) and charts. |
| `five_head_nlp.ui` | `cli:main`, `application:main`, services, `components/*` | Streamlit Story Studio wrapping generation, explanation, export, analytics, and session state. |
| `five_head_nlp.utils` | `validate_corpus` | Standalone raw corpus verification helper. |

---

## 6. Language-Model Pipeline: Training, Smoothing, Prediction, and Decoding

1. **Corpus & Boundary Rules (`CorpusLoader` + `StoryTokenizer`):** Reads `data/raw/story_corpus.txt` line-by-line (`34` sentences), isolating `.` and `,` as distinct tokens and lowercasing text (`415` total tokens, `381` lexical words, `34` terminal periods). N-grams never cross sentence boundaries (`C(".", token) == 0`).
2. **Vocabulary (`Vocabulary`):** Deterministically sorted by `(-frequency, token)` across `120` tokens (`119` words + `.`).
3. **Smoothing (`smoothing/`):** Phase 5 story generation uses `None` (`E1`), `LaplaceSmoother` (`E2`), or `BackoffSmoother(discount=0.5)` (`E3–E8`). `BackoffSmoother` subtracts `0.5` from observed trigram/bigram counts and backs off via conservation factor $\alpha(h)$ down to add-one unigrams so $\sum_{w \in \mathcal{V}} P(w \mid h) = 1.0$.
4. **Candidate Ranking & Tie-Breaking (`CandidateRanker`):** Candidates are ranked by `(-probability, token)`. Ties are always broken alphabetically (`token_a < token_b`).
5. **Decoding (`prediction/`):**
   - `GreedyDecoder`: returns `candidates[0].token` with zero randomness.
   - `TopKDecoder(k=5)`: truncates to `candidates[:5]`, renormalizes probabilities over the subset, and draws 1 token from isolated `self.rng = random.Random(seed)`.
   - `TemperatureDecoder(temperature=0.8)`: scales positive logits $\ln(P_i)/T$ minus $\max_j \ln(P_j)/T$ for numerical stability and samples via `self.rng.choices(...)`.

---

## 7. Five-Head Scoring Architecture and Linear Score Fusion

In `mode="five_head"`, `NextTokenPredictor` delegates candidate scoring to `HeadOrchestrator`:

| Head | Class | Range | Implemented Scoring Logic |
|---|---|---|---|
| **H1: Grammar Scout** | `GrammarScout` | `[0.0, 1.0]` | Conditional N-gram probability $P(w \mid h)$ from language model/smoother. |
| **H2: Character Keeper** | `CharacterKeeper` | `[0.10, 0.85]` | Named/noun character continuity (`0.85` active, `0.675` previously seen, `0.50` intro, `0.10` immediate repeat, `0.20` non-character). |
| **H3: Setting Painter** | `SettingPainter` | `[0.20, 0.85]` | Setting anchoring (`0.85` initial/active setting, `0.625` setting noun, `0.40` core setting transition, `0.20` non-setting). |
| **H4: Action Booster** | `ActionBooster` | `[0.20, 0.90]` | Verb momentum (`0.90` dynamic verb, `0.50` stative verb, `0.20` non-action token). |
| **H5: Repeat Blocker** | `RepeatBlocker` | `(0.0, 1.0]` | Exempt function words/punctuation `1.0`; immediate self-loop `0.05`; recent bigram cycle `0.20`; otherwise $\exp(-0.80 \cdot c_{\text{window}})$ over `history_window=8`. |

**Critical Mathematical Distinction:** `ScoreFusion.fuse_scores()` normalizes non-negative weights ($\sum \alpha_k = 1.0$) and computes:
$$S_{\text{fused}}(w) = \sum_{k=1}^5 \alpha_k H_k(w)$$
**Fused scores are heuristic/multi-criteria weighted scores, NOT inherently probabilities.** When `TopKDecoder` or `TemperatureDecoder` runs in five-head mode, `NextTokenPredictor` converts `fused_score` values into sampling weights by dividing by `sum(fused_score)` across candidates rather than treating raw `S_fused` outputs as normalized posterior probabilities.

---

## 8. Hamilton Largest-Remainder Allocation and Tie-Breaking

`HamiltonApportionment` ([hamilton_apportionment.py](../src/five_head_nlp/heads/hamilton_apportionment.py)) maps continuous head weights `{w_i}` into an integer budget summing to `total_points = 10` (`d10` physical classroom die representation):

1. Compute quota $q_i = 10 \cdot \frac{w_i}{\sum_j w_j}$.
2. Assign integer floor $a_i = \lfloor q_i \rfloor$ and compute remainder $r_i = q_i - a_i$.
3. Distribute surplus $S = 10 - \sum_i a_i$ (`+1` point each) in descending order of $r_i$.
4. **Deterministic Tie-Breaking Priority:** `["Grammar Scout", "Character Keeper", "Setting Painter", "Action Booster", "Repeat Blocker"]`.

For the balanced weight profile `(0.40, 0.15, 0.15, 0.10, 0.20)`, quotas are `(4.0, 1.5, 1.5, 1.0, 2.0)`. Both `Character Keeper` and `Setting Painter` have remainder `0.5`; `Character Keeper` receives the `1` surplus point due to tie priority, yielding `[4, 2, 1, 1, 2]`.

**Critical Distinction:** Hamilton `d10` integers (`[4, 2, 1, 1, 2]`) are an **explanatory classroom representation and tabletop game mapping**. Python scoring and fusion continue to use the exact continuous normalized fusion weights (`0.40, 0.15, 0.15, 0.10, 0.20`), not the rounded `d10` ratios.

---

## 9. Monte Carlo Configurations (`E1–E8`) and Canonical Output Locations

All canonical Phase 5 experiment definitions reside in [experiment_registry.py](../src/five_head_nlp/experiments/experiment_registry.py) and [configurations.json](../experiments/results/monte_carlo/configurations.json):

| ID | Name | Smoother | Decoder | Weights (`G, C, S, A, R`) | Canonical Results Location |
|---|---|---|---|---|---|
| `E1` | Baseline MLE | `none` | `greedy` | N/A | `experiments/results/monte_carlo/` (`200` stories) |
| `E2` | Laplace Smoothing | `laplace` (`alpha=1.0`) | `greedy` | N/A | `experiments/results/monte_carlo/` (`200` stories) |
| `E3` | Trigram Backoff | `backoff` (`d=0.5`) | `greedy` | N/A | `experiments/results/monte_carlo/` (`200` stories) |
| `E4` | Grammar Only | `backoff` (`d=0.5`) | `greedy` | `(1.00, 0.00, 0.00, 0.00, 0.00)` | `experiments/results/monte_carlo/` (`200` stories) |
| `E5` | Thematic Heads | `backoff` (`d=0.5`) | `greedy` | `(0.50, 0.20, 0.20, 0.10, 0.00)` | `experiments/results/monte_carlo/` (`200` stories) |
| `E6` | Full Five-Head (Top-K) | `backoff` (`d=0.5`) | **`top_k` (`k=5`)** | `(0.40, 0.15, 0.15, 0.10, 0.20)` | `experiments/results/monte_carlo/` (`200` stories) |
| `E7` | Full Five-Head (Greedy) | `backoff` (`d=0.5`) | `greedy` | `(0.40, 0.15, 0.15, 0.10, 0.20)` | `experiments/results/monte_carlo/` (`200` stories) |
| `E8` | Full Five-Head (Temperature) | `backoff` (`d=0.5`) | `temperature` (`T=0.8`) | `(0.40, 0.15, 0.15, 0.10, 0.20)` | `experiments/results/monte_carlo/` (`200` stories) |

> **Canonical E6 Confirmation:** E6 is verified unambiguously in both `experiment_registry.py` and `configurations.json` as **`top_k` with `k=5`** (not `k=3`).

Canonical outputs (`1,600` stories, `0` failures) are stored under `experiments/results/monte_carlo/`:
`configurations.json`, `generated_stories.jsonl`, `failure_records.jsonl`, `per_story_metrics.csv`, `aggregate_metrics.csv`, `configuration_comparison.csv`, `run_metadata.json`, `experiment_summary.md`, and `plots/*.png`.

---

## 10. Reproducibility Harness Usage, Seeds, Isolation, and Interpretation

`python -m five_head_nlp.experiments.reproduce_all` verifies Phase 5 without modifying canonical files:

1. **Isolation Guards (`validate_safe_output_dir`):** Writes only to `reports/reproducibility/<run_id>/` (default deterministic ID `run_b42_s1042_n200`). Any attempt to target `experiments/results`, `experiments/models`, `data`, `phase_7`, `src`, `tests`, `.git`, or repository root raises `ValueError`.
2. **Disjoint Seed Design:**
   - **Deterministic (`E1–E5, E7`):** Uses canonical `base_seed = 42` with canonical index anchoring `seed = base_seed + canonical_index * 10_000 + i`, checking exact equality of `tokens`, `text`, `seed`, and all 7 non-timing metric groups (including `readability`). Hardware `time.perf_counter()` latency fields are excluded.
   - **Stochastic (`E6, E8`):** Defaults to disjoint `stochastic_seed = 1042` (`51042..51241` and `71042..71241`) to generate independent Monte Carlo distributions ($N = 200$).
3. **Statistical Equivalence Interpretation:**
   - Uses **Welch Two One-Sided Tests (TOST)** with **Holm-Bonferroni** step-down adjustment ($\alpha = 0.05$, $m = 12$ comparisons across `6` headline metrics $\times$ `2` configs) plus Hedges' $g$.
   - **Equivalence Margins:** Implemented as **implementation-fixed margins calibrated using canonical Phase 5 dispersion prior to executing the reproduction run** (`±4.0` tokens, `±0.025` bigram repetition rate, `±0.020` TTR, `±0.060` setting consistency, `±0.040` dynamic action ratio, `±0.090` character continuity ratio). They are **not pre-registered** from Phase 5.
   - **Procedure Distinction:** Holm-adjusted TOST decisions control family-wise error via step-down bounds ($\alpha_{\text{step}, k} = 0.05/(12-k+1)$). Separately reported **`99.17%` simultaneous Bonferroni intervals** (`ci_family_bonferroni`) are stricter fixed-factor intervals; `90%` nominal Welch CIs are unadjusted single-test bounds (not `95%` simultaneous family-wise intervals).
   - **Metric Exclusions:** `latency` is excluded due to hardware dependency; `readability` and un-selected sub-metrics (`mattr_20`, trigram repetition) were omitted from stochastic TOST.

Verified local WP9.4 run (`run_b42_s1042_n200`) produced `1,200/1,200` exact matches for `E1–E5, E7` and `12/12` `STATISTICAL_EQUIVALENCE` verdicts for `E6` and `E8` without overwriting canonical artifacts.

---

## 11. Test Discovery, Coverage Measurement, and CI Workflow

- **Test Discovery ([pytest.ini](../pytest.ini)):** Collects `351` tests under `tests/` (`319` core unit/integration + `8` server lifecycle scenarios in [test_studio_lifecycle.py](../tests/test_ui/test_studio_lifecycle.py) + `24` reproducibility tests in [test_reproduce_all.py](../tests/test_experiments/test_reproduce_all.py)).
- **Coverage Measurement ([pyproject.toml](../pyproject.toml)):** Configured with `source = ["src/five_head_nlp"]`, `branch = true`:
  ```bash
  python -m pytest tests -v -rs --cov=src/five_head_nlp --cov-branch --cov-report=term-missing --cov-report=xml:reports/coverage.xml --cov-fail-under=70
  ```
- **Measured Coverage Baseline (WP9.4 Final Audit across `88` scanned modules):**
  - Statement coverage: **`76.10%`** (`3754/4933`, CI gate `>= 72.0%`)
  - Branch coverage: **`66.88%`** (`856/1280`, CI gate `>= 63.0%`)
  - Combined `pytest-cov` score: **`74.20%`** (CI gate `>= 70.0%`)
- **Explicit Discovery Limitation:** `src/five_head_nlp/experiments/` lacks `__init__.py`, so `coverage.py` skips unimported modules inside that folder. Four files remain **excluded** from the current coverage scan:
  - `src/five_head_nlp/experiments/cli_monte_carlo.py`
  - `src/five_head_nlp/experiments/cli_prediction.py`
  - `src/five_head_nlp/experiments/cli_scoring.py`
  - `src/five_head_nlp/experiments/reporting.py`
- **CI Workflow ([.github/workflows/ci.yml](../.github/workflows/ci.yml)):** Configures `9` jobs across `ubuntu-latest`, `windows-latest`, and `macos-latest`: `6` compatibility jobs on Python `3.10`/`3.11` (`343` non-lifecycle tests + CLI checks) and `3` full-suite jobs on Python `3.12` (`351` tests + `validate_sb3.py` + XML statement/branch/combined coverage gates). Local Windows execution on Python `3.12.8` passed; remote GitHub-hosted runs on Ubuntu, macOS, Windows, and Python `3.10`/`3.11` remain **pending** (no remote is configured and no push has been authorized).

---

## 12. Troubleshooting and Known Operational Limitations

1. **Windows Process-Group Containment Limit:** `test_studio_lifecycle.py` uses `CREATE_NEW_PROCESS_GROUP`, `CTRL_BREAK_EVENT`, multi-point `psutil` `(pid, create_time)` descendant tracking, and `proc.kill()` fallback. Because Windows Job Objects (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) are not used, a subprocess that explicitly detaches before `psutil` polling could evade cleanup.
2. **POSIX Process-Group Escape Limit:** On Unix/macOS, `test_studio_lifecycle.py` uses `start_new_session=True` and `os.killpg(..., SIGTERM)` / `SIGKILL`. Child processes that invoke `setsid()` or `setpgid()` escape process-group signals unless previously captured by `psutil` tree polling.
3. **Streamlit Headless & Port Collisions:** Always pass `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` and `--server.headless=true` in automated environments to avoid interactive prompts. If port `8501` is busy, use `--server.port=<PORT>`.
4. **Test-Time Disk Isolation:** Model serialization tests write strictly to pytest's `tmp_path` (`tests/test_models/test_model_serialization.py`). Never run `experiments/run_monte_carlo.py` with default `--output-dir` unless intentionally regenerating canonical `experiments/results/monte_carlo/`.
5. **Scratch Structural vs. Manual Verification:** `validate_sb3.py` certifies `.sb3` ZIP/AST/MD5/math integrity programmatically; it does not simulate manual user interaction inside the Scratch 3.0 online or desktop GUI.
6. **Closed Vocabulary:** The system operates strictly over the `120`-token vocabulary extracted from `data/raw/story_corpus.txt` and cannot generate open-domain words outside that lexicon.
