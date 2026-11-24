# Release Notes — Version 1.0.0
**Project:** The Five-Head Story-Generation System  
**Release Date:** 2026-11-24  
**License:** MIT License  

---

## 1. Executive Summary

Version 1.0.0 marks the production-ready release of **The Five-Head Story-Generation System**, a transparent, classical Natural Language Processing (NLP) framework designed to demystify generative AI and next-token prediction for educators, students (Grades 4–5), and AI researchers.

This release integrates all development phases into a standardized, installable Python package (`five-head-nlp`) with modernized packaging, cross-platform CLI console scripts, historical scaffolding preservation, automated end-to-end lifecycle verification, and zero black-box dependencies.

---

## 2. Core Capabilities

### Classical Probabilistic Language Engine
- **N-Gram Models:** Maximum Likelihood Estimation (MLE) for Unigram, Bigram, and Trigram distributions.
- **Probability Smoothing:** Laplace ($+1$), Add-$k$, Good-Turing, Absolute Discounting with Backoff, and Jelinek-Mercer Linear Interpolation.
- **Decoding Strategies:** Greedy Argmax, Top-$k$ Truncation, and Temperature-scaled Boltzmann Softmax sampling.

### The Five Transparent Scoring Heads
- **Head 1 — Grammar Scout ($G$):** Probabilistic continuation scoring from smoothed N-gram language models.
- **Head 2 — Character Keeper ($C$):** Protagonist tracking and character entity continuity bonuses.
- **Head 3 — Setting Painter ($S$):** Environmental and scenic consistency across narrative boundaries.
- **Head 4 — Action Booster ($A$):** Narrative momentum acceleration via descriptive action verbs.
- **Head 5 — Repeat Blocker ($R$):** Exponential penalty and suppression of immediate token and bigram repetition loops.
- **Linear Score Fusion:** Transparent multi-criteria score aggregation: $S(w) = \sum_{k=1}^5 \alpha_k \cdot H_k(w)$.
- **Hamilton Apportionment:** Largest-remainder method translating continuous head weights into an exact 10-point budget for 10-sided dice ($d10$) tabletop classroom activities.

### Interactive User Interfaces & Educational Tools
- **The Five-Head Story Studio:** Full-featured Streamlit web application providing real-time story generation, candidate token inspector, live head weight adjustment, and Hamilton $d10$ visualization.
- **Scratch 3.0 Educational Implementation (`five_head_story_machine.sb3`):** 100% vanilla Scratch 3.0 visual block project featuring 12 custom vector SVG sprites, 23 global variables, 4 lists, and live score displays.
- **Classroom Unplugged Curriculum:** 60-minute kinesthetic role-playing activity with printable student worksheets, teacher guides, assessment rubrics, and worked arithmetic answer keys.

---

## 3. Work Packages Completed in Phase 8

| Work Package | Title | Core Deliverables |
|---|---|---|
| **WP 8.1** | Modern Python Packaging | Authoring `pyproject.toml` with Python `>=3.10` requirement, setuptools discovery from `src/`, optional `dev` dependencies (`psutil`), and standard MIT `LICENSE`. |
| **WP 8.2** | Historical Scaffolding Preservation | Cleanly moving exploratory drafts (`scratch/`, `unplugged/`, `streamlit_app/`) into `archive/legacy_phase_1_scaffolds/` with complete provenance documentation; establishing `five_head_nlp.decoding` compatibility facade. |
| **WP 8.3** | Path Normalization & CLI Standardization | Authoring 4 canonical console script wrappers (`five-head-monte-carlo`, `five-head-scoring`, `five-head-prediction`, `five-head-studio`) with side-effect-free `--help` handling; sanitizing all absolute machine paths to relative repository paths. |
| **WP 8.4** | Isolated Virtual Environment Build | Clean build and editable installation (`pip install -e ".[dev]"`) in an isolated Python virtual environment (`test_env`); verifying package resolution and console scripts from remote working directories. |
| **WP 8.5** | Comprehensive Quality Assurance | Running 319 regression tests (100% pass), 8 Streamlit lifecycle test scenarios (100% pass), Scratch AST validation (100% valid), Phase 5 artifact consistency check (1,600 JSONL stories), and zero machine-path scan. |
| **WP 8.6** | Release Documentation & Version Tagging | Authoring release notes, updating root `README.md`, creating `CONTRIBUTING.md` and `CITATION.cff`, and preparing release tag `v1.0.0`. |

---

## 4. Quality Assurance & Verification Summary

- **Total Unit & Integration Tests:** 319 passing tests in `tests/` across 8 sub-suites with 0 failures, 0 errors, and 0 skips.
- **Decoding Compatibility Facade:** 2 passing unit tests in `tests/test_decoding/test_decoding_facade.py`.
- **End-to-End Server Lifecycle Verification:** 8 passing scenario tests in `tests/test_ui/verify_studio_lifecycle.py`:
  - Scenario 1 (Normal Lifecycle): Startup HTTP 200, 3.0s observation, graceful control signal exit within timeout, port release.
  - Scenario 2 (Startup Timeout): Timeout detected, symmetrical graceful teardown executed, resources released.
  - Scenario 3 (Early Exit): Process exit code captured, exception-safe teardown.
  - Scenario 4 (Graceful Fallback): Simulated unresponsive target terminated cleanly via fallback kill.
  - Scenario 5 (Signal Dispatch Safety): Missing process handled without uncaught exceptions.
  - Scenario 6 (Pre-Launch Port Collision): Port conflict detected, launch safely aborted.
  - Scenario 7 (Orphan Cleanup): Subprocess tree traversal detects and terminates surviving child processes.
  - Scenario 8 (Post-Shutdown Port Hold): Bounded retry polling accurately detects lingering port occupation.
- **Scratch Project Validation:** `validate_sb3.py` confirms 100% validity across all 12 targets, 4 monitors, 23 variables, 4 lists, and 12 asset MD5 checksums.
- **Phase 5 Benchmark Consistency:** Exactly 1,600 JSONL records confirmed intact across 8 experimental conditions.
- **Machine Path Scan:** Exactly 0 absolute machine paths exist across all `.py`, `.json`, and `.md` project files.

---

## 5. Verified Boundaries & Documented Limitations

### Verified Boundaries
- **Python Compatibility:** Tested and verified on Python 3.10, 3.11, and 3.12.
- **Cross-Platform CLI Operation:** All console scripts execute identically from inside the repository and from remote working directories.
- **Graceful Shutdown:** Streamlit server responds cleanly to OS interrupt signals (`CTRL_BREAK_EVENT` on Windows; `SIGTERM` on POSIX).

### Documented Platform Limitations
- **Windows Process Group Containment:** Without enclosing processes in a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`), any untracked grandchild process that deliberately breaks away before `psutil` discovery enumeration could theoretically remain detached. For v1.0.0, Windows Job Objects are formally accepted as not required; multi-point `psutil` dynamic child tracking is the accepted standard.
- **POSIX Process Groups:** Processes that deliberately invoke `setsid()` or `setpgid()` escape session process groups. The test harness detects them on a best-effort basis if discovered via `psutil` prior to detachment.
- **Scope Limitation:** The framework uses classical probabilistic NLP and transparent scoring heads; deep neural networks and black-box LLM APIs are intentionally excluded from the architecture.
