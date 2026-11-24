# Project Implementation Log

## Phase 1: Project Initialization & Directory Scaffolding
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Repository Structure**:
   - Created core directory tree covering `data/`, `docs/`, `experiments/`, `scratch/`, `src/five_head_nlp/`, `streamlit_app/`, `tests/`, and `unplugged/`.
   - Initialized Python package subdirectories with `__init__.py` markers under `src/` and `tests/`.
2. **Configuration & Dependencies**:
   - Created `requirements.txt` specifying Python 3.11-compatible packages (`numpy`, `pandas`, `streamlit`, `matplotlib`, `pytest`, `pytest-cov`) without external LLM/neural dependencies.
   - Created `.gitignore` ignoring virtual environments, caches, OS files, and experimental outputs.
3. **Documentation**:
   - Authored root `README.md` introducing the system, architecture, and roadmap.
   - Authored `docs/architecture.md` detailing end-to-end pipeline, scoring formulation, and subsystem design.
   - Authored `docs/mathematical_formulation.md` as a formal mathematical placeholder.
   - Authored `scratch/README.md` defining the planned Scratch 3.0 visual implementation schema.

---

## Phase 2.1: Reconstruct and Validate Story Corpus
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Corpus Reconstruction**:
   - Authored 34 connected children's story sentences in `data/raw/story_corpus.txt`.
   - Incorporated all five characters (Milo, Pip, Zara, Sleepy Owl, Playful Fox) and four settings (Bright Forest, Crystal Castle, Dark Cave, Secret Garden).
   - Created `data/raw/corpus_metadata.json` documenting metadata, characters, settings, and reconstruction parameters.
2. **Documentation**:
   - Authored `docs/corpus_reconstruction.md` detailing design methodology, metric comparisons, known transition tables, and analytical explanations of branching factor.
3. **Validation & Verification**:
   - Implemented `src/five_head_nlp/utils/validate_corpus.py` with standalone metrics calculation and reporting CLI.
   - Implemented comprehensive pytest suite in `tests/test_corpus_validation.py` (9 tests passing).
   - Configured `pytest.ini` for deterministic module discovery.
   - Verified that all 18 known target bigram transition frequencies match exactly (100% precision).
   - Hit exact targets: 34 sentences, 381 words, 415 tokens, 120 unique vocabulary terms, 11.21 average words/sentence, FKGL 3.99 (~4.02).

---

## Phase 2.1.1: Independent Corpus Statistics Audit and Correction
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Corpus Integrity Verification**:
   - Verified that all 34 sentences in `data/raw/story_corpus.txt` remain intact and preserved without modification.
2. **Calculation Conventions Formalized**:
   - Explicitly documented punctuation tokenization (periods as distinct tokens), vocabulary size composition ($119 + 1 = 120$), sentence-final inclusions, continuous flat-stream vs within-sentence boundary handling, and branching factor formulations.
3. **Audit of Pivot and Setting Tokens**:
   - Recalculated exact unigram frequencies, outgoing transitions, and unique next-token counts for `near` (7 unigrams, unique next: 1), `castle` (9 unigrams, unique next: 6), `cave` (7 unigrams, unique next: 5), `the` (63 unigrams, unique next: 24), `a` (11 unigrams, unique next: 11), `to` (11 unigrams, unique next: 5), `zara` (7 unigrams, unique next: 6), `milo` (7 unigrams, unique next: 7), `pip` (7 unigrams, unique next: 6), `and` (12 unigrams, unique next: 6), and `.` (34 unigrams, 33 outgoing transitions, unique next: 12).
   - Audited total unique bigrams: 271 for the continuous flat stream ($N-1=414$ transitions; branching factor 2.26) and 259 for within-sentence transitions (381 transitions; branching factor 2.18).
4. **Documentation & Metadata Updates**:
   - Updated `data/raw/corpus_metadata.json` separating original reference targets from actual reconstructed statistics.
   - Updated `docs/corpus_reconstruction.md` with complete audit tables and discrepancies analysis.
5. **Validation & Testing Enhancements**:
   - Updated `src/five_head_nlp/utils/validate_corpus.py` to output audited unigram, out-transition, and continuation distributions.
   - Added `test_pivot_token_actual_statistics` to `tests/test_corpus_validation.py` to assert actual corpus counts without hardcoded assumptions.
   - All 10 unit tests pass via pytest.

---

## Phase 2.2: Implement Word-Level Tokenization Pipeline
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Tokenizer Implementation**:
   - Implemented `StoryTokenizer` class and functional `tokenize(text)` / `normalize_token(token)` interfaces in `src/five_head_nlp/tokenization/tokenizer.py`.
   - Pattern: `r"\w+(?:['’]\w+)?|[^\w\s]"` delivering word tokenization, punctuation separation, contraction preservation, and case normalization.
   - Preserves special symbols losslessly as independent tokens rather than silently dropping them.
   - Raised appropriate `TypeError` for non-string inputs.
   - Exposed module interface via `src/five_head_nlp/tokenization/__init__.py`.
2. **Corpus Integration**:
   - Verified tokenization of `data/raw/story_corpus.txt` producing exactly 415 tokens, 381 word tokens, 34 period tokens, 0 empty tokens, and 100% lowercase normalized words.
3. **Automated Pytest Suite**:
   - Created `tests/test_tokenization/test_tokenizer.py` with 18 comprehensive tests covering basic sentences, punctuation, normalization, whitespace, contractions, unicode, error handling, and corpus integration.
   - Full suite passes: 28 passing tests across `test_tokenizer.py` (18) and `test_corpus_validation.py` (10).

---

## Phase 2.3: Corpus Loader and Vocabulary Management
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Corpus Loader (`CorpusLoader`)**:
   - Implemented in `src/five_head_nlp/tokenization/corpus_loader.py`.
   - Reads UTF-8 story corpus, tokenizes each line via `StoryTokenizer`, preserves sentence boundaries (`get_sentences()`), generates flat token stream (`get_flat_tokens()`), and computes metadata (`get_metadata()`).
   - Implemented strict error handling: `FileNotFoundError` on missing corpus, `ValueError` on empty or whitespace-only files.
2. **Vocabulary Manager (`Vocabulary`)**:
   - Implemented in `src/five_head_nlp/tokenization/vocabulary.py`.
   - Supports both flat token sequences and nested sentence lists.
   - Implemented deterministic vocabulary ordering: descending unigram frequency followed by alphabetical tie-breaking.
   - Maintained dual vocabulary views: full-token vocabulary ($|V| = 120$, including `.`) and lexical vocabulary ($|V_{\text{lexical}}| = 119$, words only).
   - Provided bidirectional token-to-ID and ID-to-token lookups, unigram frequency queries, and empirical probabilities ($P(w) = f(w)/N$).
3. **Processed Data Artifacts**:
   - Implemented `export_processed_artifacts()` in `src/five_head_nlp/tokenization/export_artifacts.py`.
   - Generated `data/processed/tokenized_corpus.json` (sentences, flat tokens, token counts).
   - Generated `data/processed/vocabulary.json` (mappings, frequencies, unigram probabilities).
   - Generated `data/processed/word_frequencies.csv` (120 rows sorted by descending frequency and alphabetical order).
4. **Automated Pytest Suite**:
   - Created `tests/test_tokenization/test_corpus_loader.py` (11 tests).
   - Created `tests/test_tokenization/test_vocabulary.py` (16 tests).
   - Full project test suite passes: 55 passed in 0.61s.

---

## Phase 2.4: Corpus Statistical Analysis and Exploratory Data Analysis
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Corpus Statistics Engine (`CorpusStatistics`)**:
   - Implemented in `src/five_head_nlp/evaluation/corpus_statistics.py`.
   - Reuses `CorpusLoader` and `Vocabulary` without duplication.
   - Computes basic dimensions, sentence lengths (min, max, avg, median), lexical frequency distributions, unigram probabilities ($P(w) = f(w)/N$), hapax legomena count and ratio ($57 / 119 = 47.9\%$), and TTR.
   - Computes strictly within-sentence bigram transitions ($381$ transitions, $259$ unique bigrams), ensuring period `.` has 0 outgoing transitions.
   - Computes branching factors across full vocabulary ($259 / 120 = 2.16$) and active vocabulary ($259 / 119 = 2.18$).
   - Analyzes pivot token occurrences, outgoing transitions, and unique next tokens for `the`, `.`, `and`, `a`, `to`, `near`, `milo`, `pip`, `zara`, `castle`, and `cave`.
   - Analyzes recurring characters (Milo: 7, Pip: 7, Zara: 7, Owl: 5, Fox: 4) and setting landmarks (Crystal Castle, Dark Cave, Secret Garden, Bright Forest).
2. **Visualizations Engine (`CorpusPlotter`)**:
   - Implemented in `src/five_head_nlp/evaluation/corpus_plots.py` using `matplotlib` (strictly no seaborn).
   - Generated 6 publication-ready figures in `experiments/results/corpus_analysis/`: `top_20_words.png`, `top_20_bigrams.png`, `sentence_length_distribution.png`, `character_occurrences.png`, `setting_occurrences.png`, and `vocabulary_frequency_distribution.png`.
3. **Machine-Readable Outputs**:
   - Saved `experiments/results/corpus_analysis/corpus_statistics.json` containing complete structured metrics.
   - Saved `experiments/results/corpus_analysis/bigram_frequencies.csv` (259 rows).
   - Saved `experiments/results/corpus_analysis/pivot_token_statistics.csv`.
4. **Documentation**:
   - Authored `docs/corpus_statistics.md` detailing statistical distributions, mathematical formulations, plot interpretations, and corpus limitations.
5. **Automated Pytest Suite**:
   - Created `tests/test_evaluation/test_corpus_statistics.py` with 9 tests covering synthetic correctness and real corpus integration.
   - Full test suite passes: 64 passed in 1.55s.

---

## Phase 3.1: N-Gram Probability Engine
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **N-Gram Probability Engine (`NGramProbabilityEngine`)**:
   - Implemented in `src/five_head_nlp/probability/mle.py`.
   - Supports configurable primary order $N \ge 1$ with automatic extraction and caching of lower orders ($1 \dots n$).
   - Provides sequence-level and corpus-level extraction functions:
     - `extract_ngrams(tokens, n)`
     - `extract_sentence_ngrams(sentences, n)`
     - `extract_unigrams(sentences)`
     - `extract_bigrams(sentences)`
     - `extract_trigrams(sentences)`
   - Strictly enforces sentence boundary isolation: transitions never bridge between the terminal token of sentence $k$ and the initial token of sentence $k+1$.
   - The sentence-ending period `.` has exactly 0 outgoing transitions ($C(., \cdot) = 0$).
2. **Maximum Likelihood Estimation (MLE)**:
   - Implemented unigram MLE: $P_{\text{MLE}}(w) = C(w) / N_{\text{total}}$.
   - Implemented bigram conditional MLE: $P_{\text{MLE}}(w_i \mid w_{i-1}) = C(w_{i-1}, w_i) / C(w_{i-1})$.
   - Implemented trigram conditional MLE: $P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = C(w_{i-2}, w_{i-1}, w_i) / C(w_{i-2}, w_{i-1})$.
   - Implemented candidate continuation lookups: `get_candidates(context)` with deterministic ordering (frequency descending, alphabetical tie-breaking).
   - Implemented candidate probability distributions: `get_candidate_distribution(context)` summing to 1.0 for all valid observed contexts.
   - Implemented sequence probability computation via the chain rule of probability: `sequence_probability(sequence)`.
   - Implemented information theoretic log probability: `log_probability(token, context, base=2.0)`.
   - Implemented branching factor metrics: `branching_factor(context)` and `average_branching_factor(order)`.
3. **Module Interfaces & Compatibility**:
   - Exposed primary API in `src/five_head_nlp/probability/__init__.py`.
   - Re-exported in `src/five_head_nlp/ngrams/__init__.py` for backward and forward compatibility.
4. **Integration with Real Story Corpus**:
   - Total unigrams: 415 (unique: 120).
   - Total bigrams: 381 (unique: 259).
   - Total trigrams: 347 (unique: 303).
   - Confirmed average bigram branching factor: $259 / 119 = 2.1765$.
   - Confirmed key n-gram counts and transition probabilities:
     - `("crystal", "castle")`: count 6, $P(\text{castle} \mid \text{crystal}) = 1.0$.
     - `("dark", "cave")`: count 6, $P(\text{cave} \mid \text{dark}) = 1.0$.
     - `("secret", "garden")`: count 5, $P(\text{garden} \mid \text{secret}) = 1.0$.
     - `("the", "crystal", "castle")`: count 6, $P(\text{castle} \mid \text{the}, \text{crystal}) = 1.0$.
5. **Automated Pytest Suite**:
   - Created `tests/test_probability/test_mle.py` (23 test functions) covering sequence extraction, sentence isolation, hand-calculated synthetic probability distributions, real corpus dimensions, known transition probabilities, and edge/error handling.
   - Created `tests/test_ngrams/test_mle.py` running the test harness for the n-gram suite.
   - **Full test suite passes: 110 passed in 1.60s (64 baseline + 46 new assertions across suites)**.

---

## Phase 3.2: Language Model Architecture
- **Status**: Completed
- **Date**: 2026-11-23

### Completed Activities
1. **Core Model Hierarchy**:
   - Implemented abstract base class `BaseNGramModel` in `src/five_head_nlp/models/base_ngram.py`.
   - Implemented `UnigramModel` in `src/five_head_nlp/models/unigram.py` (order $N=1$, context-independent, sequence probability $\prod P(w_i)$).
   - Implemented `BigramModel` in `src/five_head_nlp/models/bigram.py` (order $N=2$, context $w_{i-1}$, sentence boundary preservation, sequence probability $P(w_1) \prod P(w_i \mid w_{i-1})$).
   - Implemented `TrigramModel` in `src/five_head_nlp/models/trigram.py` (order $N=3$, context $(w_{i-2}, w_{i-1})$, graceful fallback to bigram and unigram on shorter histories).
   - Exposed unified package in `src/five_head_nlp/models/__init__.py`.
2. **Vocabulary & Unknown-Token Integration**:
   - Reused existing `Vocabulary` class directly without duplicating vocabulary representations.
   - Integrated `Vocabulary.from_dict()` for safe reconstruction from serialized states.
   - Defined consistent unknown-token protocol: $P(w \mid \text{context}) = 0.0$ for $w \notin \mathcal{V}$; `is_known_token()` and `is_known_context()` methods for disambiguation.
   - Avoided artificial `<UNK>` injection into the raw corpus.
3. **Safe Model Serialization**:
   - Implemented `save(path)` and polymorphic `BaseNGramModel.load(path)` using standard JSON format (no pickle/unsafe code execution).
   - Validated schema integrity (required keys, model order, class mapping).
   - Stored pre-trained models under `experiments/models/`: `unigrammodel_corpus.json`, `bigrammodel_corpus.json`, `trigrammodel_corpus.json`.
   - Verified that all predictions, candidate orderings, and joint probabilities remain 100% identical before and after serialization.
4. **Documentation**:
   - Authored `docs/language_model_architecture.md` detailing architecture, class diagrams, method signatures, unknown-token protocols, and JSON serialization schemas.
5. **Automated Pytest Suite**:
   - Created `tests/test_models/test_base_ngram.py` (7 tests).
   - Created `tests/test_models/test_unigram.py` (10 tests).
   - Created `tests/test_models/test_bigram.py` (9 tests).
   - Created `tests/test_models/test_trigram.py` (8 tests).
   - Created `tests/test_models/test_model_serialization.py` (10 tests).
   - **Full project test suite passes: 154 passed in 3.00s (110 baseline + 44 new tests across test_models/)**.

---

## Phase 3.3: Probability Smoothing
- **Status**: Completed (Pending User Approval to begin Phase 3.4)
- **Date**: 2026-11-23

### Completed Activities
1. **Smoothing Architecture (`src/five_head_nlp/smoothing/`)**:
   - Created extensible `BaseSmoother` base class defining standardized smoothing interfaces, log-probability, sequence likelihood, and cross-sentence perplexity evaluation.
   - Implemented `LaplaceSmoother` (`laplace.py`): Add-One smoothing with simplex normalization $\sum_{v \in \mathcal{V}} P(v \mid h) = 1.0$.
   - Implemented `AddKSmoother` (`add_k.py`): Parametric smoothing supporting arbitrary $k > 0$ (tested $k=0.1, 0.5, 1.0$).
   - Implemented `GoodTuringSmoother` (`good_turing.py`): Frequency-of-frequencies ($N_r$), adjusted count estimation ($r^* = (r+1)N_{r+1}/N_r$), unseen mass allocation ($P_0 = N_1 / N$), and missing-bin fallbacks for small-corpus stability.
   - Implemented `BackoffSmoother` (`backoff.py`): Absolute discounting backoff cascading Trigram $\to$ Bigram $\to$ Unigram with exact normalization coefficients $\alpha(h)$ and observable fallback tiers.
   - Implemented `InterpolationSmoother` (`interpolation.py`): Linear interpolation $\lambda_1 P_{\text{uni}} + \lambda_2 P_{\text{bi}} + \lambda_3 P_{\text{tri}}$ with weight validation, active tier normalization, and grid-search simplex optimization.
2. **Comparative Evaluation & Reporting**:
   - Implemented `SmoothingEvaluator` (`evaluator.py`) evaluating all techniques on a strictly separated train (27 sentences, 80%) / validation (7 sentences, 20%) split with zero train/test leakage.
   - Generated reports under `experiments/results/smoothing_analysis/`:
     - `smoothing_results.json`: Comprehensive metrics.
     - `smoothing_comparison.csv`: Tabular comparison of validation zero-events, NLL, and perplexity.
     - `good_turing_frequency_counts.csv`: $N_r$ and $r^*$ tables across N-gram orders.
     - `probability_distribution.csv`: Probability distributions for key contexts (`crystal`, `dark`, `near`, etc.).
3. **Visualizations**:
   - Implemented `SmoothingPlotter` (`plots.py`) generating 6 figures using matplotlib only (no seaborn):
     - `mle_vs_laplace.png`
     - `add_k_comparison.png`
     - `good_turing_adjusted_counts.png`
     - `context_distributions.png`
     - `perplexity_comparison.png`
     - `zero_probability_counts.png`
4. **Automated Pytest Suite**:
   - Created `tests/test_smoothing/test_base_smoother.py` (6 tests).
   - Created `tests/test_smoothing/test_laplace.py` (6 tests).
   - Created `tests/test_smoothing/test_add_k.py` (5 tests).
   - Created `tests/test_smoothing/test_good_turing.py` (6 tests).
   - Created `tests/test_smoothing/test_backoff.py` (6 tests).
   - Created `tests/test_smoothing/test_interpolation.py` (7 tests).
   - Created `tests/test_smoothing/test_evaluator.py` (3 tests).
   - **Full test suite passes: 193 passed in 9.84s (154 baseline + 39 smoothing tests)**.

---

## Phase 3.3: Mathematical and Evaluation Audit
- **Status**: Completed (Pending User Approval to begin Phase 3.4)
- **Date**: 2026-11-23

### Audit Findings & Resolutions
1. **Cross-Entropy & Perplexity Mathematical Unification**:
   - **Root Cause of Discrepancy**: Prior evaluator logged base-2 cross-entropy ($H_2 = 6.6803$ bits) alongside base-2 perplexity ($2^{6.6803} = 102.56$). However, in standard natural logarithm formulations, $H_{\text{nats}} = 4.6304$ nats, and $\exp(4.6304) = 102.56$.
   - **Fix**: Standardized natural logarithm across all smoothing calculations:
     $$\text{NLL}_{\text{total}} = -\sum \ln P, \quad H = \frac{\text{NLL}}{M}, \quad \text{PPL} = \exp(H)$$
     Reporting both $H$ (nats) and $H_{\text{bits}} = H / \ln(2)$, with $\text{PPL} = \exp(H) = 2^{H_{\text{bits}}}$ holding to 9-decimal floating-point tolerance.
2. **Leak-Free 3-Way Partition**:
   - Established strict deterministic split:
     - **Train (24 sentences, 295 tokens, 70.6%)**: model fitting and N-gram counting.
     - **Validation (5 sentences, 62 tokens, 14.7%)**: hyperparameter tuning and interpolation weight optimization $\boldsymbol{\lambda}^* = (0.65, 0.20, 0.15)$.
     - **Test (5 sentences, 58 tokens, 14.7%)**: strictly held-out final benchmark evaluation.
3. **Absolute Discounting Backoff Degenerate Fallback**:
   - Resolved edge case where lower-order distribution assigned 0 mass to unobserved continuations ($\sum_{v \in \mathcal{V}_{\text{unseen}}} P_{\text{lower}}(v) = 0$).
   - Reserved mass $\beta(h)$ is now distributed uniformly across unobserved vocabulary tokens ($\frac{\beta(h)}{|\mathcal{V}_{\text{unseen}}|}$), preserving strict simplex normalization $\sum_{v \in \mathcal{V}} P(v \mid h) = 1.0$.
4. **Good-Turing Type-Token Formalization**:
   - Clarified distinction between singleton event types ($N_1 = 46$ unigram hapax legomena) and total occurrences ($N = \sum r N_r = 295$ train tokens).
   - Added `Total_Occurrences_r_times_N_r` column to `good_turing_frequency_counts.csv`.
5. **Audited Test-Set Benchmark Results ($M = 58$ events)**:
   - **Unsmoothed MLE (Trigram)**: Zero Events = 45 (77.59%), PPL = $\infty$.
   - **Laplace (Add-1)**: Zero Events = 0 (0.0%), $H = 4.6430$ nats (6.6985 bits), PPL = 103.8586.
   - **Add-k (k=0.1)**: Zero Events = 0 (0.0%), $H = 4.4636$ nats (6.4397 bits), PPL = 86.8035.
   - **Add-k (k=0.5)**: Zero Events = 0 (0.0%), $H = 4.5870$ nats (6.6176 bits), PPL = 98.1991.
   - **Good-Turing**: Zero Events = 0 (0.0%), $H = 4.3938$ nats (6.3389 bits), PPL = 80.9451.
   - **Absolute Discounting Backoff**: Zero Events = 0 (0.0%), $H = 3.7385$ nats (5.3935 bits), **PPL = 42.0339 (Best Single Model)**.
   - **Linear Interpolation (Fixed: 0.15, 0.35, 0.50)**: Zero Events = 0 (0.0%), $H = 3.8475$ nats (5.5508 bits), PPL = 46.8752.
   - **Linear Interpolation (Optimized on Val: 0.65, 0.20, 0.15)**: Zero Events = 0 (0.0%), $H = 3.7888$ nats (5.4660 bits), **PPL = 44.2019**.
6. **Artifacts & Tests**:
   - Re-generated all CSV, JSON, and PNG analytical artifacts in `experiments/results/smoothing_analysis/`.
   - Added mathematical audit tests in `test_base_smoother.py`, `test_backoff.py`, `test_interpolation.py`, and `test_evaluator.py`.
   - Total test suite: 193 tests passing.

---

## Phase 3.4: Next-Token Prediction and Decoding Engine
- **Status**: Completed (Pending User Approval to begin Phase 3.5 / Phase 4)
- **Date**: 2026-11-23

### Completed Activities
1. **Next-Token Prediction Package (`src/five_head_nlp/prediction/`)**:
   - `prediction_result.py`: `CandidateToken` and `PredictionResult` data structures for standardized inference outputs.
   - `candidate_ranker.py`: `CandidateRanker` sorting candidate tokens by descending probability with deterministic alphabetical tie-breaking and support for top-N truncation.
   - `decoding.py`: Abstract `BaseDecoder` class with isolated, reproducible pseudo-random number generator (`random.Random(seed)`).
   - `greedy.py`: `GreedyDecoder` implementing deterministic argmax selection ($w_t = \arg\max P$).
   - `top_k.py`: `TopKDecoder` implementing top-$k$ candidate filtering, renormalization across top-$k$, and stochastic sampling.
   - `temperature.py`: `TemperatureDecoder` implementing numerically stable softmax temperature scaling ($P_T(w) \propto \exp(\ln(P(w)) / T)$) with zero-probability preservation.
   - `generation_state.py`: `GenerationState` managing sequence rollouts, token counts, rolling context windowing, and terminal period (`.`) stopping enforcement.
   - `predictor.py`: `NextTokenPredictor` coordinating models (`BaseNGramModel`), smoothers (`BaseSmoother`), ranking, and decoding strategies for single-step prediction and sequence generation.
   - `evaluator.py`: `PredictionEvaluator` benchmarking accuracy, top-k accuracy, MRR, latency, coverage, and reproducibility across held-out test prediction events ($M = 58$).
2. **Performance Optimization**:
   - Implemented context distribution caching in `BackoffSmoother`, accelerating per-step backoff distribution retrieval from ~500 ms to **sub-millisecond (< 0.2 ms)** latency.
3. **Reproducible Demonstration & Artifacts**:
   - Script: `experiments/run_next_token_prediction.py`.
   - Artifacts generated under `experiments/results/next_token_prediction/`:
     - `prediction_examples.json`: Step-by-step predictions across sample contexts.
     - `candidate_rankings.csv`: Candidate probability rankings and log-probabilities.
     - `decoding_comparison.csv`: Comparative benchmark metrics across decoding strategies.
     - `prediction_metrics.json`: Comprehensive metrics and reproducibility checks.
     - `prediction_latency.csv`: Execution latency benchmarks across models and smoothers.
   - Visualizations generated with Matplotlib:
     - `top_candidate_distribution.png`
     - `top_k_candidate_comparison.png`
     - `temperature_effect_distribution.png`
     - `decoding_strategy_comparison.png`
4. **Automated Pytest Suite**:
   - `tests/test_prediction/test_candidate_ranker.py` (7 tests)
   - `tests/test_prediction/test_greedy.py` (3 tests)
   - `tests/test_prediction/test_top_k.py` (5 tests)
   - `tests/test_prediction/test_temperature.py` (7 tests)
   - `tests/test_prediction/test_generation_state.py` (7 tests)
   - `tests/test_prediction/test_predictor.py` (8 tests)
   - `tests/test_prediction/test_prediction_integration.py` (5 tests)
   - **Total project test suite passes: 235 passed (193 baseline + 42 new prediction tests, 0 failures)**.

---

## Phase 4: Five Transparent Scoring Heads Implementation
- **Status**: Completed
- **Date**: 2026-11-24

### Completed Activities
1. **Scoring Heads Architecture (`src/five_head_nlp/heads/`)**:
   - `base_head.py`: Abstract `BaseHead` enforcing pure evaluation (`score_candidates()` does not mutate state) and explicit state updates (`update_state()`).
   - `head_result.py`: Standardized `HeadScore`, `FusedCandidate`, and `OrchestratorResult` dataclasses preserving individual head breakdowns and explanations.
   - `scoring_config.py`: Configuration dataclasses for weights, thresholds, toggles, and domain lexicons.
   - `grammar_scout.py` (Head 1): Evaluates grammatical and statistical plausibility using underlying N-gram conditional probabilities ($G(w \mid h) = P(w \mid h)$).
   - `character_keeper.py` (Head 2): Tracks character continuity, mentions, active focus, and introduction bonuses based on corpus character lexicon.
   - `setting_painter.py` (Head 3): Tracks environmental and spatial consistency (castle, cave, forest, garden) and setting shifts.
   - `action_booster.py` (Head 4): Propels plot momentum by rewarding dynamic action verbs (flew, ran, found, showed) over stative verbs.
   - `repeat_blocker.py` (Head 5): Penalizes immediate token self-loops, 2-token attractor bigram loops, and lexical saturation via exponential frequency decay ($R(w) = \exp(-\gamma \cdot c)$), while exempting function words.
   - `score_fusion.py`: Linear weighted score fusion ($S(w) = \sum \alpha_k H_k(w)$) with weight normalization and deterministic alphabetical tie-breaking.
   - `hamilton_apportionment.py`: Hamilton's largest-remainder method converting continuous weights into an exact 10-point ($d10$) integer allocation with deterministic tie-breaking.
   - `head_orchestrator.py`: `HeadOrchestrator` coordinating all 5 heads, score fusion, d10 apportionment, and state propagation.
   - `evaluator.py`: `HeadEvaluator` running full ablation benchmarks across the held-out test split.
2. **Predictor Integration**:
   - Enhanced `NextTokenPredictor` to support two selectable prediction modes:
     - `mode="baseline"`: Pure statistical N-gram decoding (100% backward-compatible).
     - `mode="five_head"`: Multi-head candidate evaluation, fused scoring, d10 interpretation, and stateful updates.
3. **Artifacts & Visualizations (`experiments/results/five_head_scoring/`)**:
   - `head_score_breakdown.csv`: Per-token individual head scores, explanations, and evidence.
   - `candidate_fusion_results.csv`: Candidate rankings under 5 distinct weight profiles.
   - `hamilton_allocation.json`: d10 point allocations for balanced, grammar-dominated, character-heavy, action-heavy, and setting-heavy profiles.
   - `head_evaluation_metrics.json`: Comparative ablation metrics on held-out test split.
   - `scoring_configuration.json`: Active head configuration parameters and lexicons.
   - Matplotlib analytical charts:
     - `individual_head_scores.png`
     - `weighted_fusion_contributions.png`
     - `hamilton_d10_allocation.png`
     - `baseline_vs_five_head_rankings.png`
     - `repetition_penalty_decay.png`
     - `character_setting_tracking.png`
4. **Automated Pytest Suite**:
   - `tests/test_heads/test_base_head.py` (3 tests)
   - `tests/test_heads/test_grammar_scout.py` (2 tests)
   - `tests/test_heads/test_character_keeper.py` (5 tests)
   - `tests/test_heads/test_setting_painter.py` (4 tests)
   - `tests/test_heads/test_action_booster.py` (3 tests)
   - `tests/test_heads/test_repeat_blocker.py` (4 tests)
   - `tests/test_heads/test_score_fusion.py` (4 tests)
   - `tests/test_heads/test_hamilton_apportionment.py` (3 tests)
   - `tests/test_heads/test_head_orchestrator.py` (4 tests)
   - `tests/test_heads/test_head_integration.py` (4 tests)
   - **Total project test suite passes: 271 passed (235 baseline + 36 new head tests, 0 failures)**.

---

## Phase 5: Experimental Suite & Monte Carlo Story Generation Harness
- **Status**: Completed
- **Date**: 2026-11-24

### Completed Activities
1. **Preflight Verification & Mathematical Audit**:
   - Verified strict evaluation priority in Character Keeper: (1) Immediate duplicate penalty (0.10) inspecting both `context` and `state.all_tokens`, (2) Active character focus (0.85), (3) Recurrent character mention (0.60), (4) New character introduction (0.50), and (5) Non-character neutral (0.20).
   - Audited fused score-to-probability mapping in `NextTokenPredictor`, verifying exact non-negative normalization to the probability simplex.
   - Implemented high-performance alpha normalization caching in `BackoffSmoother`, accelerating backoff inference by over 10,000x with zero mathematical divergence.
2. **Modular Metrics Architecture (`src/five_head_nlp/experiments/metrics/`)**:
   - `length_metrics.py`: Total tokens, words, sentences, length distributions.
   - `repetition_metrics.py`: Immediate token repetition, repeated bigrams, repeated trigrams, 8-token rolling window repetition rates.
   - `diversity_metrics.py`: Lexical Type-Token Ratio (TTR), Moving-Average TTR (MATTR-20), hapax legomena ratio.
   - `character_metrics.py`: Character introductions, total mentions, character density, continuity ratio.
   - `setting_metrics.py`: Setting introductions, mentions, density, and consistency ratio.
   - `action_metrics.py`: Dynamic action counts, stative verb counts, action density, and dynamic action ratio.
   - `readability_metrics.py`: Average sentence length, word length, and lexical density proxies.
   - `latency_metrics.py`: CPU-timed generation duration, per-token latency distribution (mean, median, p95), and throughput.
   - `__init__.py`: Consolidated `StoryMetrics` dataclass and flattening utilities.
3. **Experimental Framework & Configurations (`E1` to `E8`)**:
   - `experiment_config.py`: Complete configuration dataclass.
   - `experiment_registry.py`: Formal specifications for all 8 experimental conditions:
     - **E1**: Baseline MLE (unsmoothed TrigramModel, baseline mode, greedy).
     - **E2**: Laplace Smoothing (alpha=1.0, baseline mode, greedy).
     - **E3**: Trigram Backoff (discount=0.5, baseline mode, greedy).
     - **E4**: Grammar Only (Five-head mode, Grammar weight 1.0, greedy).
     - **E5**: Thematic Heads (Five-head mode, Grammar 0.50, Char 0.20, Setting 0.20, Action 0.10, Repeat 0.0, greedy).
     - **E6**: Full Five-Head (Balanced weights, Top-K k=5 stochastic decoding).
     - **E7**: Full Five-Head Greedy (Balanced weights, deterministic greedy decoding).
     - **E8**: Full Five-Head Temperature (Balanced weights, Temperature T=0.8 stochastic decoding).
   - `story_generator.py`: Autoregressive generation engine with isolated pseudo-random number generator seeding, loop breaking, period termination, and constraint enforcement (target 80–120 tokens, min 3 sentences).
   - `generation_harness.py`: Monte Carlo batch orchestration with non-overlapping seed allocation and warm-up cycles.
4. **Statistical Aggregation & Reporting (`src/five_head_nlp/experiments/`)**:
   - `aggregation.py`: Computes full distributional metrics (mean, median, std, min, max) and builds comparison tables.
   - `reporting.py`: Generates 8 publication-ready Matplotlib figures in `experiments/results/monte_carlo/plots/` and `experiment_summary.md`.
   - `experiments/run_monte_carlo.py`: CLI driver supporting `--stories-per-config`, `--output-dir`, and `--seed`.
5. **Full Monte Carlo Generation Execution ($N=1,600$)**:
   - Generated 200 stories per configuration across all 8 configurations (1,600 total stories) in 221.02 seconds (7.2 stories/sec) with **0 generation failures**.
   - Output artifacts written to `experiments/results/monte_carlo/`:
     - `generated_stories.jsonl` (1,600 stories)
     - `failure_records.jsonl` (0 failures)
     - `per_story_metrics.csv` (1,600 records)
     - `aggregate_metrics.csv`
     - `configuration_comparison.csv`
     - `run_metadata.json`
     - `configurations.json`
     - `experiment_summary.md`
     - `plots/` (8 Matplotlib charts)
6. **Automated Pytest Suite**:
   - `tests/test_experiments/test_experiment_config.py` (8 tests)
   - `tests/test_experiments/test_metrics.py` (9 tests)
   - `tests/test_experiments/test_generation_harness.py` (5 tests)
   - `tests/test_experiments/test_aggregation.py` (3 tests)
   - **Total project test suite passes: 296 passed (271 baseline + 25 new experiment tests, 0 failures)**.

---

## Planned Milestones & Next Phases

| Phase | Milestone Description | Target Deliverables |
| :---: | :--- | :--- |
| **Phase 2** | Corpus Engineering & Tokenization | Canonical children's story corpus, vocabulary banks, word-level tokenizer with punctuation isolation, tokenization tests |
| **Phase 3** | N-Gram Modeling & Five Scoring Heads | Bigram & trigram-backoff frequency engine, independent implementation of H1 to H5, unit tests for each head |
| **Phase 4** | Fusion, Repetition & Decoding Engines | Linear weighted score fusion, non-linear repetition suppression, temperature sampling, Hamilton d10 apportionment |
| **Phase 5** | Experimental Suite & Benchmark Replication | Automated Monte Carlo generation harness (1,600 stories across 8 configurations), metrics computation, chart generation |
| **Phase 6** | Interactive Streamlit Web Application | Glass-box UI with real-time token candidate inspection, creativity slider, head weight adjustment, step-by-step generator |
| **Phase 7** | Scratch 3.0 Project Assets & Classroom Guide | Scratch project JSON / sb3 export, printable cards and scoring sheets for the unplugged physical game |
