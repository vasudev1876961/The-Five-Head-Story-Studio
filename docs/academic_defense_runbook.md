# Academic Defense Runbook — The Five-Head Story-Generation System

**Release Version:** `v1.0.0` (Tag `v1.0.0` $\rightarrow$ `5f88c17cf230afa7e4c7e6657b96bc31fb1c87ed`, with Phase 9 CI, Reproducibility & Handover Dossier)  
**Target Domain:** Educational Explainable NLP & Glass-Box Autoregressive Story Generation

---

## 1. Problem Statement and Project Motivation

Contemporary neural language models achieve fluent generation through billions of opaque parameters, making it impossible for primary school learners (Grades 4–5) or introductory students to inspect *why* a word was chosen, and hindering controlled ablation of linguistic constraints vs. narrative heuristics. This system decomposes next-token selection into **five transparent, separately measurable scoring heads** over a closed, verifiable story corpus—eliminating neural black boxes and external APIs completely.

---

## 2. Research Objectives and System Design

1. **Interpretability ("Glass-Box" Guarantee):** Expose raw N-gram counts, smoothing distributions, per-head scores, fused rankings, and plain-English token explanations at every generation step.
2. **Educational Pedagogical Transfer:** Bridge abstract probability distributions cleanly into physical `d10` classroom dice gameplay (via Hamilton largest-remainder apportionment) and visual Scratch 3.0 block programming.
3. **Controlled Ablation & Empirical Rigor:** Isolate the contributions of grammar (`H1`), entity focus (`H2`), setting cohesion (`H3`), verb momentum (`H4`), and repetition suppression (`H5`) across a systematic 8-condition (`E1–E8`) Monte Carlo benchmark (`1,600` stories) with independent reproducibility verification.

> **Scope Disclaimer:** Proxy metrics (TTR, repetition rate, continuity ratios) measure structural properties of generated token streams; they do **not** claim human-level narrative understanding, semantic comprehension, or subjective literary quality.

---

## 3. Corpus Description and Dataset Boundaries

Ground-truth corpus: [story_corpus.txt](../data/raw/story_corpus.txt) (measured via [corpus_statistics.json](../experiments/results/corpus_analysis/corpus_statistics.json) and [corpus_statistics.md](corpus_statistics.md)):
- **Sentences:** `34` sentences, each explicitly terminating with `.` (length `10–14` tokens, mean `12.21`).
- **Tokens:** `415` total tokens (`381` lexical words + `34` period tokens `.`).
- **Vocabulary:** $|\mathcal{V}| = 120$ unique tokens (`119` lexical words + `.`), with `57` hapax legomena (`47.90%` of lexical types).
- **Readability Target:** Grade 4 readability ($\text{FKGL} \approx 4.0$).
- **Entities:** `5` named protagonists (`milo`, `pip`, `zara`, `owl`, `fox`), `6` character nouns, `4` core settings (`castle`, `cave`, `forest`, `garden`), `11` setting nouns, `15` dynamic verbs, `9` state verbs.
- **Boundary:** Closed micro-world domain; out-of-vocabulary generation is unsupported by design.

---

## 4. N-Gram Language-Model Methodology

Implemented in [mle.py](../src/five_head_nlp/probability/mle.py) and `src/five_head_nlp/models/`:
- **Intra-Sentence Boundary Isolation:** N-grams never cross sentence-ending `.`, ensuring $C(., w) = 0$ for all $w \in \mathcal{V}$ and preventing punctuation splice artifacts (`259` unique bigrams across `381` transitions; `303` unique trigrams across `347` transitions).
- **Maximum Likelihood Estimation (MLE):**
  $$P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$$
- **Zero-Frequency Defect:** On the held-out `5`-sentence test partition (`58` prediction events), unsmoothed trigram MLE suffers `45` zero-probability events (`77.59%`), yielding infinite cross-entropy and perplexity ($\text{PPL} = \infty$).

---

## 5. Smoothing Methods and Perplexity Evaluation

Evaluated on a deterministic **3-way leak-free sentence partition** (`24` train / `5` validation / `5` test sentences = `295 / 62 / 58` tokens, seed `42` in [smoothing_comparison.csv](../experiments/results/smoothing_analysis/smoothing_comparison.csv)):

| Smoothing Method | Key Parameters | Zero Events ($M=58$) | Cross-Entropy (bits) | Held-Out Perplexity |
|---|---|---:|---:|---:|
| Unsmoothed Trigram MLE | — | `45` (`77.59%`) | $\infty$ | $\infty$ |
| Laplace (Add-One) | $\alpha = 1.0$ | `0` | `6.6985` | `103.8586` |
| Add-$k$ | $k = 0.5$ | `0` | `6.6176` | `98.1991` |
| Add-$k$ | $k = 0.1$ | `0` | `6.4397` | `86.8035` |
| Simple Good-Turing | $k_{\text{threshold}} = 5$ (empirical fallback) | `0` | `6.3389` | `80.9451` |
| Linear Interpolation (Fixed) | $(0.15, 0.35, 0.50)$ | `0` | `5.5508` | `46.8752` |
| Linear Interpolation (Val-Tuned) | $(0.65, 0.20, 0.15)$ | `0` | `5.4660` | `44.2019` |
| **Absolute Discounting Backoff** | $d = 0.5$ | **`0`** | **`5.3935`** | **`42.0339`** |

**Defense Takeaways:**
1. Laplace (`+1`) over-allocates mass (`+120` pseudo-counts) in sparse contexts, giving `PPL = 103.86`.
2. **Absolute Discounting Backoff ($d = 0.5$)** outperforms all alternatives (`PPL = 42.03`), conserving exact probability mass $\sum_{w \in \mathcal{V}} P(w \mid h) = 1.0$ via $\beta(h)$ and $\alpha(h)$ normalization.

---

## 6. Next-Token Prediction and Decoding Strategies

Implemented in [predictor.py](../src/five_head_nlp/prediction/predictor.py), [candidate_ranker.py](../src/five_head_nlp/prediction/candidate_ranker.py), [greedy.py](../src/five_head_nlp/prediction/greedy.py), [top_k.py](../src/five_head_nlp/prediction/top_k.py), and [temperature.py](../src/five_head_nlp/prediction/temperature.py):
- **Deterministic Candidate Ranking:** Sorts descending by score/probability and breaks exact ties alphabetically: `key = (-score, token)`.
- **Greedy (`GreedyDecoder`):** Selects rank 1 (`candidates[0].token`), consuming zero RNG draws.
- **Top-$k$ (`TopKDecoder`):** Truncates to top $k$ (`k=5` in `E6`), renormalizes weights over the top-$k$ subset, and samples using an isolated `random.Random(seed)`.
- **Temperature (`TemperatureDecoder`):** Computes numerically stable log-sum-exp scaled distribution $P_T(w_i) \propto \exp(\ln(P_i)/T - z_{\max})$ (`T=0.8` in `E8`), preserving zero support for zero-score candidates and sampling via isolated `random.Random(seed)`.

---

## 7. Five-Head Architecture (`src/five_head_nlp/heads/`)

Each candidate $w$ is evaluated across five modular heads returning normalized scores $H_k(w) \in [0.0, 1.0]$, updating internal discourse state only *after* token commitment (`update_state()`):

1. **Grammar Scout (`GrammarScout`):** $H_1(w) = P(w \mid h)$ from the language model/smoother.
2. **Character Keeper (`CharacterKeeper`):** Rewards active protagonist continuity (`0.85`), previously seen characters (`0.675`), novel character introductions (`0.50`), non-character tokens (`0.20`), and penalizes immediate duplicate character repeats (`0.10`).
3. **Setting Painter (`SettingPainter`):** Rewards initial/current setting (`0.85`), setting nouns (`0.625`), core setting transitions (`0.40`), and non-setting tokens (`0.20`).
4. **Action Booster (`ActionBooster`):** Rewards dynamic plot-advancing verbs (`0.90`), perceptual/state verbs (`0.50`), and non-action tokens (`0.20`).
5. **Repeat Blocker (`RepeatBlocker`):** Assigns `1.0` to exempt function words/punctuation (`the`, `a`, `.`, `,`, prepositions, auxiliaries); `0.05` to immediate self-loops ($w_t = w_{t-1}$); `0.20` to recent bigram cycles over `history_window=8`; and $\exp(-0.80 \cdot c_{\text{window}}(w))$ otherwise.

---

## 8. Weighted Score Fusion and Hamilton Apportionment

- **Weighted Linear Fusion (`ScoreFusion`):**
  $$S_{\text{fused}}(w) = \sum_{k=1}^5 \alpha_k H_k(w), \quad \alpha_k \ge 0,\; \sum_{k=1}^5 \alpha_k = 1.0$$
  Default balanced weight vector: $\boldsymbol{\alpha} = (0.40, 0.15, 0.15, 0.10, 0.20)$.
  **Critical Distinction:** **Fused scores $S_{\text{fused}}(w)$ are weighted multi-criteria scores, NOT probabilities.** Only when passed to stochastic decoders (`TopKDecoder` / `TemperatureDecoder`) does `NextTokenPredictor` normalize by $\sum_v S_{\text{fused}}(v)$ to obtain positive sampling weights.
- **Hamilton Largest-Remainder Allocation (`HamiltonApportionment`):**
  Converts weights to $K=10$ integer die slots via quota $q_i = 10 w_i$, floor $a_i = \lfloor q_i \rfloor$, remainder $r_i = q_i - a_i$, and deterministic tie priority `Grammar Scout > Character Keeper > Setting Painter > Action Booster > Repeat Blocker`. Default balanced weights yield `[4, 2, 1, 1, 2]`.
  **Critical Distinction:** **Hamilton `d10` allocation is strictly an explanatory classroom representation, NOT the fusion-weight vector** used by Python candidate scoring.

---

## 9. Experimental Design and Phase 5 Configurations (`E1–E8`)

All `1,600` canonical stories (`200` per config, `0` failures, base seed `42`, $\sigma(c, i) = 42 + 10{,}000 c + i$, target `80–120` tokens, min `3` sentences) use Trigram models ([experiment_registry.py](../src/five_head_nlp/experiments/experiment_registry.py), [configurations.json](../experiments/results/monte_carlo/configurations.json)):

- **E1 (`Baseline MLE`):** `none`, `greedy`
- **E2 (`Laplace Smoothing`):** `laplace` ($\alpha=1.0$), `greedy`
- **E3 (`Trigram Backoff`):** `backoff` ($d=0.5$), `greedy`
- **E4 (`Grammar Only`):** `backoff`, `greedy`, weights `(1.0, 0.0, 0.0, 0.0, 0.0)`
- **E5 (`Thematic Heads`):** `backoff`, `greedy`, weights `(0.50, 0.20, 0.20, 0.10, 0.00)` (no `Repeat Blocker`)
- **E6 (`Full Five-Head (Top-K)`):** `backoff`, **`top_k` (`k=5`)**, weights `(0.40, 0.15, 0.15, 0.10, 0.20)`
- **E7 (`Full Five-Head (Greedy)`):** `backoff`, `greedy`, weights `(0.40, 0.15, 0.15, 0.10, 0.20)`
- **E8 (`Full Five-Head (Temperature)`):** `backoff`, `temperature` ($T=0.8$), weights `(0.40, 0.15, 0.15, 0.10, 0.20)`

---

## 10. Empirical Phase 5 Metrics, Results, and Interpretation

Exact observed values from [configuration_comparison.csv](../experiments/results/monte_carlo/configuration_comparison.csv) ($N = 200$ per condition):

| ID | Tokens (Mean±SD) | Sents | Bigram Rep % | Trigram Rep % | Win Rep % | TTR | Char Cont | Set Cons | Dyn Act % |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E1** | `85.14 ± 3.66` | `10.00` | `31.82%` | `24.71%` | `5.10%` | `0.5275` | `0.1107` | `0.3840` | `51.81%` |
| **E2** | `84.72 ± 4.01` | `9.62` | `29.94%` | `23.33%` | `4.83%` | `0.5391` | `0.0833` | `0.3781` | `54.11%` |
| **E3** | `84.94 ± 3.82` | `9.93` | `31.48%` | `24.44%` | `4.86%` | `0.5282` | `0.1000` | `0.3895` | `52.45%` |
| **E4** | `84.34 ± 3.73` | `9.82` | `31.94%` | `25.03%` | `4.83%` | `0.5255` | `0.0911` | `0.3947` | `52.36%` |
| **E5** | `93.33 ± 12.34` | `4.09` | `66.04%` | `62.42%` | `60.18%` | `0.2211` | `0.5111` | `0.7296` | `87.39%` |
| **E6** | `94.63 ± 11.10` | `3.96` | `19.17%` | `3.06%` | `3.33%` | `0.2888` | `0.3321` | `0.7327` | `98.31%` |
| **E7** | `88.95 ± 7.94` | `6.24` | `42.58%` | `35.87%` | `4.67%` | `0.3741` | `0.2549` | `0.5397` | `85.66%` |
| **E8** | `104.27 ± 11.24` | `3.15` | `0.61%` | `0.06%` | `10.43%` | `0.6780` | `0.0710` | `0.2095` | `65.96%` |

*(Note: Mean token latency/throughput in `configuration_comparison.csv` spans `0.0082 ms` / `123,390 tps` on `E1` to `3.3770 ms` / `303.5 tps` on `E8`, reflecting `time.perf_counter()` CPU timing.)*

**Measured Result vs. Design Interpretation:**
- **Ablation Proof (`E5` vs. `E6/E7/E8`):** Enabling thematic heads without `Repeat Blocker` (`E5`) causes greedy attractor lock-in (`66.04%` bigram rep, `62.42%` trigram rep, `60.18%` window rep, `0.2211` TTR). Adding `Repeat Blocker` (`E7`) cuts window repetition from `60.18%` to `4.67%`.
- **Decoding Trade-off (`E6` vs. `E8`):** `E6` (`top_k=5`) yields strong thematic continuity (`0.7327` setting consistency, `0.3321` character continuity, `98.31%` dynamic actions) with low trigram repetition (`3.06%`), while `E8` (`temperature=0.8`) maximizes lexical diversity (`0.6780` TTR, `0.61%` bigram rep) at the cost of lower entity consistency (`0.0710` character, `0.2095` setting).

---

## 11. Phase 7 Scratch Educational Implementation

Under `phase_7/`:
- **Scratch Project ([five_head_story_machine.sb3](../phase_7/scratch/five_head_story_machine.sb3)):** Validated by `validate_sb3.py` (`12` targets: 1 stage + 11 sprites, `4` monitors, `23` variables, `4` lists, `12` SVG assets verified by MD5 checksum).
- **Classroom Materials (`phase_7/classroom/`):** Unplugged `d10` simulation guide, teacher lesson plan, student worksheet, and worked arithmetic answer key.
- **Verification Scope Caveat:** **`validate_sb3.py` certifies structural ZIP, JSON AST, SVG MD5, and weight-sum validity only. Structural validation is NOT equivalent to manual interactive execution in the Scratch 3.0 online or desktop editor.**

---

## 12. Phase 8 Packaging and Release Engineering

- Standard `pyproject.toml` packaging (`five-head-nlp`, `v1.0.0`, MIT License, `CITATION.cff`), isolated `src/` layout, 4 cross-platform console scripts (`five-head-studio`, `five-head-monte-carlo`, `five-head-scoring`, `five-head-prediction`).
- Automated Streamlit process lifecycle verification (`tests/test_ui/test_studio_lifecycle.py`, `8` scenarios covering startup `/_stcore/health`, 3.0s process-tree tracking, graceful signals, port binding checks, and fallback kills).
- **Process Containment Limits:** No Windows Job Objects (pre-discovery breakaway descendants could evade `psutil` tracking); on POSIX, `setsid`/`setpgid` escapes `killpg` unless discovered prior to detachment.
- Local annotated tag `v1.0.0` targets commit `5f88c17cf230afa7e4c7e6657b96bc31fb1c87ed`; **no remote is configured and no push has been performed**.

---

## 13. Phase 9 Reproducibility, Test, Coverage, and CI Findings

- **Unified Test Suite:** **`351/351` tests passed** locally on Windows `11` / Python `3.12.8`.
- **WP9.4 Reproducibility (`run_b42_s1042_n200`, isolated under `reports/reproducibility/`):**
  - Canonical Phase 5 artifacts (`1,600` stories, models, corpus, `.sb3`) preserved **unchanged** (`11/11` SHA256 hashes identical before/after).
  - **Deterministic `E1–E5, E7` (`base_seed=42`):** **`1,200/1,200` exact matches** across tokens, text, seeds, and non-timing metrics (including `readability`).
  - **Stochastic `E6, E8` (`stochastic_seed=1042`, independent $N=200$ draws):** All `12` comparisons (`6` headline metrics $\times$ `2` configs) satisfied Holm-Bonferroni step-down Welch TOST equivalence ($p_{\text{TOST, Holm}} \le 0.03205 < 0.05$, $|g| \le 0.151$).
  - **Exact Methodological Caveats:** (a) Equivalence margins were implementation-fixed and calibrated using canonical Phase 5 dispersion prior to executing the reproduction run (**not pre-registered**); (b) Holm step-down rejection ($p_{\text{TOST, Holm}} < 0.05$) differs from stricter fixed `99.17%` simultaneous Bonferroni intervals (ranks `10–12` satisfy Holm yet cross $\pm\Delta$ under `99.17%` Bonferroni) and from unadjusted `90%` nominal CIs; (c) `readability` and secondary sub-metrics were not included in stochastic TOST; (d) `latency` was excluded due to hardware variability.
- **Coverage (`88` discovered modules):** `76.10%` statement (gate `72%`), `66.88%` branch (gate `63%`), `74.20%` combined (gate `70%`). Four unimported files under `experiments/` (`cli_monte_carlo.py`, `cli_prediction.py`, `cli_scoring.py`, `reporting.py`) remain **excluded** due to missing `experiments/__init__.py`.
- **CI Workflow (`.github/workflows/ci.yml`):** 9-job matrix across Ubuntu, Windows, macOS $\times$ Python `3.10`, `3.11`, `3.12`; **remote GitHub-hosted CI execution remains pending** (no push authorized).

---

## 14. Limitations, Threats to Validity, and Future Work

- **Internal Validity:** Closed `120`-token lexicon (`34` sentences); statistical equivalence margins calibrated from canonical $N=200$ dispersion; `E6` continuity variance (`SD=0.312`) limits tight-margin TOST power without larger sample sizes.
- **Construct Validity:** Proxy metrics (`TTR`, `CCR`, `SCR`, `DAR`, readability proxies) capture surface distributional traits and do **not** validate plot coherence or human reader enjoyment.
- **Software/CI Threats:** `4` unimported CLI/reporting files excluded from coverage scan; Ubuntu/macOS and Python `3.10`/`3.11` await remote CI execution; manual Scratch editor UI execution remains outside automated checks.
- **Future Extensions:** Adding `experiments/__init__.py` + CLI smoke tests, increasing stochastic `N` for tighter `CCR` bounds, human-evaluation studies, Modified Kneser-Ney smoothing comparisons, and remote CI activation upon authorization.

---

## 15. Likely Viva Questions with Technical Answers

1. **Q: Why did unsmoothed MLE give infinite perplexity, and why did Backoff ($d=0.5$) beat Laplace?**  
   *A:* Unsmoothed MLE assigns `0.0` to `45/58` (`77.59%`) unseen test events, yielding $\ln(0)=-\infty$. Laplace adds `120` total pseudo-counts per context, severely distorting small-count trigram transitions (`PPL = 103.86`), whereas Absolute Discounting Backoff (`PPL = 42.03`) reserves discounted mass $\beta(h)$ strictly proportional to observed context diversity and cascades smoothly to lower-order distributions.
2. **Q: Are the five-head fused scores valid probabilities?**  
   *A:* No. `ScoreFusion` returns weighted heuristic scores $S_{\text{fused}}(w) = \sum \alpha_k H_k(w) \in [0, 1]$ where components like `ActionBooster` and `SettingPainter` do not sum to 1 across candidates. Only during stochastic decoding does `NextTokenPredictor` divide by $\sum_v S_{\text{fused}}(v)$ to construct sampling weights.
3. **Q: Does Hamilton `d10` allocation alter Python generation weights?**  
   *A:* No. `HamiltonApportionment` translates normalized weights into 10 discrete classroom dice slots (`[4, 2, 1, 1, 2]`) solely for visualization and unplugged pedagogy; Python scoring uses continuous weights `(0.40, 0.15, 0.15, 0.10, 0.20)`.
4. **Q: Why can't a Kolmogorov-Smirnov test with $p > 0.05$ establish stochastic reproducibility?**  
   *A:* Failing to reject the null hypothesis of equality merely reflects lack of evidence against equality (vulnerable to low power). Welch TOST reverses the null hypothesis ($H_0: |\Delta\mu| \ge \Delta$) so rejection provides affirmative evidence that true distributional differences stay bounded inside predefined margins $\pm\Delta$, controlled across all `12` tests via Holm-Bonferroni.
5. **Q: Why do ranks `10–12` pass Holm-adjusted TOST while crossing their `99.17%` simultaneous Bonferroni CI?**  
   *A:* Step-down Holm evaluates rank $k$ against $\alpha / (12 - k + 1)$ (`0.0167`, `0.025`, `0.05`), keeping step-down CIs inside $\pm\Delta$ (`step_CI_in = True`), whereas `99.17%` Bonferroni applies the fixed worst-case $\alpha / 12$ bound to every metric regardless of rank.
6. **Q: What did WP9.4 confirm regarding E6 (`k=5` vs. `k=3`) and cross-platform CI?**  
   *A:* Canonical `experiment_registry.py`, `configurations.json`, and `experiment_summary.md` unambiguously prove E6 uses `top_k(k=5)`. On Python `3.12.8` (Windows), all `351` tests passed; Python `3.10`/`3.11` and Ubuntu/macOS CI jobs are configured in `.github/workflows/ci.yml` but remain pending remote execution.
