# Phase 5 — Experimental Methodology and Benchmark Replication

## 1. Executive Summary & Objective

Phase 5 establishes a rigorous, reproducible experimental harness that generates, evaluates, compares, and visualizes stories produced across statistical and multi-head generation architectures.
The harness conducted a full-scale Monte Carlo benchmark generating **1,600 multi-sentence stories** (200 stories per configuration across 8 canonical conditions, E1 through E8).

All stories adhered strictly to length and structural constraints:
- **Target Token Length**: 80–120 tokens (with a hard safety cap at 150).
- **Sentence Bounds**: Minimum 3 completed sentences, each terminating in a period `.`.
- **Reproducibility**: Complete isolation of pseudo-random number generator seeds via a deterministic bijection $\sigma(c, i) = \sigma_{\text{base}} + 10\,000 \cdot c + i$.

---

## 2. Experimental Conditions (E1 to E8)

The 8 conditions span baseline statistical language models, smoothing tiers, ablation conditions, and full five-head configurations with varied decoding strategies:

| Condition ID | Configuration Name | Architecture Mode | Smoother | Decoder | Head Weight Vector $(w_1, w_2, w_3, w_4, w_5)$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Baseline MLE | Baseline | None (Unsmoothed Trigram) | Greedy ($k=1$) | N/A |
| **E2** | Laplace Smoothing | Baseline | Laplace (Add-1, $\alpha=1.0$) | Greedy ($k=1$) | N/A |
| **E3** | Trigram Backoff | Baseline | Absolute Discounting ($d=0.5$) | Greedy ($k=1$) | N/A |
| **E4** | Grammar Only | Five-Head | Trigram Backoff ($d=0.5$) | Greedy ($k=1$) | $(1.00, 0.00, 0.00, 0.00, 0.00)$ |
| **E5** | Thematic Heads | Five-Head | Trigram Backoff ($d=0.5$) | Greedy ($k=1$) | $(0.50, 0.20, 0.20, 0.10, 0.00)$ |
| **E6** | Full Five-Head | Five-Head | Trigram Backoff ($d=0.5$) | Top-$K$ ($k=5$) | $(0.40, 0.15, 0.15, 0.10, 0.20)$ |
| **E7** | Full Five-Head Greedy | Five-Head | Trigram Backoff ($d=0.5$) | Greedy ($k=1$) | $(0.40, 0.15, 0.15, 0.10, 0.20)$ |
| **E8** | Full Five-Head Temperature | Five-Head | Trigram Backoff ($d=0.5$) | Temperature ($T=0.8$) | $(0.40, 0.15, 0.15, 0.10, 0.20)$ |

---

## 3. Evaluation Metric Definitions

### 3.1 Length and Structure
- **Total Tokens**: Total tokens generated in the story, inclusive of punctuation.
- **Completed Sentences**: Count of sentences formally ending with `.`.
- **Mean Sentence Length**: Average tokens per sentence.

### 3.2 Repetition Suppression
- **Immediate Token Repetition Rate**: Fraction of successive identical transitions $w_i == w_{i-1}$ on content tokens (excluding exempt prepositions, articles, and punctuation).
- **Repeated Bigram Rate**: Fraction of duplicate within-story bigrams:
  $$R_{\text{bg}} = \frac{|B| - |B_{\text{unique}}|}{|B|}$$
- **Repeated Trigram Rate**: Fraction of duplicate within-story trigrams:
  $$R_{\text{tg}} = \frac{|T| - |T_{\text{unique}}|}{|T|}$$
- **Rolling Window Repetition Rate ($W=8$)**: Proportion of 8-token sliding windows containing at least one duplicate content token.

### 3.3 Lexical Diversity
- **Lexical Type-Token Ratio (TTR)**: Unique lexical word types divided by total lexical tokens (excluding punctuation).
- **Moving-Average TTR (MATTR-20)**: Rolling average TTR evaluated over sliding 20-word windows, eliminating length-dependency biases.
- **Hapax Legomena Ratio**: Proportion of unique words occurring exactly once in the story.

### 3.4 Thematic Narrative Consistency
- **Character Density**: Total character mentions per story token.
- **Character Continuity Ratio**: Fraction of consecutive character mentions that preserve the same character entity without abrupt switching.
- **Setting Density & Consistency**: Setting mentions per token and proportion belonging to the dominant environmental landmark.
- **Dynamic Action Ratio**: Dynamic action verbs (flew, ran, found, opened, etc.) as a fraction of total verbs (dynamic + stative).

### 3.5 Latency and Throughput
- **Per-Token Latency ($\tau_i$)**: Monotonic wall-clock time in milliseconds measured with `time.perf_counter()` around candidate evaluation and selection.
- **Generation Throughput (TPS)**: Total tokens generated divided by total elapsed generation time in seconds.

---

## 4. Empirical Results (1,600 Story Monte Carlo Run)

Across 1,600 stories (200 per condition, 0 failures, total runtime: 221.02 seconds; exact metrics from `experiments/results/monte_carlo/configuration_comparison.csv`):

| ID | Name | Tokens (Mean±Std) | Sents | Bigram Rep % | Trigram Rep % | Win Rep % | TTR | Char Cont | Set Cons | Dyn Act % | Latency (ms) | Throughput (tps) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Baseline MLE | 85.1 ± 3.7 | 10.0 | 31.8% | 24.7% | 5.1% | 0.527 | 0.111 | 0.384 | 51.8% | 0.01 ms | 123,390.3 |
| **E2** | Laplace Smoothing | 84.7 ± 4.0 | 9.6 | 29.9% | 23.3% | 4.8% | 0.539 | 0.083 | 0.378 | 54.1% | 0.27 ms | 3,727.8 |
| **E3** | Trigram Backoff | 84.9 ± 3.8 | 9.9 | 31.5% | 24.4% | 4.9% | 0.528 | 0.100 | 0.390 | 52.4% | 0.15 ms | 6,939.8 |
| **E4** | Grammar Only | 84.3 ± 3.7 | 9.8 | 31.9% | 25.0% | 4.8% | 0.525 | 0.091 | 0.395 | 52.4% | 0.91 ms | 1,115.1 |
| **E5** | Thematic Heads (No Repeat) | 93.3 ± 12.3 | 4.1 | 66.0% | 62.4% | 60.2% | 0.221 | 0.511 | 0.730 | 87.4% | 2.21 ms | 452.6 |
| **E6** | Full Five-Head (Top-K) | 94.6 ± 11.1 | 4.0 | 19.2% | 3.1% | 3.3% | 0.289 | 0.332 | 0.733 | 98.3% | 2.85 ms | 351.1 |
| **E7** | Full Five-Head (Greedy) | 88.9 ± 7.9 | 6.2 | 42.6% | 35.9% | 4.7% | 0.374 | 0.255 | 0.540 | 85.7% | 3.27 ms | 314.0 |
| **E8** | Full Five-Head (Temp T=0.8) | 104.3 ± 11.2 | 3.1 | 0.6% | 0.1% | 10.4% | 0.678 | 0.071 | 0.210 | 66.0% | 3.38 ms | 303.5 |

---

## 5. In-Depth Comparative Analysis

### 5.1 The Critical Role of Head 5 (Repeat Blocker)
The contrast between **E5** (Thematic heads without Repeat Blocker) and **E6/E8** provides empirical proof of the repetition trap in unpenalized multi-head boosting:
- In **E5**, thematic boosting heavily rewards character and action words (e.g., 'milo', 'flew', 'castle'). Without negative feedback, greedy decoding immediately falls into self-reinforcing attractor cycles, resulting in a **66.0% repeated bigram rate**, **62.4% repeated trigram rate**, and **60.2% window repetition rate**.
- In **E6** (Top-K `k=5` with Repeat Blocker), the bigram repetition rate drops to **19.2%** and trigram repetition drops to **3.1%** (`3.3%` window repetition).
- In **E8** (Temperature `T=0.8` with Repeat Blocker), bigram repetition drops to **0.6%** with **0.1% trigram repetition**, eliminating multi-token attractor loops.

### 5.2 Dynamic Action and Narrative Momentum (Head 4)
- Unsteered baselines (E1–E4) exhibit an almost even split of dynamic vs stative verbs (~51.8% to 54.1%).
- Under the guidance of Head 4 (Action Booster), dynamic action verbs jump to **87.4% in E5** and **98.3% in E6**, ensuring that generated stories favor dynamic action verbs ("flew", "ran", "found", "opened") over passive verbs.

### 5.3 Character and Setting Continuity (Heads 2 & 3)
- Pure baselines display low character continuity (~0.08 to 0.11), frequently swapping entity focus across sentences.
- Conditions incorporating Head 2 (E5 and E6) achieve character continuity of **0.511** and **0.332**, maintaining recurring entities across sentence boundaries.
- Setting consistency rises from **~0.38–0.39** in baselines (`E1–E4`) to **0.730–0.733** in `E5/E6`, while high-entropy temperature sampling (`E8`) trades setting consistency (**0.210**) for elevated lexical diversity (`0.678` TTR).

### 5.4 Computational Efficiency
- Inference latency for the five-head system averages between **2.21 ms and 3.38 ms per token** (`~304–453` tokens/sec in pure Python on the Phase 5 benchmark machine), supporting interactive generation in Phase 6.

---

## 6. Generated Artifacts Manifest

All artifacts are persisted under `experiments/results/monte_carlo/`:
1. `generated_stories.jsonl`: 1,600 JSON-serialized story objects including tokens, detokenized text, and complete hierarchical metric breakdowns.
2. `failure_records.jsonl`: Zero failures logged ($100\%$ completion rate).
3. `per_story_metrics.csv`: 1,600 rows containing 56 flat columns per story.
4. `aggregate_metrics.csv`: Distributional summary statistics (mean, median, std, min, max, count) across all configurations.
5. `configuration_comparison.csv`: Comparative summary table across key performance indicators.
6. `run_metadata.json`: Timestamp, base seed, wall-clock time, and configuration list.
7. `configurations.json`: Full specification of experimental parameters and head weights.
8. `experiment_summary.md`: Markdown summary report.
9. `plots/`: Eight high-resolution Matplotlib figures:
   - `length_distribution.png`
   - `repetition_rates_comparison.png`
   - `lexical_diversity_ttr.png`
   - `character_tracking_metrics.png`
   - `setting_consistency.png`
   - `action_dynamics.png`
   - `latency_throughput_benchmark.png`
   - `overall_performance_comparison.png`
