# Corpus Reconstruction Specification & Validation (Audited Phase 2.1.1)

> **Corpus Name**: Five-Head Children's Story Corpus  
> **Classification**: Reconstructed  
> **Grade Level**: Grades 4–5 (Ages 9–11)  
> **Corpus File**: `data/raw/story_corpus.txt`  
> **Metadata**: `data/raw/corpus_metadata.json`

---

## 1. Rationale for Reconstruction

The original project documentation for the Five-Head Story-Generation System defines the mathematical architecture, classroom integer mode, decoding configurations, and statistical properties of the story corpus. However, the verbatim 34 sentences were not published in full in the available project brief.

To provide an empirical foundation for the Python reference engine, Scratch glass box, and experimental benchmarks, a **faithful reconstructed corpus** was engineered from the documented statistical signatures, entity registries, setting word banks, and transition frequencies.

This corpus is explicitly classified as **`reconstructed`**, preserving academic integrity and preventing any misrepresentation of historical source data.

---

## 2. Calculation Conventions & Statistical Definitions

To ensure transparency and consistency across the project, the following conventions are adopted:

1. **Punctuation as Distinct Tokens**:
   - The corpus text is tokenized by lowercase word splitting, treating sentence-ending periods (`.`) as individual tokens.
   - Punctuation is treated as a first-class token because next-token prediction in autoregressive language models must explicitly learn when to terminate or transition between sentences.
2. **Vocabulary Size Composition ($|\mathcal{V}| = 120$)**:
   - The reported unique vocabulary size of $120$ consists of **119 lexical words + 1 period token (`.`)**:
     $$|\mathcal{V}| = 119 + 1 = 120$$
3. **Sentence Boundaries & Continuous Autoregressive Stream**:
   - In a multi-sentence story generator, text generation does not halt at a period; it transitions from the period to sentence-initiating tokens (e.g., `.` $\to$ `the`, `.` $\to$ `milo`).
   - Therefore, the **canonical project model** treats the corpus as a continuous flat stream of $415$ tokens, producing $N - 1 = 414$ transitions:
     $$(\text{token}_1, \text{token}_2), (\text{token}_2, \text{token}_3), \dots, (\text{token}_{414}, \text{token}_{415})$$
   - This includes the $33$ inter-sentence boundary transitions $(\text{sentence}_k\text{[-1]}, \text{sentence}_{k+1}\text{[0]}) = (., w_{\text{start}})$.
   - For comparative purposes, an alternative **within-sentence disjoint model** is also tracked, which isolates each sentence and yields $381$ intra-sentence bigram transitions.
4. **Sentence-Final Occurrences**:
   - Words appearing immediately before a sentence-ending period (e.g., `"castle ."` or `"cave ."`) are fully counted in unigram occurrence totals.
   - In bigram transition tables, these words have an outgoing transition to the period token (`word` $\to$ `.`).
5. **Average Bigram Branching Factor Formula**:
   - The average branching factor measures the average out-degree of active transition nodes in the bigram transition graph:
     $$\text{Branching Factor} = \frac{\sum_{u \in \mathcal{V}_{\text{out}}} \text{OutDegree}(u)}{|\mathcal{V}_{\text{out}}|}$$
     where $\mathcal{V}_{\text{out}} = \{u \in \mathcal{V} \mid \text{Count}_{\text{out}}(u) > 0\}$ is the set of tokens that have at least one observed outgoing transition.

---

## 3. Narrative Design & Entity Coverage

The reconstructed corpus comprises exactly 34 interconnected, narrative sentences written in simple, evocative English suitable for Grades 4–5.

### 3.1 The Five Recurring Characters
1. **Milo** — The clever cat (7 unigram occurrences, active across forest, castle, and cave).
2. **Pip** — The tiny dragon (7 unigram occurrences, flying between castle, cave, and forest).
3. **Zara** — The friendly wizard (7 unigram occurrences, casting magic and reading ancient books).
4. **Sleepy Owl** — The observant owl resting in the old tree (5 unigram occurrences).
5. **Playful Fox** — The nimble explorer running across the settings (4 unigram occurrences).

### 3.2 The Four Interconnected Settings
1. **Bright Forest** — Scene of nature, old trees, green moss, and singing birds (5 unigram occurrences).
2. **Crystal Castle** — The central architectural landmark with towers, doors, and stone floors (9 unigram occurrences).
3. **Dark Cave** — Mystery location with stone walls, hidden chests, and padlocks (7 unigram occurrences).
4. **Secret Garden** — Quiet sanctuary of flowers, fountains, and ancient paths (6 unigram occurrences).

---

## 4. Statistical Audit: Reference Targets vs. Reconstructed Actuals

Below is the audited comparison separating original reference targets from actual reconstructed values:

| Metric | Original Reference Target | Actual Reconstructed Value | Difference | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Sentence Count** | 34 | 34 | 0 | **Exact Match** |
| **Word Count** | 381 | 381 | 0 | **Exact Match** |
| **Total Tokens (incl. `.`)** | 415 | 415 | 0 | **Exact Match** |
| **Unique Vocabulary** | 120 | 120 | 0 | **Exact Match** |
| **Avg Words / Sentence** | 11.21 | 11.21 | 0.00 | **Exact Match** |
| **Flesch–Kincaid Grade Level** | 4.02 | 3.99 | -0.03 | **Target Aligned (Grade 4)** |
| **Branching Factor (Flat Stream)** | 1.95 | 2.26 | +0.31 | **Audited Value** |
| **Branching Factor (Within-Sentence)** | — | 2.18 | — | **Audited Alternative** |
| **Unique Bigrams (Flat Stream)** | — | 271 | — | **Audited Total** |
| **Unique Bigrams (Within-Sentence)**| — | 259 | — | **Audited Total** |
| **Total Transitions (Flat Stream)** | 414 | 414 | 0 | **Exact Match** |
| **Total Transitions (Within-Sentence)**| 381 | 381 | 0 | **Exact Match** |

### Readability Formulation (FKGL)
Flesch–Kincaid Grade Level is calculated using the standard Kincaid et al. formula:

$$\text{FKGL} = 0.39 \cdot \left(\frac{\text{Total Words}}{\text{Total Sentences}}\right) + 11.8 \cdot \left(\frac{\text{Total Syllables}}{\text{Total Words}}\right) - 15.59$$

- Average Sentence Length ($\text{ASL}$): $\frac{381}{34} \approx 11.2059$
- Average Syllables per Word ($\text{ASW}$): $\frac{491}{381} \approx 1.2887$
- Calculated $\text{FKGL}$: $0.39(11.2059) + 11.8(1.2887) - 15.59 = 4.3703 + 15.2069 - 15.59 = 3.987 \approx 3.99$.

---

## 5. Audited Pivot Token Transition Analysis

The Phase 2.1.1 audit recalculates token occurrences, outgoing transitions, and unique continuation counts across the corpus:

| Token | Reference Occurrence | Reconstructed Unigrams | Reconstructed Out-Transitions | Reference Unique Next | Reconstructed Unique Next | Notes & Discrepancies |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `the` | 63 | 63 | 63 | 21 | 24 | Occurs 63 times; branches into 24 distinct continuations. |
| `.` | 33 | 34 | 33 | 12 | 12 | 34 unigram periods total; 33 have outgoing transitions to sentence starters. |
| `and` | 12 | 12 | 12 | 6 | 6 | Exact match on occurrences and unique continuations. |
| `a` | 11 | 11 | 11 | 10 | 11 | 11 occurrences; each occurrence followed by a distinct adjective/noun. |
| `to` | 11 | 11 | 11 | 6 | 5 | 11 occurrences; branches into `the` (5), `help` (2), `find` (2), `play` (1), `see` (1). |
| `milo` | 7 | 7 | 7 | 7 | 7 | Exact match on occurrences and unique continuations. |
| `pip` | 7 | 7 | 7 | 6 | 6 | Exact match on occurrences and unique continuations. |
| `zara` | 7 | 7 | 7 | 7 | 6 | Occurs 7 times; followed by `.` twice, producing 6 unique continuations. |
| `near` | 8 | 7 | 7 | — | 1 | Appears 7 times, always followed by `the` (`near the`: 7). |
| `castle` | 6 | 9 | 9 | — | 6 | Appears 9 times (4 sentence-final before `.`, 5 mid-sentence). |
| `cave` | 5 | 7 | 7 | — | 5 | Appears 7 times (2 sentence-final before `.`, 5 mid-sentence). |

---

## 6. Known Bigram Transition Comparison

The reconstructed corpus matches **100% of all 18 target transitions exactly**:

| Previous Token ($w_t$) | Next Token ($w_{t+1}$) | Target Frequency | Reconstructed Frequency | Match Status |
| :--- | :--- | :---: | :---: | :---: |
| `the` | `silver` | 8 | 8 | **Exact Match** |
| `the` | `crystal` | 6 | 6 | **Exact Match** |
| `the` | `old` | 5 | 5 | **Exact Match** |
| `the` | `dark` | 5 | 5 | **Exact Match** |
| `the` | `bright` | 4 | 4 | **Exact Match** |
| `the` | `brave` | 4 | 4 | **Exact Match** |
| `.` | `the` | 11 | 11 | **Exact Match** |
| `.` | `milo` | 4 | 4 | **Exact Match** |
| `.` | `pip` | 4 | 4 | **Exact Match** |
| `.` | `zara` | 4 | 4 | **Exact Match** |
| `and` | `the` | 6 | 6 | **Exact Match** |
| `and` | `flew` | 2 | 2 | **Exact Match** |
| `to` | `the` | 5 | 5 | **Exact Match** |
| `near` | `the` | 7 | 7 | **Exact Match** |
| `milo` | `lived` | 1 | 1 | **Exact Match** |
| `milo` | `liked` | 1 | 1 | **Exact Match** |
| `milo` | `found` | 1 | 1 | **Exact Match** |
| `milo` | `showed` | 1 | 1 | **Exact Match** |

---

## 7. Discrepancies, Limitations & Engineering Decisions

1. **Bigram Branching Factor**:
   - The reported target was $1.95$, while the audited value is $2.26$ for the flat stream ($2.18$ within sentences).
   - In natural English text with 120 vocabulary tokens, forcing branching down to $1.95$ requires artificial cyclic repetition. Preserving natural narrative readability and grammatical syntax naturally yielded $2.26$ branching, providing healthy diversity for the generation engine while maintaining all 18 key transition counts.
2. **Pivot Token Out-Degrees (`the: 24` vs. `21`, `a: 11` vs. `10`, `to: 5` vs. `6`, `zara: 6` vs. `7`)**:
   - The reconstructed corpus hits the exact total unigram occurrences for all pivot tokens (`the: 63`, `.: 34/33`, `and: 12`, `a: 11`, `to: 11`, `milo: 7`, `pip: 7`, `zara: 7`).
   - Slight differences in out-degrees ($\pm 1$ to $\pm 3$) reflect natural linguistic variance while satisfying the 18 hardcoded bigram frequencies.
3. **Tokens `near` (7 vs. 8), `castle` (9 vs. 6), and `cave` (7 vs. 5)**:
   - `near` appears 7 times, strictly followed by `the`, satisfying the target bigram count `("near", "the"): 7`.
   - `castle` and `cave` appear as prominent setting locations, frequently terminating sentences (e.g., `castle .` occurs 4 times; `cave .` occurs 2 times).
