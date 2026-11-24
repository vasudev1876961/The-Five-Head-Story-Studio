# Corpus Statistical Analysis and Exploratory Data Analysis (Phase 2.4)

> **Document Version**: 1.0.0  
> **Target Corpus**: Five-Head Children's Story Corpus (`data/raw/story_corpus.txt`)  
> **Generated Artifacts**: `experiments/results/corpus_analysis/`  
> **Visualizations**: 6 publication-ready figures (`.png`)

---

## 1. Corpus Overview & Architecture

The **Five-Head Children's Story Corpus** serves as the empirical ground truth for teaching autoregressive language generation to Grades 4–5 students and testing the five-head scoring engine. 

The corpus comprises:
- **34 connected narrative sentences** designed around five recurring characters and four settings.
- **Grade 4 readability** ($\text{FKGL} = 3.99 \approx 4.02$).
- **Vocabulary size** of exactly 120 unique tokens (119 lexical words and 1 punctuation token `.`).

This report details the lexical distributions, within-sentence bigram transitions, entity salience, and syntactic length profiles derived dynamically by the `CorpusStatistics` engine.

---

## 2. Basic Corpus Dimensions & Sentence-Length Profile

| Metric | Calculated Value | Reference Target | Status / Notes |
| :--- | :---: | :---: | :--- |
| **Number of Sentences** | 34 | 34 | Exact match |
| **Total Tokens ($N$, words + periods)** | 415 | 415 | Exact match |
| **Lexical Word Tokens** | 381 | 381 | Exact match |
| **Punctuation Tokens (`.`)** | 34 | 34 | Exactly one period per sentence |
| **Unique Full Vocabulary ($|V|$)** | 120 | 120 | Includes period token `.` |
| **Unique Lexical Vocabulary ($|V_{\text{lexical}}|$)** | 119 | 119 | Excludes punctuation |
| **Average Sentence Length** | 12.21 tokens | 12.21 tokens | (11.21 words + 1.00 period) |
| **Minimum Sentence Length** | 10 tokens | — | Sentence 1 & Sentence 8 |
| **Maximum Sentence Length** | 14 tokens | — | Sentence 11 & Sentence 14 |
| **Median Sentence Length** | 12.0 tokens | — | Symmetrical length distribution |
| **Type-Token Ratio (Full: $|V| / N$)** | 0.2892 | — | $120 / 415$ |
| **Type-Token Ratio (Lexical: $|V_{\text{lex}}| / N_{\text{words}}$)** | 0.3123 | — | $119 / 381$ |

### Sentence-Length Distribution
Sentence length ranges tightly between 10 and 14 tokens:
- **10 tokens**: 2 sentences (5.9%)
- **11 tokens**: 6 sentences (17.6%)
- **12 tokens**: 21 sentences (61.8%) — Modal length
- **13 tokens**: 3 sentences (8.8%)
- **14 tokens**: 2 sentences (5.9%)

The tight clustering around 12 tokens guarantees cognitive accessibility for 9–11-year-old learners while maintaining consistent syntactic pacing across generation turns.

---

## 3. Lexical Frequency & Probability Analysis

Unigram probability for each token $w$ is defined as:

$$P_{\text{unigram}}(w) = \frac{f(w)}{N}$$

where $N = 415$ is the total token count.

### Top 20 Lexical Words

| Rank | Lexical Word | Frequency ($f(w)$) | Probability ($P(w)$) | Cumulative Probability |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `the` | 63 | 0.151807 | 15.18% |
| 2 | `and` | 12 | 0.028916 | 18.07% |
| 3 | `a` | 11 | 0.026506 | 20.72% |
| 4 | `to` | 11 | 0.026506 | 23.37% |
| 5 | `castle` | 9 | 0.021687 | 25.54% |
| 6 | `silver` | 9 | 0.021687 | 27.71% |
| 7 | `key` | 8 | 0.019277 | 29.64% |
| 8 | `cave` | 7 | 0.016867 | 31.33% |
| 9 | `milo` | 7 | 0.016867 | 33.01% |
| 10 | `near` | 7 | 0.016867 | 34.70% |
| 11 | `pip` | 7 | 0.016867 | 36.39% |
| 12 | `zara` | 7 | 0.016867 | 38.07% |
| 13 | `bright` | 6 | 0.014458 | 39.52% |
| 14 | `crystal` | 6 | 0.014458 | 40.96% |
| 15 | `dark` | 6 | 0.014458 | 42.41% |
| 16 | `forest` | 5 | 0.012048 | 43.61% |
| 17 | `found` | 5 | 0.012048 | 44.82% |
| 18 | `in` | 5 | 0.012048 | 46.02% |
| 19 | `liked` | 5 | 0.012048 | 47.23% |
| 20 | `old` | 5 | 0.012048 | 48.43% |

*Key Takeaway*: The top 20 lexical words account for nearly half ($48.43\%$) of all emitted words, providing the dense statistical anchors needed for n-gram language modeling in a tiny corpus.

### Hapax Legomena (Words Occurring Exactly Once)
- **Total Hapax Legomena Count**: **57 words**
- **Hapax Ratio**: $57 / 119 = \mathbf{47.90\%}$
- Examples: `across`, `advice`, `ancient`, `birds`, `blew`, `box`, `breeze`, `cheerful`, `chest`, `cozy`, `creature`, `discovered`, `echoed`, `entrance`, `fountain`, `gate`, `harmonies`, `hidden`, `journey`, `kingdom`, `moss`, `whispered`, `wings`.

*Educational Significance*: Nearly half the vocabulary occurs once. In greedy bigram generation, these words can only be generated if their immediate single predecessor is selected, illustrating the concept of token sparsity and conditional dependency to young learners.

---

## 4. Within-Sentence Bigram Transitions

### Boundary Convention
To measure intra-sentence coherence without introducing artificial sentence concatenation artifacts:
- Bigram transitions are computed **strictly within sentence boundaries**.
- For sentence $s = (w_1, w_2, \dots, w_L, .)$, transitions are $(w_1, w_2), \dots, (w_L, .)$.
- The period `.` has **0 outgoing transitions** in this intra-sentence model.
- Total within-sentence transitions: $\sum_{i=1}^{34} (L_i - 1) = 415 - 34 = \mathbf{381}$.

### Transition Summary & Branching Factor
- **Total Within-Sentence Bigram Transitions**: **381**
- **Unique Within-Sentence Bigrams**: **259**
- **Active Tokens with Outgoing Transitions ($|\mathcal{V}_{\text{active}}|$)**: **119** (every word token; period has 0)
- **Branching Factor Across Active Vocabulary**:
  $$B_{\text{active}} = \frac{259}{119} = \mathbf{2.1765} \approx \mathbf{2.18}$$
- **Branching Factor Across Full Vocabulary ($|V| = 120$)**:
  $$B_{\text{full}} = \frac{259}{120} = \mathbf{2.1583} \approx \mathbf{2.16}$$

### Top 20 Within-Sentence Bigrams

| Rank | Bigram Sequence ($w_i \to w_{i+1}$) | Frequency ($C(w_i, w_{i+1})$) | Structural Function |
| :---: | :--- | :---: | :--- |
| 1 | `the silver` | 8 | Object modification (`silver key`) |
| 2 | `near the` | 7 | Prepositional phrase anchor |
| 3 | `the crystal` | 6 | Setting modification (`crystal castle`) |
| 4 | `crystal castle` | 6 | Setting landmark compound |
| 5 | `silver key` | 6 | Core quest object compound |
| 6 | `dark cave` | 6 | Setting landmark compound |
| 7 | `the dark` | 5 | Setting adjective phrase |
| 8 | `the old` | 5 | Setting/object adjective phrase |
| 9 | `secret garden` | 5 | Setting landmark compound |
| 10 | `to the` | 5 | Spatial directional phrase |
| 11 | `castle .` | 4 | Sentence-terminating setting phrase |
| 12 | `the brave` | 4 | Character adjective phrase |
| 13 | `the bright` | 4 | Setting adjective phrase |
| 14 | `bright forest` | 3 | Setting landmark compound |
| 15 | `the clever` | 3 | Milo descriptor phrase (`the clever cat`) |
| 16 | `clever cat` | 3 | Milo descriptor compound |
| 17 | `the tiny` | 3 | Pip descriptor phrase (`the tiny dragon`) |
| 18 | `tiny dragon` | 3 | Pip descriptor compound |
| 19 | `the secret` | 3 | Garden descriptor phrase |
| 20 | `trees .` | 3 | Sentence-terminating setting noun |

---

## 5. Audited Pivot-Token Analysis

Pivot tokens represent key decision junctions in the generation graph. The table below provides the audited counts:

| Pivot Token | Total Unigram Occurrences | Outgoing Within-Sentence Transitions | Unique Continuations ($D(a)$) | Primary Continuations & Frequencies |
| :--- | :---: | :---: | :---: | :--- |
| `the` | 63 | 63 | 24 | `silver`: 8, `crystal`: 6, `dark`: 5, `old`: 5, `secret`: 5, `brave`: 4, `bright`: 4, `clever`: 4 |
| `.` | 34 | 0 | 0 | Terminal punctuation; 0 outgoing transitions in intra-sentence model |
| `and` | 12 | 12 | 6 | `the`: 6, `flew`: 2, `liked`: 1, `shared`: 1, `showed`: 1, `walked`: 1 |
| `a` | 11 | 11 | 11 | 11 unique next tokens (1 each: `sleepy`, `playful`, `hidden`, `tiny`, `quiet`, `magic`, etc.) |
| `to` | 11 | 11 | 5 | `the`: 5, `find`: 2, `help`: 2, `play`: 1, `see`: 1 |
| `near` | 7 | 7 | 1 | Strictly branches to `the`: 7 (100% deterministic transition) |
| `milo` | 7 | 7 | 7 | 7 unique next tokens (1 each: `found`, `liked`, `lived`, `saw`, `showed`, `smiled`, `walked`) |
| `pip` | 7 | 7 | 6 | `.`: 2, `flew`: 1, `helped`: 1, `liked`: 1, `lived`: 1, `was`: 1 |
| `zara` | 7 | 7 | 6 | `.`: 2, `found`: 1, `held`: 1, `lived`: 1, `made`: 1, `read`: 1 |
| `castle` | 9 | 9 | 6 | `.`: 4, `and`: 1, `flew`: 1, `stone`: 1, `tower`: 1, `was`: 1 |
| `cave` | 7 | 7 | 5 | `and`: 2, `.`: 2, `had`: 1, `rested`: 1, `to`: 1 |

---

## 6. Character and Setting Entity Distributions

### Recurring Characters
The story corpus evenly balances its three primary protagonists, supported by two animal companions:

| Character Entity | Canonical Token | Total Unigram Occurrences | Associated Roles & Setting Alignments |
| :--- | :--- | :---: | :--- |
| **Milo** | `milo` | **7** | Clever cat; explores castle, forest, and cave; finds the silver key. |
| **Pip** | `pip` | **7** | Tiny dragon; flies to crystal castle; helps friends. |
| **Zara** | `zara` | **7** | Friendly wizard; lives near garden; casts protective magic. |
| **Sleepy Owl** | `owl` | **5** | Sits in old tree near dark cave; observes forest at night. |
| **Playful Fox** | `fox` | **4** | Runs across forest; discovers secret garden paths. |

### Settings & Landmark Coverage
Settings are tracked both as **compound phrases** (`crystal castle`) and **location nouns** (`castle`):

| Setting Location | Compound Phrase | Phrase Count | Location Noun | Noun Count |
| :--- | :--- | :---: | :--- | :---: |
| **Crystal Castle** | `crystal castle` | 6 | `castle` | 9 |
| **Dark Cave** | `dark cave` | 6 | `cave` | 7 |
| **Secret Garden** | `secret garden` | 5 | `garden` | 6 |
| **Bright Forest** | `bright forest` | 4 | `forest` | 5 |

---

## 7. Statistical Visualizations & Interpretations

Six publication-ready figures were generated by `CorpusPlotter` and saved under `experiments/results/corpus_analysis/`:

1. **[top_20_words.png](file:///c:/Users/vasud/OneDrive/Desktop/NLP-GAME-PROJECT/experiments/results/corpus_analysis/top_20_words.png)**:
   - *Description*: Horizontal bar chart of the 20 most frequent lexical words.
   - *Interpretation*: Demonstrates the dominant frequency of function words (`the: 63`, `and: 12`, `a: 11`, `to: 11`) contrasted with character and setting content nouns (`castle: 9`, `silver: 9`, `key: 8`, `cave: 7`, `milo: 7`).
2. **[top_20_bigrams.png](file:///c:/Users/vasud/OneDrive/Desktop/NLP-GAME-PROJECT/experiments/results/corpus_analysis/top_20_bigrams.png)**:
   - *Description*: Horizontal bar chart of the 20 most frequent within-sentence word pairs.
   - *Interpretation*: Reveals the structural prominence of modifying adjective-noun pairs (`the silver`, `near the`, `crystal castle`, `dark cave`).
3. **[sentence_length_distribution.png](file:///c:/Users/vasud/OneDrive/Desktop/NLP-GAME-PROJECT/experiments/results/corpus_analysis/sentence_length_distribution.png)**:
   - *Description*: Discrete bar histogram of sentence token counts.
   - *Interpretation*: Highlights the tight distribution peaked at 12 tokens (61.8% of sentences), proving controlled syntactic uniformity.
4. **[character_occurrences.png](file:///c:/Users/vasud/OneDrive/Desktop/NLP-GAME-PROJECT/experiments/results/corpus_analysis/character_occurrences.png)**:
   - *Description*: Color-coded bar comparison of character entity frequencies.
   - *Interpretation*: Confirms equal parity between Milo, Pip, and Zara (7 mentions each), with secondary support from Owl (5) and Fox (4).
5. **[setting_occurrences.png](file:///c:/Users/vasud/OneDrive/Desktop/NLP-GAME-PROJECT/experiments/results/corpus_analysis/setting_occurrences.png)**:
   - *Description*: Grouped bar chart comparing full compound phrases against individual location nouns.
   - *Interpretation*: Demonstrates that castle and cave appear both as complete phrases and as standalone nouns terminating clauses.
6. **[vocabulary_frequency_distribution.png](file:///c:/Users/vasud/OneDrive/Desktop/NLP-GAME-PROJECT/experiments/results/corpus_analysis/vocabulary_frequency_distribution.png)**:
   - *Description*: Rank-frequency curve on a log-log scale.
   - *Interpretation*: Displays the classic Zipfian power-law decay characteristic of natural language, even within a compact 415-token corpus.

---

## 8. Limitations of a Small 34-Sentence Corpus

While ideal for Grades 4–5 glass-box education and transparent inspection, several inherent limitations must be documented:

1. **High Sparsity (47.9% Hapax Legomena)**:
   - 57 out of 119 lexical words occur only once. In higher-order models (trigrams), this leads to sparse transition graphs where backoff to unigram frequencies is frequently required.
2. **Deterministic Bigram Bridges**:
   - The token `near` is followed by `the` with 100% probability ($7/7$). While natural in this story context, it forms a deterministic bridge where greedy decoding cannot branch.
3. **Context Sensitivity Constraints**:
   - Because character and setting associations are learned from only 34 sentences, subtle semantic nuances cannot be inferred statistically without the rule-based boosting provided by Heads 2, 3, and 4.
