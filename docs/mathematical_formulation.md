# Mathematical Formulation Specification

> *Note: This document serves as a formal placeholder for the complete mathematical and probabilistic definitions of the Five-Head Story-Generation System. Full formalization will be implemented in subsequent phases.*

---

## 1. N-Gram Language Modeling & Maximum Likelihood Estimation

### 1.1 Intra-Sentence Maximum Likelihood Estimation (MLE)
Given a corpus $\mathcal{C}$ structured as a collection of disjoint tokenized sentences $\mathcal{S} = \{s_1, s_2, \dots, s_K\}$, where each sentence $s_k = (w_{k,1}, w_{k,2}, \dots, w_{k,M_k})$ terminates in a sentence-boundary period `.`:

1. **Sentence Boundary Preservation**:
   No $N$-gram $(w_1, \dots, w_n)$ is permitted to bridge across sentence boundaries. For any sentence $s_k$, valid $N$-grams are strictly sub-sequences:
   $$(w_{k, j}, w_{k, j+1}, \dots, w_{k, j+n-1}), \quad \forall \, 1 \le j \le M_k - n + 1$$
   Consequently, the sentence-ending period `.` has zero outgoing transitions:
   $$C(., \cdot) = \sum_{v \in \mathcal{V}} C(., v) = 0$$

2. **Unigram Probability ($N=1$)**:
   The empirical unigram probability of token $w$ across all $N_{\text{total}} = \sum_{k=1}^K M_k$ observed tokens is:
   $$P_{\text{MLE}}(w) = \frac{C(w)}{N_{\text{total}}}, \quad \text{where } \sum_{w \in \mathcal{V}} P_{\text{MLE}}(w) = 1.0$$

3. **Bigram Conditional Probability ($N=2$)**:
   The conditional probability of token $w_i$ given preceding token $w_{i-1}$ is:
   $$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{\sum_{v \in \mathcal{V}} C(w_{i-1}, v)} = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$
   For any observed context $w_{i-1}$ having at least one valid continuation:
   $$\sum_{v \in \mathcal{V}_{\text{cand}}(w_{i-1})} P_{\text{MLE}}(v \mid w_{i-1}) = 1.0$$

4. **Trigram Conditional Probability ($N=3$)**:
   The conditional probability of token $w_i$ given preceding two-token context $(w_{i-2}, w_{i-1})$ is:
   $$P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{\sum_{v \in \mathcal{V}} C(w_{i-2}, w_{i-1}, v)} = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$$

5. **Autoregressive Joint Sequence Probability (Chain Rule)**:
   For an arbitrary token sequence $W = (w_1, w_2, \dots, w_m)$ under an $N$-gram model of order $n$:
   $$P(W) = P_{\text{MLE}}(w_1) \prod_{i=2}^m P_{\text{MLE}}\left(w_i \mid w_{\max(1, i-n+1)}^{i-1}\right)$$

6. **Information Theoretic Log-Probability**:
   The base-2 log-likelihood of token continuation is given by:
   $$\log_2 P_{\text{MLE}}(w_i \mid \text{context}) = \begin{cases} \log_2 P_{\text{MLE}}(w_i \mid \text{context}) & \text{if } P_{\text{MLE}} > 0 \\ -\infty & \text{if } P_{\text{MLE}} = 0 \end{cases}$$

### 1.2 Empirical Corpus Properties (34-Sentence Ground Truth)
- **Sentences**: $K = 34$
- **Total Tokens**: $N_{\text{total}} = 415$ (381 words + 34 periods)
- **Vocabulary Size**: $|\mathcal{V}| = 120$ unique tokens (119 words + 1 period `.`)
- **Unigram Occurrences**: 415 (120 unique types)
- **Within-Sentence Bigram Transitions**: 381 transitions (259 unique bigrams)
- **Within-Sentence Trigram Transitions**: 347 transitions (303 unique trigrams)
- **Active Contexts ($N=2$)**: 119 tokens (the period `.` has 0 outgoing transitions)
- **Average Bigram Branching Factor**: $\frac{259}{119} \approx 2.1765$ continuations per active context

---

### 1.3 Probability Smoothing Framework

To resolve the zero-frequency problem ($P_{\text{MLE}} = 0$ for unobserved N-grams) and provide full support across the vocabulary simplex $\mathcal{V}$, five probability smoothing strategies are formalized:

#### 1.3.1 Laplace (Add-One) Smoothing
Adds a uniform pseudo-count of 1 to every transition:
- **Unigram**:
  $$P_{\text{Laplace}}(w) = \frac{C(w) + 1}{N_{\text{total}} + |\mathcal{V}|}$$
- **Conditional (Bigram / Trigram)**:
  $$P_{\text{Laplace}}(w \mid h) = \frac{C(h, w) + 1}{C(h) + |\mathcal{V}|}$$
- **Normalization Proof**:
  $$\sum_{w \in \mathcal{V}} P_{\text{Laplace}}(w \mid h) = \frac{\sum_{w} C(h, w) + \sum_{w} 1}{C(h) + |\mathcal{V}|} = \frac{C(h) + |\mathcal{V}|}{C(h) + |\mathcal{V}|} = 1.0$$

#### 1.3.2 Add-$k$ Smoothing
Generalizes Laplace smoothing with positive parameter $k > 0$:
$$P_{\text{Add-}k}(w \mid h) = \frac{C(h, w) + k}{C(h) + k|\mathcal{V}|}$$
As $k \to 0^+$, $P_{\text{Add-}k} \to P_{\text{MLE}}$. For small corpora ($|\mathcal{V}| = 120$), $k \in \{0.1, 0.5\}$ prevents over-allocating excessive probability mass to unseen events.

#### 1.3.3 Good-Turing Estimation
Re-estimates counts based on frequency-of-frequency statistics:
- $N_r = |\{x : C(x) = r\}|$ (number of distinct N-gram event types occurring exactly $r$ times).
- Singleton event types (hapax legomena): $N_1$ distinct types occurring once each, accounting for $1 \times N_1 = N_1$ total occurrences.
- Total sample occurrences: $N = \sum_{r \ge 1} r \cdot N_r$.
- **Unseen Probability Mass**:
  $$P_0 = \frac{N_1}{N}$$
  allocated evenly among unseen vocabulary elements.
- **Adjusted Count**:
  $$r^* = (r + 1) \frac{N_{r+1}}{N_r}$$
- **Small-Corpus Thresholding & Missing-Bin Fallback**: For $r \le k_{\text{threshold}}$ where $N_{r+1} > 0$, $r^*$ is discounted; for missing bins ($N_{r+1} = 0$) or counts exceeding $k_{\text{threshold}}$, $r^* = r$ (Simple Good-Turing empirical fallback).

#### 1.3.4 Absolute Discounting Backoff
Subtracts fixed discount $d \in (0, 1)$ from observed events and redistributes reserved mass $\beta(h)$ over lower-order backoff distributions:
$$P_{\text{backoff}}(w \mid h) = \begin{cases} \frac{C(h, w) - d}{C(h)} & \text{if } C(h, w) > 0 \\ \alpha(h) \cdot P_{\text{backoff}}(w \mid \text{lower}(h)) & \text{if } C(h, w) == 0 \end{cases}$$
where:
$$\beta(h) = \frac{d \cdot |\mathcal{V}_{\text{cand}}(h)|}{C(h)}$$
$$\alpha(h) = \frac{\beta(h)}{\sum_{v \in \mathcal{V}_{\text{unseen}}(h)} P_{\text{backoff}}(v \mid \text{lower}(h))}$$
where $\mathcal{V}_{\text{unseen}}(h) = \mathcal{V} \setminus \mathcal{V}_{\text{cand}}(h)$.
- **Degenerate Context Resolution**: If the lower-order distribution assigns zero probability mass across all unobserved continuations ($\sum_{v \in \mathcal{V}_{\text{unseen}}(h)} P_{\text{lower}}(v) = 0$), the reserved mass $\beta(h)$ is distributed uniformly over $\mathcal{V}_{\text{unseen}}(h)$:
  $$P_{\text{backoff}}(w \mid h) = \frac{\beta(h)}{|\mathcal{V}_{\text{unseen}}(h)|}, \quad \forall w \in \mathcal{V}_{\text{unseen}}(h)$$
  This guarantees that simplex normalization $\sum_{w \in \mathcal{V}} P_{\text{backoff}}(w \mid h) = 1.0$ is strictly conserved for all degenerate contexts.

#### 1.3.5 Linear Interpolation
Convex combination of Unigram, Bigram, and Trigram distributions:
$$P_{\text{interp}}(w \mid w_{t-1}, w_t) = \lambda_1 P_{\text{uni}}(w) + \lambda_2 P_{\text{bi}}(w \mid w_t) + \lambda_3 P_{\text{tri}}(w \mid w_{t-1}, w_t)$$
subject to:
$$\lambda_1, \lambda_2, \lambda_3 \ge 0, \quad \sum_{i=1}^3 \lambda_i = 1.0$$
When higher-order contexts lack evidence ($C(h) = 0$), weights are re-normalized across active tiers to guarantee $\sum_{w \in \mathcal{V}} P_{\text{interp}}(w \mid h) = 1.0$.
Weight optimization is performed strictly on validation data to prevent evaluation data leakage.

#### 1.3.6 Cross-Entropy and Perplexity Evaluation Metrics
Given held-out evaluation sequence of $M$ prediction events:
1. **Total Negative Log-Likelihood (NLL)**:
   $$\text{NLL}_{\text{total}} = -\sum_{i=1}^M \ln P(w_i \mid \text{context}_i)$$
2. **Mean Cross-Entropy ($H$, in nats)**:
   $$H = \frac{\text{NLL}_{\text{total}}}{M} = -\frac{1}{M} \sum_{i=1}^M \ln P(w_i \mid \text{context}_i)$$
3. **Cross-Entropy in Bits ($H_{\text{bits}}$)**:
   $$H_{\text{bits}} = \frac{H}{\ln(2)} = -\frac{1}{M} \sum_{i=1}^M \log_2 P(w_i \mid \text{context}_i)$$
4. **Perplexity (PPL)**:
   $$\text{PPL} = \exp(H) = 2^{H_{\text{bits}}}$$
   If any prediction event receives zero probability ($P = 0$), $\text{NLL}_{\text{total}} = \infty$, $H = \infty$, and $\text{PPL} = \infty$.

#### 1.3.7 3-Way Leak-Free Partition
To benchmark smoothing methods with strict mathematical rigor:
- **Training Set**: 24 sentences ($N_{\text{train}} = 295$ tokens, 70.6%). Used exclusively for model parameter fitting and N-gram counts.
- **Validation Set**: 5 sentences ($N_{\text{val}} = 62$ tokens, 14.7%). Used exclusively for hyperparameter tuning and interpolation weight optimization $\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda}} H_{\text{val}}(\boldsymbol{\lambda})$.
- **Test Set**: 5 sentences ($N_{\text{test}} = 58$ tokens, 14.7%). Strictly held-out; never observed during training or weight optimization, used solely for final benchmark reporting.

---

### 1.4 Next-Token Prediction & Decoding Engine

To transition from static probability estimation to autoregressive token generation, next-token prediction decouples probability retrieval, candidate ranking, and token selection.

#### 1.4.1 Candidate Ranking and Deterministic Tie-Breaking
Given context $h = (w_{t-k+1}, \dots, w_t)$, the model or smoother outputs a probability distribution over the vocabulary $\mathcal{V}$:
$$\mathcal{D}(h) = \{(w, P(w \mid h)) \mid w \in \mathcal{V}\}$$
Candidates are sorted in descending order of probability:
$$P(w_{(1)} \mid h) \ge P(w_{(2)} \mid h) \ge \dots \ge P(w_{(|\mathcal{V}|)} \mid h)$$
**Tie-Breaking Rule**: If $P(w_a \mid h) = P(w_b \mid h)$ (within numerical tolerance $\epsilon = 10^{-9}$), tokens are ordered alphabetically by string comparison ($w_a <_{\text{lex}} w_b$).

#### 1.4.2 Greedy Decoding (Deterministic Argmax)
Greedy decoding deterministically selects the candidate with maximum probability:
$$w_{t+1}^* = \arg\max_{w \in \mathcal{V}} P(w \mid h)$$
Under deterministic tie-breaking, greedy decoding guarantees identical outputs for identical context histories without random number generation.

#### 1.4.3 Top-$k$ Candidate Filtering & Truncated Sampling
Top-$k$ decoding truncates the candidate distribution to the $k$ most likely candidates ($k \ge 1$), discarding the probability tail to suppress unlikely transitions:
1. Select candidate subset: $\mathcal{V}_k(h) = \{w_{(1)}, w_{(2)}, \dots, w_{(k)}\}$.
2. Compute subset partition sum: $Z_k = \sum_{j=1}^k P(w_{(j)} \mid h)$.
3. Renormalize probabilities across the top-$k$ support:
   $$P_{\text{top-}k}(w_{(i)} \mid h) = \frac{P(w_{(i)} \mid h)}{Z_k}, \quad \forall \, 1 \le i \le k$$
4. Sample $w_{t+1} \sim P_{\text{top-}k}(\cdot \mid h)$ using an isolated random number generator.

#### 1.4.4 Temperature-Scaled Softmax Sampling
Temperature scaling re-weights candidate probabilities via a positive temperature parameter $T > 0$:
$$P_T(w \mid h) = \begin{cases} \frac{\exp\left( \frac{\ln P(w \mid h)}{T} \right)}{\sum_{v : P(v \mid h) > 0} \exp\left( \frac{\ln P(v \mid h)}{T} \right)} & \text{if } P(w \mid h) > 0 \\ 0 & \text{if } P(w \mid h) = 0 \end{cases}$$
- **Low Temperature ($T \to 0^+$)**: $P_T$ concentrates mass onto the argmax candidate, converging to greedy selection.
- **Unit Temperature ($T = 1.0$)**: $P_T(w \mid h) = P(w \mid h)$, preserving the exact model distribution.
- **High Temperature ($T > 1.0$)**: Softens peak probabilities, flattening the distribution towards uniform exploration among candidates with non-zero probability support.
- **Numerical Stability**: Computed by subtracting the maximum scaled logit $z_{\max} = \max_j \frac{\ln P(w_j \mid h)}{T}$ prior to exponentiation:
  $$\hat{z}_i = \frac{\ln P(w_i \mid h)}{T} - z_{\max}, \quad P_T(w_i \mid h) = \frac{\exp(\hat{z}_i)}{\sum_j \exp(\hat{z}_j)}$$

#### 1.4.5 Generation State & Sentence Termination Dynamics
A lightweight generation state maintains the rolling context window and enforces sentence-boundary isolation:
1. **Context Windowing**: For an N-gram model of order $n$, active context is truncated to the rolling tail of length $\min(t, n-1)$.
2. **Terminal Period Halting**: The sentence-ending period `.` indicates terminal punctuation. Once selected, generation terminates immediately ($\text{is\_terminated} = \text{True}$), and no further prediction steps are permitted.
3. **Maximum Token Guard**: Generation enforces a hard stopping limit $\text{max\_tokens}$ to prevent infinite loops.

---

## 2. Multi-Head Scoring Equations

Each candidate token $w \in \mathcal{V}_{\text{cand}}$ evaluated by the system receives individual normalized scores in $[0.0, 1.0]$ across five transparent heads:

### 2.1 Individual Head Scoring Functions

1. **Head 1 — Grammar Scout**:
   Evaluates statistical language model plausibility:
   $$H_1(w) = P(w \mid h) \in [0.0, 1.0]$$
   where $P(w \mid h)$ is retrieved from the underlying N-gram language model or smoother.

2. **Head 2 — Character Keeper**:
   Maintains narrative character continuity across active entities:
   $$H_2(w) = \begin{cases} 0.85 & \text{if } w = \text{active\_character} \text{ (continuity bonus)} \\ 0.70 & \text{if } w \in \mathcal{C}_{\text{mentioned}} \text{ (recurrent mention)} \\ 0.50 & \text{if } w \in \mathcal{L}_{\text{char}} \setminus \mathcal{C}_{\text{mentioned}} \text{ (introduction bonus)} \\ 0.10 & \text{if } w \in \mathcal{L}_{\text{char}} \land w_{t} = w \text{ (immediate character duplicate)} \\ 0.20 & \text{otherwise (non-character neutral score)} \end{cases}$$

3. **Head 3 — Setting Painter**:
   Maintains spatial and environmental consistency:
   $$H_3(w) = \begin{cases} 0.85 & \text{if } w = \text{current\_setting} \lor \text{current\_setting is None} \text{ (consistency bonus)} \\ 0.625 & \text{if } w \in \mathcal{L}_{\text{setting\_nouns}} \text{ (associated environmental feature)} \\ 0.40 & \text{if } w \in \mathcal{L}_{\text{core\_settings}} \setminus \{\text{current\_setting}\} \text{ (setting shift)} \\ 0.20 & \text{otherwise (non-setting neutral score)} \end{cases}$$

4. **Head 4 — Action Booster**:
   Drives plot progression by rewarding dynamic verbs over passive states:
   $$H_4(w) = \begin{cases} 0.90 & \text{if } w \in \mathcal{L}_{\text{dynamic\_actions}} \text{ (movement, discovery, interaction)} \\ 0.50 & \text{if } w \in \mathcal{L}_{\text{state\_verbs}} \text{ (perception, stative)} \\ 0.20 & \text{otherwise (non-action neutral score)} \end{cases}$$

5. **Head 5 — Repeat Blocker**:
   Suppresses repetitive attractor loops and lexical saturation:
   $$H_5(w) = \begin{cases} 1.00 & \text{if } w \in \mathcal{L}_{\text{exempt}} \text{ (function words and punctuation)} \\ 0.05 & \text{if } w = w_{t} \text{ (immediate self-loop)} \\ 0.20 & \text{if } (w_{t}, w) \in \mathcal{W}_{\text{bigrams}} \text{ (repeated bigram cycle)} \\ \exp(-\gamma \cdot c_{\text{window}}(w)) & \text{otherwise (frequency decay, with } \gamma = 0.80) \end{cases}$$

### 2.2 Linear Head Fusion
Individual normalized head scores are combined via a non-negative normalized weight vector $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5)$:

$$S_{\text{fused}}(w) = \sum_{k=1}^5 \alpha_k H_k(w), \quad \text{where } \alpha_k \ge 0, \; \sum_{k=1}^5 \alpha_k = 1.0$$

Default Profile (Balanced): $\boldsymbol{\alpha} = (0.40, 0.15, 0.15, 0.10, 0.20)$.

Candidates are ranked in descending order of $S_{\text{fused}}(w)$.
Ties are broken alphabetically by token string ($w_a <_{\text{lex}} w_b$).

---

## 3. Repetition Penalty Formulation

To break high-probability attractor cycles, a non-linear decay function is applied to candidate scores:

$$S_{\text{pen}}(w) = \max\left(\epsilon, \, \frac{S_{\text{base}}(w)}{r^{\gamma(w)}}\right)$$

where:
- $r = 1.8$ (repetition penalty base factor).
- $\epsilon = 0.05$ (score floor preventing zero division or candidate starvation).
- $\gamma(w) = 2 \cdot I_{\text{bg}}(w) + c_{\text{recent}}(w)$.
- $I_{\text{bg}}(w) \in \{0, 1\}$ indicates whether the bigram $(w_t, w)$ appears in the recent context window.
- $c_{\text{recent}}(w)$ counts the occurrences of token $w$ in the sliding window.

---

## 4. Stochastic Sampling & Temperature Scaling

### 4.1 Temperature-Scaled Softmax
For continuous generation, penalized scores are mapped to a probability simplex over $\mathcal{V}_{\text{cand}}$:

$$P(w) = \frac{\left(S_{\text{pen}}(w)\right)^{1/T}}{\sum_{v \in \mathcal{V}_{\text{cand}}} \left(S_{\text{pen}}(v)\right)^{1/T}}$$

where $T > 0$ represents the temperature parameter:
- $T \to 0^+$: Deterministic Argmax (Greedy).
- $T = 1.0$: Proportional energy sampling.
- $T \to \infty$: Uniform exploration across $\mathcal{V}_{\text{cand}}$.

---

## 5. Classroom Integer Mode & Hamilton Apportionment

In unplugged or integer mode, students compute:

$$S_{\text{raw}}(w) = \sum_{k=1}^5 H_k^{\text{int}}(w)$$

When repetition is flagged by $H_5$:
$$S_{\text{int}}(w) = \max\left(1, \, \left\lfloor \frac{S_{\text{raw}}(w)}{3} \right\rfloor\right)$$

To map candidate scores $\{S_1, S_2, \dots, S_m\}$ to $K = 10$ slots of a 10-sided die ($d10$):

1. **Exact Quota**:
   $$q_i = K \cdot \frac{S_i}{\sum_{j=1}^m S_j}$$

2. **Base Integer Allocation**:
   $$a_i = \lfloor q_i \rfloor$$

3. **Remainder Ranking**:
   $$r_i = q_i - a_i$$

Remaining slots $K - \sum_{i=1}^m a_i$ are assigned one by one to candidates with the largest fractional remainders $r_i$ until $\sum a_i = 10$.

---

## 6. Phase 5 Experimental Suite & Monte Carlo Metrics

### 6.1 Multi-Sentence Story Generation Process
A generated story $\mathcal{W} = (s_1, s_2, \dots, s_M)$ is an ordered sequence of completed sentences $s_m = (w_{m,1}, \dots, w_{m, L_m}, .)$, where each sentence terminates in a period `.`.
Generation continues until the dual termination condition is satisfied:
$$\sum_{m=1}^M (L_m + 1) \ge N_{\text{min}} \quad \text{and} \quad M \ge M_{\text{min}}$$
with default target constraints $N_{\text{min}} = 80$, $N_{\text{max}} = 120$, and $M_{\text{min}} = 3$.

### 6.2 Deterministic Isolated Seed Scheduling
To ensure absolute reproducibility across independent story executions without inter-story state leakage, the random seed for condition $c$ at ordinal run index $i \in \{0, \dots, N_{\text{stories}}-1\}$ is derived via a bijection:
$$\sigma(c, i) = \sigma_{\text{base}} + 10\,000 \cdot c + i$$
where $\sigma_{\text{base}} = 42$. All stochastic samplers and starter selectors are instantiated exclusively with an isolated pseudo-random number generator $\text{RNG}(\sigma(c, i))$.

### 6.3 Repetition Metric Formulations

1. **Immediate Content Token Repetition Rate**:
   Evaluating successive identical tokens excluding exempt functional tokens $\mathcal{V}_{\text{exempt}}$:
   $$R_{\text{imm}} = \frac{\sum_{i=2}^N \mathbb{I}(w_i = w_{i-1} \land w_i \notin \mathcal{V}_{\text{exempt}})}{\sum_{i=2}^N \mathbb{I}(w_i \notin \mathcal{V}_{\text{exempt}})}$$

2. **Repeated Bigram Rate**:
   Given the multiset of within-story bigrams $B = \{(w_i, w_{i+1})\}_{i=1}^{N-1}$:
   $$R_{\text{bg}} = \frac{|B| - |B_{\text{unique}}|}{|B|} = 1 - \frac{|B_{\text{unique}}|}{N - 1}$$

3. **Repeated Trigram Rate**:
   Given the multiset of within-story trigrams $T = \{(w_i, w_{i+1}, w_{i+2})\}_{i=1}^{N-2}$:
   $$R_{\text{tg}} = \frac{|T| - |T_{\text{unique}}|}{|T|} = 1 - \frac{|T_{\text{unique}}|}{N - 2}$$

4. **Rolling Window Repetition Rate ($W=8$)**:
   For each sliding window of length $W$, checking for non-exempt duplicates:
   $$R_{\text{win}} = \frac{1}{N - W + 1} \sum_{j=1}^{N - W + 1} \mathbb{I}\left(\exists w \notin \mathcal{V}_{\text{exempt}} : \text{count}(w \in (w_j, \dots, w_{j+W-1})) > 1\right)$$

### 6.4 Lexical Diversity Formulations

1. **Lexical Type-Token Ratio (TTR)**:
   Restricted to lexical words $\mathcal{W}_{\text{words}} = \{w \in \mathcal{W} \mid w \notin \{., ,, !, ?\}\}$:
   $$\text{TTR} = \frac{|\text{set}(\mathcal{W}_{\text{words}})|}{|\mathcal{W}_{\text{words}}|}$$

2. **Moving-Average Type-Token Ratio (MATTR-$k$, $k=20$)**:
   Average TTR over rolling sub-sequences of fixed length $k$:
   $$\text{MATTR}_k = \frac{1}{|\mathcal{W}_{\text{words}}| - k + 1} \sum_{i=1}^{|\mathcal{W}_{\text{words}}| - k + 1} \frac{|\text{set}(w_i, \dots, w_{i+k-1})|}{k}$$

3. **Hapax Legomena Ratio**:
   Fraction of vocabulary types appearing exactly once:
   $$\text{HLR} = \frac{\sum_{v \in \text{set}(\mathcal{W}_{\text{words}})} \mathbb{I}(\text{count}(v) = 1)}{|\text{set}(\mathcal{W}_{\text{words}})|}$$

### 6.5 Thematic Continuity & Density Formulations

1. **Character Continuity Ratio**:
   Given ordered sequence of character mentions $(c_1, c_2, \dots, c_J)$ where $c_j \in \mathcal{V}_{\text{char}}$:
   $$\text{CCR} = \frac{\sum_{j=2}^J \mathbb{I}(c_j = c_{j-1})}{\max(1, J - 1)}$$

2. **Setting Consistency Ratio**:
   Dominant setting mentions relative to all setting mentions $\mathcal{V}_{\text{setting}}$:
   $$\text{SCR} = \frac{\max_{s \in \mathcal{V}_{\text{setting}}} \text{count}(s)}{\sum_{s \in \mathcal{V}_{\text{setting}}} \text{count}(s)}$$

3. **Dynamic Action Ratio**:
   Proportion of dynamic verbs relative to total action verbs:
   $$\text{DAR} = \frac{\sum_{w \in \mathcal{V}_{\text{dyn}}} \text{count}(w)}{\sum_{w \in \mathcal{V}_{\text{dyn}}} \text{count}(w) + \sum_{w \in \mathcal{V}_{\text{state}}} \text{count}(w)}$$

### 6.6 Latency & Throughput Benchmarking
Per-token latency $\tau_i$ is recorded using monotonic CPU timers:
$$\tau_i = \left(t_{\text{end}}(w_i) - t_{\text{start}}(w_i)\right) \times 10^3 \quad [\text{ms}]$$
Total story generation duration $T_{\text{gen}} = t_{\text{story\_end}} - t_{\text{story\_start}}$ [s], and generation throughput:
$$\text{TPS} = \frac{N_{\text{tokens}}}{T_{\text{gen}}} \quad [\text{tokens/s}]$$

