# Mathematical Formulation & Pedagogical Explanation

**Project:** The Five-Head Story-Generation System  
**Component:** Educational Mathematical Formulation  
**Target Audience:** Educators, curriculum designers, and students bridging computational thinking to NLP  

---

## 1. Executive Summary & Pedagogical Bridge

Modern natural language processing (NLP) models, such as large transformer language models, operate as complex high-dimensional probability distributions. To students and non-specialists, these models appear as impenetrable "black boxes" that generate text through inscrutable magic.

The **Five-Head Story-Generation Architecture** demystifies text generation by decomposing the next-token prediction problem into **five transparent, interpretable, and competing agents (Scoring Heads)**. 

In this educational formulation, we bridge the gap between:
1. **The Production Python Engine (`five_head_nlp`):** Full N-gram maximum likelihood estimation (MLE), probability smoothing (Laplace, Add-$k$, Good-Turing, Backoff, Interpolation), vocabulary candidate ranking across 120 tokens, and Monte Carlo multi-sentence rollouts.
2. **The Scratch 3.0 & Classroom Simulation:** An interpretable, hand-calculable micro-model where individual head scores are normalized to $[0.0, 1.0]$, combined via linear score fusion, and mapped to a 10-sided die (Hamilton d10 apportionment).

> [!IMPORTANT]
> **Pedagogical Note:** In the Scratch 3.0 project and classroom activities, head scores $h_i(w)$ are **calibrated educational demonstration values** designed to illustrate multi-agent trade-offs and arithmetic fusion. They preserve the exact behavioral logic of the production system (e.g., Grammar prioritizing valid POS tags, Repeat Blocker penalizing immediate recency) while remaining intuitive and accessible for classroom arithmetic.

---

## 2. Production Python vs. Educational Scratch Simulation

| Feature / Concept | Production Python Implementation (`five_head_nlp`) | Educational Scratch & Classroom Simulation |
|---|---|---|
| **Candidate Space** | Full corpus vocabulary ($V = 120$ tokens) | Top 5 representative candidate tokens per step |
| **Probability Scale** | Log-likelihoods ($\log P \in (-\infty, 0]$) | Normalized visual scores ($h_i \in [0.0, 1.0]$) |
| **Grammar Modeling** | Bigram/Trigram MLE with probability smoothing | Bigram syntactic transition plausibility score |
| **Discourse Tracking** | Multi-sentence entity memory & decay matrices | Contextual keyword match & prompt recency |
| **Score Combination** | Configurable weighted sum / temperature scaling | Linear score fusion: $S(w) = \sum_{i=1}^5 w_i h_i(w)$ |
| **Candidate Selection** | Greedy, Top-$k$, or Temperature sampling | Greedy argmax or Hamilton d10 physical dice roll |
| **Audience** | Engineers, researchers, data scientists | Middle school, high school, introductory AI students |

---

## 3. The Linear Score Fusion Formula

At each step $t$, the system considers a running story context $C = (w_1, w_2, \dots, w_{t-1})$ and evaluates a set of candidate tokens $w \in \mathcal{C}$.

Each candidate word receives an individual score from each of the five heads:
- $h_1(w) \in [0.0, 1.0]$ : **Grammar Scout**
- $h_2(w) \in [0.0, 1.0]$ : **Character Keeper**
- $h_3(w) \in [0.0, 1.0]$ : **Setting Painter**
- $h_4(w) \in [0.0, 1.0]$ : **Action Booster**
- $h_5(w) \in [0.0, 1.0]$ : **Repeat Blocker**

The **Fused Score** $S(w)$ is calculated as the convex linear combination:

$$S(w) = \sum_{i=1}^5 w_i \cdot h_i(w) = w_1 h_1(w) + w_2 h_2(w) + w_3 h_3(w) + w_4 h_4(w) + w_5 h_5(w)$$

### Mathematical Properties:
1. **Convexity:** The weights are non-negative ($w_i \ge 0$) and sum strictly to unity:
   $$\sum_{i=1}^5 w_i = 1.0$$
2. **Boundedness:** Since $h_i(w) \in [0.0, 1.0]$ and $\sum w_i = 1.0$:
   $$0.0 \le S(w) \le 1.0$$
3. **Monotonicity:** An increase in any individual head score $h_i(w)$ strictly increases or preserves the candidate's fused score.
4. **Interpretability:** The contribution of head $i$ to candidate $w$ is exactly $w_i \cdot h_i(w)$, allowing students to trace why a specific word won or lost.

---

## 4. The Five Scoring Heads: Simplified Rules

### Head 1: Grammar Scout ($h_1$)
- **Purpose:** Enforces grammatical syntax and transition likelihood.
- **Rule:** 
  - Follows determiner (`"the"`) with a valid noun: $h_1 \approx 0.80 - 0.95$
  - Follows determiner with a verb: $h_1 \approx 0.20 - 0.35$ (Grammar violation)
  - Follows determiner with a proper noun already introduced: $h_1 \approx 0.10 - 0.25$

### Head 2: Character Keeper ($h_2$)
- **Purpose:** Maintains narrative focus on story protagonists and characters.
- **Rule:**
  - Candidate is the primary protagonist (`"Milo"`): $h_2 \approx 0.95$
  - Candidate refers to character actions or objects: $h_2 \approx 0.30 - 0.50$
  - Candidate has no character reference: $h_2 \approx 0.10 - 0.20$

### Head 3: Setting Painter ($h_3$)
- **Purpose:** Builds world atmosphere and vivid environmental descriptions.
- **Rule:**
  - Candidate is a rich environmental location (`"forest"`, `"castle"`, `"garden"`): $h_3 \approx 0.85 - 0.95$
  - Candidate is a non-setting word (verb, pronoun): $h_3 \approx 0.10$

### Head 4: Action Booster ($h_4$)
- **Purpose:** Drives plot momentum, urgency, and conflict forward.
- **Rule:**
  - Candidate is a vigorous action verb (`"ran"`, `"leaped"`, `"grabbed"`): $h_4 \approx 0.90 - 0.95$
  - Candidate is a passive noun or location: $h_4 \approx 0.15 - 0.30$

### Head 5: Repeat Blocker ($h_5$)
- **Purpose:** Prevents repetitive loops, stuttering, and stale vocabulary.
- **Rule:** Evaluates repetition penalty $P(w) \in [0.0, 1.0]$ based on recency:
  $$h_5(w) = 1.0 - P(w)$$
  - Word appeared in the immediate prior context (e.g. `"Milo"` in `"Milo entered the"`): $P = 0.95 \implies h_5 = 0.05$ (Severe penalty!)
  - Word has not appeared in recent context: $P = 0.15 \implies h_5 = 0.85$ (High approval)

---

## 5. Weight Profiles

The system provides three calibrated profiles that alter the personality of the generated story:

| Profile | $w_1$ (Grammar) | $w_2$ (Character) | $w_3$ (Setting) | $w_4$ (Action) | $w_5$ (Repeat) | Sum | Story Aesthetic |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Balanced** | $0.40$ | $0.15$ | $0.15$ | $0.10$ | $0.20$ | $1.00$ | Coherent, fluent, well-rounded narrative |
| **Action-Oriented** | $0.25$ | $0.10$ | $0.10$ | $0.40$ | $0.15$ | $1.00$ | Fast-paced, verb-heavy adventure |
| **Narrative-Rich** | $0.20$ | $0.30$ | $0.30$ | $0.10$ | $0.10$ | $1.00$ | Descriptive, world-building, character-centric |

---

## 6. Complete Worked Walkthrough: `"Milo entered the"`

Let context $C = \text{"Milo entered the"}$. The model evaluates 5 candidate words:

$$\mathcal{C} = \{\text{"forest"}, \text{"castle"}, \text{"garden"}, \text{"ran"}, \text{"Milo"}\}$$

### Individual Head Scores:

| Candidate ($w$) | $h_1$ (Grammar) | $h_2$ (Character) | $h_3$ (Setting) | $h_4$ (Action) | $h_5$ (Repeat) |
|---|:---:|:---:|:---:|:---:|:---:|
| `forest` | $0.90$ | $0.40$ | $0.95$ | $0.30$ | $0.85$ |
| `castle` | $0.85$ | $0.35$ | $0.90$ | $0.25$ | $0.85$ |
| `garden` | $0.80$ | $0.40$ | $0.85$ | $0.30$ | $0.85$ |
| `ran` | $0.30$ | $0.20$ | $0.10$ | $0.95$ | $0.90$ |
| `Milo` | $0.15$ | $0.95$ | $0.10$ | $0.10$ | $0.05$ |

### Fused Score Calculation (Balanced Profile):
$$w = (0.40, 0.15, 0.15, 0.10, 0.20)$$

1. **`forest`:**
   $$S(\text{forest}) = (0.40 \times 0.90) + (0.15 \times 0.40) + (0.15 \times 0.95) + (0.10 \times 0.30) + (0.20 \times 0.85)$$
   $$S(\text{forest}) = 0.3600 + 0.0600 + 0.1425 + 0.0300 + 0.1700 = \mathbf{0.7625} \quad \star \text{WINNER}$$

2. **`castle`:**
   $$S(\text{castle}) = (0.40 \times 0.85) + (0.15 \times 0.35) + (0.15 \times 0.90) + (0.10 \times 0.25) + (0.20 \times 0.85)$$
   $$S(\text{castle}) = 0.3400 + 0.0525 + 0.1350 + 0.0250 + 0.1700 = \mathbf{0.7225}$$

3. **`garden`:**
   $$S(\text{garden}) = (0.40 \times 0.80) + (0.15 \times 0.40) + (0.15 \times 0.85) + (0.10 \times 0.30) + (0.20 \times 0.85)$$
   $$S(\text{garden}) = 0.3200 + 0.0600 + 0.1275 + 0.0300 + 0.1700 = \mathbf{0.7075}$$

4. **`ran`:**
   $$S(\text{ran}) = (0.40 \times 0.30) + (0.15 \times 0.20) + (0.15 \times 0.10) + (0.10 \times 0.95) + (0.20 \times 0.90)$$
   $$S(\text{ran}) = 0.1200 + 0.0300 + 0.0150 + 0.0950 + 0.1800 = \mathbf{0.4400}$$

5. **`Milo`:**
   $$S(\text{Milo}) = (0.40 \times 0.15) + (0.15 \times 0.95) + (0.15 \times 0.10) + (0.10 \times 0.10) + (0.20 \times 0.05)$$
   $$S(\text{Milo}) = 0.0600 + 0.1425 + 0.0150 + 0.0100 + 0.0100 = \mathbf{0.2375}$$

### Outcome Analysis:
- `forest` wins with **$0.7625$**, creating the natural sentence: *"Milo entered the forest."*
- `ran` receives high enthusiasm from Action Booster ($h_4 = 0.95$), but is heavily vetoed by Grammar Scout ($h_1 = 0.30$) because a verb cannot follow the determiner *"the"*.
- `Milo` receives the highest score from Character Keeper ($h_2 = 0.95$), but Repeat Blocker catches the immediate repetition of *"Milo entered the Milo"*, dropping its score to $0.05$ and giving a final fused score of only $0.2375$.

---

## 7. Hamilton d10 Largest-Remainder Apportionment

In a physical classroom setting, students may not have access to digital computers. To simulate probabilistic decision-making with physical manipulatives, we use the **Hamilton Largest-Remainder Method** to apportion a 10-sided die ($d10$) among the five scoring heads.

### The Apportionment Problem:
Given a total dice budget $D = 10$ and fractional weights $w_i \in [0, 1]$ such that $\sum w_i = 1$, find integer allocations $D_i \in \mathbb{N}$ such that:
$$\sum_{i=1}^5 D_i = 10$$

### The Algorithm:
1. **Compute Exact Quotas:**
   $$q_i = D \times w_i = 10 \times w_i$$
2. **Assign Base Integer Allocations:**
   $$b_i = \lfloor q_i \rfloor$$
3. **Calculate Fractional Remainders:**
   $$r_i = q_i - b_i$$
4. **Calculate Surplus Dice:**
   $$S = D - \sum_{i=1}^5 b_i$$
5. **Apportion Surplus:** Award one additional die face to the heads with the largest fractional remainders $r_i$ until all $S$ surplus dice are distributed.
6. **Tie-Breaking Rule:** If two heads share identical remainders $r_i = r_j$, break ties according to priority:
   $$\text{Grammar Scout} > \text{Character Keeper} > \text{Setting Painter} > \text{Action Booster} > \text{Repeat Blocker}$$

---

### Apportionment Worked Examples:

#### 1. Balanced Profile ($0.40, 0.15, 0.15, 0.10, 0.20$):
- **Exact Quotas ($q_i$):**
  - $q_1 = 10 \times 0.40 = 4.0$
  - $q_2 = 10 \times 0.15 = 1.5$
  - $q_3 = 10 \times 0.15 = 1.5$
  - $q_4 = 10 \times 0.10 = 1.0$
  - $q_5 = 10 \times 0.20 = 2.0$
- **Base Allocations ($b_i$):**
  - $b_1 = 4, \; b_2 = 1, \; b_3 = 1, \; b_4 = 1, \; b_5 = 2$
  - $\sum b_i = 4 + 1 + 1 + 1 + 2 = 9$
- **Remainders ($r_i$):**
  - $r_1 = 0.0, \; r_2 = 0.5, \; r_3 = 0.5, \; r_4 = 0.0, \; r_5 = 0.0$
- **Surplus:** $S = 10 - 9 = 1$ die face to distribute.
- **Tie-Break:** Heads 2 and 3 both have $r = 0.5$. By tie-breaking priority ($H_2 > H_3$), Character Keeper receives the surplus die!
- **Final d10 Allocation:**
  - **Grammar Scout:** $4$ faces (Faces 1, 2, 3, 4)
  - **Character Keeper:** $2$ faces (Faces 5, 6)
  - **Setting Painter:** $1$ face (Face 7)
  - **Action Booster:** $1$ face (Face 8)
  - **Repeat Blocker:** $2$ faces (Faces 9, 10)
  - **Total:** $4 + 2 + 1 + 1 + 2 = 10$ faces.

#### 2. Action-Oriented Profile ($0.25, 0.10, 0.10, 0.40, 0.15$):
- Quotas: $q = [2.5, 1.0, 1.0, 4.0, 1.5]$
- Base: $b = [2, 1, 1, 4, 1]$ (Sum = 9)
- Remainders: $r = [0.5, 0.0, 0.0, 0.0, 0.5]$
- Surplus: $S = 10 - 9 = 1$. Tie between $H_1$ and $H_5$. Priority: $H_1 > H_5$. Grammar Scout receives surplus.
- Final Allocation:
  - Grammar Scout: $3$ faces
  - Character Keeper: $1$ face
  - Setting Painter: $1$ face
  - Action Booster: $4$ faces
  - Repeat Blocker: $1$ face

#### 3. Narrative-Rich Profile ($0.20, 0.30, 0.30, 0.10, 0.10$):
- Quotas: $q = [2.0, 3.0, 3.0, 1.0, 1.0]$
- Base: $b = [2, 3, 3, 1, 1]$ (Sum = 10)
- Remainders: $r = [0.0, 0.0, 0.0, 0.0, 0.0]$
- Surplus: $S = 10 - 10 = 0$.
- Final Allocation:
  - Grammar Scout: $2$ faces
  - Character Keeper: $3$ faces
  - Setting Painter: $3$ faces
  - Action Booster: $1$ face
  - Repeat Blocker: $1$ face

---

## 8. Summary of Classroom Pedagogical Outcomes

By engaging with this mathematical formulation, students master five foundational concepts:
1. **Multi-Agent Coordination:** Understanding how complex systems emerge from specialized independent agents.
2. **Convex Combination & Weighting:** Applying weighted averages to trade off competing objectives.
3. **Discrete vs. Continuous Modeling:** Experiencing how continuous probabilities can be faithfully discretized onto physical game mechanics (Hamilton d10).
4. **Veto Mechanics in AI Safety:** Observing how safety components like Repeat Blocker operate as critical guardrails against degenerative behavior.
5. **Algorithmic Explainability:** Learning that AI decisions can be completely transparent and open to inspection.
