# Scratch 3.0 Glass-Box Implementation Specification

> **Educational Subsystem: Grades 4–5 (Ages 9–11)**

---

## 🎯 Pedagogical Objective

The Scratch 3.0 implementation serves as a visual, interactive "glass box" that exposes the internal mechanisms of autoregressive language modeling. Instead of hiding computations in compiled code, every step—tokenization, bigram lookups, scoring across the five heads, repetition penalty, and roulette-wheel sampling—is performed using visible lists, monitors, sliders, and custom blocks on the Scratch stage.

---

## 🧱 Data Structures (Parallel Lists)

Because Scratch lacks native dictionaries, multi-dimensional arrays, or key-value hash maps, the system uses synchronized 1D parallel lists:

| List Name | Role / Content | Example Entry |
| :--- | :--- | :--- |
| `Story_Tokens` | Full tokenized training corpus | `["milo", "the", "clever", "cat", ...]` |
| `BG_Prev` | Previous word in observed bigram transition | `["milo", "milo", "the", "crystal"]` |
| `BG_Next` | Next word in observed bigram transition | `["lived", "liked", "silver", "castle"]` |
| `BG_Count` | Frequency of transition `(BG_Prev -> BG_Next)` | `[1, 1, 8, 3]` |
| `Bank_Milo` | Milo character entity word bank | `["cat", "clever", "whiskers", "paws", ...]` |
| `Bank_Pip` | Pip character entity word bank | `["dragon", "tiny", "wings", "fire", ...]` |
| `Bank_Zara` | Zara character entity word bank | `["wizard", "friendly", "wand", "spell", ...]` |
| `Bank_Setting` | Setting vocabulary (castle, forest, cave) | `["crystal", "dark", "forest", "trees", ...]` |
| `Bank_Action` | Action and sensory vocabulary | `["flew", "climbed", "glowing", "magic", ...]` |
| `Cand_Words` | Dynamic candidate next words for current token | `["lived", "liked", "found"]` |
| `Cand_Scores` | Combined scores for each candidate word | `[11, 8, 14]` |
| `Cand_Probs` | Probability or d10 slot ranges | `[3, 2, 5]` |
| `Generated_Story`| Tokens emitted in the current generation run | `["the", "clever", "cat", "milo", "lived"]` |

---

## 🧩 Planned Custom Blocks (`My Blocks`)

The Scratch project will be organized into modular custom procedures (with *"Run without screen refresh"* enabled for performance-critical loops):

```scratch
define Tokenize_Corpus (raw_text)
define Train_Bigram_Transitions
define Train_Trigram_Transitions
define Get_Candidates_For (current_word)
define Score_Head_1_Grammar (cand_index)
define Score_Head_2_Character (cand_index)
define Score_Head_3_Setting (cand_index)
define Score_Head_4_Action (cand_index)
define Score_Head_5_Repetition (cand_index)
define Combine_Five_Heads
define Apply_Repetition_Penalty
define Apportion_D10_Slots (Largest Remainder Method)
define Sample_Roulette_Wheel
define Generate_Next_Token
```

---

## 🎛️ Stage Layout & Interactive Controls

1. **User Sliders**:
   - `Creativity Dial (Temperature)`: Range `0.5` to `2.0` (or `Low / Medium / High`).
   - `Weight_Grammar (H1)`: Default `40%`.
   - `Weight_Character (H2)`: Default `15%`.
   - `Weight_Setting (H3)`: Default `15%`.
   - `Weight_Action (H4)`: Default `10%`.
   - `Weight_Repetition (H5)`: Default `20%`.
2. **Visual Feedback Displays**:
   - Dynamic **Candidate Score Bars**: Sprites whose heights reflect intermediate candidate scores.
   - **The 10-Sided Die Roller**: A sprite that animates a roll from 1 to 10 and highlights the winning candidate slot.
   - **Story Scroll**: Text display rendering the emerging sentence token by token.

---

## 🔬 Algorithmic Parity with Python Engine

To maintain scientific validity:
- The Scratch engine implements the same **Hamilton Apportionment Algorithm** for d10 die mapping.
- Trigram backoff will be supported via coordinated 3-token lists (`TG_W1`, `TG_W2`, `TG_Next`, `TG_Count`).
- Running the same initial prompt with identical deterministic seeds will produce matching token paths in both Scratch and Python.
