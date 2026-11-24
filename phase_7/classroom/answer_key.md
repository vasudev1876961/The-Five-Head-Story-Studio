# Teacher Answer Key: The Five-Head Story Machine Worksheet

This document provides complete solutions, worked calculations, and pedagogical notes for the student activity worksheet.

---

## Part 1: Meet the Five AI Helpers (Matching)

- **1. Grammar Scout (Owl)** $\longrightarrow$ *Makes sure words fit natural language rules*
- **2. Character Keeper (Guardian)** $\longrightarrow$ *Keeps track of who the hero is*
- **3. Setting Painter (Artist)** $\longrightarrow$ *Keeps the world in the same castle or forest*
- **4. Action Booster (Runner)** $\longrightarrow$ *Injects excitement with dynamic action verbs*
- **5. Repeat Blocker (Shield)** $\longrightarrow$ *Stops the story from repeating words in loops*

---

## Part 2: Candidate Word Scoring Table

Completed values in bold:

| Candidate Word | 🦉 Grammar Scout | 👤 Character Keeper | 🏰 Setting Painter | ⚡ Action Booster | 🛡️ Repeat Blocker |
|---|:---:|:---:|:---:|:---:|:---:|
| `forest` | 0.90 | 0.40 | 0.95 | 0.30 | 0.85 |
| `castle` | 0.85 | 0.35 | 0.90 | 0.25 | 0.85 |
| `garden` | 0.80 | 0.40 | 0.85 | 0.30 | 0.85 |
| `Milo` | 0.15 | **0.95** | 0.10 | 0.10 | **0.05** |
| `ran` | 0.30 | 0.20 | 0.10 | **0.95** | 0.90 |

### Scoring Rule Explanations:
- `Milo` under **Character Keeper**: While `Milo` has high protagonist salience (**0.95**), choosing `Milo` creates the duplicate phrase *"Milo entered the Milo"*.
- `Milo` under **Repeat Blocker**: Because `Milo` was just mentioned in the same clause, Repeat Blocker severely penalizes it with **0.05**.
- `ran` under **Action Booster**: `ran` is a dynamic past-tense action verb, so Action Booster awards it the high bonus of **0.95**.

---

## Part 3: Weighted Fused Score Calculations

Formula:
$$S(w) = 0.40(H_1) + 0.15(H_2) + 0.15(H_3) + 0.10(H_4) + 0.20(H_5)$$

### 1. Calculation for `forest`:
- Grammar: $0.40 \times 0.90 = 0.3600$
- Character: $0.15 \times 0.40 = 0.0600$
- Setting: $0.15 \times 0.95 = 0.1425$
- Action: $0.10 \times 0.30 = 0.0300$
- Repeat: $0.20 \times 0.85 = 0.1700$
- **Total Fused Score for `forest`** = $0.3600 + 0.0600 + 0.1425 + 0.0300 + 0.1700 =$ **`0.7625`**

### 2. Calculation for `castle`:
- Grammar: $0.40 \times 0.85 = 0.3400$
- Character: $0.15 \times 0.35 = 0.0525$
- Setting: $0.15 \times 0.90 = 0.1350$
- Action: $0.10 \times 0.25 = 0.0250$
- Repeat: $0.20 \times 0.85 = 0.1700$
- **Total Fused Score for `castle`** = $0.3400 + 0.0525 + 0.1350 + 0.0250 + 0.1700 =$ **`0.7225`**

### 3. Calculation for `garden`:
- Grammar: $0.40 \times 0.80 = 0.3200$
- Character: $0.15 \times 0.40 = 0.0600$
- Setting: $0.15 \times 0.85 = 0.1275$
- Action: $0.10 \times 0.30 = 0.0300$
- Repeat: $0.20 \times 0.85 = 0.1700$
- **Total Fused Score for `garden`** = $0.3200 + 0.0600 + 0.1275 + 0.0300 + 0.1700 =$ **`0.7075`**

### 4. Calculation for `ran`:
- Grammar: $0.40 \times 0.30 = 0.1200$
- Character: $0.15 \times 0.20 = 0.0300$
- Setting: $0.15 \times 0.10 = 0.0150$
- Action: $0.10 \times 0.95 = 0.0950$
- Repeat: $0.20 \times 0.90 = 0.1800$
- **Total Fused Score for `ran`** = $0.1200 + 0.0300 + 0.0150 + 0.0950 + 0.1800 =$ **`0.4400`**

### 5. Calculation for `Milo`:
- Grammar: $0.40 \times 0.15 = 0.0600$
- Character: $0.15 \times 0.95 = 0.1425$
- Setting: $0.15 \times 0.10 = 0.0150$
- Action: $0.10 \times 0.10 = 0.0100$
- Repeat: $0.20 \times 0.05 = 0.0100$
- **Total Fused Score for `Milo`** = $0.0600 + 0.1425 + 0.0150 + 0.0100 + 0.0100 =$ **`0.2375`**

### Winning Word:
- **`forest`** wins with **`0.7625`**!
- Completed sentence: *"Milo entered the forest."*

---

## Part 4: Hamilton d10 Allocation Solutions

1. Quotas: Grammar = **4.0**, Character = **1.5**, Setting = **1.5**, Action = **1.0**, Repeat = **2.0**.
2. Base Allocation (Floor):
   - Grammar: **4**
   - Character: **1**
   - Setting: **1**
   - Action: **1**
   - Repeat: **2**
   - Base Sum = $4 + 1 + 1 + 1 + 2 =$ **9 tokens**
3. Surplus: $10 - 9 =$ **1 surplus token**.
   - Character remainder = $0.5$
   - Setting remainder = $0.5$
   - By priority tie-breaker (*Grammar > Character > Setting > Action > Repeat*), **Character Keeper** receives the surplus point!
4. Final Die Face Allocation:
   - Faces 1, 2, 3, 4: **Grammar Scout** (4 faces)
   - Faces 5, 6: **Character Keeper** (2 faces)
   - Face 7: **Setting Painter** (1 face)
   - Face 8: **Action Booster** (1 face)
   - Faces 9, 10: **Repeat Blocker** (2 faces)
   - **Total = Exactly 10 Faces.**

---

## Part 5: Spot the Repetition Trap Solutions

1. **Repeating Phrase**: *"Pip flew near the crystal castle."* (A 6-token attractor cycle repeating 3 times).
2. **Sleeping Helper**: **Repeat Blocker (Head 5)**.
3. **How it stops it**: Repeat Blocker detects words that appeared in the recent 8-token sliding window and gives them an **exponential penalty** (reducing candidate scores to 0.20 or 0.05). This forces the AI to choose a fresh continuation like *"and Zara smiled"* or *"to find a golden key"*.

---

## Part 6: Reflection Questions Sample Answers

1. **Why didn't `Milo` win?**
   - *Sample Student Answer: "Because repeating 'Milo' right after 'the' sounds wrong in English ('Milo entered the Milo'), and Repeat Blocker gave it a huge penalty (0.05) to stop words from repeating."*
2. **High-speed chase scene weight adjustment:**
   - *Sample Student Answer: "Action Booster! Because in a chase scene, we need dynamic verbs like 'ran', 'dashed', 'jumped', and 'flew' to make the story fast and thrilling."*
3. **Why multiple helpers are better than just grammar:**
   - *Sample Student Answer: "Grammar only knows what words sound legal together, but it doesn't care who the main character is, where they are, or if exciting things are happening. The other helpers make sure the story has a hero, a world, action, and no boring repetitive loops!"*
