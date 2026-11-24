# Classroom Unplugged Activity: Can Five AI Helpers Write a Story Together?

**Target Audience:** Primary & Middle School Students (Grades 4–6, Ages 9–12)  
**Session Duration:** 45–60 minutes  
**Format:** Unplugged cooperative classroom game (Zero computers required)  
**Core Subject:** Natural Language Processing (NLP), Artificial Intelligence, Creative Writing

---

## 1. Learning Objectives

By the end of this activity, students will understand:
1. **Next-Token Prediction**: How generative AI language systems choose one word at a time based on preceding context.
2. **Multi-Signal Evaluation**: Why modern AI relies on multiple independent evaluation criteria rather than a single guessing rule.
3. **Character Continuity**: How an AI keeps track of the main character without forgetting who is acting.
4. **Setting Consistency**: How AI maintains a consistent physical world (castle, cave, forest, garden).
5. **Dynamic Action**: Why exciting stories need action verbs (*flew, ran, opened*) rather than passive repetition.
6. **Repetition Suppression**: How systems stop themselves from repeating the same words endlessly.
7. **Weighted Decisions**: How different importance weights combine multiple scores into one final decision.

---

## 2. Materials Required

- **Five Head Role Badges** (Printable cards or colored lanyards):
  - 🦉 **Grammar Scout** (Blue)
  - 👤 **Character Keeper** (Pink)
  - 🏰 **Setting Painter** (Green)
  - ⚡ **Action Booster** (Orange)
  - 🛡️ **Repeat Blocker** (Yellow)
- **Candidate Word Cards** (Set of index cards with candidate words printed):
  - Card 1: `forest`
  - Card 2: `castle`
  - Card 3: `garden`
  - Card 4: `Milo`
  - Card 5: `ran`
- **10 Paper Tokens / Counters** (Representing the 10-sided die budget for Hamilton Apportionment).
- **Printable Student Scoring Worksheets** (1 per student or group).
- **Pencils & Erasers**.
- **Whiteboard / Chalkboard** for the teacher to record class scores.

---

## 3. The Five AI Helpers (Roles)

Divide the classroom into **five teams**. Each team embodies one scoring head:

| Helper | Mascot | Team Mission | Question They Ask |
|---|---|---|---|
| **Head 1: Grammar Scout** | 🦉 Owl | Checks language plausibility | *"Does this word sound natural after the context?"* |
| **Head 2: Character Keeper** | 👤 Guardian | Maintains protagonist focus | *"Does this keep our main character Milo in the spotlight?"* |
| **Head 3: Setting Painter** | 🏰 Artist | Keeps the world consistent | *"Does this word match the place where the scene happens?"* |
| **Head 4: Action Booster** | ⚡ Runner | Drives plot momentum | *"Is something exciting happening right now?"* |
| **Head 5: Repeat Blocker** | 🛡️ Shield | Discourages repetition | *"Did we just say this word? Stop repetitive loops!"* |

---

## 4. Simplified Scoring Rules for Students

To make the mathematics intuitive for primary school students, each helper assigns a score between **0.00 and 1.00** using clear guidelines:

### 🦉 Head 1: Grammar Scout
- Words that complete the sentence smoothly: **0.80 to 0.90**
- Words that are possible verbs: **0.65**
- Words that make the grammar awkward: **0.20**

### 👤 Head 2: Character Keeper
- Current active protagonist (*Milo*): **0.85**
- Previously mentioned character (*Pip*, *Zara*): **0.60**
- Brand new character (*Owl*, *Fox*): **0.50**
- Repeating the character name immediately (*"Milo Milo"*): **0.10** (Penalty!)
- Non-character words: **0.20**

### 🏰 Head 3: Setting Painter
- Matching the current scene location (*forest*): **0.85**
- Related nature/scenery nouns (*trees*, *path*): **0.625**
- Shifting to a new valid location (*castle*): **0.40**
- Non-setting words: **0.20**

### ⚡ Head 4: Action Booster
- Dynamic action verbs (*flew, ran, opened, walked*): **0.90**
- Descriptive/state verbs (*saw, liked, rested*): **0.50**
- Nouns and other non-action words: **0.20**

### 🛡️ Head 5: Repeat Blocker
- Safe helper words (*the, a, and, to, in*): **1.00** (Exempt!)
- New content words not seen recently: **0.80**
- Repeated phrase or bigram: **0.20**
- Immediate duplicate word (*"the the"*): **0.05** (Heavy Penalty!)

---

## 5. Walkthrough Classroom Example

### Starting Sentence Context:
> **"Milo entered the _____"**

### Candidate Words on the Table:
1. `forest`
2. `castle`
3. `garden`
4. `Milo`
5. `ran`

### Team Scoring Round:
Each group discusses and announces its score for all 5 candidates:

| Candidate | Grammar Scout (w=0.40) | Character Keeper (w=0.15) | Setting Painter (w=0.15) | Action Booster (w=0.10) | Repeat Blocker (w=0.20) |
|---|:---:|:---:|:---:|:---:|:---:|
| **forest** | 0.90 | 0.40 | 0.95 | 0.30 | 0.85 |
| **castle** | 0.85 | 0.35 | 0.90 | 0.25 | 0.85 |
| **garden** | 0.80 | 0.40 | 0.85 | 0.30 | 0.85 |
| **Milo** | 0.15 | 0.95 | 0.10 | 0.10 | 0.05 |
| **ran** | 0.30 | 0.20 | 0.10 | 0.95 | 0.90 |

### Calculating the Fused Score:
The class uses the **Balanced Weight Profile**:
- Grammar Scout Weight ($\alpha_1$) = **0.40**
- Character Keeper Weight ($\alpha_2$) = **0.15**
- Setting Painter Weight ($\alpha_3$) = **0.15**
- Action Booster Weight ($\alpha_4$) = **0.10**
- Repeat Blocker Weight ($\alpha_5$) = **0.20**

$$\text{Fused Score} = (0.40 \times H_1) + (0.15 \times H_2) + (0.15 \times H_3) + (0.10 \times H_4) + (0.20 \times H_5)$$

Let's calculate candidate **`forest`**:
- Grammar: $0.40 \times 0.90 = 0.3600$
- Character: $0.15 \times 0.40 = 0.0600$
- Setting: $0.15 \times 0.95 = 0.1425$
- Action: $0.10 \times 0.30 = 0.0300$
- Repeat: $0.20 \times 0.85 = 0.1700$
- **Total Fused Score for `forest`** = $0.3600 + 0.0600 + 0.1425 + 0.0300 + 0.1700 =$ **`0.7625`**

Let's calculate candidate **`castle`**:
- Grammar: $0.40 \times 0.85 = 0.3400$
- Character: $0.15 \times 0.35 = 0.0525$
- Setting: $0.15 \times 0.90 = 0.1350$
- Action: $0.10 \times 0.25 = 0.0250$
- Repeat: $0.20 \times 0.85 = 0.1700$
- **Total Fused Score for `castle`** = $0.3400 + 0.0525 + 0.1350 + 0.0250 + 0.1700 =$ **`0.7225`**

Let's calculate candidate **`Milo`**:
- Grammar: $0.40 \times 0.15 = 0.0600$
- Character: $0.15 \times 0.95 = 0.1425$ (High hero salience!)
- Setting: $0.15 \times 0.10 = 0.0150$
- Action: $0.10 \times 0.10 = 0.0100$
- Repeat: $0.20 \times 0.05 = 0.0100$ (Severely penalized duplicate!)
- **Total Fused Score for `Milo`** = $0.0600 + 0.1425 + 0.0150 + 0.0100 + 0.0100 =$ **`0.2375`**

### The Decision:
Candidate **`forest`** wins with **0.7625**!  
The teacher writes `forest` on the board:
> **"Milo entered the forest"**

---

## 6. Hamilton d10 Dice Budget Game

In tabletop roleplaying and board games, we roll a **10-sided die (d10)**.  
How do we convert our continuous weights into 10 physical die faces?

### The Hamilton Largest-Remainder Method:
1. **Give each head a quota** ($q_i = 10 \times \alpha_i$):
   - Grammar Scout ($0.40 \times 10$) = **4.0**
   - Character Keeper ($0.15 \times 10$) = **1.5**
   - Setting Painter ($0.15 \times 10$) = **1.5**
   - Action Booster ($0.10 \times 10$) = **1.0**
   - Repeat Blocker ($0.20 \times 10$) = **2.0**
2. **Assign Base Integer Tokens** (Take the whole number):
   - Grammar Scout = **4 tokens**
   - Character Keeper = **1 token**
   - Setting Painter = **1 token**
   - Action Booster = **1 token**
   - Repeat Blocker = **2 tokens**
   - *Total tokens given out so far* = $4 + 1 + 1 + 1 + 2 =$ **9 tokens**.
3. **Distribute the 1 Surplus Token**:
   - Look at the fractional remainders:
     - Character Keeper has **0.5** leftover.
     - Setting Painter has **0.5** leftover.
   - Tie-breaker rule: Character Keeper gets the extra token!
4. **Final 10-Sided Die Allocation**:
   - Faces 1, 2, 3, 4: **Grammar Scout** (4 faces)
   - Faces 5, 6: **Character Keeper** (2 faces)
   - Face 7: **Setting Painter** (1 face)
   - Face 8: **Action Booster** (1 face)
   - Faces 9, 10: **Repeat Blocker** (2 faces)
   - **Total = Exactly 10 Faces!**

---

## 7. Classroom Discussion Questions

1. **Why didn't `Milo` win even though Milo is the main character?**
   - *Answer: Because saying "Milo entered the Milo" is both grammatically incorrect and an immediate repetition. Both Grammar Scout and Repeat Blocker heavily penalized it.*
2. **What would happen if we turned OFF Repeat Blocker?**
   - *Answer: The story could get stuck in an endless loop, repeating words over and over.*
3. **What happens if we give Action Booster a giant weight (0.60)?**
   - *Answer: The system would prioritize dynamic action verbs like "ran" even if the sentence sounds rushed.*
4. **Why do real AI models need multiple helpers?**
   - *Answer: A single helper only looks at one thing (e.g. grammar or action). Multiple helpers ensure stories are grammatical, logical, exciting, and varied.*
