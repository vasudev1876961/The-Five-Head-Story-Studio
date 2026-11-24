# Teacher Guide: The Five-Head Story-Generation System

**Target Level:** Grades 4–6 (Primary & Early Secondary Education)  
**Curriculum Connections:** Computer Science / Computational Thinking, Language Arts, Mathematics  
**Estimated Lesson Time:** 45–60 minutes  
**Preparation Time:** 15 minutes (Printing role badges and worksheets)

---

## 1. Pedagogical Overview

Large Language Models (LLMs) such as ChatGPT or Gemini are frequently presented to students as mysterious "magic brains." This lesson breaks down autoregressive generative AI into an inspectable, cooperative decision-making process.

By embodying the **Five Transparent Scoring Heads**, students discover that:
- AI does not generate whole stories instantaneously; it generates **one word at a time** (next-token prediction).
- Token selection is governed by **multi-criteria evaluation**: grammar, character focus, world consistency, pacing, and repetition suppression.
- Decisions are determined through **weighted linear score fusion**, showing that mathematics directly shapes creativity.

---

## 2. Lesson Plan & Timeline (60 Minutes)

### Phase 1: The Hook — How Does AI Think? (8 Minutes)
1. Write on the board:
   > *"The sleepy cat climbed up the ______"*
2. Ask the class: *"What word comes next?"*
   - Students will suggest: *tree, roof, wall, couch, stairs*.
3. Ask: *"How did your brain choose those words? Did you just guess randomly?"*
4. Explain: *"Your brain checked grammar, remembered the cat, imagined the setting, and made sure you didn't say 'cat' twice! That is exactly what our Five AI Helpers do!"*

### Phase 2: Role Assignment & Introduction (7 Minutes)
1. Divide students into five teams. Hand out the **Five Head Role Badges**:
   - **Team 1: Grammar Scout** (Checks language plausibility)
   - **Team 2: Character Keeper** (Checks character focus)
   - **Team 3: Setting Painter** (Checks setting consistency)
   - **Team 4: Action Booster** (Encourages dynamic verbs)
   - **Team 5: Repeat Blocker** (Penalizes repetitive loops)
2. Explain the scoring scale (0.00 to 1.00):
   - $1.00$ = Perfect match!
   - $0.50$ = Okay / neutral.
   - $0.10$ or $0.05$ = Big penalty!

### Phase 3: The Cooperative Word-Choice Round (20 Minutes)
1. Present the starter sentence:
   > *"Milo entered the ____"*
2. Hand out the 5 candidate cards: `forest`, `castle`, `garden`, `Milo`, `ran`.
3. Have each team confer and decide their score for every word using the rules in their worksheet.
4. Each team sends a spokesperson to write their score on the classroom board table.
5. Guide the class through calculating the **Fused Score** using the **Balanced Profile**:
   $$\text{Fused Score} = 0.40(H_1) + 0.15(H_2) + 0.15(H_3) + 0.10(H_4) + 0.20(H_5)$$
6. Compare the fused scores:
   - `forest`: **0.7625** (Winner!)
   - `castle`: **0.7225**
   - `garden`: **0.7075**
   - `ran`: **0.4400**
   - `Milo`: **0.2375**
7. Discuss why `forest` beat `castle` (both had high setting scores, but `forest` fit the grammar slightly better), and why `Milo` was heavily penalized by Repeat Blocker.

### Phase 4: Hamilton d10 Die Allocation (10 Minutes)
1. Explain: *"In computer code we use decimals like 0.40, but in tabletop games we roll a 10-sided die! How do we share 10 dice faces among 5 heads?"*
2. Demonstrate **Hamilton's Largest-Remainder Method**:
   - Multiply each weight by 10 to get the exact quota: Grammar = 4.0, Character = 1.5, Setting = 1.5, Action = 1.0, Repeat = 2.0.
   - Hand out 10 physical counters / paper tokens.
   - Give each head its whole number (Base allocation = $4 + 1 + 1 + 1 + 2 = 9$).
   - 1 token remains! Character and Setting both have remainder $0.5$.
   - Apply tie-breaker: Character Keeper wins the 10th face!
   - Final d10 Allocation: **Grammar (4), Character (2), Setting (1), Action (1), Repeat (2)**.

### Phase 5: Reflection & Assessment (15 Minutes)
1. Students complete the reflection section on their worksheets.
2. Group debrief:
   - *"What would happen to the story if Repeat Blocker was asleep?"*
   - *"Can the highest grammar score alone write an exciting story?"*

---

## 3. Assessment Rubric

| Criteria | Developing (1 pt) | Proficient (2 pts) | Advanced (3 pts) |
|---|---|---|---|
| **Role Understanding** | Confuses the responsibilities of the five heads. | Accurately explains the function of their assigned head. | Explains how all five heads interact to shape story quality. |
| **Scoring Application** | Assigns random scores without following rules. | Correctly applies scoring rules to candidate words. | Explains nuances (e.g. why immediate repeats receive heavy penalties). |
| **Mathematical Calculation** | Makes arithmetic errors in weighted fusion. | Accurately calculates weighted scores with teacher guidance. | Independently performs weighted fusion and Hamilton d10 apportionment. |
| **Conceptual Understanding** | Treats AI as an all-knowing single brain. | Explains that AI chooses words using multiple criteria. | Articulates trade-offs between grammar, action, continuity, and diversity. |

---

## 4. Common Misunderstandings & Teacher Guidance

1. **Misconception: "AI knows the whole story before writing sentence 1."**
   - *Correction*: Emphasize that autoregressive language models predict **one token at a time** conditioning only on prior history.
2. **Misconception: "The word with the highest grammar score always wins."**
   - *Correction*: Show how Action Booster and Setting Painter can push an exciting word past a purely common grammatical continuation.
3. **Misconception: "Repeat Blocker hates all common words."**
   - *Correction*: Clarify that helper words (*the, a, and, in*) are **exempt**; Repeat Blocker specifically targets duplicate content words and loop attractor cycles.

---

## 5. Extensions & Cross-Curricular Connections

- **Computer Science / Scratch Connection**: Open `five_head_story_machine.sb3` in the Scratch 3.0 editor. Students can test different weight profiles digitally and compare them against their unplugged calculations!
- **Creative Writing**: Have students write the next 3 sentences of Milo's adventure using their newly selected word.
- **Mathematics**: Explore what happens to candidate rankings when students create an *Action-Heavy* profile ($\alpha_4 = 0.50$) or a *Character-Heavy* profile ($\alpha_2 = 0.50$).
