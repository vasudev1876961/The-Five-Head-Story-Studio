# Student Activity Worksheet: The Five-Head Story Machine

**Name:** ___________________________ **Date:** ______________ **Class / Group:** ____________  
**My Assigned Role:** __________________________________ (e.g., Grammar Scout, Repeat Blocker)

---

## 🎯 Part 1: Meet the Five AI Helpers

Draw a line connecting each AI Helper to its mission and question:

| AI Helper | Mascot | What It Cares About |
|---|---|---|
| **1. Grammar Scout** | 👤 Guardian | Keeps track of who the hero is |
| **2. Character Keeper** | ⚡ Runner | Makes sure words fit natural language rules |
| **3. Setting Painter** | 🛡️ Shield | Injects excitement with dynamic action verbs |
| **4. Action Booster** | 🦉 Owl | Keeps the world in the same castle or forest |
| **5. Repeat Blocker** | 🏰 Artist | Stops the story from repeating words in loops |

---

## 📝 Part 2: Score the Candidate Words!

The story so far:
> **"Milo entered the ____"**

Your team must score the five candidate words below using your helper's rules.

### Helper Scoring Rules:
- **🦉 Grammar Scout**: Smooth continuation = `0.80 - 0.90`, Verb = `0.65`, Awkward = `0.20`
- **👤 Character Keeper**: Active hero (Milo) = `0.85`, Immediate duplicate ("Milo Milo") = `0.10`, Non-character = `0.20`
- **🏰 Setting Painter**: Matches scene (forest) = `0.85`, Other setting = `0.40`, Non-setting = `0.20`
- **⚡ Action Booster**: Dynamic action verb = `0.90`, State/Sense verb = `0.50`, Non-action = `0.20`
- **🛡️ Repeat Blocker**: Safe word = `1.00`, Fresh content word = `0.80`, Immediate duplicate word = `0.05`

### Complete the Team Scores:
*(Fill in the scores assigned by each helper)*

| Candidate Word | 🦉 Grammar Scout | 👤 Character Keeper | 🏰 Setting Painter | ⚡ Action Booster | 🛡️ Repeat Blocker |
|---|:---:|:---:|:---:|:---:|:---:|
| `forest` | 0.90 | 0.40 | 0.95 | 0.30 | 0.85 |
| `castle` | 0.85 | 0.35 | 0.90 | 0.25 | 0.85 |
| `garden` | 0.80 | 0.40 | 0.85 | 0.30 | 0.85 |
| `Milo` | 0.15 | ______ | 0.10 | 0.10 | ______ |
| `ran` | 0.30 | 0.20 | 0.10 | ______ | 0.90 |

---

## 🧮 Part 3: Calculate the Fused Score!

The class is using the **Balanced Profile**:
- Grammar Weight ($\alpha_1$) = **0.40**
- Character Weight ($\alpha_2$) = **0.15**
- Setting Weight ($\alpha_3$) = **0.15**
- Action Weight ($\alpha_4$) = **0.10**
- Repeat Weight ($\alpha_5$) = **0.20**

$$\text{Fused Score} = (0.40 \times H_1) + (0.15 \times H_2) + (0.15 \times H_3) + (0.10 \times H_4) + (0.20 \times H_5)$$

### 1. Calculate Candidate `forest`:
- Grammar: $0.40 \times 0.90 =$ ________
- Character: $0.15 \times 0.40 =$ ________
- Setting: $0.15 \times 0.95 =$ ________
- Action: $0.10 \times 0.30 =$ ________
- Repeat: $0.20 \times 0.85 =$ ________
- **Total Fused Score for `forest`** = ________________________

### 2. Calculate Candidate `Milo`:
- Grammar: $0.40 \times 0.15 =$ ________
- Character: $0.15 \times 0.95 =$ ________
- Setting: $0.15 \times 0.10 =$ ________
- Action: $0.10 \times 0.10 =$ ________
- Repeat: $0.20 \times 0.05 =$ ________
- **Total Fused Score for `Milo`** = ________________________

### 3. Which word won?
Winning word: **________________________** with a fused score of **________________________**!

The story now reads:
> **"Milo entered the ________________________"**

---

## 🎲 Part 4: The 10-Sided Die (d10) Challenge!

We want to play this unplugged using a **10-sided die (d10)**!  
Convert the weights into 10 physical die faces using Hamilton's largest-remainder method:

1. **Calculate the Quotas** (Weight $\times$ 10):
   - Grammar: $0.40 \times 10 =$ **4.0**
   - Character: $0.15 \times 10 =$ **1.5**
   - Setting: $0.15 \times 10 =$ **1.5**
   - Action: $0.10 \times 10 =$ **1.0**
   - Repeat: $0.20 \times 10 =$ **2.0**

2. **Give Out Base Tokens** (Whole numbers):
   - Grammar: _____ tokens
   - Character: _____ token
   - Setting: _____ token
   - Action: _____ token
   - Repeat: _____ tokens
   - *Total given out:* _____ tokens (Out of 10)

3. **Surplus Token**:
   - How many tokens are left to reach 10? _____ token.
   - Character has a remainder of **0.5**. Setting has a remainder of **0.5**.
   - By our tie-break rule, who gets the extra token? ________________________

4. **Draw or Color Your 10-Sided Die Faces**:
   Color or label the 10 die face boxes below:
   `[ 1 ]` `[ 2 ]` `[ 3 ]` `[ 4 ]` `[ 5 ]` `[ 6 ]` `[ 7 ]` `[ 8 ]` `[ 9 ]` `[ 10 ]`

---

## 🔍 Part 5: Spot the Repetition Trap!

Read this bad AI story:
> *"Pip flew near the crystal castle. Pip flew near the crystal castle. Pip flew near the crystal castle."*

1. What phrase is repeating over and over? __________________________________________________
2. Which AI helper was asleep? __________________________________________________
3. How does that helper stop this from happening?
   ___________________________________________________________________________________________

---

## 💡 Part 6: Reflection Questions

1. Why didn't `Milo` win the next-word contest, even though Milo is the hero of the story?
   ___________________________________________________________________________________________
   ___________________________________________________________________________________________

2. If you were writing a high-speed chase scene, which helper would you give the most power (weight) to? Why?
   ___________________________________________________________________________________________
   ___________________________________________________________________________________________

3. Why is an AI with five cooperative helpers better than an AI that only looks at grammar?
   ___________________________________________________________________________________________
   ___________________________________________________________________________________________
