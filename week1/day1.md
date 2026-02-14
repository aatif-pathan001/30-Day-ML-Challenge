# Day 1 - February 14, 2026

## 🎯 Goals for Today
- [x] Create 3 GitHub repositories
- [x] Understand RNN theoretical foundations
- [x] Solve 3 LeetCode problems
- [x] Document learnings

## 📚 What I Learned

### RNN (Recurrent Neural Networks)

**Core Concept:**
RNNs process sequential data by maintaining a "hidden state" that acts as memory. Unlike feedforward networks, RNNs have connections that loop back, allowing information to persist.

**Key Points I Understood:**
1. **Hidden State**: Acts as the network's memory, updated at each time step
2. **Sequential Processing**: Processes input one element at a time (character by character, word by word)
3. **Parameter Sharing**: Same weights used across all time steps (efficient!)
4. **Vanishing Gradient Problem**: Main limitation - RNN struggles with long sequences because gradients become very small during backpropagation

**Formula (simple form):**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y

Where:
- h_t = hidden state at time t
- x_t = input at time t
- y_t = output at time t
- W = weight matrices
- b = bias terms
```

**When to use RNN:**
- Text generation
- Sentiment analysis
- Time series prediction
- Any sequential data where order matters

**Limitations:**
- Vanishing/exploding gradients
- Difficulty learning long-term dependencies
- Training can be slow

**Solution → LSTM **

### Resources I Used:
- my own notes from 1/4th lab lectures.
- PyTorch documentation

**Analogies that helped me:**
- Hidden state = short-term memory
- Like reading a sentence - you remember previous words to understand current word
- Each time step is like one frame in a video sequence

---

## 💻 LeetCode Problems Solved

### Problem 1: Two Sum
**Difficulty**: Easy
**Time**: 15 minutes
**Approach**: Hash map for O(n) solution
**Key Learning**: Trading space for time complexity

**My Solution Logic:**
```
Store seen numbers in dictionary: {number: index}
For each number, check if (target - number) exists in dictionary
If yes, return both indices
```

**Mistakes I made:**
- Initially tried brute force (nested loops) and used lists. - O(n²)
- Realized hash map is faster and makes it O(n)

**Time Complexity**: O(n)
**Space Complexity**: O(n)

---

### Problem 2: Valid Anagram
**Difficulty**: Easy
**Time**: 12 minutes
**Approach**: Character frequency counting

**Key Learning**: Python's Counter is very useful for frequency problems

**My Solution Logic:**
```
Count frequency of each character in both strings
Compare if frequencies match
```

**Time Complexity**: O(n)
**Space Complexity**: O(1) - only 26 possible characters

---

### Problem 3: Contains Duplicate
**Difficulty**: Easy
**Time**: 8 minutes
**Approach**: Set comparison

**Key Learning**: Set operations are powerful and efficient

**My Solution Logic:**
```
Convert list to set (removes duplicates)
Compare lengths - if different, duplicates exist
```

**Time Complexity**: O(n)
**Space Complexity**: O(n)

---

## 🏗️ Projects Created


---

## 💪 Challenges Faced

### Challenge 1: Understanding backpropagation through time
**Problem**: Initially confused about how gradients flow backward through the sequence
**Solution**: Drew diagrams showing unrolled RNN, understood it's like a very deep network
**Time spent**: 30 minutes

### Challenge 2: LeetCode - Valid Anagram
**Problem**: First approach was converting the string to a list and Using inbuilt python sort function.
**Solution**: Realized I can use hash map to track seen char and remove back to empty it.
**Learning**: Always think "can I store something to avoid nested loops?"

---

## 📈 Progress Tracking

**Week 1 Goals:**
- [x] Day 1: RNN theory ✓
- [ ] Day 2: LSTM implementation
- [ ] Day 3: Model optimization
- [ ] Day 4: Regularization
- [ ] Day 5: Transfer learning
- [ ] Day 6: Transfer learning projects
- [ ] Day 7: Week consolidation + blog post

**Skill Development:**
- RNN/LSTM: 20% → understanding theory ✓
- LeetCode: 3/150 problems solved
- GitHub: 3 repositories created

---

## 💭 Reflections

**What went well:**
- Created all 3 repositories as planned
- Understood RNN concepts clearly
- Solved LeetCode problems within time limits
- Maintained focus and discipline

**What could be better:**
- Need to start coding RNN implementation (will do tomorrow)
- Could have read more about LSTM today
- Should set up better note-taking system

**Energy Level**: 8/10 - Excited and motivated!

**Time Spent Today**: ~5 hours
- Theory: 2 hours
- LeetCode: 1 hour
- Repository setup: 1 hour
- Documentation: 1 hour

---

## 🔥 Motivation Check

**Why am I doing this:**
To transition into an ML/DL Engineer role and build a career in AI.

**What I'm grateful for:**
- Clear roadmap to follow
- Structured learning plan
- Supportive guidance

**Quote for today:**
"The expert in anything was once a beginner." - Keep pushing forward!
