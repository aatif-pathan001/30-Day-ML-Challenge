# Day 2 - February 19, 2026

## 🎯 Goals for Today
- [x] Deep understanding of optimizers
- [x] Working OptimizedLSTM with all techniques
- [x] Model better than Day 2 (lower loss)
- [x] 3 Medium LeetCode problems solved
- [x] Document learnings

## 📚 What I Learned

## 💻 LeetCode Problems Solved

### Problem 1: Group Anagrams
**Difficulty**: Medium
**Time**: - minutes
**Approach**: Hash map for O(n) solution
**Key Learning**: Trading space for time complexity

**My Solution Logic:**
```
Store seen numbers in dictionary: {number: index}
For each number, check if (target - number) exists in dictionary
If yes, return both indices
```

**Mistakes I made:**
- Initially tried brute force (nested loops) - O(n²)
- Realized hash map makes it O(n)

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
