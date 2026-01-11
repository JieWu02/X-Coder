# Dual-Verify Accuracy Reports

## Test Output Labeling Accuracy

### Overall Accuracy

| n (# solutions) | Test Output Labeling Acc |
|:---------------:|:------------------------:|
| 4 | 91.75% |
| 8 | 92.08% |
| 16 | 92.47% |

### Accuracy by Source

| Source | n=4 | n=8 | n=16 |
|:------:|:---:|:---:|:----:|
| atcoder | 94.75% | 95.00% | 96.61% |
| codechef | 92.80% | 92.80% | 92.80% |
| codeforces | 94.44% | 94.81% | 95.06% |

## Golden Solution Accuracy

### Overall Performance Summary

| n | Avg Pass Rate | Full Pass Count | Full Pass Rate |
|:-:|:-------------:|:---------------:|:--------------:|
| 4 | 91.79% | 421/500 | 84.20% |
| 8 | 92.15% | 425/500 | 85.00% |
| 16 | 92.50% | 429/500 | 85.80% |

### Performance by Difficulty (Avg Pass / Full Pass)

- EASY (215): n=4 92.37% / 88.84%, n=8 92.37% / 88.84%, n=16 92.37% / 88.84%
- MEDIUM (73): n=4 94.73% / 89.04%, n=8 94.73% / 89.04%, n=16 94.73% / 89.04%
- HARD (94): n=4 88.14% / 73.40%, n=8 89.20% / 77.66%, n=16 88.94% / 77.66%
- MEDIUM_HARD (106): n=4 92.78% / 83.96%, n=8 93.54% / 83.96%, n=16 95.42% / 87.74%
- VERY_HARD (12): n=4 83.33% / 58.33%, n=8 83.33% / 58.33%, n=16 83.33% / 58.33%

### Performance by Source (Avg Pass / Full Pass)

- atcoder (59): n=4 95.00% / 89.83%, n=8 95.00% / 89.83%, n=16 96.53% / 93.22%
- codechef (25): n=4 92.80% / 52.00%, n=8 92.80% / 52.00%, n=16 92.80% / 52.00%
- codeforces (402): n=4 94.45% / 88.31%, n=8 94.90% / 89.30%, n=16 95.11% / 89.80%

### Key Findings

1. Increasing the number of sampled solutions improves overall pass rates, with diminishing returns beyond n=16.
2. MEDIUM_HARD benefits the most from higher n (95.42% avg, 87.74% full pass at n=16).
3. Golden selection achieves 84-86% full-pass success over ground-truth tests.
