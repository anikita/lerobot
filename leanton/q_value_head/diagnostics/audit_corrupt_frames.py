#!/usr/bin/env python3
"""
Run AFTER 1_extract_features.py completes.
Counts corrupt frames by diffing expected vs actual frame count,
and checks which rounds are affected.
"""
import torch
from pathlib import Path

OUTPUT = Path(__file__).parent / "cached_hidden_states.pt"
ROUND_SIZES = [
    788, 1457, 1667, 7812, 9932, 6192, 2702, 1870, 2142,
    20955, 3351, 5753, 6862, 7300, 5028, 22570, 3118, 3703,
    4125, 4216, 6344,
]
EXPECTED = sum(ROUND_SIZES)

data = torch.load(OUTPUT, weights_only=True, map_location="cpu")
actual = data["hidden_states"].shape[0]
missing = EXPECTED - actual

print(f"Expected frames: {EXPECTED:,}")
print(f"Actual frames:   {actual:,}")
print(f"Missing (corrupt): {missing:,} ({missing/EXPECTED*100:.2f}%)")

# Per-round audit
print(f"\nPer-round breakdown:")
cumulative = 0
for i, size in enumerate(ROUND_SIZES):
    r = i + 1
    start = cumulative
    end = cumulative + size
    round_indices = data["round_indices"]
    actual_in_round = ((round_indices == i).sum().item())
    missing_in_round = size - actual_in_round
    if missing_in_round > 0:
        print(f"  r{r:02d}: expected {size:,}, got {actual_in_round:,} — MISSING {missing_in_round:,} ({missing_in_round/size*100:.1f}%)")
    cumulative += size

# Check Q-target coverage
q = data["q_targets"]
n_nonzero = (q.abs() > 0.05).sum().item()
print(f"\nNonzero Q frames in result: {n_nonzero:,}")
print(f"(Original dataset had 17,507 nonzero Q frames)")
