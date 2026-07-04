# 3. Temporal TCN Q-Head Plan

## Goal

Replace single-frame mean-pooled features with a **90-frame temporal window** of mean-pooled suffix features `[B, 90, 720]`, processed by a causal TCN that learns velocity/acceleration patterns across the feature trajectory. Pure Q regression — no reconstruction.

## Why suffix (720-dim) over prefix (960-dim)

Lower dimensionality → smaller temporal windows. The diagnostic showed suffix has marginally better class separation (0.004 gap vs 0.001 for prefix) despite still being collapsed per-frame. The trajectory across frames may amplify this.

## Files

| File | Action |
|:-----|:-------|
| `model.py` | **Add** `TemporalTCNQHead` class (~60 lines) |
| `2_train_q_head_v3.py` | **New file** (~400 lines, clean rewrite) |

Existing files untouched — v2 is preserved.

---

## 1. `TemporalTCNQHead` — model.py addition

```
Input:  [B, T, D]     T=90 temporal window, D=720 (suffix) or 960 (prefix)

1. Input Proj:   Linear(D → 256)                            → [B, T, 256]
2. Transpose:    [B, 256, T]   (Conv1d channel-first)
3. 5× Causal Conv1d blocks, kernel=5, channel dim=256:
   ┌────────────────────────────────────────────────────────┐
   │ Block i:  dilation = 2^i                               │
   │   CausalPad(left = (k-1)*dilation, right = 0)          │
   │   Conv1d(256 → 256, k=5, dilation)                     │
   │   BatchNorm1d(256)                                     │
   │   ReLU                                                 │
   │   Dropout(0.1)                                         │
   └────────────────────────────────────────────────────────┘
   dilations: [1, 2, 4, 8, 16]
   Receptive field: Σ(k-1)*d_i = 124 frames  (covers T=90 ✓)

4. Global mean pool over T:  mean(dim=-1)                    → [B, 256]
5. MLP head:   256 → 64 → ReLU → Dropout(0.1) → 1 → tanh    → [B, 1]

~0.3M params
```

**Why causal?** Frame t sees only frames ≤ t. At inference, buffer 90 frames, predict Q for current frame. Real-time compatible.

**Why BatchNorm not LayerNorm?** Natural fit for `[B, C, T]` Conv1d layout. Batch=64 is large enough for stable statistics.

**Why sinusoidal position encoding is NOT needed?** Convolutions are translation-equivariant by design — the same kernel applied at every temporal position. The model learns patterns like "feature X increasing over 10 frames" regardless of where in the window that happens. Unlike transformers which need explicit position info to know frame order, convolutions have temporal locality built in.

## 2. `2_train_q_head_v3.py` — clean rewrite

### Data loading

1. Load `features_v3/{round}_with_q_suffix.pt` → field `"suffix_states"`, shape `[N, 720]`
2. All in RAM (12 rounds × ~7 MB avg = ~80 MB total)
3. Pre-pad each round with T-1=89 zero frames at the start → first real frame has 90-frame history (89 zeros + itself)
4. Track round boundaries for window extraction — windows NEVER cross rounds
5. CLI `--val` specifies held-out rounds (default: r5, r18)

### Window sampling (balanced)

- Map every frame index → `(round_idx, local_frame_idx)`
- Classify frames: Q < -0.05 = neg, Q > 0.05 = pos, |Q| ≤ 0.05 = neu
- Per batch: sample N/3 neg, N/3 pos, N/3 neu frame indices
- For each frame at local position `i` in round `r`: extract window `[i-T+1 : i+1, :]` → shape `[T, 720]`
- Stack → `[B, T, 720]`, shuffle batch order

### Validation

- Batch-process val rounds (same pattern as v2) — no pre-extraction
- Process val frames in batches of 64, build windows on the fly

### Training loop

Same structure as v2:
- Adam + CosineAnnealingLR
- Balanced sampling per epoch
- Metrics: neg MSE, pos MSE, neu MAE, intervention point-biserial correlation
- Print epoch timing, loss breakdown
- Same plotting functions

### CLI

```bash
python 2_train_q_head_v3.py --model tcn --temporal-window 90 \
    --features-dir features_v3 --feature-field suffix \
    --val r5,r18 --epochs 200 --batch-size 64 --lr 1e-4
```

`--feature-field suffix` → loads `_suffix.pt` (720-dim).
`--feature-field prefix` → loads `.pt` mean-pooled (960-dim). Allows comparison.

### Memory budget

- Data in RAM: ~80 MB (12 rounds of suffix)
- Training windows per batch: 64 × 90 × 720 × 4 bytes = ~16 MB
- Model: ~0.3M × 4 bytes = ~1.2 MB
- Total GPU: <100 MB

## 3. What this tests

The TCN's conv kernels are **learned discrete derivative operators**. A kernel like `[-0.2, -0.1, 0.0, +0.1, +0.2]` detects features trending upward over 5 frames. Stacked with dilations, higher layers detect velocity and acceleration patterns at multiple timescales.

If the feature *trajectory* carries failure information that the feature *point* does not — if "the suffix features are drifting in a particular direction across the last 90 frames" is the signal — the TCN will find it. If even the trajectory is collapsed (all frames project to the same point, so there's no movement to detect), the conv kernels will have nothing to differentiate and we get the same ~1.0× null result. That would confirm we need Path A (earlier layers).
