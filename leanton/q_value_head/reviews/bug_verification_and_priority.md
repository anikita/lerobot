# Bug Verification Report — Reviewer GLM-5.2 Rev 1

**Date:** 2026-06-30
**Verdict:** All 12 claims confirmed. No false positives. The review was thorough and accurate on every factual claim.

## Priority List

| # | Priority | Severity | Issue | Affects Conclusion? | Why / Why Not |
|:--|:---------|:---------|:------|:--------------------|:--------------|
| 1 | **P0** | 🔴 HIGH | `hidden_dim` crash without `--dead-zone` (v5:328-356) | **No** | Prevents running `--dead-zone 0`, but all v5 experiments used `--dead-zone > 0`. Doesn't invalidate results, just means nobody could test without dead zone. |
| 2 | **P0** | 🟡 MEDIUM | Dead zone relabels boundary frames as neutral instead of truly excluding them (v5:352,509,541) | **Yes — inflates breakthrough** | The 1.05× improvement is partly explained by: (a) hard boundary frames (|Q|≈1.0) relabeled to 0, making neg/pos subsets easier; (b) neutral class inflated with ambiguous frames. The anti-correlation breaking might still be real, but the magnitude is overstated. |
| 3 | **P1** | 🟡 MEDIUM | Trailing A-segments forced to Q=0 assumes every episode ends in success (build_q_targets:121) | **Yes — conservative bias** | If any episode ends in failure with a trailing A-segment, those frames are strong-negative-Q mislabeled as neutral. This contaminates the null baseline with failure states. A model that correctly predicts negative Q on those frames gets *penalized*. This biases toward the null baseline, making the negative result **conservative** — fixing it could reveal signal. |
| 4 | **P1** | 🟡 MEDIUM | Cosine collapse computed on mean-pooled features — pooling induces similarity | **Partially** | One of three evidence pillars is weakened. But (a) SigLIP features (no VLM pooling) also fail, (b) architecture-agnostic failure holds without cosine, (c) frame position shortcut is independent of pooling. The conclusion likely holds even with this gap. |
| 5 | **P2** | 🟡 MEDIUM | RLTQHead param count: 3.3M actual vs 110M in README table | **No** | The capacity range narrows from "961K–110M" to "0.5M–3.3M", which actually *strengthens* the argument (all architectures fail within a tighter capacity band). The table number is wrong but the conclusion is unaffected. |
| 6 | **P2** | 🟡 MEDIUM | TCN `pool="mean"` dilutes last-position readout (13.5× less sensitive than `pool="last"`) | **No** | The TCN was configured suboptimally, but the single-frame MLP (no temporal context, no pooling issue) also fails at 1.0×. The negative result doesn't depend on the TCN's performance. |
| 7 | **P2** | 🟡 MEDIUM | `BatchNorm1d` train/eval mismatch under balanced sampling | **No** | Could only hurt the TCN's numbers. Fixing it would make TCN look slightly better, not worse. Doesn't change the conclusion. |
| 8 | **P3** | 🟢 LOW | `T==1` silently routes all models to MLP (v5:882) | **No** | All v5 temporal experiments used T≥6. No experiment was accidentally downgraded. |
| 9 | **P3** | 🟢 LOW | `min_length=2` drops 1-frame correction blips, contaminates adjacent labels | **Unlikely** | If 1-frame blips are rare (likely in 30Hz data), impact is negligible. The dead zone (±10 frames) would mask most affected frames anyway. |
| 10 | **P3** | 🟢 LOW | Onset detection duplicated and inconsistent (build_q_targets vs loader) | **Unlikely** | Difference is at most a few frames at the boundary, and dead zone (±10) absorbs the discrepancy. |
| 11 | **P3** | 🟢 LOW | TCN docstring says 124, property computes 125 | **No** | Cosmetic. |
| 12 | **P3** | 🟢 LOW | Hard `break` at |Q|<0.05 creates label step discontinuity | **No** | Step is 0.05 magnitude; MSE is insensitive. Dead zone masks tail frames anyway. |

## Impact on the Core Claim

**The negative result survives**: Q-from-intervention-onset is not learnable from frozen SmolVLA features. None of the confirmed bugs could plausibly flip this — the single-frame MLP (no temporal issues, no pooling, no BatchNorm, no dead zone) converging to 1.0× on contiguous holdout is the decisive observation, and it is immune to every bug on this list.

**What changes:**
- The dead-zone "breakthrough" (1.05×) is downgraded from "first symmetric improvement" to "expected consequence of relabeling hard frames." The anti-correlation breaking is still interesting but the README framing is wrong.
- The capacity table (961K–110M) is corrected to (0.5M–3.3M). The conclusion is unchanged.
- The trailing-A assumption needs verification. If episodes end in failure with trailing A-segments, the null baseline is contaminated — which means our negative result is *conservative* (the true signal would be harder to miss if we fixed the labels).

## Fix Plan for v6

| Fix | Bug# | What | Effort |
|:----|:-----|:-----|:-------|
| Guard `hidden_dim` init | #1 | `elif hidden_dim is None:` before dim check | 1 line |
| True dead-zone exclusion | #2 | Skip dead-zone frames in window builder + val, not relabel | ~10 lines |
| TCN pool="last" default | #3 | Change default + add `--tcn-pool` CLI flag | ~5 lines |
| LayerNorm in TCN | #4 | Replace BatchNorm1d → LayerNorm in CausalConvBlock | 1 line |
| Correct README param table | #5 | Measure all models, update table | ~5 lines |
| VLM linear probe | #6 | Run linear probe on prefix/suffix features (separate script) | ~20 lines |
| Error on T<2 for temporal | #7 | Raise ValueError for causal_tf/tcn with T<2 | 3 lines |
| Fix TCN docstring | #8 | 124 → 125 | 1 character |
| min_length=1 | #9 | Lower filter or merge sub-min segments | ~5 lines |
| Shared onset detection | #10 | Extract `detect_onsets()` function, import in both | ~10 lines |
| Verify trailing-A assumption | #12 | Check episode outcomes, tag trailing A-segments | Investigation |
