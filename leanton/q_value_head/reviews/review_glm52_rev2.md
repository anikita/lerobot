# Review rev2 — Verifying v6 Fixes and Responding to the Rebuttal

**Reviewer:** roboticist / ML reviewer (GLM-5.2, rev2)
**Date:** 2026-06-30
**Inputs:** `2_train_q_head_v6.py`, updated `model.py`, updated `build_q_targets.py`, and `review_glm52_rev1_rebuttal.md`.

---

## 0. Net Position

The v6 fixes are correct and carefully implemented; I verified each one against the diff. The rebuttal is strong and changed my mind on one point (the clock-vs-state distinction). But it contains one internal contradiction on the SigLIP evidence that, once corrected, actually *weakens* the "features are the bottleneck" attribution and *strengthens* the "run the controls before concluding" recommendation. We have converged on the right next experiments either way.

---

## 1. Fix Verification (v6)

I read the v5→v6 diff and the updated `model.py` / `build_q_targets.py`. Status:

| # | Fix | Verified | Notes |
|:--|:----|:--------:|:------|
| 1 | `hidden_dim` guarded init | Yes | `if hidden_dim is None: hidden_dim = h.shape[1]` then `elif h.shape[1] != hidden_dim`. First file sets, rest compare. Correct. Loader now runs without `--dead-zone`. |
| 2 | Dead zone true exclusion | Yes | `dead_mask` tracked per round, applied via `alive = ~dead_mask` to flat train arrays, target-frame skip in window builder, and val masks. Frames are now genuinely removed from training and metrics, not relabeled. Correct. |
| 3 | TCN `pool="last"` default | Yes | `model.py:412` and CLI default. Matches transformer readout. |
| 4 | `BatchNorm1d` → `LayerNorm` | Yes | `CausalConvBlock` uses `nn.LayerNorm(channels)` with correct `[B,C,T]→[B,T,C]` transpose so normalization runs over the channel dim. No train/eval stats drift. Correct. |
| 5 | RLTQHead 110M → corrected | Yes (README) | Per rebuttal. |
| 7 | Temporal models error on `T<2` | Yes | `causal_tf`/`tcn`/`transformer` now refuse `T<2` instead of silently building an MLP. |
| 8 | TCN docstring 124→125 | Yes | `model.py:397`. |
| 9 | `min_length=1` | Yes | `build_q_targets.py:51`. Now aligns with the loader's raw rising-edge onset detection, so the two scripts agree on what an onset is. Good side effect. |

### One residual inconsistency in v6 (minor, new)

The within-round holdout metric block does **not** apply the dead-zone exclusion, while train and val do. At `2_train_q_head_v6.py:782-795`:

```python
hold_h = torch.cat([th["h"] for th in train_holdout], dim=0)
hold_q = torch.cat([th["q"] for th in train_holdout], dim=0)
...
hn = hold_q < -0.05; hpos = hold_q > 0.05; hneu = ~hn & ~hpos
```

The holdout dict *does* carry `dead_mask` (line 432) but the masks `hn/hpos/hneu` never intersect with `~dead_mask`. So holdout MSE is computed over dead-zone frames (whose `q` was zeroed), while train/val exclude them. This makes the holdout numbers not directly comparable to the val numbers, and it also means the holdout null is contaminated by relabeled boundary frames, the exact thing fix #2 was meant to remove.

Suggested fix:

```python
hold_dm = torch.cat([th["dead_mask"] for th in train_holdout], dim=0)
hold_alive = ~hold_dm
hn  = (hold_q < -0.05) & hold_alive
hpos = (hold_q >  0.05) & hold_alive
hneu = hold_q.abs().le(0.05) & hold_alive
```

Low severity, but worth closing so all three evaluation surfaces use one definition of "eligible frame."

### One still-open evidence gap (not in the bug list, not addressed in v6)

Rev1 §3.6/§4 flagged that the cosine-similarity collapse diagnostic is compromised if computed on **mean-pooled** features, because pooling heterogeneous tokens (vision + language + state) toward their mean mechanically inflates cosine similarity. The rebuttal leans on the SigLIP linear probe instead, but never actually runs a per-token cosine or a VLM prefix/suffix linear probe under the contiguous holdout. The SigLIP probe is a different feature source, not a fix for the VLM diagnostic. This remains open: before citing "0.97 to 0.99 cosine similarity" as proof the VLM features are collapsed, recompute it per token (or at least on the state token alone) and run a VLM linear probe. You already built `estimate_ceiling.py`; point it at the VLM features.

---

## 2. Engaging the Four Disagreements

### Disagreement 1 (Q is a clock) — I largely concede, with one caveat

Your counter is good and corrects an overstatement in my rev1. The distinction between *how the label is generated* (a clock from onsets) and *what the model must learn* (a state→Q mapping) is real. A model that merely memorized a per-episode clock would not transfer, and the fact that the frame-position shortcut hits 1.0× on the contiguous holdout does show that "transferable clock" is not available either. So you are right that I should not have framed Q as "merely a clock that any encoder can be excused for failing." The learning problem is state→Q, and failing it is meaningful. I retract that framing.

**The caveat I will not concede:** the contiguous-holdout null does not isolate *which* of three things is true:

1. state is absent from the features,
2. state is present but the frozen features do not encode it,
3. state is present and encoded, but Q is not determined by state alone (because human intent contributes).

All three produce 1.0× on the contiguous holdout. Your rebuttal asserts (1) and (3)-is-false, but (1) is the quantity under test and (3) is the label-proxy concern. The physical-failure-is-causally-upstream prior (gripper slip → human perceives → human intervenes) is a reasonable argument that onsets carry state-correlated signal, and I gave you that in rev1. But it is a *prior*, not a measurement. It raises the prior that (1) is the culprit; it does not settle it. So: concede the framing, hold the line on "the holdout null underdetermines the attribution."

### Disagreement 2 (onset anchoring, +Q/−Q) — agree to agree

(a) Onset as the least-bad observable boundary, with peaking-slightly-late being conservative for an early-warning system: I fully accept. That was a caveat in rev1, not an objection.

(b) +Q and −Q as two sides of the same transition, learned by one distance-to-boundary mechanism: reasonable, and empirically testable. One caveat: you cite the symmetric neg/pos improvement under the dead zone as support. But rev1 §2.2 and your own §6.1 concede that the v5 dead-zone "improvement" was partly a relabeling artifact. v6 re-runs with true exclusion have not been reported yet. So the symmetry-under-dead-zone evidence is currently *pending*, not confirmed. Re-run v6 first, then cite symmetry.

### Disagreement 3 (gate metric) — converged

Your multi-metric point is correct and refines my rev1. A constant negative bias `c` improves `val_neg` but worsens `val_pos`, so the joint (neg, pos, null) pattern does distinguish "learned Q" from "learned an offset." I was too quick to call the ratio gameable in isolation; embedded in the three-metric panel it is much less so. We agree: lead with the R² ceiling and a rank AUC, keep the ratio as a first-pass sanity check. Conceded.

### Disagreement 4 (narrowing the claim) — converged on the claim, one sharp pushback

You adopt the narrower claim and commit to the raw-state and outcome-label controls. That is exactly the right outcome and I accept it.

**The pushback is on your strongest leg, the SigLIP argument, and it contains an internal contradiction.**

Rebuttal §5.2 says, verbatim: the SigLIP linear probe "also shows *no in-sample signal either* (1.6× in-sample, 1.0× on holdout)" and then concludes "the complete absence of even in-sample signal from raw vision suggests the physical state visible to the camera genuinely doesn't determine Q."

That conclusion does not follow from your own number. **1.6× in-sample is signal.** It is weak, and it does not generalize, but 1.6× > 1.0× means a linear readout on raw pixels does beat the null within-episode. "No in-sample signal" and "1.6× in-sample" cannot both be true. Which reading is correct matters:

- If the true in-sample result were ~1.0× (no signal at all), your argument holds: the camera genuinely does not see Q, full stop.
- At 1.6× in-sample collapsing to 1.0× on holdout, the correct interpretation is the opposite of "absence." It is the textbook signature of a **real but weak, non-transferable signal**: the pixels contain *some* state information correlated with Q, but per-episode factors (the operator's specific intervention timing, anticipation horizon, patience) dominate and do not transfer. That is precisely the label-proxy / human-intent picture I raised in rev1, and it is consistent with your own §6.5 sentence "human intent is not in the camera."

So the SigLIP result, read honestly, is **mild evidence against** "the camera cannot see Q at all" and **mild evidence for** "there is a weak state signal drowned by non-transferable human-timing variance." That re-weights the attribution away from (1) "state absent" and toward (3) "Q is not determined by state alone." It does not overturn your negative result; it relocates the likely cause.

This is why the two controls matter more than the rebuttal suggests:

- **Raw-state control (joint angles + object pose):** if low-dimensional state *also* shows the 1.6×-in-sample / 1.0×-holdout pattern, that confirms the signal is real but non-transferable, and pins the cause on the human-intent variance (label), not on the features. If raw state generalizes where pixels do not, the VLM is the bottleneck after all.
- **Outcome-based Q:** if relabeling failure from task outcome (instead of takeover) makes the in-sample signal *transfer*, that is direct evidence the takeover proxy was injecting the non-transferable variance. If it makes no difference, the features were genuinely the limit.

Either control settles the attribution your rebuttal currently asserts by prior. Run them before promoting "features are the bottleneck" from hypothesis to conclusion.

---

## 3. What I Changed My Mind On

1. **The clock framing was too strong.** Your state-vs-clock distinction is correct. The contiguous holdout is a legitimate test of state→Q transfer, not just of clock memorization. Rev1 overstated this.
2. **The ratio is less gameable than I implied** when read inside the three-metric panel. R²/AUC still preferred, but the ratio is a fair first-pass gate.
3. **The dead-zone "exclusion" was the right thing to fix**, and v6 fixed it correctly. The v5 "1.05× breakthrough" should be treated as unconfirmed until v6 re-runs.

## 4. What I Still Want Before Signing Off

1. **Re-run v6 and report honest dead-zone numbers.** The symmetry argument and the 1.05× both depend on it.
2. **Fix the holdout dead-mask inconsistency** (§1 residual).
3. **Run a VLM prefix/suffix linear probe under the contiguous holdout**, and recompute cosine per token. Close the VLM-side evidence gap rather than borrowing the SigLIP probe.
4. **Correct the SigLIP wording** in the rebuttal/README: 1.6× in-sample is weak signal, not "absence of signal." The non-generalization is the finding, not the absence.
5. **Run the raw-state and outcome-label controls.** These convert "Q-from-takeover is not learnable from frozen features" into a real attribution. Until then, the narrower claim is the honest headline.

## 5. Bottom Line

The code is now in good shape and the evaluation is honest. The remaining disagreement is narrow and empirical: I read your SigLIP 1.6× as "weak non-transferable signal, label-proxy likely," you read it (in the rebuttal's prose) as "no signal, features innocent." The data itself, once the contradiction is resolved, will probably read closer to my interpretation, but that is exactly what the two controls will settle. We agree on what to do next; the only open question is whether the conclusion you want to publish is earned before or after those controls. My position: after.
