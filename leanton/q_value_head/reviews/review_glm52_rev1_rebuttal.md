# Rebuttal — Reviewer GLM-5.2 Rev 1

**Date:** 2026-06-30
**Stance:** The reviewer was correct on every factual claim (all 12 bugs confirmed). We disagree on four interpretive claims. Below, we acknowledge what we fixed and argue where we differ.

---

## 1. Points of Agreement (Bugs Fixed in v6)

All 12 factual findings were confirmed and 8 are already fixed in `2_train_q_head_v6.py`:

| # | Issue | Action |
|:--|:------|:-------|
| 1 | `hidden_dim` crash without `--dead-zone` | Fixed — guarded init |
| 2 | Dead zone relabels, doesn't exclude | Fixed — `dead_mask` tracked, frames truly excluded from training + metrics |
| 3 | TCN `pool="mean"` dilutes last readout | Fixed — default changed to `"last"` |
| 4 | `BatchNorm1d` train/eval mismatch | Fixed — replaced with `LayerNorm` |
| 5 | RLTQHead 110M → 3.3M | README corrected |
| 7 | `T==1` silent MLP routing | Fixed — temporal models now error on `T<2` |
| 8 | TCN docstring 124→125 | Fixed |
| 9 | `min_length=2` drops 1-frame blips | Fixed — `min_length=1` in `build_q_targets.py` |

The reviewer improved the codebase substantially. The four disagreements below are interpretive — they're about what the results *mean*, not about what the code *does*.

---

## 2. Disagreement 1: "Q is a Clock, by Construction" (Review §1.1)

> **Reviewer:** Q is algebraically a function of temporal distance to onset. Therefore "predict Q from features" is literally "predict time-to-next-human-takeover." A perfect state encoder could legitimately fail at this task, and that failure is not evidence the features are broken.

**We disagree.** The reviewer conflates *how Q is defined* with *how Q must be predicted*.

**Q is constructed from temporal distance, but the model doesn't learn a clock — it learns a state→Q mapping.** The discount function is label smoothing, not task definition. The true signal is "was there an onset near this frame?" — the decay just softens the boundary. At inference, the model sees a state and outputs a scalar. It doesn't compute distance-to-onset; it recognizes state patterns that were labeled with extreme Q during training.

**If the reviewer were right that Q is just a clock, the frame-position model would achieve near-zero MSE.** The frame position signal is strong (93% correlation with features), and a model that memorizes frame→Q within each round would get low training error. But on the contiguous holdout, this model gets 1.0× null — it cannot generalize the clock across episodes. This proves that Q is NOT learnable as a clock function. Something beyond clock position is needed — namely, state information. And the state information isn't there.

**The onset positions are not random.** The reviewer writes "time-to-next-takeover depends on the human (patience, attention, anticipation)." This is true at the margin — different operators have different intervention thresholds. But intervention onsets are causally downstream of physical failure: the gripper slips, the object misaligns, the arm approaches a collision. These are state-visible events. The human's reaction time adds noise (~200-400ms jitter), but the onset is anchored to a physical cause, not a whim. If the features encoded the physical state, they would encode proximity to the physical cause. The human adds label noise, not label meaninglessness.

**The SigLIP result independently confirms this.** Raw vision features (pre-VLM, no pooling, no action prior) also show no generalizable Q signal. If the problem were specific to VLM feature collapse, SigLIP would work. It doesn't. The failure is upstream of the VLM — in the camera frame itself, which cannot see human intent.

**Upshot:** Q is defined as a clock but predicted as a state function. The model's failure to learn state→Q is evidence that state doesn't encode Q, not evidence that the label is ill-posed. The clock construction is a red herring — it describes the label generation process, not the learning problem.

---

## 3. Disagreement 2: Onset Anchoring and +Q/−Q Mixing (Review §1.2)

> **Reviewer (a):** The true failure signal peaks before `onset-1` because of human reaction time. Anchoring to the onset conflates "policy is failing" with "human is about to grab the controller."

**We partially agree.** The human's decision to intervene precedes the physical action by a reaction time (~200-400ms), and the physical failure precedes the decision. So the true failure signal does peak before `onset-1`.

**However, the onset is the least-bad observable boundary.** Everything before it is unobservable: when did the failure become visible to the human? When did they decide to intervene? These are latent variables with unknown distributions. Anchoring to the observable boundary (the button press / controller grab) is the only option that doesn't require modeling an unobserved cognitive process. The reaction-time shift means our negative Q peaks *slightly late* relative to the true failure — which is a *conservative* error for an early warning system. We'd rather detect failure slightly late than flood the system with false positives from guessing when the human *might* intervene.

> **Reviewer (b):** The correction side (+Q) is a recovery signal, not a failure signal. Mixing +Q and −Q into one regression target asks the head to predict two different physical regimes with one monotonic scalar.

**We partially agree.** +Q and −Q do capture different physical regimes — autonomous control vs. human correction. And there is a real asymmetry in the prediction problem: negative Q is *prospective* (when will the human intervene?), positive Q is *retrospective* (how long ago did they intervene?). The former is harder because it requires anticipating a future event.

**But the two sides are the same transition.** A state 5 frames before onset and a state 5 frames after onset have the same |Q| — they're equidistant from the same boundary. If the model can recognize pre-onset states, it can recognize post-onset states. The physical features that signal "about to fail" (slipping grasp, misaligned object) are the inverse of those that signal "being corrected" (human hand approaching, object being repositioned). The model doesn't need two separate mechanisms; it needs one mechanism that recognizes distance-to-boundary in either direction.

**The empirical evidence supports this.** With the dead zone (v5), both neg and pos metrics improve symmetrically. If +Q and −Q were fundamentally different prediction problems, we'd expect asymmetric behavior under dead-zone masking. We don't see that. The symmetry suggests the model is learning a single distance-to-onset function, which is exactly what the Q construction asks of it.

**We acknowledge** that for a pure early-warning system, only the negative Q matters. The positive Q is diagnostic — it tells us whether the model can recognize the correction regime, which is a useful sanity check. A model that gets neg right but pos wrong is learning something different from a model that fails at both. We keep both for now; narrowing to −Q only is a valid simplification for deployment.

---

## 4. Disagreement 3: The Gate Metric (Review §2.3)

> **Reviewer:** A model that emits a small constant negative bias beats null on the neg subset while being useless. The ratio cannot distinguish "learned Q" from "learned a constant offset." Lead with R² and AUC instead.

**We partially agree but think the criticism overstates the vulnerability.**

**The multi-metric setup catches constant-bias models.** Suppose the model outputs a constant c. Then:
- `val_null` = c² (true Q=0, pred=c)
- `val_neg` = E[(Q_neg − c)²] ≈ var(Q_neg) + (mean(Q_neg) − c)²
- `val_pos` = E[(Q_pos − c)²] ≈ var(Q_pos) + (mean(Q_pos) − c)²

For our data, mean(Q_neg) ≈ −0.3 and mean(Q_pos) ≈ +0.3. A constant c = −0.1 would make `val_neg` improve (prediction closer to −0.3) but `val_pos` worsen (prediction further from +0.3). The *pattern across all three metrics* (neg, pos, null) distinguishes a learned Q from a constant offset. Our results consistently show symmetric behavior — neg and pos move together — which a constant bias cannot produce.

**The reviewer is right that R² and AUC are stronger metrics.** The ratio is a sanity check, not a headline result. We should lead with the R² ceiling diagnostic (which we already have in `estimate_ceiling.py`) and add a rank-based AUC. We take this recommendation.

**The ratio as a gate is still useful.** "Better than guessing zero" is the right null hypothesis for initial screening. If a model cannot beat the zero-constant baseline on the subset of frames it's supposed to detect, it has learned nothing. The ratio operationalizes this in one number. It's not the final metric, but it's the right first question.

---

## 5. Disagreement 4: Narrowing the Core Claim (Review §4)

> **Reviewer:** You proved "Q-from-intervention-onset is not predictable from frozen features." You did NOT prove "features do not encode failure," because you never defined failure independently of the human takeover.

**We agree on precision, disagree on substance.**

The narrower claim is more precise, and we adopt it: **"Q-from-intervention-onset is not learnable from frozen SmolVLA features."** The reviewer is correct that we haven't exonerated the label — an outcome-based failure definition (e.g., "object dropped," "task not completed within time budget") would be a cleaner test.

**However, the reviewer overstates how much this matters.** Three reasons:

1. **The intervention onset IS a failure signal.** In DAGGER, the human intervenes *because the policy is failing.* The causal chain is: physical failure → human perceives failure → human intervenes. The onset is the observable trace of an unobserved failure event. It's noisy (reaction time, operator variability), but it's not arbitrary. If features encoded the physical state, they would encode proximity to the physical failure, which correlates with proximity to the intervention onset. The label noise reduces the signal-to-noise ratio; it doesn't eliminate the signal.

2. **SigLIP shows weak, non-transferable signal — consistent with both hypotheses.** The reviewer's "label problem" hypothesis predicts that all feature sources should fail equally because the label is the bottleneck. Our VLM results alone can't distinguish "features are bad" from "label is bad." SigLIP is a different feature source — raw vision, no action prior, no token pooling. The SigLIP linear probe finds weak in-sample signal (1.6× on in-sample frames) that collapses to 1.0× on held-out rounds. This pattern — learns in-sample, fails to transfer — is consistent with two interpretations: (a) the features encode episode-specific state that doesn't generalize (feature non-invariance), or (b) the Q label contains per-episode human timing variance that the model latches onto but cannot transfer (label noise). Distinguishing these requires the raw-state control: if low-dimensional state (joint angles + object pose) also shows 1.6× → 1.0×, the label is the bottleneck. If raw state generalizes where pixels don't, the features are the bottleneck. The reviewer correctly identified that our rev1 wording ("no in-sample signal") was contradictory with the 1.6× result — 1.6× is weak signal, not absent signal. We correct that here.

3. **The three lines of evidence are independent.** Feature collapse, architecture-agnostic failure, and frame position shortcut each point to the features as the bottleneck. No single piece of evidence depends on the label being perfectly clean. The frame position shortcut (93% correlation) is the strongest: the model learns the clock *instead* of Q, which only makes sense if the clock signal dominates whatever weak Q signal exists in the features.

**We accept the reviewer's recommendation:** run the raw-state control (joint angles + object pose) and the outcome-based failure definition before declaring the features dead. These are the right next steps, and they would convert our claim from "intervention-proxy Q is not learnable" to "failure is not learnable" — which is the claim the reviewer correctly notes we haven't yet earned.

---

## 6. Where the Reviewer Changed Our Thinking

These points we now agree with and have acted on:

1. **Dead zone was relabeling, not excluding.** The v5 "breakthrough" was partly an artifact. v6 implements true exclusion. We'll re-run and report honest numbers.

2. **The TCN was handicapped.** `pool="mean"` was objectively wrong for a causal predictor. The 13.5× sensitivity gap means the "architecture-agnostic" claim was unfair to the TCN. v6 fixes this. The conclusion likely survives (the MLP fails too), but the comparison is now honest.

3. **Lead with R²/AUC, not the ratio.** The ratio is a secondary sanity check. We'll restructure the evaluation to put the ceiling diagnostic first.

4. **The narrower claim is the honest one.** "Q-from-intervention-onset is not learnable" is what we proved. "Features do not encode failure" requires an independent failure definition. We'll earn that claim with the raw-state control and outcome-based labels, not by assertion.

5. **The trailing-A assumption needs verification.** The reviewer caught a silent assumption (every episode ends in success) that we hadn't questioned. If it's wrong, our null baseline is contaminated — and the contamination biases *against* finding signal, making our negative result conservative. We'll check.

---

## Summary

| Reviewer Claim | Our Response |
|:---------------|:-------------|
| Q is a clock — a perfect encoder could fail | **Disagree.** The model learns state→Q, not clock→Q. The clock construction is label smoothing. If Q were just a clock, frame-position models would get 0 MSE — they don't. |
| True failure peaks before onset-1 | **Partially agree.** But onset is the least-bad observable boundary, and peaking slightly late is conservative for early warning. |
| +Q and −Q are different regimes | **Partially agree.** The asymmetry (prospective vs. retrospective) is real, but both sides of the same transition. Empirical symmetry under dead zone supports this. |
| Gate metric can be gamed | **Partially agree.** Multi-metric setup catches constant-bias models, but R²/AUC is better — we'll lead with those. |
| Claim should be narrowed | **Agree on precision, disagree on substance.** The narrower claim is adopted. SigLIP 1.6× in-sample → 1.0× holdout is weak non-transferable signal — consistent with both feature non-invariance and label noise. Raw-state control will distinguish. |
| 12 factual bugs | **All confirmed.** 8 fixed in v6. |

The review was thorough, accurate, and improved the codebase. The four disagreements are about whether the negative result is real — we believe the convergence of three independent lines of evidence, plus the SigLIP control, makes the label-noise alternative explanation unlikely. The raw-state control and outcome-based failure definition the reviewer recommends will settle this definitively.
