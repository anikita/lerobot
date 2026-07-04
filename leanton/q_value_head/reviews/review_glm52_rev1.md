# Review: Q-Value Head for SmolVLA Early Failure Detection

**Reviewer:** roboticist / ML reviewer (GLM-5.2, rev1)
**Date:** 2026-06-30
**Scope:** `build_q_targets.py`, `model.py`, `2_train_q_head_v5.py`, `README.md` (the four core files per `ask_for_review.md`).
**Stance:** fresh eyes on the fundamentals. Short verdict first, then the detail.

---

## TL;DR (Verdict)

**The negative result is real, but the conclusion "the features are the bottleneck" is only two-thirds supported. There is a co-conspirator you have not fully exonerated: the Q-target definition itself.** Q is a deterministic function of *temporal distance to a human-intervention onset*, which makes "predict Q from features" identical to "predict time-to-next-takeover from the current frame." That is a future-prediction task whose target depends on the human's policy, not purely on robot state. So even a perfect state encoder could legitimately fail, and that failure is not evidence the features are broken.

The strongest piece of evidence you have is **feature collapse (cosine 0.97 to 0.99)**, and it is close to decisive, but it has a methodological hole (mean-pooling induces similarity) that you should close with a linear probe before declaring the VLM features dead. You already ran the right probe on SigLIP; you have not shown it for the VLM prefix/suffix features.

The code is clean and the causal masking is correct. I found **one real latent bug** (the feature loader cannot run without `--dead-zone`, which is why all your v5 runs use it), **one misleading claim in the dead-zone writeup** (frames are relabeled neutral, not excluded from the loss), and several second-order methodology issues that together could be overstating the "architecture-agnostic" part of the claim.

Recommended order of action: (1) close the linear-probe gap, (2) run the raw-state control, (3) only then decide between contrastive fine-tuning and reframing as anomaly detection. Detail below.

---

## 1. Q-Target Construction (`build_q_targets.py`)

### 1.1 The framing problem: Q is a clock, by construction

This is the most important point in the whole review and it is not in your three lines of evidence.

Your Q-target is, algebraically, a function of *one variable*: the signed distance to the nearest intervention onset.

```
A-type:  Q[t] = -gamma^(onset - t - 1)      (autonomous, approaching takeover)
B-type:  Q[t] = +gamma^(t - onset)          (correction, after takeover)
```

There is no other term. No state, no action, no outcome. So "predict Q from features" is literally "predict time-to-next-human-takeover from the current observation." Two consequences fall out of this:

1. **The frame-position shortcut (your evidence #3, 93% correlation) is not a leak you can patch. It is the dominant signal because Q is defined by position.** A model that learns the clock is solving the task as you posed it. The contiguous holdout "exposes" the shortcut only because the clock is not transferable across episodes, which is a different (and weaker) statement than "the features do not encode failure."

2. **A model can fail this task for a reason that has nothing to do with feature quality:** time-to-next-takeover depends on the human (patience, attention, anticipation, whether they want to demonstrate a sub-task, end-of-episode). That quantity is not a deterministic function of robot state. Your Section 6.5 conclusion ("human intent is not in the camera") is correct, but it implies the *label* is the problem at least as much as the features. The current writeup blames the features exclusively.

**Recommendation:** restate the core claim as "Q-from-intervention-onset is not predictable from frozen features," which is what you actually proved. Do not over-claim "features do not encode failure," because you never defined failure independently of the human.

### 1.2 Is anchoring Q to the onset boundary sound?

Mostly yes, with two caveats.

- **The peak negative Q lands one frame before the human acts** (`Q[onset-1] = -1.0`). The human's *decision* to intervene precedes the *action* by a reaction time, and the physical failure usually precedes the decision. So the true failure signal peaks *before* `onset-1`. Your construction therefore labels the moments just before takeover as maximally negative, which is reasonable, but it conflates "policy is failing" with "human is about to grab the controller." These differ by a variable, unobserved delay. A fixed gamma cannot represent that spread.
- **The correction side (`B-type`, +Q) is a recovery signal, not a failure signal.** It measures "human is currently fixing things," which is a consequence of prior failure, not future failure. Mixing +Q and -Q into one regression target asks the head to predict two different physical regimes with one monotonic scalar. For an early-warning system you may want only the -Q (pre-onset) side; the +Q side mostly adds label noise on the correction frames.

### 1.3 Discount range (gamma=0.965, ~3 s at 30 Hz)

The arithmetic checks out: `ln(0.05)/ln(0.965) ~ 84.5 frames ~ 2.8 s`. The range is defensible for a pick-and-place task, but it assumes a *single* characteristic timescale. The right way to set gamma is empirical: measure the distribution of "time from first visible failure to human takeover" across rounds, then set the decay so the 95th percentile of failures is still above threshold. A single gamma implicitly assumes all failures evolve at the same rate, which is false for a task with distinct failure modes (gripper slip vs. misalignment vs. collision).

Also note the hard cutoff at `|Q| < 0.05`: `compute_q_targets` writes `Q[t]` until the value decays below 0.05, then `break`s. So frame at `dist=84` gets `Q=-0.053` and `dist=85` gets exactly `0.0`. That is a step discontinuity in the label at the boundary. Harmless for MSE, but it creates a synthetic "edge" the model could in principle chase. Smoother to let it decay to zero naturally or taper with a cosine window.

### 1.4 Trailing autonomous segments (A-type with no onset ahead) get Q=0

You asked whether this is correct. **It is an assumption, not a fact, and it is a source of silent label noise.** `compute_q_targets` leaves trailing A-segments at Q=0 (the `has_onset == False` branch, line 121). That is correct *only if every episode ends in success*. For DAGGER data where the expert intervenes on failure, trailing autonomous usually does mean "task done," so it is probably mostly fine, but:

- If any episode ends because the human gave up, or the time budget ran out, or the object was dropped and not recovered, those final frames are strong-negative-Q mislabeled as neutral.
- It is cheap to check: correlate episode outcome (success/fail, which you have or can derive) with whether the episode ends in an A-segment. If failed episodes ever trail with A-segments, your neutral class is contaminated exactly where it matters most (the "everything is fine" baseline).

**Recommendation:** tag each episode with a terminal outcome and either (a) give trailing A-segments of failed episodes negative Q, or (b) drop them entirely from training.

### 1.5 Minor construction issues

- **`min_length=2` filter silently drops 1-frame segments** (line 74). A 1-frame correction blip is dropped, so that correction frame is never assigned +Q and defaults to 0. Worse, the preceding A-segment still anchors its onset at the *position* of the dropped blip (`onset_frame = end`), so you can get a strong -Q label immediately adjacent to a genuine correction frame that was relabeled to 0. These are exactly the boundary frames the dead zone is later trying to suppress. Consider merging sub-min_length segments into their neighbors, or lowering the filter to 1.
- **Onset detection in the loader is independent of `build_q_targets`** (`2_train_q_head_v5.py:346`): the dead zone uses `np.where(interv_np[1:] & ~interv_np[:-1])[0] + 1` with no `min_length` filtering, so the two scripts can disagree on what counts as an onset. They should share one onset-detection function.

---

## 2. Training Methodology (`2_train_q_head_v5.py`)

### 2.1 Contiguous holdout: correct instinct, correct interpretation

The contiguous last-20% holdout is the right call for temporally autocorrelated data, and your interpretation is sound: if Q were a function of state (and state were encoded), the model would generalize across time within a round, because Q is not about wall-clock time. Failing to generalize temporally therefore *does* argue that the per-frame features do not carry state-correlated Q. The cross-round holdout (r18 fully held out) gives you a second, independent generalization regime that also fails, which makes the negative result robust to "the holdout is just distributionally weird." Good.

One tiny boundary note (not a real leak): for `T>1` windows, the first `T-1` val predictions in an `is_both` round draw history from the train segment. Negligible, but if you ever push T large, mention it.

### 2.2 Dead zone: you are relabeling, not excluding (this matters)

The README says the dead zone "zeroes out Q and *excludes from training*" the +/-10 frames around each onset. **That is not what the code does.** Trace it:

- `load_feature_data` sets `q[~mask] = 0.0` for dead-zone frames (line 352). They stay in `train_rounds`.
- In `train`, windows are built over *all* frames (`for i in range(T-1, n)`, line 509), including dead-zone ones.
- The balanced sampler classifies windows by their q value into neg / pos / **neu** (`|q| <= 0.05`). Dead-zone frames now have `q = 0`, so they fall into the **neutral class and are actively trained on** as neutral targets (lines 539 to 541, 580).

So the dead zone does not remove the hard examples from the loss; it relabels them as "neutral" and trains the model to output ~0 on exactly the ambiguous boundary frames. That has two effects:

1. It inflates the neutral class with the hardest, most label-noisy frames.
2. The remaining neg/pos frames are the *easy* ones (further from the onset), so their MSE drops. The 1.05x improvement you celebrate is therefore consistent with "we removed the hard examples by relabeling them," which is exactly the worry you raised. The symmetric neg/pos improvement is expected from symmetric relabeling and is not by itself evidence the model learned anything physical.

This is not necessarily wrong (relabling ambiguous boundary frames as neutral is a defensible choice), but the framing in the README is inaccurate and the 1.05x should not be reported as a breakthrough. If you want true exclusion, mask dead-zone frames out of the window index builder (skip `i` where any frame in the window, or the target frame, is in the dead zone) and out of the val masks.

### 2.3 The gate metric (val_neg / val_null)

"Better than guessing zero" is a weak but *meaningful* bar for a frame-conditional detector, because the model does not know at inference which frames are neg. There is a real confound though: **a model that emits a small constant negative bias on every frame beats null on the neg subset while being useless.** Your SigLIP result ("1.6x in-sample, weak negative bias, does not generalize") is exactly this failure mode, and the neg/pos ratio cannot distinguish "learned Q" from "learned a constant offset." The ratio is also unstable under `--q-renorm` because the threshold 0.05 is in raw-Q units while the targets are rescaled.

The honest, threshold-free metric is the one you already have a diagnostic for: **R-squared between predicted and true Q over all frames, or a rank metric (AUC of "does |Q_pred| rank imminent-correction frames above neutral frames").** Lead your results with the R-squared ceiling from `estimate_ceiling.py` and an AUC, not the ratio. The ratio is fine as a secondary sanity check.

Also: the `>10x means learning` threshold (line 786) is arbitrary and you never hit it on an honest split. Either drop it or justify the 10x.

### 2.4 BUG: the loader cannot run without `--dead-zone` (latent, high-impact)

This is the most concrete bug in the codebase and it explains why every v5 run in the README uses `--dead-zone 10`.

In `load_feature_data`, `hidden_dim` is initialized to `None` (line 328). The dim-handling branch is:

```python
if dead_zone > 0 and interv is not None:
    ...
    hidden_dim = h.shape[1]          # line 355: only set here
elif h.shape[1] != hidden_dim:       # line 356: compares int != None
    print("dim mismatch -- skipping")
    continue
```

When `dead_zone == 0`, the `elif` runs. On the *first* file, `hidden_dim` is still `None`, so `h.shape[1] != None` is `True`, the file is skipped, and `hidden_dim` is never set. Every subsequent file hits the same comparison and is also skipped. The function then raises `"No training rounds loaded"`.

So with the current v5 script, any `--dead-zone 0` run crashes. This means:

- All reported v5 numbers were produced with the dead zone on. The "without dead zone" column in your dead-zone comparison table must come from a different (pre-v5) loader, not from this script. Confirm that.
- Anyone reproducing a non-dead-zone baseline with this file will hit a crash.

**Fix:** guard the elif so the first file sets `hidden_dim`:

```python
elif hidden_dim is None:
    hidden_dim = h.shape[1]
elif h.shape[1] != hidden_dim:
    print(f"  dim mismatch -- skipping"); continue
```

### 2.5 Silent model routing at T=1

Line 882: `if model_name == "mlp" or T == 1:` builds a `QValueHead` regardless of `--model`. So `--model causal_tf --temporal-window 1` and `--model tcn --temporal-window 1` silently produce an MLP. If you ever intended a single-frame TCN/transformer run, you did not get one. Either error out for temporal models at `T<2`, or document that `T=1` always means MLP.

### 2.6 Normalization, balanced sampling, and BatchNorm

- Normalization stats are computed from training data only (lines 399 to 401) and applied to val. Good, no leak.
- Balanced sampling oversamples the minority neg/pos classes, so training batches are distributionally shifted from the natural frame distribution. The TCN uses `BatchNorm1d` (line 373), whose running statistics are estimated from these balanced batches and then applied at eval to a natural distribution. That is a train/eval mismatch that can only hurt the temporal models' reported numbers. Prefer `LayerNorm` or `GroupNorm` in the TCN blocks, or at minimum flag that BN stats come from balanced batches.
- `batches_per_epoch = n_per_class * 3 // batch_size` (line 546). With heavily imbalanced classes this can be small; each epoch then sees only a fraction of the data via sampling-with-replacement. Not wrong, but check that epochs-per-effective-epoch is high enough that the cosine LR schedule does not collapse before the data is covered.

---

## 3. Architectures (`model.py`)

### 3.1 Causal masking is correct

`CausalTemporalTransformer` registers

```python
torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
```

and passes `mask=causal[:T,:T]` to `nn.TransformerEncoder`. For PyTorch's boolean attention mask, `True` means "do not attend," so the upper triangle (j>i) is blocked: position i attends to j<=i only. Readout is `x[:, -1, :]`, the last position, which causally depends only on the past. Verified. No bug here. The transformer's pre-LN (`norm_first=True`) is the stable choice.

One stylistic note: because windows are always presented with the predicted frame at the *last* position, positional embedding `T-1` is always the "current" frame. The model can learn a position-specific readout bias, which is fine and consistent, but it means the causal transformer is not strictly forced to use attention over history (the last pos_embed alone could carry a lot). Not a bug.

### 3.2 TCN: mean-pool is the wrong readout for a causal predictor

The TCN ends with `x.mean(dim=-1)` over time (default `pool="mean"`, line 474). For a *causal* conv stack, output position 0 has seen only 1 input frame, position 1 has seen 2, and only the last position has the full receptive field. Mean-pooling dilutes the fully-informed last-position output with a dozen barely-informed early positions. For predicting "Q at the current frame," the principled readout is `pool="last"` (mirror what the causal transformer does) or last-plus-mean. The fact that the transformer (last readout) and the TCN (mean readout) are then compared as "two temporal architectures" is unfair to the TCN; the TCN may be underperforming its own capacity. This does not rescue the negative result (the single-frame MLP also fails), but it means the "architecture-agnostic" table has one row that is not configured apples-to-apples.

### 3.3 TCN receptive field: comment drift

The `receptive_field` property computes `1 + 4*(1+2+4+8+16) = 125` frames. The README and the "Model Zoo" table say 125. But the `TemporalTCNQHead` docstring (line 395) says "receptive field = 124 frames." Stale comment. 125 frames at 30 Hz is ~4.2 s, adequate for the 1 to 4 s failure horizon, so the design is fine; just fix the comment.

Also: the `CausalConvBlock` has no residual connection (line 376 onward). At depth 5 with ReLU it survives the smoke test, but adding residuals (`x = x + block(x)`) is cheap insurance and standard for TCNs.

### 3.4 CausalConvBlock is correct

Left-only padding `(kernel-1)*dilation` then conv gives a strictly causal conv with output length equal to input length. Verified.

### 3.5 The 110M RLTQHead number is almost certainly wrong

The README's architecture table lists `RLTQHead (bottleneck AE) | 110M`. But `RLTQHead` in `model.py` is a pure PyTorch module: input_proj (960x256), a 256-dim 2-layer encoder, a 1-layer decoder, output_proj (256x960), pos/cls/query params, and a tiny q_head. That is on the order of 2 to 4M parameters, not 110M. The only way to reach 110M is if the number *includes the unfrozen VLM adapter layers* attached at training time. If so, the param column is comparing apples (standalone heads) to oranges (head plus adapter), which undermines the framing "from 961K to 110M params, all fail." The conclusion (capacity does not help) still holds because the MLP at 961K already fails, but the 110M figure should be corrected or footnoted, otherwise a reader will catch it and doubt the table.

### 3.6 The adapter operates on already-pooled features, which is irreversible

Both `VLMAdapter` and `QValueHead` consume *mean-pooled* prefix features. The SmolVLA prefix mixes vision tokens, language tokens, and a proprioceptive state token. Mean-pooling across heterogeneous modalities is a lossy, irreversible step: once you average the state token with the vision tokens, no later VLM layer can recover the per-modality detail. So "unfreezing later VLM layers as an adapter" cannot undo what pooling destroyed. `RLTQHead` was your attempt to avoid pooling, but it too collapses the sequence to a single CLS/RL token before reading Q. If you want to test whether per-token information survives, the head must attend over the *full token sequence* (as `TransformerQHead` does), not over a pooled vector or a single token.

This also feeds back into evidence #1: if your cosine-similarity diagnostic was computed on mean-pooled vectors, the 0.97 to 0.99 similarity is partly an artifact of pooling heterogeneous tokens toward their mean. **Pooling induces similarity.** Re-run the collapse diagnostic on individual tokens (or at least on the state token alone) before citing 0.97 as proof the features are dead.

---

## 4. The Core Claim: Is the Negative Result Real?

**Yes, with a caveat on attribution.** Let me go through your specific challenges.

*Could the contiguous holdout be too harsh?* No. The fully-held-out round (r18) gives the same 1.0x, and that is a separate generalization regime. Two independent regimes agreeing is strong.

*Could Q be too sparse (only ~15% nonzero)?* Sparsity makes the task harder but is not by itself fatal; the balanced sampler exists precisely to handle it. Sparsity is a contributing headwind, not an alternative explanation. The single-frame MLP failing at 1.0x is the decisive observation, because it is configuration-robust: no temporal model, no pooling choice, no BN issue can be blamed.

*Could the SigLIP features be compromised by ImageNet pre-training?* Possibly, but you tested SigLIP directly and it also fails to generalize (1.6x in-sample only). Two independent feature sources (VLM prefix/suffix and raw SigLIP) failing the same way is strong evidence the problem is upstream of the features, i.e., in the label or in "state does not determine human intent."

**The caveat on attribution.** Your three lines of evidence establish "Q-from-intervention-onset is not predictable from these features." They do *not* establish "the features do not encode failure," because you never defined failure independently of the human takeover. To convert your negative result from "features are bad" to "features are bad *even when failure is defined cleanly*," you need one more experiment: define failure from task outcome (not intervention), then re-test. Until then, the honest headline is the narrower one.

**The one piece of evidence I would not publish as-is:** feature collapse via mean-pooled cosine similarity. Replace it with (a) per-token cosine, and (b) a linear probe on the VLM prefix/suffix features under the contiguous holdout. You ran the probe on SigLIP and got the right answer; run it on the VLM features too. If a linear probe on VLM features also cannot beat null on the contiguous holdout, that closes the gap and the collapse claim becomes rigorous.

---

## 5. Where to Go Next

Ranked by expected value, not by how interesting the idea is.

1. **Define failure from outcome, not intervention (fix the label).** This is the highest-value move and it is cheap to reason about: tag each episode (or sub-goal) as success or failure, and build Q from outcome (e.g., negative Q discounted backward from a verified failure time, positive Q from recovery to a verified sub-goal). If the negative result *persists* under an outcome-based label, then you have genuinely exonerated the label and the blame sits on the features. If it *disappears*, the intervention proxy was the culprit all along. Either way you learn something decisive. Do this before any fine-tuning.

2. **Raw-state control (joint angles plus object pose).** Bypass the VLM entirely and train the same heads on low-dimensional robot state plus a detector-based object pose. This cleanly separates two hypotheses:
   - If raw state *also* fails on the contiguous holdout, the signal genuinely is not in the observable state (label problem, see #1).
   - If raw state *succeeds*, the VLM is the bottleneck and contrastive fine-tuning is justified.
   This is the single most informative experiment you have not run, and it is cheaper than fine-tuning a VLA.

3. **Linear probe on VLM features (close evidence gap).** As above. Decides whether the collapse claim is rigorous.

4. **Reframe as anomaly / OOD detection.** Drop the continuous Q. Train a normalizing flow or a feature-density model on successful-rollout frames under the frozen backbone, and flag frames that fall in low-density regions. This sidesteps the intervention-proxy problem entirely (you only need "does this look like success," not "when will the human intervene"), is cheap, and maps directly onto your reasoning-tower use case (cheap critic gates the expensive tower). My favorite of the "reframe" options for your architecture.

5. **Contrastive / auxiliary-loss fine-tuning of the backbone.** Only worth the cost after (2) shows the state carries signal that the VLM is discarding. Otherwise you risk spending GPU to confirm what the linear probe already told you.

6. **Online RL (RL^T style).** The most expensive and the most principled for learning a real critic, because the target becomes empirical return rather than a human-proxy. Reasonable endgame if the cheap diagnostics above point at "the features could encode this but do not."

---

## Concrete Bug / Issue Checklist

| # | File:line | Severity | Issue |
|:--|:----------|:---------|:------|
| 1 | `2_train_q_head_v5.py:328,356` | **High** | `hidden_dim` never set when `dead_zone==0`; loader skips every file and crashes. All v5 runs survive only because they pass `--dead-zone`. |
| 2 | `2_train_q_head_v5.py:343-358` vs README | Medium | Dead zone relabels boundary frames as neutral and trains on them; README says "excluded from loss." Inaccurate. |
| 3 | `model.py:474` / training | Medium | TCN default `pool="mean"` dilutes the causal last-position readout; unfair vs. transformer's last readout. Use `pool="last"`. |
| 4 | `model.py:373` | Medium | `BatchNorm1d` under balanced sampling causes train/eval distribution mismatch in TCN. Prefer LayerNorm. |
| 5 | README results table | Medium | `RLTQHead 110M` likely includes the VLM adapter, not the head alone. Apples-to-oranges in the capacity argument. |
| 6 | Cosine-collapse diagnostic | Medium | If computed on mean-pooled features, 0.97 to 0.99 is partly a pooling artifact. Re-run per-token and add a VLM linear probe. |
| 7 | `2_train_q_head_v5.py:882` | Low | `T==1` silently routes every `--model` to `QValueHead`. Document or error. |
| 8 | `model.py:395` | Low | TCN docstring says receptive field 124; property and README say 125. Stale comment. |
| 9 | `build_q_targets.py:74` | Low | `min_length=2` drops 1-frame correction blips, mislabeling genuine correction frames as Q=0 while neighbors get peak -Q. |
| 10 | `build_q_targets.py` vs loader | Low | Onset detection duplicated and inconsistent (min_length filter vs raw rising edge). Share one function. |
| 11 | `build_q_targets.py:117-119` | Low | Hard `break` at `|Q|<0.05` creates a label step discontinuity. Taper instead. |
| 12 | Conceptual | Medium | Trailing A-segments forced to Q=0 assumes every episode ends in success. Verify against episode outcome. |

---

## What Is Good

- The causal masking in both the transformer and the TCN is correct, including the PyTorch bool-mask convention and left-only causal padding. This is the easiest thing to get wrong and you got it right.
- Contiguous plus cross-round holdout is the right evaluation design, and the frame-position-shortcut discovery (random split inflates to 10.7x, contiguous exposes 1.0x) is a clean, honest finding that many people miss.
- The balanced sampler, the null baseline, the R-squared ceiling diagnostic, and the leak-test harness show genuine methodological discipline. You built the right diagnostics; you just are not leading with the strongest one (R-squared / AUC over the ratio).
- The code is readable, the CLI is well-designed, and the separation of build / extract / train is clean.

---

## Bottom Line

You have strong evidence that **Q-from-intervention-onset is not learnable from frozen SmolVLA features**, and the architecture work is sound. But before you conclude "frozen features are a dead end" and spend effort on contrastive fine-tuning, close two gaps that are cheaper than fine-tuning:

1. **Exonerate the label** by redefining failure from task outcome (or run the raw-state control). Your current label is a human-takeout clock, and a clock that depends on another agent's policy is not something any state encoder should be expected to predict.
2. **Exonerate the diagnostic** by replacing mean-pooled cosine with a per-token probe plus a VLM linear probe.

If both gaps close and the negative result holds, the dead-end claim is bulletproof and the next step is anomaly detection or online RL. If either gap opens, you may have a real signal you are currently masking.
