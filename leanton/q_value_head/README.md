# Q-Value Head — Training Pipeline

Standalone project that trains a lightweight Q-value head on frozen SmolVLA backbone features. The goal: predict a continuous failure score `q_target ∈ [-1, 1]` from visual and proprioceptive features, giving the robot an **early warning system** to self-correct before policy failure.

**Status (June 2026):** Q from frozen features is a dead end — every architecture converges to the null baseline (1.0×) on held-out data. The bottleneck is the features themselves, not the model. See [Results So Far](#results-so-far).

## Scope

The Q-head is the **descriptive** half of the Reasoning Tower Paradox: a cheap (~1-2M param) critic that watches the frozen VLA backbone and flags imminent failure, so the expensive reasoning tower only activates when needed. The **prescriptive** half (instruction head) is blocked until the descriptive side is resolved.

For the full architectural motivation, see `Knowledge_Drafts/LeRobot/Design_Strategies/the_reasoning_tower_paradox/` (Chapters 1-7).

## Files

### First-Class (Reproduce the Experiment)

| File | Purpose | Dependencies |
|:-----|:--------|:-------------|
| `build_q_targets.py` | Compute Q-targets from intervention arrays using the symmetric onset-anchored discount policy (γ=0.965). Works with or without LeRobot — pass `--all` for the full 21-round dataset or `--intervention-file` for a raw bool array. | numpy (+ lerobot for HF dataset access) |
| `1_extract_vlm_features.py` | Load v2 dataset rounds via LeRobot, run the SmolVLA backbone, save hidden states (prefix, suffix, mid-layer) as .pt files. | torch, lerobot, datasets, HF |
| `1_extract_siglip.py` | Extract raw SigLIP vision features (pre-VLM, per-camera) — bypasses the collapsed VLM features. | torch, lerobot, transformers |
| `2_train_q_head_v6.py` | Train a Q-head on cached features. Supports MLP, TCN (causal Conv1d, pool=last, LayerNorm), and CausalTemporalTransformer (causal self-attention). Full control over train/val rounds, dead-zone exclusion (true exclusion, not relabeling), Q renormalization, temporal window, feature field. v6 incorporates reviewer fixes #1-#8. | torch, numpy |
| `model.py` | All Q-head architectures: `QValueHead` (MLP), `RLTQHead` (bottleneck autoencoder), `TemporalTCNQHead` (causal Conv1d blocks), `CausalTemporalTransformer` (causal self-attention with positional embeddings). | torch |
| `requirements.txt` | Python dependencies. | — |

### Diagnostics (`diagnostics/`)

Ancillary scripts used during investigation — not needed for reproduction:

| File | What it diagnosed |
|:-----|:------------------|
| `diagnostic_feature_similarity.py` | Cosine-similarity analysis proving feature collapse (Ch 6.1) |
| `estimate_ceiling.py` | R² ceiling diagnostic — model-free upper bound on explainable variance (Ch 7.3) |
| `validate_features.py` | Feature file integrity and shape validation |
| `silly_leak_test.py` | TCN temporal leak test — verifies architecture can extract signal when it exists (Ch 6.4) |
| `audit_corrupt_frames.py` | Detect corrupt feature files after extraction |
| `convert_to_npy.py` | Convert .pt feature files to .npy for external tools |

### Archive (`archive/`)

Previous versions of training scripts (v1-v4) and feature extractors (v1-v3), plus all pre-v5 weight files. Kept for provenance; not needed for current work.

## Reproduction Pipeline

### Step 0: Build Q Targets

```bash
# Build Q for all 21 rounds (requires lerobot + HF token)
python build_q_targets.py --all --out-dir ./q_targets/

# Or for specific rounds
python build_q_targets.py --rounds 17,19,20,21 --out-dir ./q_targets/
```

This produces `q_targets_r{NN}.npy` — float32 arrays of shape `(n_frames,)` with values in `[-1, 1]`.

**How Q is defined:** The correction onset is the ground-truth boundary. A-type segments (autonomous) decay Q **backwards** from the onset at Q=-1.0; B-type segments (correction) decay Q **forwards** from the onset at Q=+1.0. Both decay with γ=0.965. There is no ramp — the onset boundary is a clean ±1.0 jump. See Chapter 3 for the full design rationale.

### Step 1: Extract Features

```bash
# VLM backbone features (prefix, suffix, mid-layer)
python 1_extract_vlm_features.py r01 --layers 0,4,8,12

# Raw SigLIP vision features (per-camera, pre-VLM)
python 1_extract_siglip.py r01
```

Output lands in `features_v4/` or `features_siglip/` as `{round_key}_{field}.pt` files. Each .pt contains:
- `hidden_states`: feature tensor `[n_frames, dim]`
- `q_targets`: Q-target tensor `[n_frames]` (merged from Step 0 output)
- `interventions`: bool tensor `[n_frames]`
- `frame_indices`: int64 tensor `[n_frames]`
- `metadata`: dict with round info, extraction params

### Step 2: Train

```bash
# Causal Transformer on SigLIP features — the current best configuration
python 2_train_q_head_v5.py \
    --model causal_tf \
    --feature-field siglip_cat \
    --no-adapter \
    --temporal-window 120 \
    --epochs 100 \
    --train r17,r19,r20,r21 \
    --val r18 \
    --dead-zone 10 \
    --q-renorm

# TCN on VLM layer-12 features (for comparison)
python 2_train_q_head_v5.py \
    --model tcn \
    --feature-field layer_12 \
    --temporal-window 90 \
    --epochs 50 \
    --train r1,r17,r19,r20,r21 \
    --val r18
```

Key CLI flags:
| Flag | Effect |
|:-----|:-------|
| `--model` | `mlp`, `tcn`, `transformer`, `rlt`, `causal_tf` |
| `--feature-field` | `prefix`, `suffix`, `layer_{N}`, `siglip_cat`, `siglip_{cam}` |
| `--temporal-window` | Number of consecutive frames per training sample |
| `--train / --val` | Explicit round assignment (comma-separated). Same round in both = 80/20 contiguous split |
| `--dead-zone N` | Exclude ±N frames around each intervention onset from training and metrics (v6: true exclusion, not relabeling) |
| `--q-renorm` | Renormalize Q by `max(\|Q\|)` after dead-zone masking |
| `--balanced-sampling` | Equal negative/positive samples per batch |
| `--no-adapter` | Skip VLM adapter fine-tuning (pure feature readout) |

### Step 3: Evaluate

The script reports four metrics per epoch:
- **val_neg**: MSE on frames with Q < -0.05 (autonomous, approaching failure)
- **val_pos**: MSE on frames with Q > 0.05 (correction, recovering)
- **val_null**: MSE on frames with |Q| < 0.05 (neutral — the baseline)
- **val_all**: MSE on all frames

The gate metric is **val_neg / val_null** — can the model predict Q better than guessing zero? A ratio > 1.0 means the model extracts signal; 1.0 means it's no better than the null baseline.

## Results So Far

### The Pipeline Works

The Q-target construction, feature extraction, and training pipeline are all verified and reproducible. The training script supports 5 architectures, 4 feature fields, explicit round control, dead-zone masking, and Q renormalization. A full 21-round extraction takes ~2h on an L40S; training converges in seconds to minutes.

### The Features Are the Bottleneck

Every experiment converges to the same conclusion: **frozen SmolVLA backbone features do not encode the Q failure signal.** Three independent lines of evidence:

**1. Feature Collapse (Chapter 6.1).** Cosine similarity across all Q-values is 0.97–0.99 for both prefix (960-dim) and suffix (720-dim) features. A frame 2 seconds before a crash looks nearly identical to a frame mid-correction. The backbone is optimized for action prediction, not state evaluation — it compresses away exactly the information the Q-head needs.

**2. Architecture-Agnostic Failure (Chapter 6.3).** All architectures converge to 1.0× the null baseline on contiguous holdout:

| Architecture | Params | Temporal | × Over Null |
|:-------------|-------:|:---------|:------------|
| MLP (single frame) | 0.52M | No | 1.00× |
| Transformer (TW=3) | 1.9M | Yes | 1.00× |
| RLTQHead (bottleneck AE) | 3.3M | No | 1.00× |
| TCN (TW=90) | 1.9M | Yes | 1.00× |
| Causal Transformer (TW=120) | 1.9M | Yes | 1.00× |

Models don't even overfit — they converge to the same floor regardless of capacity or temporal context.

**3. Frame Position Shortcut (Chapter 6.4).** Features encode frame position at 93% correlation. On random (non-contiguous) train/test splits, models appear to learn Q (10.7× at frame t), but this is an artifact — they learn the clock, not the failure signal. The contiguous holdout (last 20% of each round) is the only honest metric, and it exposes the shortcut.

### SigLIP: No Generalizable Signal (Chapter 6.5)

Raw SigLIP vision features (960-dim per camera, pre-VLM) were tested as a bypass. A linear probe finds weak negative bias (1.6× on in-sample frames) — the robot's configuration does carry some state information — but this does not generalize to held-out rounds. The physically observable state (object position, gripper pose) is simply not predictive of imminent correction: **human intent is not in the camera.**

### Dead Zone: v5 Artifact, v6 Truth

**v5 (relabeling):** Introducing a dead zone (±10 frames around each intervention onset, with boundary frames *relabeled* to Q=0 and still trained on as neutral) appeared to produce the first symmetric improvement — both neg and pos dropping below null at ~1.05×.

**v6 (true exclusion, reviewer-verified):** The dead-zone frames are now genuinely excluded from training, window building, and all metrics. Re-running the same experiments with true exclusion:

| # | Model | Features | Dead Zone | Renorm | val_neg/null | val_pos/null | Pattern |
|:--|:------|:---------|:---------:|:------|:------------:|:------------:|:--------|
| A | MLP | VLM prefix | 0 | no | 1.53× | 0.67× | Anti-correlated |
| B | MLP | VLM prefix | 10 | no | 1.03× | 0.96× | **Symmetric ~1.0×** |
| C | Causal TF | SigLIP cat | 10 | yes | 1.17× | 0.78× | Anti-correlated |

Three findings:

1. **The v5 "breakthrough" was a relabeling artifact.** With true exclusion, no configuration beats 1.0× on val_neg/null. The reviewer correctly predicted this: relabeling hard boundary frames (|Q|≈1.0) as neutral (Q=0) made the remaining neg/pos subsets easier.

2. **Dead zone kills anti-correlation for VLM features, not SigLIP.** MLP on prefix goes from anti-correlated (1.53/0.67) to symmetric (1.03/0.96). The anti-correlation was partly caused by the onset-boundary ±1.0 jump — removing those frames restores symmetry. But SigLIP + causal transformer stays anti-correlated (1.17/0.78) even with dead zone, suggesting a second, architecture-specific source of anti-correlation that dead zone doesn't fix.

3. **The negative result is robust.** Across three configurations — different architectures, feature sources, dead zone settings, renorm — val_neg/null never exceeds 1.0× on contiguous holdout. The cleanest result is MLP on prefix with dead_zone=10: 1.03× and 0.96×, effectively dead-on the null baseline in both directions. The model converges to predicting zero everywhere.

### Where We Stand

**Q-from-intervention-onset is not learnable from frozen SmolVLA features.** The evidence: 5 architectures (MLP, Transformer, RLT, TCN, Causal Transformer), 16 VLM layers, 2 feature sources (VLM prefix/suffix + raw SigLIP), with and without temporal context (1-120 frame windows), with and without dead-zone exclusion — all converge to ~1.0× the null baseline on contiguous holdout. The pipeline is verified, the architectures are correct (independently reviewed, 8 bugs fixed in v6), and the result holds.

**Why?** The negative result could mean the features don't encode the relevant physical state (feature bottleneck), or the Q label — derived from human intervention timing — contains non-transferable per-operator variance (label bottleneck). But we can't cleanly distinguish these with the data we have:

- **Proprioception-only control is invalid.** Joint angles alone don't capture object position or gripper-object geometry, which are precisely what drive intervention timing. Both we and the reviewer agree this path is a dead end.
- **Episode-level outcome labels don't exist in DAGGER.** The human intervenes precisely to prevent failure. Every episode ends successfully. The intervention IS the outcome — there are no dropped objects or collisions to label from, because the human prevented them. Building Q from "task outcomes" when interventions preempt failures is circular.

**The intervention onset is the only observable failure boundary in DAGGER data.** It is noisy (operator timing varies), but it's what we have. The negative result — that this signal cannot be extracted from frozen features — is real. The remaining question is whether the noise is in the label (H₂: different operators have different intervention thresholds, so Q doesn't transfer across episodes) or in the features (H₁: the VLM discards the state that correlates with imminent intervention). Both are plausible, and the current data cannot separate them.

## Next Steps

Two paths forward. Neither requires resolving H₁ vs H₂:

### Path A: Anomaly Detection (cheap, deployment-ready)

Drop continuous Q entirely. The reasoning tower needs a critic that decides when to wake — it doesn't need to predict *when* the human will intervene, it needs to detect *whether the current state looks unfamiliar*. This reframes the problem:

- **Train a density model** on frozen features over autonomous (non-intervention) frames. Fit a Mahalanobis distance, one-class SVM, or normalizing flow on "what success looks like."
- **Flag low-density frames at inference.** A frame that falls outside the distribution of autonomous operation is anomalous — the robot is in unfamiliar territory, whether or not a human would intervene.
- **Evaluate:** AUC of "does feature density drop before intervention onset" on held-out rounds.

This sidesteps the human-intent problem entirely. It doesn't need to predict when someone will take over — it just needs to model "does this look like the kind of state the policy can handle." It's cheap (no fine-tuning, no new labels), and it maps directly onto the architecture's use case (cheap critic gates the expensive tower). The reviewer's Experiment D.

### Path B: Online RL (expensive, principled)

If the frozen backbone genuinely doesn't encode task quality, learn a critic from actual task returns via online reinforcement learning. The RL^T paper (Physical Intelligence, 2025) demonstrated this for VLA policies: a small bottleneck-autoencoder critic trained with online rollouts can learn to predict task success from backbone features, even when those features weren't trained for it.

- **Why it could work where offline Q fails:** Online RL provides the critic with varied outcomes (successes AND failures) that don't exist in DAGGER data. The critic learns from empirical returns — "did the task succeed after this state?" — rather than from a human-proxy label. The features adapt through the RL updates.
- **Cost:** Requires online robot rollouts. Non-trivial infrastructure.
- **When:** Only after Path A has been exhausted, since Path A is cheaper and may solve the practical problem.

### Path C: Contrastive Fine-Tuning (only if needed)

If Path A succeeds, fine-tuning is unnecessary — anomaly detection on frozen features is sufficient. If Path A fails, and Path B is too expensive, contrastive fine-tuning of the backbone (or an auxiliary Q-prediction loss during imitation learning) is the middle ground. But this should not be run until the cheaper paths are exhausted, per the reviewer's recommendation.

### Decision Logic

```
Path A: Anomaly detection on frozen features
├─ Works (AUC >> 0.5 on held-out rounds)
│   └─ Deploy: anomaly score gates the reasoning tower
│       The H₁ vs H₂ question is academic — the system works
└─ Fails (AUC ≈ 0.5)
    └─ Path B or C depending on resources
        The frozen features genuinely don't carry the signal
```

**Recommendation:** Run Path A first. It's the cheapest experiment, it's the reviewer's top recommendation, and it's the most direct path to a working system — the reasoning tower doesn't need a Q-value, it needs a wake-up signal.

## Architecture Reference

### Q-Target Policy

```
A-type (autonomous):  Q = -γ^dist   backwards from onset, Q = -1.0 at onset-1
B-type (correction):  Q = +γ^k      forwards from onset,  Q = +1.0 at onset
γ = 0.965, decay threshold = 0.05 (~3 seconds to neutral)
```

### Model Zoo

| Model | Class | Mechanism | Receptive Field |
|:------|:------|:----------|:---------------|
| MLP | `QValueHead` | 3-layer MLP on mean-pooled or single-frame features | 1 frame |
| RLT | `RLTQHead` | Bottleneck autoencoder (encode→decode) + residual | 1 frame |
| TCN | `TemporalTCNQHead` | Causal Conv1d blocks (LayerNorm), dilations [1,2,4,8,16], kernel=5, last-timestep readout | 125 frames |
| Causal TF | `CausalTemporalTransformer` | 2-layer transformer, upper-triangular mask, last-position readout, learned position embeddings | Configurable |

## Related Documents

- **Design documents:** `Knowledge_Drafts/LeRobot/Design_Strategies/the_reasoning_tower_paradox/`
- **Q-target pipeline:** `Knowledge_Drafts/LeRobot/Experiments/2026-06-26_intervention-field-recovery/`
- **v2 Dataset:** `anikitakis/pick_n_place_dagger_3cam_dagger_only_v2` on HuggingFace
