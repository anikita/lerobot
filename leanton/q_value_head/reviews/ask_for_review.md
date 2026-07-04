# Review Request: Q-Value Head for SmolVLA Early Failure Detection

## What This Is

A ~2K-line standalone project (7 first-class files) that trains a lightweight Q-value head on frozen SmolVLA backbone features. The goal: predict a continuous failure score `q_target ∈ [-1, 1]` from visual + proprioceptive features, giving the robot an early warning system to self-correct before policy failure.

This is the **descriptive half** of a larger architecture (the Reasoning Tower Paradox). The Q-head is a cheap critic that watches the frozen VLA backbone; when it detects imminent failure, it wakes an expensive reasoning module. The prescriptive half (instruction head) is on hold until this side is resolved.

**Repo:** `~/lerobot/leanton/q_value_head/` (the `leanton` branch of a LeRobot fork)
**Primary SSoT:** `README.md` in that directory — read it first (~5 min)
**Full context:** `Knowledge_Drafts/LeRobot/Design_Strategies/the_reasoning_tower_paradox/` (Chapters 1-7, ~25 pages)

## What I Need From You

I've been staring at this for weeks. I need a fresh pair of eyes on the fundamentals — is this negative result real, or did I miss something?

### 1. Q-Target Construction (`build_q_targets.py`)

The Q signal is derived from human intervention onsets: autonomous segments decay Q **backwards** from -1.0, correction segments decay Q **forwards** from +1.0 (γ=0.965, no ramp, clean ±1.0 jump at onset).

- Is this construction logically sound? The onset is the moment the human took over — is anchoring Q to that boundary the right framing?
- The discount factor (γ=0.965) gives ~3 seconds of non-zero Q around each onset. Is this range appropriate for a 30Hz pick-and-place task?
- A-type segments after the last correction (trailing autonomous, no onset ahead) get Q=0 — is that correct, or should they have negative Q too?

### 2. Training Methodology (`2_train_q_head_v5.py`)

- **Contiguous holdout:** The last 20% of each round is held out temporally (not random split). Is this the right evaluation? Does it prove temporal generalization, or am I leaking something else?
- **Dead zone:** ±10 frames around each intervention onset are zeroed out (excluded from loss). This breaks the anti-correlation pattern (neg and pos both drop below null for the first time), but the improvement is tiny (~1.05×). Is this methodologically sound, or am I just removing the hard examples?
- **Metrics:** The gate metric is `val_neg / val_null` — MSE on negative-Q frames divided by MSE on neutral frames. Is "better than guessing zero" the right bar, or is there a more meaningful baseline?

### 3. Architectures (`model.py`)

- **CausalTemporalTransformer:** 2-layer causal self-attention with learned positional embeddings, upper-triangular mask, last-position readout. Is the causal masking correct? The mask uses `torch.triu(..., diagonal=1)` — does this properly prevent information leakage from future frames?
- **TCN:** Causal Conv1d blocks, kernel=5, dilations [1,2,4,8,16], receptive field=125 frames. Is the dilation pattern sufficient for the timescales involved (~1-4 seconds between onset and failure)?
- **Are there bugs in either architecture?** I've smoke-tested gradients, shapes, and causality. But I may have missed something subtle — incorrect padding, off-by-one in the temporal window, positional embedding initialization.

### 4. The Core Claim: Q from Frozen Features is a Dead End

Three independent lines of evidence converge on this:

1. **Feature collapse:** Cosine similarity 0.97-0.99 across all Q-values for both prefix (960-dim) and suffix (720-dim) features
2. **Architecture-agnostic failure:** MLP, Transformer, RLT (bottleneck AE), TCN, and Causal Transformer all converge to 1.0× the null baseline on contiguous holdout — from 961K to 110M params
3. **Frame position shortcut:** Features encode frame position at 93% correlation; models learn the clock, not Q

**The question:** Is this evidence sufficient to conclude the features don't encode the signal? Or could there be a training methodology flaw that masks a real but weak signal across all architectures simultaneously?

Specific challenges:
- Could the contiguous holdout be too harsh (throwing out frames the model needs to generalize)?
- Could the Q targets be too sparse (only ~15% of frames have |Q| ≥ 0.05)?
- Could the SigLIP features (pre-VLM, raw vision) also be compromised — e.g., by ImageNet pre-training that's irrelevant to robot state?

### 5. Where Would You Go Next?

Assuming the negative result holds, what direction would you pursue?

- **Contrastive fine-tuning** of the backbone to surface state differences?
- **Raw state features** (joint angles + object pose from a detector) — bypass the VLM entirely?
- **Online RL** (the RL^T paper approach — train the critic with actual task outcomes)?
- **Reframe as anomaly detection** — forget Q, just detect out-of-distribution states?
- **Multi-task pressure** — force the backbone to predict Q as an auxiliary loss during imitation learning, so features adapt?

## What NOT to Review

- The feature extraction scripts (`1_extract_vlm_features.py`, `1_extract_siglip.py`) — they're LeRobot data loaders, one-time executables, not algorithmic
- The diagnostic scripts in `diagnostics/` — they're investigation artifacts, not part of the pipeline
- The archived files in `archive/` — old versions kept for provenance
- The design documents in the vault — they're context, not code; the README is self-contained

## Time Budget

The README is the entry point, the 4 core files to review are `build_q_targets.py`, `model.py`, `2_train_q_head_v5.py`, and `README.md`. Don't read the vault chapters unless you want the full argument — the README summarizes the results.
