# Next Experiments — Reviewer GLM-5.2

**Date:** 2026-06-30
**Stance:** the negative result lives inside a confound with exactly two axes, the **label** (Q derived from human takeover) and the **features** (frozen VLM). Every useful next experiment varies exactly one of those while holding the other fixed, so no run is wasted. Order below is the order I would actually run them in, with what each decides.

---

## The Framing

Right now you cannot tell these apart, and they all produce 1.0× on the contiguous holdout:

1. The state is absent from the features.
2. The state is present but the frozen VLM discards it.
3. Q is not determined by state alone, because human-takeover timing adds non-transferable variance.

Experiment A splits (1+2) from (3). Experiment B then splits (1) from (2). That is the whole game.

> **Rev1 correction (per author pushback):** an earlier draft proposed a "training-free, proprioception-only" Step 0 that clustered frames by joint angles + gripper and read off takeover-timing variance. That diagnostic is **invalid**. Proprioception is an incomplete state key: it does not contain object position or gripper-object geometry, which are precisely what drive takeover timing. High within-cluster variance would therefore be the trivially-expected signature of clustering while blind, and it cannot distinguish "takeover timing is not determined by state" from "proprioception is not the relevant state." It is strictly less informative than the VLM/SigLIP failure already on hand, which at least saw the camera. There is no free, proprioception-only shortcut to the label question. The valid core of that idea (does full observable state determine takeover timing?) is just Experiment A with full-state inputs, so Step 0 has been folded into A below.

---

## 1. Experiment A: state control (the single most informative run)

Hold the label fixed (same intervention-onset Q), replace the frozen VLM features with **structured state**: proprioception (joint angles + gripper width) plus the **2D object centroid per camera**. The object-centroid detector is on the critical path, because proprioception alone is blind to where the object is, and object position is the one scene quantity not derivable from the joints. No gripper detector is needed (see below). Same heads (MLP, causal_tf), same contiguous holdout. See the "Practical A" spec in the conclusions section for the minimal input vector.

Raw state is the **ground truth** of what the robot is doing. It is the ceiling for "anything state-predictable." So:

- Raw state shows the same ~1.6× in-sample / 1.0× holdout pattern as SigLIP means the signal is real but non-transferable, and the bottleneck is the **label's human-timing variance**, not any representation. Stop trying to fix features. Go to B.
- Raw state generalizes (>~2× holdout) where VLM/SigLIP do not means the frozen VLM is genuinely discarding state. Contrastive/auxiliary fine-tuning is justified. Go to E.
- Raw state shows no signal even in-sample (~1.0×) means observable state simply does not determine Q. Strong, clean negative. Go to D.

This is cheap (small head on ~14 dims) and it cuts the search space in half regardless of outcome.

---

## 2. Experiment B: outcome-based Q (exonerate the label)

Hold features fixed (VLM *and* raw state), change the label. Redefine Q from task outcome: negative Q discounted backward from a verified failure (object dropped via segmentation, collision, timeout), positive Q from verified recovery.

- Outcome-Q transfers where takeover-Q did not means the takeover proxy was the noise source. The label was the bottleneck.
- Outcome-Q also fails to transfer on both feature sources means Q, however you define it, is not in the observable state. The negative result is now bulletproof and publishable as stated.

B only matters if A points at the label, but it is the experiment that converts "Q-from-takeover is not learnable" into a real attribution.

---

## 3. Experiment C: decompose the VLM features (close the diagnostic gap, run in parallel)

This is diagnostic and cheap, no retraining. Run two linear probes on the frozen VLM prefix/suffix under the contiguous holdout:

- Predict Q (the probe never run on VLM features; it was only run on SigLIP).
- **Predict raw state** from VLM features (a state-decoding probe).

The second probe is the sharp one. It decomposes "features are bad" into two different failures:

- If VLM features cannot linearly reconstruct joint angles / object pose, the backbone compressed away state. Feature collapse is real **and causal**, not a pooling artifact. Justifies fine-tuning.
- If VLM features can reconstruct state but cannot predict Q, state is preserved and Q-specific information is what is missing. Points back at the label.

Also recompute the cosine diagnostic per token (and on the state token alone), not on the mean-pooled vector. Pooling inflates cosine similarity, and 0.97 is currently cited on pooled features.

---

## 4. Experiment D: anomaly/OOD reframe (the pragmatic path, run regardless)

Drop continuous Q entirely. Fit a density model on frozen features over **successful-rollout** frames only (start with Mahalanobis / one-class SVM before reaching for a normalizing flow), and flag low-density frames at inference. Evaluate AUC against imminent takeover.

This sidesteps the human-intent problem: it models "does this look like success," not "when will the human act." It is cheap, needs no fine-tuning, and it is exactly the gate the reasoning tower wants (cheap critic decides when to wake the expensive module). Even if A and B confirm the negative result, D may still work, because OOD-ness and "Q from takeover" are different questions.

---

## 5. Experiment E: backbone fine-tuning (only after A or C justifies it)

Contrastive fine-tuning or a Q auxiliary loss during imitation learning, so the features actually adapt. Expensive. Do not run this until A or C tells you the state is in the world but the frozen VLM threw it away. Otherwise you risk burning GPU to rediscover what a linear probe would have told you in minutes.

---

## Recommended Order

1. **Prerequisite:** get the 2D object centroid per frame, per camera (off-the-shelf or color/shape detector on the 3-cam recordings). This is the only non-trivial data-prep step and the only detector you need. Proprioception alone is not a valid state key for this task; the object centroid is the missing piece.
2. **A** (state control) and **C** (VLM probes) in parallel. A tells you where the signal lives; C tells you whether the VLM kept the state.
3. **B** (outcome-Q) if A points at the label.
4. **D** (anomaly) as the always-worth-it deployment hedge.
5. **E** (fine-tuning) only if A or C says the frozen features are the problem.

The single highest-value run is **A**, because full ground-truth state is the upper bound on anything state-predictable, and the 1.6×/1.0× gap already present on SigLIP is the fingerprint A would reproduce if the label is the culprit. If A reproduces that gap on ground-truth state (proprioception + object pose), the attribution is settled without needing B at all.

---

## Practical Note: Object Detection and Camera Calibration (2026-06-30)

**We already have the detector.** The `lerobot-target-annotator` tool (`leanton/lerobot-target-annotator/`) provides multi-stream ZMQ annotation with click-to-mark object centroids. For the pick-and-place station, the object is a distinct colored cube on a uniform background — a simple color-threshold or contour detector is sufficient for 2D centroid extraction, no deep-learning model needed. The detection half of the prerequisite is done.

**The blocker is calibration.** The scene cameras are not perfectly overhead — they're mounted at an angle, producing perspective distortion. This means:

- **2D pixel centroids are not a rotation/translation-invariant state representation.** The same object position in world coordinates maps to different pixel coordinates depending on camera angle and distance. A state vector built from raw 2D centroids would contain camera-specific distortion that doesn't generalize across episodes or camera configurations — exactly the kind of non-transferable variance we're trying to diagnose.

- **Calibrated 3D coordinates solve this.** With camera intrinsics and extrinsics (or a homography from a known calibration grid), 2D centroids from two or more cameras triangulate to a 3D object position in robot-frame coordinates. This 3D position, combined with the 6 joint angles, forms a ~9-dimensional state vector that is the ground-truth physical description of the robot+object system — invariant to viewpoint, lighting, and episode-specific camera placement.

**What calibration requires:**

1. **Intrinsic calibration** (camera matrix + distortion coefficients): a one-time per-camera procedure using a checkerboard pattern. Standard OpenCV tooling (`cv2.calibrateCamera`).
2. **Extrinsic calibration** (camera pose relative to robot base): requires known 3D points visible to both the camera and the robot. If the pick-and-place station has fixed fiducials or the robot can touch known points, this is a one-time solve. Alternatively, a homography from a ground-plane calibration grid is sufficient if the object moves in a plane (likely for a tabletop pick-and-place task).
3. **Triangulation:** with two calibrated cameras, 2D centroids → 3D position via `cv2.triangulatePoints`. With one calibrated camera + known table height, a ray-plane intersection gives 3D from a single 2D point.

**Fallback if calibration is blocked:** 2D centroids from a single camera, normalized by the table surface area visible in that camera (to remove scale variance), may be a usable intermediate representation for Experiment A. The signal will be noisier than with calibrated 3D coordinates, and a negative result (1.0×) would be inconclusive — it could mean either "state doesn't determine Q" or "2D distortion masked the signal." A positive result (>~1.5×), however, would be informative despite the distortion.

**Status:** Detector available. Calibration pending. This is the concrete blocker for Experiment A.

---

## Conclusions and Practical Pivot (post-discussion)

### Data constraint acknowledged

The 3D / metric version of A is blocked. The data is real (not sim), and metric object pose would require a calibrated multi-camera system with a registered table frame, which is not available now. So "full ground-truth 3D state" as an input is off the table for the time being. That is accepted.

### But the discriminating core of A survives in 2D image space

A's purpose is not metrology, it is to test whether the grasp-relevant geometry was compressed away by the frozen encoders. That geometry is largely **2D-visible with no calibration at all**. The failure modes that drive takeovers (object slipping out, misaligned grasp, object dropped, drift off target) all show up in a single view as the relative position of the object centroid and the gripper. "Object sliding out of the gripper" does not need metric depth or a world frame; it needs object-in-pixels and gripper-in-pixels.

**Practical A (minimal, no calibration, no table frame):** feed the head, per frame, proprioception (joint angles + gripper width, already in the dataset) plus, for each camera, the 2D pixel position of the **object centroid** only (optionally object mask area / aspect ratio as a rough scale proxy). That is the entire input. No gripper detector, no calibration, no ArUco board.

**Why no gripper endpoint is needed:** forward kinematics (joint angles to gripper 3D pose) is an exact, smooth function an MLP learns trivially, and under a static camera rig the projection to image pixels is also a fixed smooth function. So the model recovers the gripper's image position implicitly from the joints and can compute the gripper-to-object relationship for itself. The camera rig must be static across episodes (safe assumption for a fixed 3-cam setup). The gripper endpoint can be added later as a sample-efficiency ablation if the minimal run is noisy, but it is not on the critical path.

**Why the object centroid is the one irreducible input:** the object is external. There is no function from joint angles to object position, because the same arm configuration can coexist with the object anywhere. So the object centroid, however crude, is the one piece of scene information that must come from the pixels. This is the non-redundant part of A.

This input is an **un-compressed representation of the scene with respect to Q**: nothing in this pipeline was optimized for action prediction or image-text contrastive, so nothing had an incentive to discard object location. That is the property A needs.

### Tiered fallback

- **Tier 0 (free today):** proprioception only. As a *clustering* test it is confounded (see Rev1 correction), but as a *predictor input* it is a valid lower bound. If it surprisingly predicts Q, the signal is in the arm and we are done. If it fails, it is uninformative on its own but costs zero.
- **Tier 1 (cheap, no calibration):** add the 2D object centroid per camera (the only detector you actually need). **This is the real A.**
- **Tier 2 (deferred):** calibrated 3D, if a one-time calibration ever becomes available. The same 2D detector outputs feed straight in.

Crude detection is fine here: we are testing whether the signal exists at all, not measuring pose to the millimeter. A noisy 2D detector adds some noise but cannot fabricate an absent signal or hide a strong one.

### Honest weakening of the 2D version

2D image-space state is no longer the theoretical ceiling, so a null result is slightly less airtight than the 3D version: strictly, "2D object + proprioception fails" leaves a small gap where one could argue the missing metric depth was the relevant dimension. In practice that gap is small (the dominant failure modes are 2D-visible, and SigLIP already fails for the same reason). The **success** branch is not weakened at all: if 2D object geometry predicts Q where the VLM and SigLIP do not, the frozen encoders compressed away object location, full stop.

### Correction: temporal depth is 120 frames, not 12

The temporal context already explored was **~120 frames (causal transformer) and ~90 frames (TCN)**, i.e. roughly 3 to 4 seconds at 30 Hz, not 12 frames. This removes a confound I had raised: the temporal models already had ample context to cover the 1 to 3 s failure horizon, so "insufficient temporal depth" is **not** an explanation for the negative result. If anything, the temporal models failing with adequate depth slightly strengthens the negative result. Practical consequence for A: the 120-frame window used today can be reused as-is on the low-dim state vector (15 dims times 120 is trivially small), and explicit joint/object velocities can be appended as extra input dims so a single frame already carries the "is it slipping" signal.

### What A still decides (unchanged)

- **2D state generalizes on the contiguous holdout** → the VLM/SigLIP encoders compressed away object-location geometry. Features are the bottleneck. Justifies E.
- **2D state shows the SigLIP fingerprint (weak in-sample, 1.0× holdout)** → signal is real but non-transferable; the human-takeover label injects per-episode variance no representation can capture. Go to B.
- **2D state shows nothing, not even in-sample** → Q is not in the observable state. Strong negative. Go to D.

The single highest-value run is still **A**, now in its Tier 1 (2D, no-calibration) form, runnable today without any new camera hardware.
