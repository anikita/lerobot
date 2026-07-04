#!/usr/bin/env python3
"""
estimate_ceiling.py — Model-free R²(Q | state+phase) ceiling diagnostic.

Answers: how much of Q's variance can be explained by physical state and task phase,
without training any model? This is the Bayes-optimal ceiling — the best any
predictor could achieve from state+phase observations.

METHOD:
    1. Load raw observation.state and q_target from all rounds.
    2. Normalize state per-round (joint angles have different zero-points).
    3. Bucket frames by state (k-means, 50 clusters) and phase (frame-position decile).
    4. Compute within-bucket Q variance → R² = 1 - Var_within / Var_total.
    5. Report R² decomposed by state alone, phase alone, and combined.

INTERPRETATION:
    R² > 0.5: Q is mostly state-determined. Pursue better features.
    R² < 0.2: Q is mostly not in state+phase. Redirect to intent-free detectors.
    0.2–0.5: Calibrated pursuit against measured ceiling.

USAGE:
    python estimate_ceiling.py
    python estimate_ceiling.py --good-rounds-only  # only rounds with >20% autonomous
"""

import sys, time
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
DEVICE = "cpu"  # no GPU needed


def load_state_and_q(good_rounds_only=True):
    """Load raw observation.state and q_target from all _with_q rounds."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    all_rounds = [f"r{i}_with_q" for i in range(1, 22)]  # r1..r21
    if good_rounds_only:
        # Filter to rounds with >20% autonomous (from 6.1)
        good = {f"r{i}_with_q" for i in range(1, 10)} | \
               {f"r{i}_with_q" for i in range(17, 22)}
        all_rounds = [r for r in all_rounds if r in good]

    states = []       # [total_frames, 32]
    q_values = []     # [total_frames]
    phases = []       # [total_frames] — normalized frame position [0, 1]
    round_ids = []    # [total_frames] — which round
    interventions = []  # [total_frames] — bool

    round_boundaries = []  # (round_name, start_idx, n_frames)

    print(f"Loading state and Q from {len(all_rounds)} rounds...")
    offset = 0

    for round_name in all_rounds:
        rn = round_name.lstrip("r").split("_")[0]
        repo = f"anikitakis/rollout_pick_n_place_dagger_r{rn}_with_q"

        try:
            ds = LeRobotDataset(repo, revision="main")
        except Exception as e:
            print(f"  SKIP {round_name}: {e}")
            continue

        n = len(ds)

        # Load state and Q from HF dataset columns (no video decoding)
        state_col = ds.hf_dataset["observation.state"][:n]
        q_col = ds.hf_dataset["q_target"][:n]
        interv_col = ds.hf_dataset["intervention"][:n]

        # Convert to numpy
        if hasattr(state_col, "numpy"):
            state_np = np.array(state_col)
        else:
            state_np = np.array(list(state_col))

        q_np = np.array(list(q_col), dtype=np.float32)
        interv_np = np.array(list(interv_col), dtype=bool)

        # Phase: normalized frame position within round [0, 1]
        phase_np = np.arange(n) / max(n - 1, 1)

        # Per-round state normalization (zero mean, unit variance)
        state_mean = state_np.mean(axis=0, keepdims=True)
        state_std = state_np.std(axis=0, keepdims=True) + 1e-8
        state_np = (state_np - state_mean) / state_std

        states.append(state_np)
        q_values.append(q_np)
        phases.append(phase_np)
        round_ids.append(np.full(n, len(round_boundaries), dtype=np.int32))
        interventions.append(interv_np)

        nz = (np.abs(q_np) > 0.05).sum()
        print(f"  {round_name:20s}  {n:5d} frames  "
              f"Q∈[{q_np.min():.1f},{q_np.max():.1f}]  nonzero={nz}  "
              f"state_dim={state_np.shape[1]}")

        round_boundaries.append((round_name, offset, n))
        offset += n

    states_all = np.concatenate(states, axis=0)
    q_all = np.concatenate(q_values, axis=0)
    phases_all = np.concatenate(phases, axis=0)
    round_ids_all = np.concatenate(round_ids, axis=0)
    interv_all = np.concatenate(interventions, axis=0)

    print(f"\nTotal: {len(q_all):,} frames across {len(round_boundaries)} rounds")
    print(f"  Q mean={q_all.mean():.4f}  std={q_all.std():.4f}  "
          f"nonzero={(np.abs(q_all)>0.05).sum():,}")
    print(f"  Interventions: {interv_all.sum():,} frames "
          f"({100*interv_all.mean():.1f}%)")

    return states_all, q_all, phases_all, round_ids_all, interv_all, round_boundaries


def compute_r_squared(q_all, bucket_ids):
    """Compute R² = 1 - Var_within / Var_total.

    Args:
        q_all: [N] — Q values
        bucket_ids: [N] — integer bucket labels

    Returns:
        r_squared: float
        var_total: total Q variance
        var_within: mean within-bucket Q variance
        n_buckets: number of unique buckets
        mean_samples_per_bucket: average frames per bucket
    """
    var_total = np.var(q_all)

    unique_buckets = np.unique(bucket_ids)
    within_vars = []
    bucket_sizes = []

    for b in unique_buckets:
        mask = bucket_ids == b
        n_in_bucket = mask.sum()
        if n_in_bucket >= 5:  # minimum samples for reliable variance estimate
            within_vars.append(np.var(q_all[mask]))
            bucket_sizes.append(n_in_bucket)

    if not within_vars:
        return 0.0, var_total, 0.0, 0, 0

    var_within = np.mean(within_vars)
    r_squared = 1.0 - var_within / var_total

    return r_squared, var_total, var_within, len(within_vars), int(np.mean(bucket_sizes))


def main():
    good_only = "--good-rounds-only" in sys.argv

    print("=" * 65)
    print("R²(Q | state+phase) Ceiling Diagnostic")
    print("=" * 65)
    print(f"  Good rounds only: {good_only}")
    print()

    # Load
    t0 = time.time()
    states, q_all, phases, round_ids, interv, round_info = load_state_and_q(good_only)
    N = len(q_all)
    D_state = states.shape[1]
    print(f"  Load time: {time.time() - t0:.0f}s")

    # ── 1. R²(Q | state) — physical state only ──
    print(f"\n--- R²(Q | state) ---")
    print(f"  Bucketing {N:,} frames × {D_state}-dim state with equal-width grid...")
    t1 = time.time()

    # Simple grid bucketing: discretize each state dimension into 3 equal-width bins
    # → 3^D_state buckets. For 6-dim: 729 buckets. For higher dims, use only top PCA dims.
    n_bins_per_dim = 3
    if D_state > 6:
        # Too many buckets — use random projection or just top dims
        state_for_bucket = states[:, :6]
    else:
        state_for_bucket = states

    state_clusters = np.zeros(N, dtype=np.int32)
    for d in range(state_for_bucket.shape[1]):
        col = state_for_bucket[:, d]
        # Equal-width bins: split range into n_bins_per_dim
        lo, hi = np.percentile(col, [1, 99])  # robust range
        bin_idx = np.clip(((col - lo) / (hi - lo + 1e-8) * n_bins_per_dim).astype(int),
                          0, n_bins_per_dim - 1)
        state_clusters = state_clusters * n_bins_per_dim + bin_idx

    r2_state, var_tot, var_within, n_buckets, avg_size = compute_r_squared(
        q_all, state_clusters)
    print(f"  R²(Q | state)        = {r2_state:.4f}")
    print(f"  Var_total             = {var_tot:.4f}")
    print(f"  Var_within (mean)     = {var_within:.4f}")
    print(f"  Buckets (≥5 samples)  = {n_buckets}")
    print(f"  Avg frames/bucket     = {avg_size}")
    print(f"  Cluster time: {time.time() - t1:.0f}s")

    # ── 2. R²(Q | phase) — temporal position only ──
    print(f"\n--- R²(Q | phase) ---")
    phase_deciles = np.clip((phases * 10).astype(int), 0, 9)
    r2_phase, _, var_within_phase, n_buckets_p, avg_size_p = compute_r_squared(
        q_all, phase_deciles)
    print(f"  R²(Q | phase)        = {r2_phase:.4f}")
    print(f"  Var_total             = {var_tot:.4f}")
    print(f"  Var_within (mean)     = {var_within_phase:.4f}")
    print(f"  Buckets (≥5 samples)  = {n_buckets_p}")
    print(f"  Avg frames/bucket     = {avg_size_p}")

    # ── 3. R²(Q | state + phase) — combined ──
    print(f"\n--- R²(Q | state + phase) ---")
    # Each bucket = state_cluster * 10 + phase_decile → up to 500 buckets
    combined_buckets = state_clusters * 10 + phase_deciles
    r2_combined, _, var_within_comb, n_buckets_c, avg_size_c = compute_r_squared(
        q_all, combined_buckets)
    print(f"  R²(Q | state+phase)  = {r2_combined:.4f}")
    print(f"  Var_total             = {var_tot:.4f}")
    print(f"  Var_within (mean)     = {var_within_comb:.4f}")
    print(f"  Buckets (≥5 samples)  = {n_buckets_c}")
    print(f"  Avg frames/bucket     = {avg_size_c}")

    # ── 4. Per-round R²(Q | state) — check consistency ──
    print(f"\n--- Per-round R²(Q | state) ---")
    print(f"  {'Round':<20s} {'N':>6s} {'R²':>8s} {'Var_total':>10s}")
    print(f"  {'-'*46}")
    for ri, (name, start, n) in enumerate(round_info):
        mask = round_ids == ri
        if mask.sum() < 50:
            continue
        # Cluster within this round's state
        round_states = states[mask]
        round_q = q_all[mask]
        if len(round_q) < 100:
            continue
        # Grid bucketing for this round
        round_clusters = np.zeros(len(round_q), dtype=np.int32)
        for d in range(min(round_states.shape[1], 6)):
            col = round_states[:, d]
            lo, hi = np.percentile(col, [1, 99])
            bin_idx = np.clip(((col - lo) / (hi - lo + 1e-8) * 3).astype(int), 0, 2)
            round_clusters = round_clusters * 3 + bin_idx
        r2_r, _, _, nb, sz = compute_r_squared(round_q, round_clusters)
        print(f"  {name:<20s} {n:>6d} {r2_r:>8.4f} {np.var(round_q):>10.4f}")

    # ── 5. R² with operator identity (round_id as proxy) ──
    print(f"\n--- R²(Q | round_id) [operator proxy] ---")
    r2_round, _, var_within_round, nb_r, sz_r = compute_r_squared(q_all, round_ids)
    print(f"  R²(Q | round_id)     = {r2_round:.4f}")
    print(f"  (round_id proxies operator state + task difficulty + lighting + ...)")

    # ── 6. Subset analysis: only nonzero Q frames ──
    nz_mask = np.abs(q_all) > 0.05
    if nz_mask.sum() > 100:
        print(f"\n--- Subset: nonzero Q frames ({nz_mask.sum():,} frames) ---")
        # State bucketing for nonzero subset
        nz_states = states[nz_mask]
        nz_clusters = np.zeros(nz_mask.sum(), dtype=np.int32)
        for d in range(min(nz_states.shape[1], 6)):
            col = nz_states[:, d]
            lo, hi = np.percentile(col, [1, 99])
            bin_idx = np.clip(((col - lo) / (hi - lo + 1e-8) * 3).astype(int), 0, 2)
            nz_clusters = nz_clusters * 3 + bin_idx

        for label, buckets in [
            ("state", nz_clusters),
            ("phase", phase_deciles[nz_mask]),
            ("state+phase", nz_clusters * 10 + phase_deciles[nz_mask]),
        ]:
            r2_nz, _, _, nb_nz, sz_nz = compute_r_squared(q_all[nz_mask], buckets)
            print(f"  R²(Q_nz | {label:<12s}) = {r2_nz:.4f}  "
                  f"(buckets={nb_nz}, avg_size={sz_nz})")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")
    print(f"  R²(Q | state)         = {r2_state:.4f}")
    print(f"  R²(Q | phase)         = {r2_phase:.4f}")
    print(f"  R²(Q | state+phase)   = {r2_combined:.4f}")
    print(f"  R²(Q | round_id)      = {r2_round:.4f}")
    print()
    print(f"  Total Q variance:     {var_tot:.4f}  (MSE null baseline)")
    print(f"  Residual (state+phase): {var_within_comb:.4f}  "
          f"(best achievable MSE from state+phase)")

    ceiling_mse = var_within_comb
    null_mse = var_tot
    improvable = null_mse - ceiling_mse
    print(f"  Improvable fraction:  {improvable:.4f}  "
          f"({100*improvable/null_mse:.1f}% of null, "
          f"max {null_mse/ceiling_mse:.1f}× over null)")

    if r2_combined > 0.5:
        print(f"\n  ✅ R² > 0.5 — Q is mostly state-determined.")
        print(f"     Pursue features: RLTQHead, LoRA, spatial object position.")
        print(f"     Ceiling improvement: {null_mse/ceiling_mse:.1f}× over null.")
    elif r2_combined > 0.2:
        print(f"\n  ⚠️  R² 0.2–0.5 — Mixed. Some state signal, substantial unmodeled variance.")
        print(f"     Calibrated pursuit. Ceiling improvement: {null_mse/ceiling_mse:.1f}× over null.")
    else:
        print(f"\n  ❌ R² < 0.2 — Most Q-variance is not in state+phase.")
        print(f"     Redirect to intent-free detectors (self-consistency, action-disagreement).")
        print(f"     Ceiling improvement: {null_mse/ceiling_mse:.1f}× over null.")

    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
