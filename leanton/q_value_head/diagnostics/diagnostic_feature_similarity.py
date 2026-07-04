#!/usr/bin/env python3
"""
Diagnostic: does the feature space encode failure state or round identity?

For same-sign Q frames, compares within-round vs cross-round cosine similarity.
If within-round >> cross-round for the same Q → features dominated by round identity.
"""

import sys
from pathlib import Path
import torch
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_similarities(h_all, q_all, r_all):
    """Compute within-round vs cross-round cosine sim per Q class."""
    N = h_all.shape[0]

    neg_mask = q_all < -0.05
    pos_mask = q_all > 0.05
    neu_mask = q_all.abs() <= 0.05

    h_norm = h_all / (h_all.norm(dim=1, keepdim=True) + 1e-8)
    n_pairs = 20000

    results = {}
    # Same-class comparisons
    for name, mask in [("neg", neg_mask), ("pos", pos_mask), ("neu", neu_mask)]:
        idx = torch.where(mask)[0]
        if len(idx) < 100:
            results[name] = {"within": 0, "cross": 0, "ratio": 0}
            continue

        i = idx[torch.randint(0, len(idx), (n_pairs,))]
        j = idx[torch.randint(0, len(idx), (n_pairs,))]

        same_round = r_all[i] == r_all[j]
        sim = (h_norm[i] * h_norm[j]).sum(dim=1)

        results[name] = {
            "within": sim[same_round].mean().item(),
            "cross": sim[~same_round].mean().item(),
            "ratio": sim[same_round].mean().item() / sim[~same_round].mean().item()
                     if sim[~same_round].mean().item() > 0 else float("inf"),
        }

    # Cross-class: neg vs pos
    neg_idx = torch.where(neg_mask)[0]
    pos_idx = torch.where(pos_mask)[0]
    if len(neg_idx) >= 100 and len(pos_idx) >= 100:
        ni = neg_idx[torch.randint(0, len(neg_idx), (n_pairs,))]
        pi = pos_idx[torch.randint(0, len(pos_idx), (n_pairs,))]
        same_round_x = r_all[ni] == r_all[pi]
        sim_x = (h_norm[ni] * h_norm[pi]).sum(dim=1)
        results["neg→pos"] = {
            "within": sim_x[same_round_x].mean().item(),
            "cross": sim_x[~same_round_x].mean().item(),
            "ratio": sim_x[same_round_x].mean().item() / sim_x[~same_round_x].mean().item()
                     if sim_x[~same_round_x].mean().item() > 0 else float("inf"),
        }
    return results


def print_results(results, title):
    print(f"\n  {title}")
    print(f"  {'Comparison':<12} {'Within':>10} {'Cross':>10} {'Ratio':>8}  {'Verdict'}")
    print(f"  {'-'*56}")
    for name in ["neg", "pos", "neu", "neg→pos"]:
        r = results.get(name)
        if r is None: continue
        if name == "neg→pos":
            tag = "DIFF CLASS"
        elif r["ratio"] > 2.0:
            tag = "⚠️ round ID"
        else:
            tag = "✅ invariant"
        print(f"  {name:<12} {r['within']:>10.4f} {r['cross']:>10.4f} {r['ratio']:>8.2f}  {tag}")


def main():
    import time
    t0 = time.time()

    for features_dir_name, file_pattern, field, dim_name, pool_dim in [
        ("data/features_v2", "*_seq.pt", "hidden_states_seq", "prefix", 960),
        ("data/features_v3", "*_suffix.pt", "suffix_states", "suffix", 720),
    ]:
        features_dir = SCRIPT_DIR / features_dir_name
        files = sorted(features_dir.glob(file_pattern))
        if len(files) < 2:
            print(f"\n{features_dir_name}: {len(files)} file(s) found — need 2+ for cross-round test")
            continue

        print(f"\n{'='*50}")
        print(f"  {features_dir_name}/{file_pattern}  ({dim_name}, {pool_dim}-dim)")
        print(f"  {len(files)} rounds")
        print(f"{'='*50}")

        all_h, all_q, all_round = [], [], []

        for ri, f in enumerate(files):
            data = torch.load(f, weights_only=True, mmap=True, map_location="cpu")
            h = data[field]          # keep as mmap, don't .float()
            q = data["q_targets"]

            n_sample = min(2000, h.shape[0])
            idx = torch.randperm(h.shape[0])[:n_sample]

            # Index into mmap and mean-pool — only loads sampled rows
            h_pooled = h[idx].float().mean(dim=1) if h.dim() == 3 else h[idx].float()
            all_h.append(h_pooled)
            all_q.append(q[idx])
            all_round.append(torch.full((n_sample,), ri, dtype=torch.long))

        h_all = torch.cat(all_h, dim=0)
        q_all = torch.cat(all_q, dim=0)
        r_all = torch.cat(all_round, dim=0)
        print(f"  Total frames sampled: {h_all.shape[0]:,}")

        results = compute_similarities(h_all, q_all, r_all)
        print_results(results, f"{dim_name} features (mean-pooled, {pool_dim}-dim)")

    print(f"\nDone in {time.time()-t0:.1f}s")

    # --- Class masks ---
    neg_mask = q_all < -0.05      # ~15% of nonzero Q
    pos_mask = q_all > 0.05
    neu_mask = q_all.abs() <= 0.05

    # --- Normalize ---
    h_norm = h_all / (h_all.norm(dim=1, keepdim=True) + 1e-8)

    # --- Sample pairs ---
    n_pairs = 20000  # sample this many pairs per condition

    import time
    t0 = time.time()

    results = {}
    for name, mask in [("neg", neg_mask), ("pos", pos_mask), ("neu", neu_mask)]:
        idx = torch.where(mask)[0]
        if len(idx) < 100:
            results[name] = {"within": 0, "cross": 0, "n": 0}
            continue

        # Random pairs
        i = idx[torch.randint(0, len(idx), (n_pairs,))]
        j = idx[torch.randint(0, len(idx), (n_pairs,))]

        same_round = r_all[i] == r_all[j]
        diff_round = ~same_round

        sim = (h_norm[i] * h_norm[j]).sum(dim=1)  # cosine similarity

        within_sim = sim[same_round].mean().item()
        cross_sim = sim[diff_round].mean().item()

        n_same = same_round.sum().item()
        n_diff = diff_round.sum().item()

        results[name] = {
            "within": within_sim,
            "cross": cross_sim,
            "ratio": within_sim / cross_sim if cross_sim > 0 else float("inf"),
            "n_same": n_same,
            "n_diff": n_diff,
        }

    # --- Print ---
    print(f"{'Class':<8} {'Within-Round CosSim':>20} {'Cross-Round CosSim':>20} {'Ratio (W/C)':>14}")
    print("-" * 65)
    for name in ["neg", "pos", "neu"]:
        r = results[name]
        tag = "⚠️ ROUND IDENTITY" if r["ratio"] > 2.0 else "✅ failure encoding"
        print(f"{name:<8} {r['within']:>20.4f} {r['cross']:>20.4f} {r['ratio']:>14.2f}  {tag}")

    print(f"\n{'='*50}")
    print(f"  Interpretation:")
    print(f"    Ratio > 2:  features dominated by round identity")
    print(f"    Ratio ~ 1:  features encode state invariantly")
    print(f"    Lower absolute cos_sim = more discriminative features")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
