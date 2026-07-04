#!/usr/bin/env python3
"""
build_q_targets.py — Compute Q-targets from intervention arrays using the symmetric
onset-anchored discount policy.

POLICY (see Chapter 3):
    A-Type (autonomous): discount BACKWARDS from onset, Q = -γ^dist, Q = -1.0 at onset-1.
    B-Type (correction):  discount FORWARDS from onset,  Q = +γ^k,    Q = +1.0 at onset.
    Both decay to |Q| < 0.05 within ~3s (γ=0.965).

    No ramp — the onset boundary is a clean ±1.0 jump (policy failure → human takeover).

USAGE:
    # Build Q for all 21 rounds (requires lerobot + HF token)
    python build_q_targets.py --all

    # Build Q for specific rounds
    python build_q_targets.py --rounds 1,17,19,20,21

    # Build Q from a raw intervention numpy array (no LeRobot needed)
    python build_q_targets.py --intervention-file intervention.npy --n-frames 5000

    # Save Q to a specific directory
    python build_q_targets.py --all --out-dir ./q_targets/

OUTPUT:
    q_targets_r{NN}.npy — float32 array of shape (n_frames,) per round
"""

import argparse
import sys
from pathlib import Path

import numpy as np

GAMMA = 0.965
DECAY_THRESHOLD = 0.05

# ── Git SHAs that preserve the intervention column for each round ──
INTERVENTION_COMMITS = {
    1: "c797adf803b9", 2: "83219b18943c", 3: "2ec5f11696bd",
    4: "68c23464107a", 5: "59368530680e", 6: "d5c3d680ca9d",
    7: "87b8437392b5", 8: "06d416f032fa", 9: "7243a7f43897",
    10: "af176244849f", 11: "1aad77928a77", 12: "27e24ee172b4",
    13: "c1e519f03cb3", 14: "b205d2abf488", 15: "97dfc02110c7",
    16: "4f3089621e88", 17: "5939b69b7cc1", 18: "bea57c52b8f3",
    19: "07c975caee23", 20: "af4d7e44c4e0", 21: "152d5112f0c4",
}


def extract_segments(intervention: np.ndarray, min_length: int = 1) -> list[dict]:
    """Parse intervention array into alternating A-type and B-type segments.

    Args:
        intervention: bool array, True = correction (B), False = autonomous (A).
        min_length: minimum segment length to keep (default 1; v6: was 2, dropped 1-frame corrections).

    Returns:
        List of segment dicts with keys: type, start, end, length, onset_frame, has_onset.
    """
    arr = np.array(intervention, dtype=bool)
    n = len(arr)
    segments = []
    i = 0

    while i < n:
        val = arr[i]
        start = i
        while i < n and arr[i] == val:
            i += 1
        end = i
        length = end - start

        if length < min_length:
            continue

        if not val:
            # A-type (autonomous): onset is the frame AFTER this segment ends
            onset_frame = end if end < n else None
            segments.append({
                "type": "A", "start": start, "end": end, "length": length,
                "onset_frame": onset_frame,
                "has_onset": onset_frame is not None,
            })
        else:
            # B-type (correction): onset IS the start of this segment
            segments.append({
                "type": "B", "start": start, "end": end, "length": length,
                "onset_frame": start,
            })

    return segments


def compute_q_targets(n_frames: int, segments: list[dict]) -> np.ndarray:
    """Compute q_target for every frame from segment structure.

    A-type: Q decays backwards from onset: Q[t] = -γ^(onset - t - 1)
    B-type: Q decays forwards from onset:  Q[t] = +γ^(t - onset)

    Args:
        n_frames: total number of frames.
        segments: output of extract_segments().

    Returns:
        float32 array of shape (n_frames,) with Q ∈ [-1, 1].
    """
    q = np.zeros(n_frames, dtype=np.float32)

    for seg in segments:
        if seg["type"] == "A" and seg["has_onset"]:
            onset = seg["onset_frame"]
            # Backward decay: Q[onset-1] = -1.0, Q[onset-2] = -γ, ...
            for t in range(onset - 1, seg["start"] - 1, -1):
                dist = onset - t - 1
                val = -1.0 * (GAMMA ** dist)
                if abs(val) < DECAY_THRESHOLD:
                    break
                q[t] = float(val)

        elif seg["type"] == "A" and not seg["has_onset"]:
            pass  # trailing autonomous, Q = 0

        elif seg["type"] == "B":
            onset = seg["onset_frame"]
            # Forward decay: Q[onset] = +1.0, Q[onset+1] = +γ, ...
            for t in range(onset, seg["end"]):
                k = t - onset
                val = GAMMA ** k
                if val < DECAY_THRESHOLD:
                    break
                q[t] = float(val)

    return q


def build_from_dataset(rounds: list[int], out_dir: Path) -> dict[int, np.ndarray]:
    """Build Q targets by loading intervention arrays from the HF dataset.

    Requires: pip install lerobot datasets huggingface_hub
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("ERROR: lerobot not installed. pip install lerobot")
        print("Or use --intervention-file to provide a raw intervention array.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for r in rounds:
        sha = INTERVENTION_COMMITS[r]
        repo_id = f"anikitakis/rollout_pick_n_place_dagger_r{r}"

        print(f"r{r:02d} @ {sha} ... ", end="", flush=True)
        ds = LeRobotDataset(repo_id, revision=sha)
        intervention = np.array(ds.hf_dataset["intervention"], dtype=bool)
        n_frames = ds.num_frames

        segments = extract_segments(intervention)
        q_targets = compute_q_targets(n_frames, segments)

        out_path = out_dir / f"q_targets_r{r:02d}.npy"
        np.save(out_path, q_targets)

        a_count = sum(1 for s in segments if s["type"] == "A")
        b_count = sum(1 for s in segments if s["type"] == "B")
        nonzero = int((np.abs(q_targets) >= DECAY_THRESHOLD).sum())
        neg = int((q_targets <= -DECAY_THRESHOLD).sum())
        pos = int((q_targets >= DECAY_THRESHOLD).sum())

        print(f"frames={n_frames:>5}  A={a_count:>2}  B={b_count:>2}  "
              f"Q≠0={nonzero:>4} ({nonzero/n_frames*100:4.1f}%)  "
              f"neg={neg:>4}  pos={pos:>4}  → {out_path.name}")

        results[r] = q_targets

    return results


def build_from_array(intervention: np.ndarray, out_dir: Path, label: str = "custom") -> np.ndarray:
    """Build Q targets from a raw intervention array (no LeRobot needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(intervention)

    segments = extract_segments(intervention)
    q_targets = compute_q_targets(n_frames, segments)

    out_path = out_dir / f"q_targets_{label}.npy"
    np.save(out_path, q_targets)

    a_count = sum(1 for s in segments if s["type"] == "A")
    b_count = sum(1 for s in segments if s["type"] == "B")
    nonzero = int((np.abs(q_targets) >= DECAY_THRESHOLD).sum())
    neg = int((q_targets <= -DECAY_THRESHOLD).sum())
    pos = int((q_targets >= DECAY_THRESHOLD).sum())

    print(f"frames={n_frames:>5}  A={a_count:>2}  B={b_count:>2}  "
          f"Q≠0={nonzero:>4} ({nonzero/n_frames*100:4.1f}%)  "
          f"neg={neg:>4}  pos={pos:>4}  → {out_path.name}")

    return q_targets


def print_summary(all_q: dict[int, np.ndarray]) -> None:
    """Print aggregate statistics across all rounds."""
    combined = np.concatenate(list(all_q.values()))
    nonzero_q = combined[np.abs(combined) >= DECAY_THRESHOLD]

    print(f"\n{'=' * 60}")
    print("Q-TARGET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  γ = {GAMMA}  |  decay threshold = {DECAY_THRESHOLD}")
    print(f"  Total frames:          {len(combined):>6}")
    print(f"  Nonzero Q (|Q|≥0.05):   {len(nonzero_q):>6} ({len(nonzero_q)/len(combined)*100:5.2f}%)")
    print(f"  Neutral Q (|Q|<0.05):   {len(combined)-len(nonzero_q):>6} ({(len(combined)-len(nonzero_q))/len(combined)*100:5.2f}%)")

    if len(nonzero_q) > 0:
        print(f"\n  Q distribution (non-zero frames):")
        print(f"    Strong neg  [-1.0, -0.5): {int(((nonzero_q >= -1.0) & (nonzero_q < -0.5)).sum()):>5}")
        print(f"    Medium neg  [-0.5, -0.1): {int(((nonzero_q >= -0.5) & (nonzero_q < -0.1)).sum()):>5}")
        print(f"    Weak neg   [-0.1, -0.05): {int(((nonzero_q >= -0.1) & (nonzero_q < -0.05)).sum()):>5}")
        print(f"    Weak pos    (0.05, 0.1]: {int(((nonzero_q > 0.05) & (nonzero_q <= 0.1)).sum()):>5}")
        print(f"    Medium pos  (0.1, 0.5]:  {int(((nonzero_q > 0.1) & (nonzero_q <= 0.5)).sum()):>5}")
        print(f"    Strong pos  (0.5, 1.0]:  {int(((nonzero_q > 0.5) & (nonzero_q <= 1.0)).sum()):>5}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Q-targets from intervention arrays (onset-anchored discount policy)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Build Q for all 21 rounds")
    group.add_argument("--rounds", type=str, help="Comma-separated round numbers, e.g. 1,17,19")
    group.add_argument("--intervention-file", type=str, help="Path to a .npy file with a raw intervention bool array")
    parser.add_argument("--n-frames", type=int, help="Total frames (only with --intervention-file)")
    parser.add_argument("--out-dir", type=str, default="./q_targets", help="Output directory (default: ./q_targets)")
    parser.add_argument("--summary-only", action="store_true", help="Print summary of existing .npy files and exit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.summary_only:
        existing = sorted(out_dir.glob("q_targets_r*.npy"))
        if not existing:
            print(f"No q_targets_r*.npy files found in {out_dir}")
            return
        all_q = {}
        for p in existing:
            r = int(p.stem.split("_r")[-1])
            all_q[r] = np.load(p)
        print_summary(all_q)
        return

    if args.intervention_file:
        intervention = np.load(args.intervention_file)
        intervention = np.array(intervention, dtype=bool)
        label = Path(args.intervention_file).stem
        q = build_from_array(intervention, out_dir, label)
        print_summary({label: q})

    else:
        rounds = list(range(1, 22)) if args.all else [int(x.strip()) for x in args.rounds.split(",")]
        results = build_from_dataset(rounds, out_dir)
        print_summary(results)


if __name__ == "__main__":
    main()
