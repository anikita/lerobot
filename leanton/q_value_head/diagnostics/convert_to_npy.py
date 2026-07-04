#!/usr/bin/env python3
"""
convert_to_npy.py — Convert existing _seq.pt files to memory-mappable .npy.

Reads features_v2/*_seq.pt, saves:
    {key}_seq.npy       — hidden_states_seq as float16 .npy (memory-mappable)
    {key}_seq_meta.pt   — q_targets, interventions, frame_indices, metadata (tiny)

Original .pt files are KEPT untouched.

USAGE:
    python convert_to_npy.py                     # convert all
    python convert_to_npy.py r1_with_q            # single round
"""

import sys
from pathlib import Path
import torch
import numpy as np

SCRIPT_DIR = Path(__file__).parent
FEATURES_DIR = SCRIPT_DIR / "data/features_v2"
SUFFIX = "_seq.pt"
NEW_SUFFIX = "_seq.npy"
META_SUFFIX = "_seq_meta.pt"


def convert_one(pt_file):
    key = pt_file.stem  # e.g. "r1_with_q_seq"
    npy_file = FEATURES_DIR / (key.replace("_seq", "") + "_seq.npy")
    meta_file = FEATURES_DIR / (key.replace("_seq", "") + "_seq_meta.pt")

    print(f"  {pt_file.name} → {npy_file.name} + {meta_file.name}")

    data = torch.load(pt_file, weights_only=True, map_location="cpu")
    h_seq = data["hidden_states_seq"]  # [N, S, D], bf16

    # Convert to float16 numpy (bf16 → float32 → float16 for .npy compatibility)
    arr = h_seq.float().numpy().astype(np.float16)
    np.save(npy_file, arr)
    size_mb = npy_file.stat().st_size / (1024 * 1024)
    print(f"    .npy: {arr.shape} float16, {size_mb:.0f} MB")

    # Save tiny metadata
    meta = {
        "q_targets": data["q_targets"],
        "interventions": data["interventions"],
        "frame_indices": data["frame_indices"],
        "metadata": data.get("metadata", {}),
    }
    torch.save(meta, meta_file)
    meta_mb = meta_file.stat().st_size / (1024 * 1024)
    print(f"    .meta: {meta_mb:.1f} MB")


def main():
    files = sorted(FEATURES_DIR.glob(f"*{SUFFIX}"))
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if not key.endswith("_seq"):
            key = key + "_seq"
        if not key.endswith(".pt"):
            key = key + ".pt"
        target = FEATURES_DIR / key
        if not target.exists():
            print(f"Not found: {target}")
            return
        files = [target]

    if not files:
        print(f"No {SUFFIX} files in {FEATURES_DIR}")
        return

    print(f"Converting {len(files)} files from {FEATURES_DIR}\n")
    for f in files:
        convert_one(f)

    print(f"\nDone. {len(files)} files converted. Original .pt files untouched.")


if __name__ == "__main__":
    main()
