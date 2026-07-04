#!/usr/bin/env python3
"""
1_extract_siglip.py — Extract SigLIP vision features BEFORE the VLM.

SigLIP processes each camera image into patch embeddings. These are the raw visual
features — before any language model, state fusion, or action-prediction compression.
They see the gripper position relative to the target directly in pixel space.

WHAT THIS SAVES (features_siglip/):

  {round}_siglip_cam{N}.pt  — mean-pooled SigLIP per camera  [N, 960]
  {round}_siglip_cat.pt      — all cameras concatenated        [N, n_cam×960]

The per-camera files enable ablation: "only camera 1", "camera 2+3", etc.
The concatenated file is ready-to-train: --feature-field siglip_cat

POOLING: each camera image → SigLIP → [64, 960] patch embeddings.
Mean-pool collapses the 64 patches into one [960] vector per camera:
  pooled[j] = mean(embedding[0], embedding[1], ..., embedding[63])
This loses spatial structure (which patch is the gripper?) but preserves the
global visual content of each camera view.

CONCAT: per-camera pooled vectors are stacked into one flat feature vector:
  camera1 [960] + camera2 [960] + camera3 [960] = [n_cam×960]
The model sees all camera views in one vector. Dimension varies by round:
  2-camera rounds: 1920-dim
  3-camera rounds: 2880-dim
Mixed training requires padding or same-camera rounds per batch.

USAGE:
    python 1_extract_siglip.py r1       # single round
    for r in 01 02 03 04 05 06 07 08 09 17 18 19 20 21; do
        python 1_extract_siglip.py r$r
    done

OUTPUT:
    features_siglip/
"""

from pathlib import Path
import time, sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# ── CONFIG ────────────────────────────────────────────────────────────────────
POLICY_PATH = "anikitakis/vla_so101_pick_n_place_full_expert"
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data/features_siglip"
CHECKPOINT_DIR = SCRIPT_DIR / "data/checkpoints_siglip"

BATCH_SIZE = 8
CHECKPOINT_EVERY = 1000
MAX_FRAMES = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RENAME_MAP = {
    "observation.images.annotated":   "observation.images.camera1",
    "observation.images.front":       "observation.images.camera2",
    "observation.images.target_patch": "observation.images.camera3",
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract SigLIP vision features (pre-VLM, per-camera).")
    parser.add_argument("round", nargs="?", default=None,
                        help="Round identifier (e.g. r01, r1)")
    parser.add_argument("--raw-patches", action="store_true",
                        help="Also save full patch embeddings [N, n_cam, 64, 960]")
    args = parser.parse_args()

    # --- Resolve repo ---
    repo = "anikitakis/rollout_pick_n_place_dagger_r1_with_q"
    if args.round:
        arg = args.round
        if arg.startswith("anikitakis/"):
            repo = arg
        elif arg.startswith("r") and not arg.startswith("r_"):
            rn = arg.lstrip("r").lstrip("0")
            repo = f"anikitakis/rollout_pick_n_place_dagger_r{rn}_with_q"
        else:
            repo = f"anikitakis/rollout_pick_n_place_dagger_{arg}"
    round_key = repo.rstrip("/").split("/")[-1].replace("rollout_pick_n_place_dagger_", "")

    OUTPUT_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    print("=" * 65)
    print(f"1_extract_siglip.py — SigLIP Vision Feature Extraction")
    print("=" * 65)
    print(f"  HF repo:       {repo}")
    print(f"  Round key:     {round_key}")
    print(f"  Policy:        {POLICY_PATH}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print(f"  Raw patches:   {args.raw_patches}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Device:        {DEVICE}")
    print()

    # ── STEP 1: Load dataset ──
    print("[1/4] Loading dataset...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo, revision="main")
    N = len(ds) if MAX_FRAMES is None else min(len(ds), MAX_FRAMES)
    dur_min = N / 20 / 60
    print(f"       {N:,} frames ({dur_min:.0f} min at 20 fps)")
    has_q = "q_target" in ds.hf_dataset.column_names
    print(f"       q_target present: {has_q}")
    if not has_q:
        print("       ❌ q_target column missing — this round is not _with_q!")
        return

    # Determine camera keys present in this round
    sample_item = ds[0]
    camera_keys = sorted(
        [k for k in sample_item.keys() if k.startswith("observation.images.")],
        key=lambda x: RENAME_MAP.get(x, x))
    n_cameras = len(camera_keys)
    print(f"       Cameras: {n_cameras}  ({[RENAME_MAP.get(k, k) for k in camera_keys]})")

    # ── STEP 2: Load policy ──
    print(f"\n[2/4] Loading SmolVLA policy...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy.model.eval()
    policy.model.to(DEVICE)
    for p in policy.model.parameters():
        p.requires_grad = False

    # Check the SigLIP output dimension
    test_item = ds[0]
    test_batch = {}
    for k in camera_keys:
        v = test_item[k]
        test_batch[RENAME_MAP.get(k, k)] = v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v
    for old, new in RENAME_MAP.items():
        if old in test_batch:
            test_batch[new] = test_batch.pop(old)
    images_test, _ = policy.prepare_images(test_batch)
    with torch.no_grad():
        test_emb = policy.model.vlm_with_expert.embed_image(images_test[0])
    siglip_dim = test_emb.shape[-1]
    siglip_patches = test_emb.shape[1]
    print(f"       SigLIP output: [{siglip_patches}, {siglip_dim}] per camera")
    print(f"       Per-camera pooled: [{siglip_dim}]")
    print(f"       Concat ({n_cameras} cams): [{n_cameras * siglip_dim}]")

    metadata = {
        "hf_repo": repo, "policy_path": POLICY_PATH,
        "siglip_dim": siglip_dim, "siglip_patches": siglip_patches,
        "n_cameras": n_cameras,
        "camera_keys": [RENAME_MAP.get(k, k) for k in camera_keys],
        "num_frames": N, "round_key": round_key,
    }

    # ── Resume ──
    ckpt_path, start_batch = find_checkpoint(round_key)

    def collate_to_device(batch_list):
        batch = {}
        for key in batch_list[0].keys():
            vals = [item[key] for item in batch_list]
            if isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals).to(DEVICE)
            elif isinstance(vals[0], str):
                batch[key] = vals
            else:
                batch[key] = torch.tensor(vals).to(DEVICE)
        return batch

    # ── STEP 3: Extract ──
    print(f"\n[3/4] Extracting SigLIP features...")
    print(f"       Frames: {N:,}")
    if start_batch > 0:
        print(f"       Resuming from batch {start_batch}")

    # Initialize accumulators
    if start_batch > 0:
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        all_cam = {i: [ckpt[f"cam_{i}"]] for i in range(n_cameras)
                   if f"cam_{i}" in ckpt}
        all_cat = [ckpt["cat"]] if "cat" in ckpt else []
        all_q = [ckpt["q_targets"]]
        all_interv = [ckpt["interventions"]]
        all_frame_idx = [ckpt["frame_indices"]]
    else:
        all_cam = {i: [] for i in range(n_cameras)}
        all_cat = []
        all_q, all_interv, all_frame_idx = [], [], []

    if start_batch > 0:
        from torch.utils.data import Subset
        ds_work = Subset(ds, range(start_batch * BATCH_SIZE, N))
    else:
        ds_work = ds

    loader = DataLoader(ds_work, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_to_device, num_workers=0)
    start_frame = start_batch * BATCH_SIZE
    pbar = tqdm(total=N, initial=start_frame, unit="frames", desc="       ", ncols=80)
    processed = 0
    t_batch_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            real_batch = start_batch + batch_idx

            if MAX_FRAMES is not None and pbar.n >= MAX_FRAMES:
                break
            bsize = len(batch["action"])

            # Rename camera keys
            for old_key, new_key in RENAME_MAP.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)

            # Rename camera keys to match policy expectations
            cam_batch = {}
            for k in camera_keys:
                new_k = RENAME_MAP.get(k, k)
                if new_k in batch:
                    cam_batch[k] = batch[new_k]

            # Prepare and run SigLIP per camera
            images, _ = policy.prepare_images(batch)

            per_cam_pooled = []
            for cam_idx, img in enumerate(images):
                # SigLIP embed → [B, 64, 960]
                emb = policy.model.vlm_with_expert.embed_image(img)
                # Mean-pool over patches → [B, 960]
                pooled = emb.float().mean(dim=1)
                per_cam_pooled.append(pooled.cpu())
                all_cam[cam_idx].append(pooled.cpu())

            # Concatenate all cameras → [B, n_cam×960]
            cat_features = torch.cat(per_cam_pooled, dim=1)
            all_cat.append(cat_features)

            all_q.append(batch["q_target"].cpu())
            all_interv.append(batch["intervention"].cpu())
            all_frame_idx.append(batch["frame_index"].cpu())

            pbar.update(bsize)
            processed += bsize

            if real_batch % 10 == 0:
                elapsed = time.time() - t_batch_start
                fps = processed / elapsed if elapsed > 0 else 0
                remaining = N - pbar.n
                eta = remaining / fps if fps > 0 else 0
                pbar.set_postfix({"fps": f"{fps:.1f}", "eta": f"{eta/60:.0f}m"})

            if real_batch > 0 and real_batch % CHECKPOINT_EVERY == 0:
                save_checkpoint(round_key, real_batch,
                                all_cam, all_cat, all_q, all_interv, all_frame_idx,
                                metadata, n_cameras)

    pbar.close()
    t_extract = time.time() - t_batch_start
    print(f"\n       Done: {pbar.n:,} frames in {t_extract/60:.0f}m "
          f"({pbar.n/t_extract:.1f} fps)")

    # ── STEP 4: Save ──
    print(f"\n[4/4] Saving...")
    q_t = torch.cat(all_q, dim=0)
    interv_t = torch.cat(all_interv, dim=0)
    frame_t = torch.cat(all_frame_idx, dim=0)
    N_final = q_t.shape[0]

    base = {
        "q_targets": q_t,
        "interventions": interv_t,
        "frame_indices": frame_t,
        "metadata": metadata,
    }

    # Per-camera files
    for cam_idx in range(n_cameras):
        cam_h = torch.cat(all_cam[cam_idx], dim=0)  # [N, 960]
        cam_data = {**base, "hidden_states": cam_h}
        path = OUTPUT_DIR / f"{round_key}_siglip_cam{cam_idx}.pt"
        torch.save(cam_data, path)
        print(f"  Camera {cam_idx}   → {path}  "
              f"({path.stat().st_size/1024**2:.0f} MB)  [{list(cam_h.shape)}]")

    # Concatenated file
    cat_h = torch.cat(all_cat, dim=0)  # [N, n_cam×960]
    cat_data = {**base, "hidden_states": cat_h}
    path = OUTPUT_DIR / f"{round_key}_siglip_cat.pt"
    torch.save(cat_data, path)
    print(f"  Concatenated → {path}  "
          f"({path.stat().st_size/1024**2:.0f} MB)  [{list(cat_h.shape)}]")

    cleanup_checkpoints(round_key)
    t_total = time.time() - t_start
    print(f"\n✅ {round_key}: {N_final:,} frames → {OUTPUT_DIR}/")
    print(f"   Time: {t_total/60:.0f}m  |  Throughput: {N_final/t_total:.1f} fps")


# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────

def find_checkpoint(round_key):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    prefix = f"checkpoint_{round_key}_"
    existing = sorted(CHECKPOINT_DIR.glob(f"{prefix}*.pt"))
    if not existing:
        return None, 0
    latest = existing[-1]
    try:
        ckpt = torch.load(latest, weights_only=True, map_location="cpu")
        return latest, ckpt.get("batch_idx", 0)
    except Exception:
        return None, 0


def save_checkpoint(round_key, batch_idx, all_cam, all_cat,
                    all_q, all_interv, all_frame_idx, metadata, n_cameras):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"checkpoint_{round_key}_{batch_idx:05d}.pt"
    result = {
        "q_targets": torch.cat(all_q, dim=0),
        "interventions": torch.cat(all_interv, dim=0),
        "frame_indices": torch.cat(all_frame_idx, dim=0),
        "metadata": metadata,
        "batch_idx": batch_idx,
        "cat": torch.cat(all_cat, dim=0),
    }
    for cam_idx in range(n_cameras):
        if all_cam[cam_idx]:
            result[f"cam_{cam_idx}"] = torch.cat(all_cam[cam_idx], dim=0)
    torch.save(result, ckpt_path)
    # Keep last 3
    prefix = f"checkpoint_{round_key}_"
    for old in sorted(CHECKPOINT_DIR.glob(f"{prefix}*.pt"))[:-3]:
        old.unlink()
    return ckpt_path


def cleanup_checkpoints(round_key):
    for ckpt in CHECKPOINT_DIR.glob(f"checkpoint_{round_key}_*.pt"):
        ckpt.unlink()


if __name__ == "__main__":
    main()
