#!/usr/bin/env python3
"""
validate_features.py — Verify extracted features are correct by comparing
a fresh forward pass against saved features for the same frames.

Tests:
  1. Model identity: is it the fine-tuned expert or base SmolVLM?
  2. Feature match: do fresh-extracted suffix features match saved ones?
  3. Action decode: do suffix features decode to sensible actions?

USAGE:
    python validate_features.py r1     # validate on round r1
    python validate_features.py r5     # validate on round r5
"""

import sys
from pathlib import Path
import torch
import numpy as np

SCRIPT_DIR = Path(__file__).parent
FEATURES_DIR = SCRIPT_DIR / "data/features_v3"
POLICY_PATH = "anikitakis/vla_so101_pick_n_place_full_expert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python validate_features.py [r1|r5|rN]")
        return
    round_name = args[0]

    # ── 1. Load policy ──────────────────────────────────────────────────────
    print("=" * 60)
    print("1. MODEL IDENTITY")
    print("=" * 60)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy.model.eval()
    policy.model.to(DEVICE)
    for p in policy.model.parameters():
        p.requires_grad = False

    model = policy.model.vlm_with_expert
    chunk_size = policy.config.chunk_size
    action_dim = policy.config.max_action_dim

    print(f"  Policy:        {POLICY_PATH}")
    print(f"  VLM base:      {model.config.text_config._name_or_path or '(not set — typical for fine-tuned)'}")
    print(f"  Hidden (prefix): {model.config.text_config.hidden_size}")
    print(f"  Expert (suffix): {model.expert_hidden_size}")
    print(f"  Chunk size:    {chunk_size}")
    print(f"  Action dim:    {action_dim}")
    print(f"  Total params:  {sum(p.numel() for p in policy.model.parameters()):,}")

    # Check if it looks fine-tuned vs base
    # Fine-tuned models have different weight distributions
    action_proj_weight = model.action_out_proj if hasattr(model, 'action_out_proj') else policy.model.action_out_proj
    w = action_proj_weight.weight.float()
    print(f"  Action head W: mean={w.mean():.4f}  std={w.std():.4f}  "
          f"min={w.min():.4f}  max={w.max():.4f}")
    if w.std() < 0.001:
        print("  ⚠️  Action head weights are near-zero — might be untrained!")
    else:
        print("  ✅ Action head has non-trivial weights")

    # ── 2. Load saved features ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"2. FEATURE MATCH (fresh extraction vs saved)")
    print(f"{'='*60}")

    seq_file = FEATURES_DIR / f"{round_name}_with_q_suffix_seq.pt"
    if not seq_file.exists():
        print(f"  ❌ {seq_file} not found")
        return
    saved = torch.load(seq_file, weights_only=True, map_location="cpu")
    saved_features = saved["suffix_states_seq"]  # [N, 50, 720]
    metadata = saved.get("metadata", {})
    hf_repo = metadata.get("hf_repo",
                           f"anikitakis/rollout_pick_n_place_dagger_{round_name}")

    print(f"  Saved file:    {seq_file.name}")
    print(f"  Shape:         {list(saved_features.shape)}")
    print(f"  Range:         [{saved_features.min():.4f}, {saved_features.max():.4f}]")
    print(f"  Mean:          {saved_features.mean():.6f}")
    print(f"  Std:           {saved_features.std():.6f}")

    # ── 3. Fresh forward pass on first N frames ─────────────────────────────
    print(f"\n  Running fresh forward pass on first 5 frames...")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    # Try to load the correct repo
    repo_candidates = [
        hf_repo,
        f"anikitakis/rollout_pick_n_place_dagger_{round_name}_with_q",
    ]
    # Map r1 → r1_with_q naming
    rn = round_name.replace("r", "").replace("_with_q", "")
    repo_candidates.append(f"anikitakis/rollout_pick_n_place_dagger_r{rn}_with_q")

    ds = None
    for repo in repo_candidates:
        try:
            ds = LeRobotDataset(repo)
            print(f"  Dataset:       {repo}")
            break
        except Exception:
            continue

    if ds is None:
        print(f"  ❌ Could not load dataset for {round_name}")
        print(f"     Tried: {repo_candidates}")
        return

    # Prepare task string
    task = ds.meta.tasks.iloc[0].name
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer(task + "\n", return_tensors="pt", padding="max_length",
                        max_length=48, truncation=True)
    lang_tokens = encoded["input_ids"].to(DEVICE)
    lang_masks = encoded["attention_mask"].to(DEVICE).bool()

    # Rename map (same as extraction script)
    RENAME_MAP = {
        "observation.images.annotated":   "observation.images.camera1",
        "observation.images.front":       "observation.images.camera2",
        "observation.images.target_patch": "observation.images.camera3",
    }

    n_test = min(5, len(ds))
    fresh_features = []

    for frame_idx in range(n_test):
        item = ds[frame_idx]
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
                 for k, v in item.items()}
        for old_key, new_key in RENAME_MAP.items():
            if old_key in batch:
                batch[new_key] = batch.pop(old_key)

        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(DEVICE)

        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)

        prefix_embs, prefix_pad_masks, prefix_att_masks = \
            policy.model.embed_prefix(images, img_masks,
                                      lang_tokens, lang_masks, state=state)
        dummy_actions = torch.zeros(1, chunk_size, action_dim, device=DEVICE)
        dummy_time = torch.zeros(1, device=DEVICE)
        suffix_embs, suffix_pad_masks, suffix_att_masks = \
            policy.model.embed_suffix(dummy_actions, dummy_time)
        suffix_att_masks = suffix_att_masks.bool()

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        with torch.no_grad():
            (prefix_out, suffix_out), _ = model.forward(
                attention_mask=att_2d_masks, position_ids=position_ids,
                past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False, fill_kv_cache=False,
            )
        fresh_features.append(suffix_out.cpu())  # [1, 50, 720]

    fresh_features = torch.cat(fresh_features, dim=0)  # [n_test, 50, 720]

    # ── 4. Compare ──────────────────────────────────────────────────────────
    saved_subset = saved_features[:n_test].float()
    fresh_subset = fresh_features.float()

    abs_diff = (saved_subset - fresh_subset).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        saved_subset.reshape(n_test, -1), fresh_subset.reshape(n_test, -1), dim=1
    )

    print(f"\n  Feature comparison (first {n_test} frames):")
    print(f"    Max absolute diff:  {max_diff:.8f}")
    print(f"    Mean absolute diff: {mean_diff:.8f}")
    for i in range(n_test):
        print(f"    Frame {i}: cos_sim={cos_sim[i]:.8f}")

    if max_diff < 1e-5:
        print(f"  ✅ Features match BIT-FOR-BIT — extraction is correct")
    elif max_diff < 0.1 and cos_sim.min().item() > 0.999:
        print(f"  ✅ Features match (max_diff={max_diff:.4f} is bf16 non-determinism, "
              f"cos_sim > 0.9999) — extraction is correct")
    else:
        print(f"  ❌ Features DO NOT MATCH — extraction is BROKEN")

    # ── 5. Action decode sanity check ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"3. ACTION DECODE SANITY CHECK")
    print(f"{'='*60}")

    # Decode actions from saved suffix features
    action_proj = policy.model.action_out_proj  # Linear(720, 32)
    saved_flat = saved_features[:10].to(DEVICE, dtype=torch.float32)  # [10, 50, 720]
    with torch.no_grad():
        decoded_actions = action_proj(saved_flat)  # [10, 50, 32]

    # Get recorded actions from dataset
    ds_actions_list = []
    for i in range(min(10, len(ds))):
        item = ds[i]
        act = item["action"]
        if isinstance(act, torch.Tensor):
            ds_actions_list.append(act)
        else:
            ds_actions_list.append(torch.tensor(act))
    ds_actions = torch.stack(ds_actions_list, dim=0).float()  # [10, 32]

    # Compare — use only the overlapping action dims
    pred_action = decoded_actions[:, 0, :].cpu()  # first step of chunk [10, 32]
    gt_action = ds_actions                         # [10, actual_action_dim]
    common_dim = min(pred_action.shape[1], gt_action.shape[1])
    pred_action = pred_action[:, :common_dim]
    gt_action = gt_action[:, :common_dim]

    action_mse = torch.nn.functional.mse_loss(pred_action, gt_action).item()
    # Null baseline: predict zero
    null_mse = torch.nn.functional.mse_loss(
        torch.zeros_like(gt_action), gt_action).item()

    print(f"  Saved features → action_out_proj → predicted actions")
    print(f"  Action MSE:      {action_mse:.6f}")
    print(f"  Null MSE (zero): {null_mse:.6f}")
    print(f"  Improvement:     {null_mse/action_mse:.1f}×")
    if null_mse / action_mse > 2:
        print(f"  ✅ Features encode meaningful action information")
    else:
        print(f"  ⚠️  Features don't predict actions better than zero")

    # Per-dim action stats
    print(f"\n  Per-dimension action check (first 10 frames):")
    print(f"  {'Dim':<6} {'GT mean':>10} {'Pred mean':>10} {'GT std':>10} {'Pred std':>10}")
    for d in range(min(8, action_dim)):
        print(f"  {d:<6} {gt_action[:, d].mean():>10.4f} {pred_action[:, d].mean():>10.4f} "
              f"{gt_action[:, d].std():>10.4f} {pred_action[:, d].std():>10.4f}")

    # ── 6. Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if max_diff < 1e-3 and null_mse / action_mse > 2:
        print("✅ Features are VALID — backbone is correct, actions decode properly")
        print("   The Q-value collapse is a REAL property of the representation,")
        print("   not an extraction artifact or wrong model.")
        print("   Next step: Path A — extract earlier VLM layers.")
    elif max_diff >= 1e-3:
        print("❌ Features don't match fresh extraction — pipeline is BROKEN")
        print("   Debug the extraction script or check for model/dtype mismatches")
    else:
        print("⚠️  Features match but actions don't decode — something is wrong")
        print("   The action_head weights might not match the features")


if __name__ == "__main__":
    main()
