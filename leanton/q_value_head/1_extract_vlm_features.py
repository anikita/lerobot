#!/usr/bin/env python3
"""
1_extract_vlm_features.py — Extract VLM backbone hidden states (prefix + suffix + mid-layer).

WHAT THIS DOES:
    Like v3 (saves prefix + suffix mean-pooled and sequences), PLUS:
      {round_key}_layer{L}.pt  — mean-pooled prefix hidden states at VLM layer L

    Mid-layer features preserve state information that the final layer compresses
    toward the action prior. Path A: find the layer where neg→neg diverges from neg→pos.

USAGE:
    python 1_extract_vlm_features.py r01              # single round
    python 1_extract_vlm_features.py r01 --layers 0,4,8,12
    python 1_extract_vlm_features.py r01 --prefix-only  # skip suffix (but still layers)
    python 1_extract_vlm_features.py r01 --no-layers     # v3-compatible only

OUTPUT (features_v4/):
    {round_key}.pt              — mean-pooled prefix   [N, 960]   (v2/v3-compatible)
    {round_key}_seq.pt          — full prefix sequence  [N, S, 960]
    {round_key}_suffix.pt       — mean-pooled suffix    [N, 720]   (v3-compatible)
    {round_key}_suffix_seq.pt   — full suffix sequence  [N, 50, 720]
    {round_key}_layer{L}.pt     — mean-pooled prefix at layer L  [N, 960]  ← NEW

    The layer file contains:
      - hidden_states:       [N, 960]  mean-pooled prefix at that layer
      - q_targets:           [N]       Q values (copied from round)
      - interventions:       [N]       intervention flags
      - frame_indices:       [N]       frame numbers
      - metadata.layer_idx:  int       which VLM layer this is
"""

from pathlib import Path
import time, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_REPO = "anikitakis/rollout_pick_n_place_dagger_r1_with_q"
HF_REVISION = "main"
POLICY_PATH = "anikitakis/vla_so101_pick_n_place_full_expert"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data/features_v4"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints_v4"

BATCH_SIZE = 8
CHECKPOINT_EVERY = 1000
MAX_FRAMES = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RENAME_MAP = {
    "observation.images.annotated":   "observation.images.camera1",
    "observation.images.front":       "observation.images.camera2",
    "observation.images.target_patch": "observation.images.camera3",
}

DEFAULT_LAYERS = (0, 4, 8, 12)


# ── MONKEY-PATCH: collect intermediate hidden states ──────────────────────────

def patch_forward_for_hidden_states(model, layer_indices):
    """Monkey-patch model.forward to collect prefix hidden states at given layers.

    The SmolVLMWithExpertModel.forward() manually loops over layers and discards
    intermediates. This patch intercepts outputs_embeds[0] (prefix hidden states)
    after each layer's MLP and stores them in model._collected_hidden_states.

    Args:
        model:    SmolVLMWithExpertModel instance
        layer_indices: set of layer indices to collect (e.g. {0, 4, 8, 12})
    """
    import types

    model._collect_layers = layer_indices
    model._collected_hidden_states = {}  # layer_idx → [B, S, 960] (or list of batches)

    original_forward = model.forward

    def patched_forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        fill_kv_cache=None,
    ):
        # ── Copy of original forward, with collection hooks ──
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids,
                    attention_mask, batch_size, head_dim,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids,
                    attention_mask, batch_size, head_dim,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

            # ── COLLECTION HOOK ──
            if layer_idx in self._collect_layers:
                # outputs_embeds[0] = prefix hidden states at this layer
                prefix_h = outputs_embeds[0].detach()  # [B, S, 960]
                if layer_idx not in self._collected_hidden_states:
                    self._collected_hidden_states[layer_idx] = []
                self._collected_hidden_states[layer_idx].append(prefix_h.cpu())

        # Final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    model.forward = types.MethodType(patched_forward, model)
    return model


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
        batch_idx = ckpt.get("batch_idx", 0)
        n_frames = ckpt["q_targets"].shape[0]
        print(f"\n📂 Found checkpoint: {latest.name} ({n_frames:,} frames)")
        return latest, batch_idx
    except Exception as e:
        print(f"\n⚠️  Corrupt checkpoint, starting fresh ({e})")
        return None, 0


def save_checkpoint(round_key, batch_idx,
                    all_prefix_pooled, all_prefix_seq,
                    all_suffix_pooled, all_suffix_seq,
                    all_layer_data,  # dict: layer_idx → list of [batch, 960]
                    all_q_target, all_intervention, all_frame_idx, metadata,
                    prefix_seq_len, suffix_seq_len,
                    save_prefix, save_suffix, save_layers):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"checkpoint_{round_key}_{batch_idx:05d}.pt"
    result = {
        "q_targets":      torch.cat(all_q_target, dim=0),
        "interventions":  torch.cat(all_intervention, dim=0),
        "frame_indices":  torch.cat(all_frame_idx, dim=0),
        "metadata":       metadata,
        "batch_idx":      batch_idx,
        "prefix_seq_len": prefix_seq_len,
        "suffix_seq_len": suffix_seq_len,
    }
    if save_prefix and all_prefix_pooled:
        result["hidden_states"] = torch.cat(all_prefix_pooled, dim=0)
    if save_prefix and all_prefix_seq:
        result["hidden_states_seq"] = torch.cat(all_prefix_seq, dim=0)
    if save_suffix and all_suffix_pooled:
        result["suffix_states"] = torch.cat(all_suffix_pooled, dim=0)
    if save_suffix and all_suffix_seq:
        result["suffix_states_seq"] = torch.cat(all_suffix_seq, dim=0)
    if save_layers and all_layer_data:
        for layer_idx, parts in all_layer_data.items():
            if parts:
                result[f"layer_{layer_idx}_hidden_states"] = torch.cat(parts, dim=0)

    torch.save(result, ckpt_path)
    # Keep last 3 checkpoints
    prefix = f"checkpoint_{round_key}_"
    all_ckpts = sorted(CHECKPOINT_DIR.glob(f"{prefix}*.pt"))
    for old in all_ckpts[:-3]:
        old.unlink()
    return ckpt_path


def cleanup_checkpoints(round_key):
    prefix = f"checkpoint_{round_key}_"
    for ckpt in CHECKPOINT_DIR.glob(f"{prefix}*.pt"):
        ckpt.unlink()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v4: extract prefix + suffix + mid-layer hidden states."
    )
    parser.add_argument("round", nargs="?", default=None,
                        help="Round identifier (e.g. r01, r1, r1_with_q)")
    parser.add_argument("--prefix-only", action="store_true",
                        help="Skip suffix extraction")
    parser.add_argument("--no-layers", action="store_true",
                        help="Skip mid-layer extraction (v3-compatible only)")
    parser.add_argument("--layers", type=str, default="0,4,8,12",
                        help="Comma-separated layer indices (default: 0,4,8,12)")
    args = parser.parse_args()

    save_prefix = True
    save_suffix = not args.prefix_only
    save_layers = not args.no_layers

    layer_indices = set()
    if save_layers:
        for s in args.layers.split(","):
            layer_indices.add(int(s.strip()))
        layer_indices = sorted(layer_indices)

    # --- Resolve repo ---
    repo = HF_REPO
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
    print(f"1_extract_vlm_features.py — Prefix + Suffix + Mid-Layer Extraction")
    print("=" * 65)
    print(f"  HF repo:       {repo}")
    print(f"  Round key:     {round_key}")
    print(f"  Policy:        {POLICY_PATH}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print(f"  Save prefix:   {save_prefix}")
    print(f"  Save suffix:   {save_suffix}")
    print(f"  Save layers:   {save_layers} → {layer_indices}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Device:        {DEVICE}")
    print()

    # ── STEP 1: Load dataset ──────────────────────────────────────────────────
    print("[1/5] Loading dataset...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo, revision=HF_REVISION)
    N = len(ds) if MAX_FRAMES is None else min(len(ds), MAX_FRAMES)
    dur_min = N / 20 / 60
    print(f"       {N:,} frames ({dur_min:.0f} min at 20 fps)")
    has_q = "q_target" in ds.hf_dataset.column_names
    print(f"       q_target present: {has_q}")
    if not has_q:
        print("       ❌ q_target column missing — this round is not _with_q!")
        return

    n_cameras = sum(1 for col in ds.hf_dataset.column_names
                    if col.startswith("observation.images."))
    print(f"       Cameras: {n_cameras}")

    # ── STEP 2: Load policy ───────────────────────────────────────────────────
    print(f"\n[2/5] Loading SmolVLA policy from {POLICY_PATH}...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy.model.eval()
    policy.model.to(DEVICE)
    for p in policy.model.parameters():
        p.requires_grad = False

    hidden_dim = policy.model.vlm_with_expert.config.text_config.hidden_size
    expert_dim = policy.model.vlm_with_expert.expert_hidden_size
    chunk_size = policy.config.chunk_size
    action_dim = policy.config.max_action_dim
    num_vlm_layers = policy.model.vlm_with_expert.num_vlm_layers
    print(f"       Model:    SmolVLM-500M, {num_vlm_layers} VLM layers, "
          f"prefix_dim={hidden_dim}, suffix_dim={expert_dim}")

    # Validate layer indices
    for li in layer_indices:
        if li < 0 or li >= num_vlm_layers:
            print(f"       ❌ Layer {li} out of range [0, {num_vlm_layers-1}]")
            return
    print(f"       Extracting layers: {layer_indices}")

    # ── Monkey-patch for hidden state collection ──
    if save_layers:
        patch_forward_for_hidden_states(policy.model.vlm_with_expert, set(layer_indices))
        print(f"       Patched forward for hidden state collection")

    # ── Task tokens ──
    task = ds.meta.tasks.iloc[0].name
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer(task + "\n", return_tensors="pt", padding="max_length",
                        max_length=48, truncation=True)
    lang_tokens_all = encoded["input_ids"].to(DEVICE)
    lang_masks_all = encoded["attention_mask"].to(DEVICE).bool()
    print(f"       Task:     \"{task}\"")

    metadata = {
        "hf_repo": repo, "hf_revision": HF_REVISION,
        "policy_path": POLICY_PATH,
        "prefix_hidden_dim": hidden_dim, "suffix_hidden_dim": expert_dim,
        "gamma": 0.965, "num_frames": N,
        "round_key": round_key, "chunk_size": chunk_size,
        "n_cameras": n_cameras,
        "num_vlm_layers": num_vlm_layers,
        "extracted_layers": list(layer_indices) if save_layers else [],
    }

    # ── Resume ──────────────────────────────────────────────────────────────
    ckpt_path, start_batch = find_checkpoint(round_key)
    start_frame = start_batch * BATCH_SIZE

    def collate_to_device(batch_list):
        batch = {}
        for key in batch_list[0].keys():
            values = [item[key] for item in batch_list]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values).to(DEVICE)
            elif isinstance(values[0], str):
                batch[key] = values
            else:
                batch[key] = torch.tensor(values).to(DEVICE)
        return batch

    # ── STEP 3: Extract ───────────────────────────────────────────────────────
    print(f"\n[3/5] Extracting hidden states...")
    print(f"       Frames: {N:,}")
    if start_frame > 0:
        print(f"       Resuming from frame {start_frame:,}")

    if start_frame > 0:
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        all_prefix_pooled = [ckpt.get("hidden_states")] if "hidden_states" in ckpt else []
        all_prefix_seq = [ckpt.get("hidden_states_seq")] if "hidden_states_seq" in ckpt else []
        all_suffix_pooled = [ckpt.get("suffix_states")] if "suffix_states" in ckpt else []
        all_suffix_seq = [ckpt.get("suffix_states_seq")] if "suffix_states_seq" in ckpt else []
        all_layer_data = {}
        for li in layer_indices:
            key = f"layer_{li}_hidden_states"
            all_layer_data[li] = [ckpt[key]] if key in ckpt else []
        all_q_target = [ckpt["q_targets"]]
        all_intervention = [ckpt["interventions"]]
        all_frame_idx = [ckpt["frame_indices"]]
    else:
        all_prefix_pooled, all_prefix_seq = [], []
        all_suffix_pooled, all_suffix_seq = [], []
        all_layer_data = {li: [] for li in layer_indices}
        all_q_target, all_intervention, all_frame_idx = [], [], []

    if start_frame > 0:
        from torch.utils.data import Subset
        ds_work = Subset(ds, range(start_frame, N))
    else:
        ds_work = ds

    loader = DataLoader(ds_work, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_to_device, num_workers=0)

    pbar = tqdm(total=N, initial=start_frame, unit="frames", desc="       ", ncols=80)
    last_checkpoint_batch = start_batch
    processed_this_run = 0
    t_batch_start = time.time()
    actual_prefix_seq_len = None
    actual_suffix_seq_len = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            real_batch = start_batch + batch_idx

            for old_key, new_key in RENAME_MAP.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)

            if MAX_FRAMES is not None and pbar.n >= MAX_FRAMES:
                break
            bsize = len(batch["action"])

            # ── Forward pass ──
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = lang_tokens_all.expand(bsize, -1)
            lang_masks = lang_masks_all.expand(bsize, -1)

            prefix_embs, prefix_pad_masks, prefix_att_masks = \
                policy.model.embed_prefix(images, img_masks,
                                          lang_tokens, lang_masks, state=state)
            dummy_actions = torch.zeros(bsize, chunk_size, action_dim, device=DEVICE)
            dummy_time = torch.zeros(bsize, device=DEVICE)
            suffix_embs, suffix_pad_masks, suffix_att_masks = \
                policy.model.embed_suffix(dummy_actions, dummy_time)
            suffix_att_masks = suffix_att_masks.bool()

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            # Clear previous collected states
            if save_layers:
                policy.model.vlm_with_expert._collected_hidden_states = {}

            (prefix_out, suffix_out), _ = policy.model.vlm_with_expert.forward(
                attention_mask=att_2d_masks, position_ids=position_ids,
                past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False, fill_kv_cache=False,
            )

            if actual_prefix_seq_len is None:
                actual_prefix_seq_len = prefix_out.shape[1]
                actual_suffix_seq_len = suffix_out.shape[1]
                print(f"\n       Prefix seq_len: {actual_prefix_seq_len}  "
                      f"Suffix seq_len: {actual_suffix_seq_len}")

            # ── Collect ──
            if save_prefix:
                all_prefix_pooled.append(prefix_out.mean(dim=1).cpu())
                all_prefix_seq.append(prefix_out.cpu())
            if save_suffix:
                all_suffix_pooled.append(suffix_out.mean(dim=1).cpu())
                all_suffix_seq.append(suffix_out.cpu())
            if save_layers:
                collected = policy.model.vlm_with_expert._collected_hidden_states
                for li in layer_indices:
                    if li in collected:
                        # Each entry is [B, S, 960]; mean-pool to [B, 960]
                        for h in collected[li]:
                            all_layer_data[li].append(h.float().mean(dim=1))

            all_q_target.append(batch["q_target"].cpu())
            all_intervention.append(batch["intervention"].cpu())
            all_frame_idx.append(batch["frame_index"].cpu())

            pbar.update(bsize)
            processed_this_run += bsize

            if real_batch % 10 == 0:
                elapsed = time.time() - t_batch_start
                fps = processed_this_run / elapsed if elapsed > 0 else 0
                remaining = N - pbar.n
                eta = remaining / fps if fps > 0 else 0
                pbar.set_postfix({"fps": f"{fps:.1f}", "eta": f"{eta/60:.0f}m"})

            if real_batch > 0 and real_batch % CHECKPOINT_EVERY == 0 \
               and real_batch != last_checkpoint_batch:
                save_checkpoint(round_key, real_batch,
                                all_prefix_pooled, all_prefix_seq,
                                all_suffix_pooled, all_suffix_seq,
                                all_layer_data,
                                all_q_target, all_intervention, all_frame_idx,
                                metadata, actual_prefix_seq_len, actual_suffix_seq_len,
                                save_prefix, save_suffix, save_layers)
                last_checkpoint_batch = real_batch
                pbar.write(f"       💾 Checkpoint — {pbar.n:,} frames saved")

    pbar.close()
    t_extract = time.time() - t_batch_start
    print(f"\n       Done: {pbar.n:,} frames in {t_extract/60:.0f}m "
          f"({pbar.n/t_extract:.1f} fps)")

    # ── STEP 4: Consolidate ───────────────────────────────────────────────────
    print(f"\n[4/5] Consolidating...")
    q_t = torch.cat(all_q_target, dim=0)
    interv_t = torch.cat(all_intervention, dim=0)
    frame_t = torch.cat(all_frame_idx, dim=0)
    N_final = q_t.shape[0]

    metadata["prefix_seq_len"] = actual_prefix_seq_len
    metadata["suffix_seq_len"] = actual_suffix_seq_len

    # ── STEP 5: Save ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Verifying & saving...")
    print(f"       Frames:      {N_final:,}")
    print(f"       Prefix dim:  {hidden_dim}  (seq_len={actual_prefix_seq_len})")
    print(f"       Suffix dim:  {expert_dim}  (seq_len={actual_suffix_seq_len})")
    print(f"       Q range:     [{q_t.min():.2f}, {q_t.max():.2f}]")
    n_nz = (q_t.abs() > 0.05).sum().item()
    print(f"       Nonzero Q:   {n_nz:,} ({100*n_nz/N_final:.1f}%)")

    assert N_final == N, f"Frame count mismatch: {N_final} vs {N}"
    for i in range(min(N_final, 100)):
        assert frame_t[i] == i, f"Order error at {i}"
    print(f"       Frame order: ✅ sequential")

    # Verify layers were collected
    if save_layers:
        for li in layer_indices:
            if all_layer_data[li]:
                n_layer = sum(h.shape[0] for h in all_layer_data[li])
                print(f"       Layer {li}: {n_layer:,} frames collected ✅")
            else:
                print(f"       Layer {li}: ❌ NOTHING COLLECTED")

    base = {
        "q_targets": q_t,
        "interventions": interv_t,
        "frame_indices": frame_t,
        "metadata": metadata,
    }

    # ---- Prefix files ----
    if save_prefix:
        r_mean = {**base, "hidden_states": torch.cat(all_prefix_pooled, dim=0)}
        path = OUTPUT_DIR / f"{round_key}.pt"
        torch.save(r_mean, path)
        print(f"\n       Prefix mean  → {path}  ({path.stat().st_size/1024**2:.0f} MB)")

        r_seq = {**base, "hidden_states_seq": torch.cat(all_prefix_seq, dim=0),
                 "prefix_seq_len": actual_prefix_seq_len}
        path = OUTPUT_DIR / f"{round_key}_seq.pt"
        torch.save(r_seq, path)
        print(f"       Prefix seq   → {path}  ({path.stat().st_size/1024**2:.0f} MB)")

    # ---- Suffix files ----
    if save_suffix:
        r_suf_mean = {**base, "suffix_states": torch.cat(all_suffix_pooled, dim=0)}
        path = OUTPUT_DIR / f"{round_key}_suffix.pt"
        torch.save(r_suf_mean, path)
        print(f"       Suffix mean  → {path}  ({path.stat().st_size/1024**2:.0f} MB)")

        r_suf_seq = {**base, "suffix_states_seq": torch.cat(all_suffix_seq, dim=0),
                     "suffix_seq_len": actual_suffix_seq_len}
        path = OUTPUT_DIR / f"{round_key}_suffix_seq.pt"
        torch.save(r_suf_seq, path)
        print(f"       Suffix seq   → {path}  ({path.stat().st_size/1024**2:.0f} MB)")

    # ---- Layer files ----
    if save_layers:
        for li in layer_indices:
            if not all_layer_data[li]:
                print(f"       Layer {li}: skipped (no data)")
                continue
            layer_h = torch.cat(all_layer_data[li], dim=0)  # [N, 960]
            layer_data = {
                "hidden_states": layer_h,
                "q_targets": q_t,
                "interventions": interv_t,
                "frame_indices": frame_t,
                "metadata": {**metadata, "layer_idx": li},
            }
            path = OUTPUT_DIR / f"{round_key}_layer{li}.pt"
            torch.save(layer_data, path)
            print(f"       Layer {li}    → {path}  "
                  f"({path.stat().st_size/1024**2:.0f} MB)  "
                  f"[{list(layer_h.shape)}]")

    cleanup_checkpoints(round_key)
    print(f"       Checkpoints cleaned")

    t_total = time.time() - t_start
    print(f"\n✅ {round_key}: {N_final:,} frames → {OUTPUT_DIR}/")
    print(f"   Time: {t_total/60:.0f}m  |  Throughput: {N_final/t_total:.1f} fps")


if __name__ == "__main__":
    main()
