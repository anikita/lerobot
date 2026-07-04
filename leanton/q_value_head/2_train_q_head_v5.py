#!/usr/bin/env python3
"""
2_train_q_head_v5.py — Train Q-head with a trainable VLM feature adapter.

Loads pre-extracted features from an early VLM layer (e.g. layer 11), runs them
through the remaining VLM layers (trainable), then through a Q-head (MLP, TCN,
or CausalTemporalTransformer).

This is Path C-lite: the VLM layers act as a feature adapter, learning to surface
Q-relevant information that the frozen early layers don't express. No video
decoding, no SigLIP — works directly on pre-extracted features.

New in v5: CausalTemporalTransformer (--model causal_tf). Unlike TCN's fixed
kernel and dilations, the causal transformer learns content-dependent attention
weights over the full temporal window. Tests whether TCN's fixed temporal
pattern was the bottleneck.

USAGE:
    # TCN (v4-compatible)
    python 2_train_q_head_v5.py --model tcn --feature-field layer_11 \
        --temporal-window 90 --epochs 50 --val r5,r18

    # Causal Transformer
    python 2_train_q_head_v5.py --model causal_tf --feature-field layer_11 \
        --temporal-window 90 --epochs 50 --val r5,r18

    # Causal Transformer on SigLIP features
    python 2_train_q_head_v5.py --model causal_tf --feature-field siglip_cat \
        --no-adapter --temporal-window 90 --epochs 50 --val r18

    # With VLM adapter + causal transformer
    python 2_train_q_head_v5.py --model causal_tf --feature-field layer_4 \
        --unfreeze-layers 5,6,7,8,9,10,11,12,13,14,15 --vlm-lr 1e-6 \
        --temporal-window 90 --epochs 50 --val r5,r18

OUTPUT:
    weights/q_head_weights_v5_*.pt     — Q-head weights
    weights/vlm_adapter_weights_*.pt   — fine-tuned VLM layer weights
"""

import sys, time, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt

from model import (TemporalTCNQHead, QValueHead, CausalTemporalTransformer,
                      count_params)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FEATURES_DIR = SCRIPT_DIR / "data/features_v4"
PLOTS_DIR = SCRIPT_DIR / "plots"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
POLICY_PATH = "anikitakis/vla_so101_pick_n_place_full_expert"

_DEFAULT_VAL_ROUNDS = {"r5_with_q", "r18_with_q"}

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
VLM_LR = 1e-6
TEMPORAL_WINDOW = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_cli():
    argv = sys.argv[1:]

    def _get(k, default):
        for i, arg in enumerate(argv):
            if arg == k and i + 1 < len(argv):
                return argv[i + 1]
        return default

    model_name = _get("--model", "mlp").lower()
    if model_name not in ("tcn", "mlp", "causal_tf"):
        raise ValueError(f"Unknown --model: {model_name}")

    features_dir = Path(_get("--features-dir", str(FEATURES_DIR)))
    feature_field = _get("--feature-field", "layer_11").lower()

    def _parse_rounds(arg, default=None):
        """Parse comma-separated round list from CLI. Returns None if not provided."""
        val = _get(arg, None)
        if val is None:
            return default
        rounds = set()
        for r in val.split(","):
            r = r.strip()
            if not r.endswith("_with_q"):
                r = f"{r}_with_q"
            rounds.add(r)
        return rounds

    train_rounds_set = _parse_rounds("--train", None)  # None = auto (all non-val)
    val_rounds = _parse_rounds("--val", _DEFAULT_VAL_ROUNDS
                               if train_rounds_set is None else set())

    epochs = int(_get("--epochs", str(EPOCHS)))
    batch_size = int(_get("--batch-size", str(BATCH_SIZE)))
    lr = float(_get("--lr", str(LR)))
    vlm_lr = float(_get("--vlm-lr", str(VLM_LR)))
    temporal_window = int(_get("--temporal-window", str(TEMPORAL_WINDOW)))

    unfreeze_str = _get("--unfreeze-layers", "12,13,14,15")
    unfreeze_layers = []
    for s in unfreeze_str.split(","):
        s = s.strip()
        if s:
            unfreeze_layers.append(int(s))

    no_adapter = "--no-adapter" in argv
    no_balanced = "--no-balanced" in argv
    train_split = float(_get("--train-split", "0.0"))
    dead_zone = int(_get("--dead-zone", "0"))
    q_renorm = "--q-renorm" in argv
    hidden_layers = _get("--hidden-layers", "")
    dropout = float(_get("--dropout", "0.0"))
    tcn_pool = _get("--tcn-pool", "mean")
    leak_target = _get("--leak-target", "none")
    leak_random = False
    leak_offset = None
    if leak_target != "none":
        if leak_target == "current":
            leak_offset = 0
        elif leak_target == "random":
            leak_random = True
            leak_offset = 0
        elif leak_target.startswith("history_"):
            parts = leak_target.split("_", 1)[1]
            if parts == "random":
                leak_random = True
                leak_offset = 50  # default depth for random test
            else:
                leak_offset = int(parts)
        else:
            raise ValueError(f"Unknown --leak-target: {leak_target}. "
                             f"Use 'none', 'current', 'random', or 'history_N'.")

    return (model_name, features_dir, feature_field, val_rounds, train_rounds_set,
            epochs, batch_size, lr, vlm_lr, temporal_window,
            sorted(unfreeze_layers), leak_offset, leak_random, no_adapter, train_split,
            tcn_pool, no_balanced, hidden_layers, dropout, dead_zone, q_renorm)


# ── VLM ADAPTER ───────────────────────────────────────────────────────────────

class VLMAdapter(nn.Module):
    """Trainable VLM layers that adapt frozen early-layer features for Q prediction.

    Takes pre-extracted features from a frozen layer L, runs them through
    VLM layers [L+1 ... 15] + final norm. The early layers and SigLIP are skipped.
    """

    def __init__(self, policy, input_layer, unfreeze_layers):
        super().__init__()
        vlm_model = policy.model.vlm_with_expert
        vlm_layers = vlm_model.get_vlm_model().text_model.layers
        num_vlm_layers = len(vlm_layers)
        self.final_norm = vlm_model.get_vlm_model().text_model.norm

        # Cache GQA config
        self.n_q_heads = vlm_model.config.text_config.num_attention_heads
        self.n_kv_heads = vlm_model.config.text_config.num_key_value_heads
        self.head_dim = vlm_layers[0].self_attn.head_dim

        expected_start = input_layer + 1
        if unfreeze_layers and unfreeze_layers[0] != expected_start:
            print(f"  ⚠️  First unfrozen layer is {unfreeze_layers[0]}, "
                  f"but input features are from layer {input_layer}. "
                  f"Expected: {expected_start}")

        self.adapter_layers = nn.ModuleList()
        for li in unfreeze_layers:
            if 0 <= li < num_vlm_layers:
                self.adapter_layers.append(vlm_layers[li])

        self.hidden_dim = vlm_model.config.text_config.hidden_size

        for layer in self.adapter_layers:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.final_norm.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  VLM Adapter: {trainable:,} trainable / {total:,} total params")
        print(f"  Layers: {unfreeze_layers} + final_norm")

    def _single_token_attn(self, layer, x_normed):
        """Self-attention for a single token with GQA.

        For seq_len=1, softmax(QK^T/sqrt(d)) = 1.0 always, so output = o_proj(V).
        With GQA, KV heads are repeated to match Q heads.
        """
        b = x_normed.shape[0]
        w_dtype = layer.self_attn.q_proj.weight.dtype

        xn = x_normed.to(w_dtype).unsqueeze(1)  # [B, 1, D]

        # Q: [B, 1, n_q_heads, head_dim]
        q = layer.self_attn.q_proj(xn).view(b, 1, self.n_q_heads, self.head_dim)
        # K, V: [B, 1, n_kv_heads, head_dim]
        k = layer.self_attn.k_proj(xn).view(b, 1, self.n_kv_heads, self.head_dim)
        v = layer.self_attn.v_proj(xn).view(b, 1, self.n_kv_heads, self.head_dim)

        n_rep = self.n_q_heads // self.n_kv_heads
        # Repeat KV heads to match Q heads
        k = k.unsqueeze(3).expand(b, 1, self.n_kv_heads, n_rep, self.head_dim)
        k = k.reshape(b, 1, self.n_q_heads, self.head_dim)
        v = v.unsqueeze(3).expand(b, 1, self.n_kv_heads, n_rep, self.head_dim)
        v = v.reshape(b, 1, self.n_q_heads, self.head_dim)

        # Single token: softmax(QK^T/sqrt(d)) = 1.0 always
        # attn_out = V (weighted by softmax=1.0)
        attn_out = v.reshape(b, 1, self.n_q_heads * self.head_dim)  # [B, 1, 960]
        attn_out = layer.self_attn.o_proj(attn_out).squeeze(1)  # [B, 960]
        return attn_out

    def forward(self, x):
        """Run mean-pooled features through adapter layers.

        Args:
            x: [B, T, D] or [B, D] — features at the input layer

        Returns:
            [B, T, D] or [B, D] — adapted features
        """
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(1)  # [B, 1, D]
        B, T, D = x.shape
        x = x.reshape(B * T, D).float()

        for layer in self.adapter_layers:
            residual = x
            x = residual + self._single_token_attn(layer, layer.input_layernorm(x))
            residual2 = x
            w_dtype = layer.mlp.gate_proj.weight.dtype
            x_normed = layer.post_attention_layernorm(x).to(w_dtype)
            x = residual2 + layer.mlp(x_normed).to(torch.float32)

        x = self.final_norm(x).float()
        x = x.reshape(B, T, D)
        if squeeze:
            x = x.squeeze(1)
        return x


def load_vlm_adapter(policy, input_layer, unfreeze_layers):
    """Create VLMAdapter and transfer unfrozen layer weights from policy."""
    adapter = VLMAdapter(policy, input_layer, unfreeze_layers)
    # Weights already shared — adapter.layers[i] IS the policy's layer
    return adapter


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_feature_data(features_dir, feature_field, val_rounds, train_split=0.0,
                      train_rounds_set=None, dead_zone=0, q_renorm=False):
    """Load pre-extracted features per round — preserves round boundaries for T>1.

    Args:
        train_split: if > 0, fraction of frames to hold out WITHIN each training round.
        train_rounds_set: if provided, ONLY these rounds go to training.
                          If None (default), all non-val rounds go to training.

    Returns:
        train_rounds: list of {"name": str, "h": [N, D], "q": [N]}
        val_rounds:   list of same (cross-round)
        train_holdout: list of same (within-round, from training rounds, unseen frames)
        hidden_dim:   int
        q_std:        float
    """
    if feature_field == "siglip":
        file_suffix = "siglip_cat"
        field_name = "hidden_states"
        file_pattern = "*_siglip_cat.pt"
        features_dir = Path(str(features_dir).replace("data/features_v4", "data/features_siglip"))
    elif feature_field.startswith("siglip_cam"):
        cam = feature_field.split("_")[-1]  # "cam0", "cam1", etc.
        file_suffix = f"siglip_{cam}"
        field_name = "hidden_states"
        file_pattern = f"*_siglip_{cam}.pt"
        features_dir = Path(str(features_dir).replace("data/features_v4", "data/features_siglip"))
    elif feature_field.startswith("layer_"):
        layer_num = feature_field.split("_", 1)[1]
        file_suffix = f"layer{layer_num}"
        field_name = "hidden_states"
        file_pattern = f"*_{file_suffix}.pt"
    elif feature_field == "suffix":
        file_suffix = "suffix"
        field_name = "suffix_states"
        file_pattern = "*_suffix.pt"
    else:
        file_suffix = ""
        field_name = "hidden_states"
        file_pattern = "*.pt"

    all_files = sorted(features_dir.glob(file_pattern))
    if not all_files:
        raise FileNotFoundError(f"No {file_pattern} files in {features_dir}")

    if file_suffix:
        files = [f for f in all_files if f.stem.endswith(f"_{file_suffix}")]
    else:
        files = [f for f in all_files
                 if not f.stem.endswith("_seq")
                 and not f.stem.endswith("_suffix")
                 and not f.stem.endswith("_suffix_seq")
                 and "_layer" not in f.stem]

    if not files:
        raise FileNotFoundError(
            f"No matching files for --feature-field {feature_field} in {features_dir}")

    print(f"Loading {feature_field} features from {features_dir}/")
    print(f"  Pattern: {file_pattern}  Field: {field_name}")

    train_rounds, val_rounds_list = [], []
    hidden_dim = None
    all_train_h = []

    for f in files:
        data = torch.load(f, weights_only=True, map_location="cpu")
        if field_name not in data:
            print(f"  ⚠️  {f.name}: no '{field_name}' — skipping")
            continue

        h = data[field_name].float()
        q = data["q_targets"].float()
        interv = data.get("interventions", None)

        # Dead zone: zero out Q around intervention onsets
        n_dead = 0
        if dead_zone > 0 and interv is not None:
            interv_list = list(interv)
            interv_np = np.array(interv_list)
            onsets = np.where(interv_np[1:] & ~interv_np[:-1])[0] + 1
            mask = np.ones(len(q), dtype=bool)
            for onset in onsets:
                lo = max(0, onset - dead_zone)
                hi = min(len(q), onset + dead_zone)
                mask[lo:hi] = False
            q[~mask] = 0.0
            n_dead = (~mask).sum()
            n_onsets = len(onsets)
            hidden_dim = h.shape[1]
        elif h.shape[1] != hidden_dim:
            print(f"  ⚠️  {f.name}: dim mismatch — skipping")
            continue

        key_clean = re.sub(r'(_layer\d+|_siglip_cat|_siglip_cam\d+)$', '',
                           f.stem.replace("_suffix", ""))
        is_val = key_clean in val_rounds or f.stem in val_rounds
        is_train = key_clean in train_rounds_set if train_rounds_set is not None \
                   else not is_val
        is_both = is_train and is_val  # round in both — split temporally

        n_frames = h.shape[0]
        nz = (q.abs() > 0.05).sum().item()
        dz_str = f"  dead={n_dead}" if n_dead > 0 else ""

        if is_both:
            # Split: first 80% → train, last 20% → val (contiguous, temporal order)
            n_train_frames = int(n_frames * 0.8)
            train_h, val_h_split = h[:n_train_frames], h[n_train_frames:]
            train_q, val_q_split = q[:n_train_frames], q[n_train_frames:]
            tag = "BOTH "
            train_rounds.append({"name": f.stem + "_tr", "h": train_h, "q": train_q})
            all_train_h.append(train_h)
            val_rounds_list.append({"name": f.stem + "_val", "h": val_h_split, "q": val_q_split})
        elif is_val:
            tag = "VAL  "
            val_rounds_list.append({"name": key_clean, "h": h, "q": q})
        elif is_train:
            tag = "TRAIN"
            train_rounds.append({"name": f.stem, "h": h, "q": q})
            all_train_h.append(h)
        else:
            tag = "SKIP "

        print(f"  {tag} {f.stem:30s}  [{n_frames}, {hidden_dim}]  "
              f"Q∈[{q.min():.1f},{q.max():.1f}]  nonzero={nz}{dz_str}")
        if is_both:
            print(f"       ↳ train [{n_train_frames}] + val [{n_frames - n_train_frames}]")

    if not train_rounds:
        raise RuntimeError("No training rounds loaded")

    # Normalization stats from training data
    all_h = torch.cat(all_train_h, dim=0)
    h_mean = all_h.mean(dim=0, keepdim=True)
    h_std = all_h.std(dim=0, keepdim=True) + 1e-8

    # Apply normalization
    for tr in train_rounds:
        tr["h"] = (tr["h"] - h_mean) / h_std
    for vr in val_rounds_list:
        vr["h"] = (vr["h"] - h_mean) / h_std

    # Within-round holdout split: contiguous tail of each round (tests memorization)
    # Using contiguous tail avoids the temporal autocorrelation cheat — train frames
    # never see frames temporally adjacent to holdout frames.
    train_holdout = []
    if train_split > 0:
        for tr in train_rounds:
            n = tr["h"].shape[0]
            n_holdout = int(n * train_split)
            if n_holdout > 0:
                train_holdout.append({
                    "name": tr["name"] + "_holdout",
                    "h": tr["h"][-n_holdout:],
                    "q": tr["q"][-n_holdout:],
                })
                tr["h"] = tr["h"][:-n_holdout]
                tr["q"] = tr["q"][:-n_holdout]

    n_train = sum(tr["h"].shape[0] for tr in train_rounds)
    n_holdout = sum(th["h"].shape[0] for th in train_holdout)
    n_val = sum(vr["h"].shape[0] for vr in val_rounds_list) if val_rounds_list else 0
    nz_train = sum((tr["q"].abs() > 0.05).sum().item() for tr in train_rounds)
    print(f"\nTrain: {n_train:,} frames ({len(train_rounds)} rounds)  nonzero Q: {nz_train:,}")
    if train_holdout:
        nz_holdout = sum((th["q"].abs() > 0.05).sum().item() for th in train_holdout)
        print(f"Holdout: {n_holdout:,} frames (within-round, unseen)  nonzero Q: {nz_holdout:,}")
    if val_rounds_list:
        nz_val = sum((vr["q"].abs() > 0.05).sum().item() for vr in val_rounds_list)
        print(f"Val:   {n_val:,} frames ({len(val_rounds_list)} rounds)  nonzero Q: {nz_val:,}")
    else:
        print(f"Val:   none")

    # Q standard deviation for leak normalization
    all_q = torch.cat([tr["q"] for tr in train_rounds])
    q_std = all_q.std().item() + 1e-8

    # Optional: renormalize Q to restore gradient magnitude after dead zone.
    # Use max(|Q|) not q_std — Q is 85% zeros, so std is misleadingly small.
    if q_renorm:
        all_q = torch.cat([tr["q"] for tr in train_rounds])
        q_max = all_q.abs().max().item() + 1e-8
        for tr in train_rounds:
            tr["q"] = tr["q"] / q_max
        for vr in val_rounds_list:
            vr["q"] = vr["q"] / q_max
        for th in train_holdout:
            th["q"] = th["q"] / q_max
        all_q = torch.cat([tr["q"] for tr in train_rounds])
        q_std = all_q.std().item() + 1e-8
        print(f"  Q renormalized by max|Q|={q_max:.4f}: "
              f"new max|Q|={all_q.abs().max():.2f}, new std={q_std:.4f}")

    return train_rounds, val_rounds_list, train_holdout, hidden_dim, q_std


# ── BATCH SAMPLING ────────────────────────────────────────────────────────────

def make_balanced_sampler(q_tensor):
    neg_idx = torch.where(q_tensor < -0.05)[0]
    pos_idx = torch.where(q_tensor > 0.05)[0]
    neu_idx = torch.where(q_tensor.abs() <= 0.05)[0]
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise RuntimeError(
            f"Need both negative and positive Q frames. "
            f"Got neg={len(neg_idx)}, pos={len(pos_idx)}, neu={len(neu_idx)}.")
    n_per_class = min(len(neg_idx), len(pos_idx), len(neu_idx))
    batches_per_epoch = max(1, n_per_class * 3 // BATCH_SIZE)
    return neg_idx, pos_idx, neu_idx, n_per_class, batches_per_epoch


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train(adapter, q_head, train_rounds, val_rounds_list, train_holdout, T,
          epochs, lr, vlm_lr, batch_size, model_name, leak_offset=None, q_std=1.0,
          leak_random=False, no_balanced=False):
    """Train with adapter + Q-head. Supports T=1 (single-frame) and T>1 (temporal windows).

    For T>1: history frames use frozen pre-extracted features. Only the LAST frame
    in each window goes through the trainable adapter.
    """
    def leak_value(q_ref):
        """Return (scaled) Q target for leak, or random noise if leak_random."""
        if leak_random:
            return torch.randn_like(q_ref)
        return q_ref / q_std

    is_mlp = (model_name == "mlp")
    is_temporal = (model_name in ("tcn", "causal_tf"))

    # ── Build flat training arrays for balanced sampling ──
    # For T=1: sample individual frames
    # For T>1: sample windows from consecutive frames within rounds
    if T == 1:
        # Flatten all training frames
        train_h_flat = torch.cat([tr["h"] for tr in train_rounds], dim=0)
        train_q_flat = torch.cat([tr["q"] for tr in train_rounds], dim=0)
    else:
        # Build window index: (round_idx, last_frame_idx, q_target)
        train_windows = []
        for ri, tr in enumerate(train_rounds):
            n = tr["h"].shape[0]
            for i in range(T - 1, n):
                train_windows.append((ri, i, tr["q"][i].item()))
        print(f"  Training windows: {len(train_windows):,}  (T={T})")

    # ── Build flat val arrays ──
    if val_rounds_list:
        val_h_flat = torch.cat([vr["h"] for vr in val_rounds_list], dim=0)
        val_q_flat = torch.cat([vr["q"] for vr in val_rounds_list], dim=0)
    else:
        # Fallback: 80/20 split from train
        all_h = torch.cat([tr["h"] for tr in train_rounds], dim=0)
        all_q = torch.cat([tr["q"] for tr in train_rounds], dim=0)
        n_train = int(0.8 * len(all_h))
        perm = torch.randperm(len(all_h))
        val_h_flat = all_h[perm[n_train:]]
        val_q_flat = all_q[perm[n_train:]]
        # Keep only training portion
        train_mask = torch.zeros(len(all_h), dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        # Rebuild train_rounds with training subset — complex, skip for now
        # Use flat indices for T=1; for T>1 with fallback, rebuild windows
        if T == 1:
            train_h_flat = all_h[perm[:n_train]]
            train_q_flat = all_q[perm[:n_train]]
        else:
            print("  ⚠️  80/20 fallback with T>1 not supported — use --val")
            return None

    # ── Balanced sampler ──
    train_q_ref = train_q_flat if T == 1 else torch.tensor([w[2] for w in train_windows])
    neg_idx = torch.where(train_q_ref < -0.05)[0]
    pos_idx = torch.where(train_q_ref > 0.05)[0]
    neu_idx = torch.where(train_q_ref.abs() <= 0.05)[0]
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        print("  ⚠️  Need both neg and pos Q frames")
        return None
    n_per_class = min(len(neg_idx), len(pos_idx), len(neu_idx))
    batches_per_epoch = max(1, n_per_class * 3 // batch_size)

    # ── Optimizer ──
    adapter_params = list(adapter.parameters()) if adapter else []
    head_params = list(q_head.parameters())
    param_groups = [{"params": head_params, "lr": lr}]
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": vlm_lr})
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # ── Null baseline ──
    val_neg_mask = val_q_flat < -0.05
    val_pos_mask = val_q_flat > 0.05
    null_neg = (val_q_flat[val_neg_mask]**2).mean().item()
    null_pos = (val_q_flat[val_pos_mask]**2).mean().item()
    print(f"\nNull baseline — Neg: {null_neg:.4f}  Pos: {null_pos:.4f}")

    history = {"train_loss": [], "val_mse": [], "val_neg_mse": [],
               "val_pos_mse": [], "val_neu_mae": []}

    print(f"\nTraining {epochs} epochs...")
    t_start = time.time()

    for epoch in range(epochs):
        if adapter: adapter.train()
        q_head.train()
        epoch_loss = 0.0

        for _ in range(batches_per_epoch):
            if no_balanced:
                batch_indices = torch.randint(0, len(train_q_ref), (batch_size,))
            else:
                ni = neg_idx[torch.randint(0, len(neg_idx), (batch_size // 3,))]
                pi = pos_idx[torch.randint(0, len(pos_idx), (batch_size // 3,))]
                ui = neu_idx[torch.randint(0, len(neu_idx), (batch_size - len(ni) - len(pi),))]
                batch_indices = torch.cat([ni, pi, ui])[torch.randperm(batch_size)]

            if T == 1:
                # Single frame: adapter on every frame
                x = train_h_flat[batch_indices].to(DEVICE, dtype=torch.float32)
                if adapter:
                    x = adapter(x)  # [B, 960]
                q_batch = train_q_flat[batch_indices].to(DEVICE, dtype=torch.float32)
                if leak_offset is not None:
                    x = torch.cat([x, leak_value(q_batch).unsqueeze(1)], dim=1)
                if is_mlp:
                    pred = q_head(x).squeeze(-1)
                else:
                    pred = q_head(x.unsqueeze(1)).squeeze(-1)
            else:
                # Temporal window: history frozen, current goes through adapter
                B = len(batch_indices)
                window_h = torch.zeros(B, T, train_rounds[0]["h"].shape[1],
                                       dtype=torch.float32, device=DEVICE)

                # Collect current frames for batched adapter forward
                current_frames_list = []
                for j, wi in enumerate(batch_indices.tolist()):
                    ri, last_idx, _ = train_windows[wi]
                    tr = train_rounds[ri]
                    start = last_idx - T + 1
                    window_h[j, :T-1] = tr["h"][start:last_idx].to(DEVICE)
                    current_frames_list.append(tr["h"][last_idx])
                current_frames = torch.stack(current_frames_list).to(DEVICE, dtype=torch.float32)
                window_h[:, T-1] = adapter(current_frames) if adapter else current_frames

                # Leak Q target into temporal window
                q_batch = torch.tensor([train_windows[wi][2] for wi in batch_indices.tolist()],
                                       dtype=torch.float32, device=DEVICE)
                if leak_offset is not None:
                    leak = torch.zeros(B, T, 1, device=DEVICE)
                    pos = T - 1 - leak_offset
                    if 0 <= pos < T:
                        leak[:, pos, 0] = leak_value(q_batch)
                    window_h = torch.cat([window_h, leak], dim=-1)

                pred = q_head(window_h).squeeze(-1)  # TCN / causal_tf

            loss = F.mse_loss(pred, q_batch)
            optimizer.zero_grad()
            loss.backward()
            all_params = head_params + adapter_params
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # ── Validation ──
        if adapter: adapter.eval()
        q_head.eval()
        with torch.no_grad():
            if T == 1 or is_mlp:
                val_preds = []
                for i in range(0, len(val_h_flat), batch_size):
                    idx = slice(i, min(i + batch_size, len(val_h_flat)))
                    vx = val_h_flat[idx].to(DEVICE, dtype=torch.float32)
                    if adapter:
                        vx = adapter(vx)
                    if leak_offset is not None:
                        vq = val_q_flat[idx].to(DEVICE, dtype=torch.float32)
                        vx = torch.cat([vx, leak_value(vq).unsqueeze(1)], dim=1)
                    if is_mlp:
                        vp = q_head(vx).cpu().squeeze(-1)
                    else:
                        vp = q_head(vx.unsqueeze(1)).cpu().squeeze(-1)
                    val_preds.append(vp)
                val_pred = torch.cat(val_preds)
            else:
                # Val with T>1: batched adapter on current frames
                val_preds = []
                for vr in val_rounds_list:
                    vh = vr["h"]
                    n = vh.shape[0]
                    for i in range(T - 1, n, batch_size):
                        end = min(i + batch_size, n)
                        B_actual = end - i
                        wh = torch.zeros(B_actual, T, vh.shape[1], dtype=torch.float32)
                        # Batched adapter forward (or raw features if no adapter)
                        if adapter:
                            adapted = adapter(
                                vh[i:end].to(DEVICE, dtype=torch.float32)
                            ).cpu()
                        else:
                            adapted = vh[i:end]
                        for j in range(B_actual):
                            frame_idx = i + j
                            start = frame_idx - T + 1
                            wh[j, :T-1] = vh[start:frame_idx]
                            wh[j, T-1] = adapted[j]
                        # Leak Q into validation window
                        if leak_offset is not None:
                            vq_window = vr["q"][i:end]  # Q for current frames
                            leak = torch.zeros(B_actual, T, 1)
                            pos = T - 1 - leak_offset
                            if 0 <= pos < T:
                                leak[:, pos, 0] = leak_value(vq_window)
                            wh = torch.cat([wh, leak], dim=-1)
                        vp = q_head(wh.to(DEVICE, dtype=torch.float32)).cpu().squeeze(-1)
                        val_preds.append(vp)
                val_pred = torch.cat(val_preds)
                # Trim val_q to match (first T-1 frames have no prediction)
                val_q_eval = torch.cat([vr["q"][T-1:] for vr in val_rounds_list])
                val_neg_mask = val_q_eval < -0.05
                val_pos_mask = val_q_eval > 0.05
                val_neu_mask = ~val_neg_mask & ~val_pos_mask

            if T == 1 or is_mlp:
                val_q_eval = val_q_flat
                val_neu_mask = val_q_flat.abs() <= 0.05

            val_mse = F.mse_loss(val_pred, val_q_eval).item()
            neg_mse = (F.mse_loss(val_pred[val_neg_mask], val_q_eval[val_neg_mask]).item()
                       if val_neg_mask.any() else 0)
            pos_mse = (F.mse_loss(val_pred[val_pos_mask], val_q_eval[val_pos_mask]).item()
                       if val_pos_mask.any() else 0)
            neu_mae = (F.l1_loss(val_pred[val_neu_mask], val_q_eval[val_neu_mask]).item()
                       if val_neu_mask.any() else 0)

        avg_loss = epoch_loss / max(1, batches_per_epoch)
        history["train_loss"].append(avg_loss)
        history["val_mse"].append(val_mse)
        history["val_neg_mse"].append(neg_mse)
        history["val_pos_mse"].append(pos_mse)
        history["val_neu_mae"].append(neu_mae)

        # ── Training metrics (on a sample) ──
        train_neg, train_pos, train_neu_mae = -1.0, -1.0, -1.0
        with torch.no_grad():
            n_sample = min(2000, len(train_q_ref))
            sample_idx = torch.randperm(len(train_q_ref))[:n_sample]
            if T == 1:
                tx = train_h_flat[sample_idx].to(DEVICE, dtype=torch.float32)
                if adapter:
                    tx = adapter(tx)
                if is_mlp:
                    tp = q_head(tx).cpu().squeeze(-1)
                else:
                    tp = q_head(tx.unsqueeze(1)).cpu().squeeze(-1)
            else:
                # Build windows for sampled frames
                tp_list = []
                for si in sample_idx.tolist():
                    ri, last_idx, _ = train_windows[si]
                    tr = train_rounds[ri]
                    start = last_idx - T + 1
                    wh = torch.zeros(1, T, tr["h"].shape[1], dtype=torch.float32)
                    wh[0, :T-1] = tr["h"][start:last_idx]
                    cf = tr["h"][last_idx:last_idx+1].to(DEVICE, dtype=torch.float32)
                    wh[0, T-1] = adapter(cf).cpu().squeeze(0) if adapter else cf.cpu().squeeze(0)
                    tp_list.append(q_head(wh.to(DEVICE, dtype=torch.float32)).cpu().squeeze(-1))
                tp = torch.cat(tp_list)
            tq = train_q_ref[sample_idx]
            tn_mask = tq < -0.05; tpos_mask = tq > 0.05; tneu_mask = ~tn_mask & ~tpos_mask
            train_neg = F.mse_loss(tp[tn_mask], tq[tn_mask]).item() if tn_mask.any() else 0
            train_pos = F.mse_loss(tp[tpos_mask], tq[tpos_mask]).item() if tpos_mask.any() else 0
            train_neu_mae = F.l1_loss(tp[tneu_mask], tq[tneu_mask]).item() if tneu_mask.any() else 0

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            # Within-round holdout metrics
            hold_neg, hold_pos, hold_neu = -1.0, -1.0, -1.0
            if train_holdout:
                hold_h = torch.cat([th["h"] for th in train_holdout], dim=0)
                hold_q = torch.cat([th["q"] for th in train_holdout], dim=0)
                with torch.no_grad():
                    hp_list = []
                    for i in range(0, len(hold_h), batch_size):
                        idx = slice(i, min(i + batch_size, len(hold_h)))
                        hx = hold_h[idx].to(DEVICE, dtype=torch.float32)
                        if adapter: hx = adapter(hx)
                        if is_mlp:
                            hp_list.append(q_head(hx).cpu().squeeze(-1))
                        else:
                            hp_list.append(q_head(hx.unsqueeze(1)).cpu().squeeze(-1))
                    hp = torch.cat(hp_list)
                    hn = hold_q < -0.05; hpos = hold_q > 0.05; hneu = ~hn & ~hpos
                    hold_neg = F.mse_loss(hp[hn], hold_q[hn]).item() if hn.any() else 0
                    hold_pos = F.mse_loss(hp[hpos], hold_q[hpos]).item() if hpos.any() else 0
                    hold_neu = F.l1_loss(hp[hneu], hold_q[hneu]).item() if hneu.any() else 0

            t_elapsed = time.time() - t_start
            t_per = t_elapsed / (epoch + 1)
            t_rem = t_per * (epochs - epoch - 1)
            parts = [f"Epoch {epoch:3d} | loss={avg_loss:.4f}"]
            parts.append(f"tr_neg={train_neg:.4f} tr_pos={train_pos:.4f} tr_neu={train_neu_mae:.4f}")
            if train_holdout:
                parts.append(f"hold_neg={hold_neg:.4f} hold_pos={hold_pos:.4f} hold_neu={hold_neu:.4f}")
            parts.append(f"val_neg={neg_mse:.4f} val_pos={pos_mse:.4f} val_neu={neu_mae:.4f}")
            parts.append(f"[{t_per:.1f}s/ep, {t_rem/60:.0f}m left]")
            print("  " + " | ".join(parts))

    # ── Final report ──
    print(f"\n--- Final ---")
    improvement_neg = null_neg / neg_mse if neg_mse > 0 else float("inf")
    improvement_pos = null_pos / pos_mse if pos_mse > 0 else float("inf")
    print(f"Null baseline MSE: neg={null_neg:.4f}  pos={null_pos:.4f}")
    print(f"Trained     MSE:  neg={neg_mse:.4f}  pos={pos_mse:.4f}")
    print(f"Improvement:      neg={improvement_neg:.0f}×  pos={improvement_pos:.0f}×")
    if improvement_neg > 10:
        print("✅ >10× — Q-head is learning")
    else:
        print("⚠️  <10× — check features or architecture")

    return history


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot_training_curves(history, tag):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"Train Loss ({tag})")
    axes[0].set_xlabel("Epoch"); axes[0].grid(True)
    axes[1].plot(history["val_neg_mse"], label="neg MSE", color="red")
    axes[1].plot(history["val_pos_mse"], label="pos MSE", color="green")
    axes[1].set_title("Val Class MSE"); axes[1].legend()
    axes[1].set_xlabel("Epoch"); axes[1].grid(True)
    axes[2].plot(history["val_neu_mae"], label="neu MAE", color="gray")
    axes[2].set_title("Val Neutral MAE")
    axes[2].set_xlabel("Epoch"); axes[2].grid(True)
    fig.tight_layout()
    path = PLOTS_DIR / f"training_curves_{tag}.png"
    fig.savefig(path)
    print(f"Saved {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    (model_name, features_dir, feature_field, val_rounds, train_rounds_set,
     epochs, batch_size, lr, vlm_lr, temporal_window,
     unfreeze_layers, leak_offset, leak_random, no_adapter, train_split,
     tcn_pool, no_balanced, hidden_layers, dropout, dead_zone, q_renorm) = _parse_cli()

    T = temporal_window

    print("=" * 65)
    print(f"2_train_q_head_v5.py — VLM Adapter + Q-Head Training")
    print("=" * 65)
    print(f"  Model:           {model_name}")
    print(f"  Feature field:   {feature_field}")
    print(f"  Features dir:    {features_dir}")
    print(f"  Unfreeze layers: {unfreeze_layers}")
    print(f"  VLM LR:          {vlm_lr}")
    print(f"  Head LR:         {lr}")
    print(f"  Temporal window: {T}")
    if leak_offset is not None:
        tag = "RANDOM" if leak_random else (
            "current" if leak_offset == 0 else f"t-{leak_offset}")
        print(f"  Leak target:     {tag}")
    if model_name == "tcn":
        print(f"  TCN pool:        {tcn_pool}")
    if model_name == "causal_tf":
        print(f"  Causal TF layers: 2  d_model: 256  heads: 8")
    if train_rounds_set is not None:
        print(f"  Train rounds:    {train_rounds_set}")
    print(f"  Val rounds:      {val_rounds}")
    if dead_zone > 0:
        print(f"  Dead zone:       ±{dead_zone} frames around intervention onsets")
    if q_renorm:
        print(f"  Q renormalize:   yes (divide by training q_std)")
    print(f"  Epochs: {epochs} | Batch: {batch_size}")
    print(f"  Device: {DEVICE}")
    print()

    # Load features
    train_rounds, val_rounds_list, train_holdout, hidden_dim, q_std = load_feature_data(
        features_dir, feature_field, val_rounds, train_split, train_rounds_set, dead_zone,
        q_renorm)

    # Determine input layer from feature field
    input_layer = 15
    if feature_field.startswith("layer_"):
        input_layer = int(feature_field.split("_")[1])

    print(f"\nInput features: layer {input_layer} (frozen)")
    if no_adapter:
        print(f"Adapter:         SKIPPED (--no-adapter)")
    else:
        print(f"Adapter layers: {unfreeze_layers} (trainable)")

    # Load policy for adapter (skip if --no-adapter)
    adapter = None
    if not no_adapter:
        print(f"\nLoading policy for VLM adapter...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
        policy.model.eval()
        adapter = load_vlm_adapter(policy, input_layer, unfreeze_layers)
        adapter.to(DEVICE, dtype=torch.float32)

    # Create Q-head (extra dim for leak channel if enabled)
    head_input_dim = hidden_dim + (1 if leak_offset is not None else 0)
    if model_name == "mlp" or T == 1:
        # Parse --hidden-layers: comma-separated list of hidden dims
        h_layers_raw = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()] if hidden_layers else None
        if h_layers_raw is None:
            h_layers = (512, 64)  # default
        elif h_layers_raw == [0]:
            h_layers = ()  # linear probe
        else:
            h_layers = tuple(x for x in h_layers_raw if x > 0)
        q_head = QValueHead(head_input_dim, hidden_layers=h_layers,
                            dropout=dropout, use_tanh=False).to(DEVICE, dtype=torch.float32)
    elif model_name == "causal_tf":
        if T < 2:
            print("❌ Causal transformer requires --temporal-window >= 2")
            return
        q_head = CausalTemporalTransformer(
            hidden_dim=head_input_dim,
            d_model=256,
            num_layers=2,
            num_heads=8,
            dropout=dropout,
            max_seq_len=T + 10,
            use_tanh=False,
        ).to(DEVICE, dtype=torch.float32)
    else:
        q_head = TemporalTCNQHead(hidden_dim=head_input_dim, use_tanh=False,
                                   pool=tcn_pool, dropout=dropout).to(DEVICE, dtype=torch.float32)

    total_h, trainable_h = count_params(q_head)
    adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad) if adapter else 0
    print(f"\nQ-head:     {total_h:,} params ({trainable_h:,} trainable)")
    if adapter:
        print(f"Adapter:    {adapter_params:,} trainable")
        print(f"Total:      {trainable_h + adapter_params:,} trainable")

    # Train
    history = train(
        adapter, q_head, train_rounds, val_rounds_list, train_holdout, T,
        epochs, lr, vlm_lr, batch_size, model_name, leak_offset, q_std, leak_random,
        no_balanced)

    if history is None:
        print("❌ Training aborted — see error above")
        return

    # Save
    tag = f"v5_{feature_field}_{model_name}_t{T}"
    if leak_offset is not None:
        tag += f"_leak{leak_offset}"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(q_head.state_dict(), WEIGHTS_DIR / f"q_head_weights_{tag}.pt")
    if adapter:
        torch.save({
            "adapter_state": {name: p.data.clone() for name, p in adapter.named_parameters()},
            "unfreeze_layers": unfreeze_layers,
            "input_layer": input_layer,
        }, WEIGHTS_DIR / f"vlm_adapter_weights_{tag}.pt")
        print(f"\nWeights saved: weights/q_head_weights_{tag}.pt, weights/vlm_adapter_weights_{tag}.pt")
    else:
        print(f"\nWeights saved: weights/q_head_weights_{tag}.pt")

    if len(history["train_loss"]) > 0:
        plot_training_curves(history, tag)

    print("\nDone.")



if __name__ == "__main__":
    main()
