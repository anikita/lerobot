#!/usr/bin/env python3
"""
model.py — Q-value head architectures for the reasoning tower paradox.

Classes:
    QValueHead         — MLP on mean-pooled features [B, 960] (v1-compatible)
    TransformerQHead   — Small transformer on full prefix sequence [B, seq_len, 960]

All models output a scalar Q ∈ [-1, 1] via tanh. Zero LeRobot dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── QValueHead (v1-compatible MLP on mean-pooled features) ────────────────────

class QValueHead(nn.Module):
    """Predict q_target from a mean-pooled backbone hidden state.

    Input:  [B, 960]    — mean-pooled over all prefix tokens
    Output: [B, 1]      — scalar Q-value, bounded to [-1, 1] via tanh

    Architecture: 960 → 512 → 64 → 1 + tanh  (~0.52M params)
    Forward pass: ~0.01ms on GPU.
    """

    def __init__(self, hidden_dim=960, hidden_layers=(512, 64), dropout=0.0,
                 use_tanh=True):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        if use_tanh:
            layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)
        self.use_tanh = use_tanh

    def forward(self, x):
        # x: [B, hidden_dim]
        return self.mlp(x)  # [B, 1]


# ── TransformerQHead (full sequence, cross-modal attention) ───────────────────

class TransformerQHead(nn.Module):
    """Predict q_target from the FULL prefix token sequence.

    Input:  [B, seq_len, 960]    — all prefix tokens (vision + lang + state)
    Output: [B, 1]               — scalar Q-value, bounded to [-1, 1]

    Architecture:
        960 → linear projection → d_model (256)
        + learnable position embeddings
        + learnable CLS token
        → 2-3 transformer encoder layers (self-attention, GELU, layer norm)
        → extract CLS token
        → MLP: d_model → 64 → 1 + tanh

    ~1.9M params at default config. Forward pass ~0.1ms on GPU.

    The transformer learns which token positions carry the failure signal —
    no hand-crafted camera splitting, no assumption about which modality
    encodes what. Self-attention routes information across cameras, language,
    and proprioception automatically.
    """

    def __init__(
        self,
        hidden_dim=960,
        d_model=256,
        num_layers=2,
        num_heads=8,
        dim_feedforward=None,
        dropout=0.1,
        max_seq_len=256,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Project backbone hidden dim → model dim
        self.input_proj = nn.Linear(hidden_dim, d_model)

        # Learnable position embeddings (shared across all sequences)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Initialize with small weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.input_proj, *self.head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, seq_len, hidden_dim] — full prefix hidden states
            mask: [B, seq_len] or None — True for positions to IGNORE (padding)

        Returns:
            q: [B, 1] — predicted Q-value ∈ [-1, 1]
        """
        B, S, _ = x.shape

        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}. "
                f"Re-create the model with max_seq_len={S}."
            )

        # Project to model dimension
        x = self.input_proj(x)  # [B, S, d_model]

        # Add position embeddings
        x = x + self.pos_embed[:, :S, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+S, d_model]

        # Build attention mask: CLS + sequence
        if mask is not None:
            # mask: [B, S] True=ignore → prepend False for CLS
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, mask], dim=1)  # [B, 1+S]
        else:
            src_key_padding_mask = None

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, 1+S, d_model]

        # Extract CLS token output
        cls_out = x[:, 0, :]  # [B, d_model]

        # Output head
        return self.head(cls_out)  # [B, 1]


# ── RLTQHead (RL Token: autoencoder bottleneck + Q-value) ────────────────────

class RLTQHead(nn.Module):
    """RL-Token Q-Head: bottleneck representation with reconstruction auxiliary loss.

    Architecture inspired by Physical Intelligence's RL^T paper:
      1. Project input: [B, S, 960] → [B, S, d_model]
      2. Transformer encoder + CLS token → RL token [B, d_model]
      3. Transformer decoder: RL token → reconstructs original [B, S, 960]
      4. Q-head on RL token → [B, 1]

    The reconstruction loss forces the RL token to preserve per-token, per-camera
    information that mean-pooling destroys. The Q-head gets a richer signal.

    Training:
        q, reconstructed = model(x, mask)
        loss, q_loss, recon_loss = model.compute_loss(q, q_target, reconstructed, x_original)
        loss.backward()  # total = MSE(Q) + λ * MSE(reconstruction)
    """

    def __init__(
        self,
        hidden_dim=960,
        d_model=256,
        num_enc_layers=2,
        num_dec_layers=1,
        num_heads=8,
        dim_feedforward=None,
        dropout=0.1,
        max_seq_len=256,
        recon_weight=0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.recon_weight = recon_weight

        # --- Input projection ---
        self.input_proj = nn.Linear(hidden_dim, d_model)

        # --- Position embeddings ---
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # --- CLS token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Transformer encoder (same as TransformerQHead) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # --- Decoder: RL token → reconstructed sequence ---
        # Learned query positions that cross-attend to the RL token
        self.decoder_queries = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        # --- Output projection (back to original hidden dim) ---
        self.output_proj = nn.Linear(d_model, hidden_dim)

        # --- Q-head on RL token ---
        self.q_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for mod in [self.input_proj, self.output_proj, *self.q_head]:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight, gain=0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, S, hidden_dim] — full prefix hidden states
            mask: [B, S] or None — True for positions to IGNORE (padding)

        Returns:
            q:            [B, 1] — predicted Q-value
            reconstructed: [B, S, hidden_dim] — decoder output (for recon loss)
        """
        B, S, _ = x.shape

        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}."
            )

        # Project
        x_proj = self.input_proj(x)  # [B, S, d_model]
        x_proj = x_proj + self.pos_embed[:, :S, :]

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x_enc = torch.cat([cls, x_proj], dim=1)  # [B, 1+S, d_model]

        # Mask
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            enc_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            enc_mask = None

        # Encode
        enc_out = self.encoder(x_enc, src_key_padding_mask=enc_mask)  # [B, 1+S, d_model]

        # RL token = CLS output
        rlt = enc_out[:, 0, :]  # [B, d_model]

        # Decode: reconstruct original sequence from RL token
        memory = rlt.unsqueeze(1)  # [B, 1, d_model] — the RL token is the memory
        queries = self.decoder_queries[:, :S, :].expand(B, -1, -1)  # [B, S, d_model]
        decoded = self.decoder(
            tgt=queries, memory=memory,
            tgt_key_padding_mask=mask,
        )  # [B, S, d_model]

        # Project back to original dimension
        reconstructed = self.output_proj(decoded)  # [B, S, hidden_dim]

        # Q-value from RL token
        q = self.q_head(rlt)  # [B, 1]

        return q, reconstructed

    def compute_loss(self, q_pred, q_target, reconstructed, original_seq, mask=None):
        """Combined loss: MSE(Q) + λ * MSE(reconstruction)."""
        q_loss = F.mse_loss(q_pred.squeeze(), q_target)

        # Reconstruction loss — only on non-padded positions
        if mask is not None:
            valid = ~mask  # [B, S], True = real token
            n_valid = valid.sum()
            if n_valid > 0:
                recon_loss = (F.mse_loss(
                    reconstructed[valid], original_seq[valid], reduction='sum'
                ) / n_valid)
            else:
                recon_loss = torch.tensor(0.0, device=q_pred.device)
        else:
            recon_loss = F.mse_loss(reconstructed, original_seq)

        total = q_loss + self.recon_weight * recon_loss
        return total, q_loss.detach(), recon_loss.detach()

    @torch.no_grad()
    def predict(self, x, mask=None):
        """Inference-only: return Q without reconstruction."""
        q, _ = self.forward(x, mask)
        return q


# ── TemporalTCNQHead (causal TCN over temporal window) ──────────────────────────

class CausalConvBlock(nn.Module):
    """Single causal conv block: pad left → Conv1d → LayerNorm → ReLU → Dropout."""

    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_left = (kernel_size - 1) * dilation  # causal: only pad past

        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, bias=False)
        self.norm = nn.LayerNorm(channels)  # LayerNorm: no train/eval mismatch under balanced sampling
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad_left, 0))        # causal pad — left only
        x = self.conv(x)                         # [B, C, T]
        x = x.transpose(1, 2)                    # [B, T, C] for LayerNorm over channels
        x = self.norm(x)
        x = x.transpose(1, 2)                    # [B, C, T]
        x = F.relu(x)
        x = self.dropout(x)
        return x


class TemporalTCNQHead(nn.Module):
    """Predict Q from a temporal window of mean-pooled features using causal TCN.

    Input:  [B, T, D]    — T consecutive mean-pooled frames, oldest→newest
    Output: [B, 1]       — scalar Q ∈ [-1, 1] for the last frame

    Architecture:
        D → Linear → 256              project feature dim
        → 5× CausalConv1d blocks      learn velocity/acceleration patterns
          kernel=5, dilations=[1,2,4,8,16], receptive field = 125 frames
        → Global mean pool over T
        → 256 → 64 → 1 + tanh

    ~0.3M params. Causal: frame t sees only frames ≤ t (real-time compatible).
    """

    def __init__(
        self,
        hidden_dim=720,
        tcn_channels=256,
        kernel_size=5,
        dilations=(1, 2, 4, 8, 16),
        dropout=0.1,
        use_tanh=True,
        pool="last",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tcn_channels = tcn_channels
        self.dilations = dilations
        self.pool = pool

        # Input projection: feature dim → tcn channels
        self.input_proj = nn.Linear(hidden_dim, tcn_channels)

        # Causal conv stack
        self.blocks = nn.ModuleList([
            CausalConvBlock(tcn_channels, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        # Output head
        head_layers = [
            nn.Linear(tcn_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        ]
        if use_tanh:
            head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self):
        for mod in [self.input_proj, *self.head]:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight, gain=0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    @property
    def receptive_field(self):
        """Total left-context frames the TCN can see."""
        return 1 + sum((self.blocks[0].kernel_size - 1) * d
                       for d in self.dilations)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] — temporal window of mean-pooled features

        Returns:
            q: [B, 1] — predicted Q-value ∈ [-1, 1]
        """
        B, T, D = x.shape

        # Project feature dim
        x = self.input_proj(x)              # [B, T, 256]

        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)               # [B, 256, T]

        # Causal conv stack
        for block in self.blocks:
            x = block(x)                     # [B, 256, T]

        # Pool over time
        if self.pool == "mean":
            x = x.mean(dim=-1)               # [B, 256]
        elif self.pool == "last":
            x = x[:, :, -1]                  # [B, 256]
        elif self.pool == "max":
            x = x.max(dim=-1).values         # [B, 256]

        # Output head
        return self.head(x)                  # [B, 1]


# ── CausalTemporalTransformer (causal self-attention over time) ────────────────

class CausalTemporalTransformer(nn.Module):
    """Predict Q from a temporal window using causal self-attention.

    Input:  [B, T, D]    — T consecutive feature vectors, oldest → newest
    Output: [B, 1]       — scalar Q ∈ [-1, 1] for the last frame

    Architecture:
        D → Linear → d_model                    project feature dim
        + learnable position embeddings
        → N× TransformerEncoderLayer            causal self-attention over time
          (causal mask: frame t attends to ≤ t)
        → readout from last position (t = T-1)
        → MLP: d_model → 64 → 1 + tanh         output head

    Unlike TCN (fixed kernel, fixed dilations), the transformer learns which
    historical frames are relevant through content-dependent attention weights.
    Global receptive field from the first layer. ~1.9M params at default config.

    Causal mask ensures real-time compatibility: no future frames leak into
    the current prediction.
    """

    def __init__(
        self,
        hidden_dim=960,
        d_model=256,
        num_layers=2,
        num_heads=8,
        dim_feedforward=None,
        dropout=0.1,
        max_seq_len=256,
        use_tanh=True,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection: feature dim → model dim
        self.input_proj = nn.Linear(hidden_dim, d_model)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Transformer encoder layers with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Readout: linear head on last position
        head_layers = [
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        ]
        if use_tanh:
            head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)

        # Precompute causal mask (constant, not a parameter)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        for mod in self.head:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight, gain=0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] — temporal window of feature vectors

        Returns:
            q: [B, 1] — predicted Q-value ∈ [-1, 1] for the last frame
        """
        B, T, D = x.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}."
            )

        # Project to model dimension
        x = self.input_proj(x)  # [B, T, d_model]

        # Add position embeddings
        x = x + self.pos_embed[:, :T, :]

        # Causal mask: frame t can attend to frames 0..t (not t+1..T-1)
        causal = self.causal_mask[:T, :T]  # [T, T], True = MASK (ignore)

        # Transformer encoder with causal mask
        # src_mask: [T, T] — True means "do not attend"
        x = self.encoder(x, mask=causal)  # [B, T, d_model]

        # Readout from last position (the one we're predicting Q for)
        last = x[:, -1, :]  # [B, d_model]

        return self.head(last)  # [B, 1]


# ── Utility ───────────────────────────────────────────────────────────────────

def count_params(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick smoke test
    print("model.py — Smoke test\n")

    # QValueHead
    mlp = QValueHead()
    total, trainable = count_params(mlp)
    print(f"QValueHead:     {total:,} params ({trainable:,} trainable)")
    x_pooled = torch.randn(4, 960)
    out = mlp(x_pooled)
    print(f"  input:  {list(x_pooled.shape)}")
    print(f"  output: {list(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")

    # TransformerQHead
    tf = TransformerQHead()
    total, trainable = count_params(tf)
    print(f"\nTransformerQHead: {total:,} params ({trainable:,} trainable)")
    x_seq = torch.randn(4, 177, 960)
    out = tf(x_seq)
    print(f"  input:  {list(x_seq.shape)}")
    print(f"  output: {list(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")

    # With mask
    mask = torch.zeros(4, 177, dtype=torch.bool)
    mask[:, 170:] = True  # pretend last 7 tokens are padding
    out_masked = tf(x_seq, mask=mask)
    print(f"  masked output: {list(out_masked.shape)}  range=[{out_masked.min():.3f}, {out_masked.max():.3f}]")

    # RLTQHead
    rlt = RLTQHead()
    total, trainable = count_params(rlt)
    print(f"\nRLTQHead:       {total:,} params ({trainable:,} trainable)")
    q, recon = rlt(x_seq)
    print(f"  input:  {list(x_seq.shape)}")
    print(f"  q:      {list(q.shape)}  range=[{q.min():.3f}, {q.max():.3f}]")
    print(f"  recon:  {list(recon.shape)}")

    # Compute loss
    total_loss, q_loss, recon_loss = rlt.compute_loss(
        q.squeeze(), torch.randn(4), recon, x_seq, mask=mask
    )
    print(f"  total_loss={total_loss:.4f}  q_loss={q_loss:.4f}  recon_loss={recon_loss:.4f}")

    # Inference mode
    q_infer = rlt.predict(x_seq, mask=mask)
    print(f"  predict: {list(q_infer.shape)}")

    # TemporalTCNQHead
    tcn = TemporalTCNQHead()
    total, trainable = count_params(tcn)
    rf = tcn.receptive_field
    print(f"\nTemporalTCNQHead: {total:,} params ({trainable:,} trainable)  "
          f"receptive_field={rf}")
    x_temporal = torch.randn(4, 90, 720)
    out = tcn(x_temporal)
    print(f"  input:  {list(x_temporal.shape)}")
    print(f"  output: {list(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")
    # Test with prefix dim (960)
    tcn2 = TemporalTCNQHead(hidden_dim=960)
    x_temporal2 = torch.randn(4, 90, 960)
    out2 = tcn2(x_temporal2)
    print(f"  prefix (960-dim): input {list(x_temporal2.shape)} → "
          f"output {list(out2.shape)}  range=[{out2.min():.3f}, {out2.max():.3f}]")

    # CausalTemporalTransformer
    ctf = CausalTemporalTransformer()
    total, trainable = count_params(ctf)
    print(f"\nCausalTemporalTransformer: {total:,} params ({trainable:,} trainable)")
    x_temp3 = torch.randn(4, 90, 960)
    out3 = ctf(x_temp3)
    print(f"  input:  {list(x_temp3.shape)}")
    print(f"  output: {list(out3.shape)}  range=[{out3.min():.3f}, {out3.max():.3f}]")
    # Verify causality: output should depend on early frames
    x_test = torch.randn(2, 10, 960)
    x_test_mod = x_test.clone()
    x_test_mod[:, 0, :] = 999.0  # modify first frame
    out_a = ctf(x_test)
    out_b = ctf(x_test_mod)
    diff = (out_a - out_b).abs().mean().item()
    print(f"  causality check: modifying frame 0 changes output by {diff:.6f} "
          f"(should be >0 — early frames affect last-position readout)")
    # Verify forward pass with different sequence lengths
    ctf2 = CausalTemporalTransformer(hidden_dim=720)
    x_temporal2 = torch.randn(4, 60, 720)
    out_tf2 = ctf2(x_temporal2)
    print(f"  720-dim, T=60: input {list(x_temporal2.shape)} → "
          f"output {list(out_tf2.shape)}  range=[{out_tf2.min():.3f}, {out_tf2.max():.3f}]")

    print("\n✅ All models pass smoke test")
