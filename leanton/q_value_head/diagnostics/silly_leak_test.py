#!/usr/bin/env python3
"""Silly validation: leak the Q target into features. If the pipeline works,
the model should achieve near-zero loss immediately."""

import torch, torch.nn as nn, re
from pathlib import Path

d = Path(__file__).parent / "data/features_v4"
files = sorted(d.glob("*_layer4.pt"))
val_rounds = {"r5_with_q", "r18_with_q"}

train_h, train_q = [], []
val_h, val_q = [], []

for f in files:
    data = torch.load(f, weights_only=True, map_location="cpu")
    h = data["hidden_states"].float()
    q = data["q_targets"].float()
    key = re.sub(r"_layer\d+$", "", f.stem)
    if key in val_rounds:
        val_h.append(h); val_q.append(q)
    else:
        train_h.append(h); train_q.append(q)

train_h = torch.cat(train_h); train_q = torch.cat(train_q)
val_h = torch.cat(val_h); val_q = torch.cat(val_q)

# Leak Q into features
train_h = torch.cat([train_h, train_q.unsqueeze(1)], dim=1)
val_h = torch.cat([val_h, val_q.unsqueeze(1)], dim=1)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP(961).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

neg_mask = train_q < -0.05; pos_mask = train_q > 0.05; neu_mask = ~neg_mask & ~pos_mask
neg_idx = torch.where(neg_mask)[0]; pos_idx = torch.where(pos_mask)[0]; neu_idx = torch.where(neu_mask)[0]
n_per_class = min(len(neg_idx), len(pos_idx), len(neu_idx))
batches_per_epoch = max(1, n_per_class * 3 // 64)

for ep in range(30):
    model.train()
    for _ in range(batches_per_epoch):
        ni = neg_idx[torch.randint(0, len(neg_idx), (21,))]
        pi = pos_idx[torch.randint(0, len(pos_idx), (21,))]
        ui = neu_idx[torch.randint(0, len(neu_idx), (22,))]
        idx = torch.cat([ni, pi, ui])[torch.randperm(64)]

        x = train_h[idx].cuda(); y = train_q[idx].cuda()
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

    if ep % 5 == 0:
        model.eval()
        with torch.no_grad():
            vp = model(val_h.cuda()).cpu()
            vm = nn.functional.mse_loss(vp, val_q).item()
            neg_m = nn.functional.mse_loss(vp[val_q < -0.05], val_q[val_q < -0.05]).item()
            pos_m = nn.functional.mse_loss(vp[val_q > 0.05], val_q[val_q > 0.05]).item()
        print(f"Epoch {ep:3d}: val_mse={vm:.6f}  neg={neg_m:.6f}  pos={pos_m:.6f}")

# Null baseline
nz = val_q < -0.05; pz = val_q > 0.05
null_neg = nn.functional.mse_loss(torch.zeros(nz.sum()), val_q[nz]).item()
null_pos = nn.functional.mse_loss(torch.zeros(pz.sum()), val_q[pz]).item()
print(f"\nNull: neg={null_neg:.4f} pos={null_pos:.4f}")
print(f"Best neg: {neg_m:.6f} ({null_neg/neg_m:.0f}x)" if neg_m > 0 else "neg=0")
print(f"Best pos: {pos_m:.6f} ({null_pos/pos_m:.0f}x)" if pos_m > 0 else "pos=0")
