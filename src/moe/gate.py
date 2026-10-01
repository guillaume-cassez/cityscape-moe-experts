"""Gating heads for the MoE harness. Input = stacked expert logits (B, K, C, H, W).
Output = per-pixel weights (B, K, H, W) summing to 1 over the K experts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGate(nn.Module):
    """Lightweight conv gate on the concatenated expert logits."""

    def __init__(self, n_experts: int, n_classes: int = 19, hidden: int = 128):
        super().__init__()
        self.n_experts = n_experts
        self.n_classes = n_classes
        in_ch = n_experts * n_classes
        groups = 16 if hidden % 16 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_experts, 1),
        )

    def forward(self, stacked):  # (B, K, C, H, W)
        b, k, c, h, w = stacked.shape
        x = stacked.reshape(b, k * c, h, w)
        return F.softmax(self.net(x), dim=1)  # (B, K, H, W)


class TransformerGate(nn.Module):
    """Per-pixel attention over the K expert tokens (each token = an expert's logit vector).

    Cheap for small K but O(H*W) sequences — intended for crop-based training. Kept as the
    'transformer' arm of the CNN-vs-transformer comparison (Keep note).
    """

    def __init__(self, n_experts: int, n_classes: int = 19, dim: int = 64, heads: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.embed = nn.Linear(n_classes, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.to_w = nn.Linear(dim, 1)

    def forward(self, stacked):  # (B, K, C, H, W)
        b, k, c, hh, ww = stacked.shape
        tok = stacked.permute(0, 3, 4, 1, 2).reshape(b * hh * ww, k, c)  # (N, K, C)
        tok = self.embed(tok)
        att, _ = self.attn(tok, tok, tok)
        att = self.norm(att + tok)
        wt = self.to_w(att).squeeze(-1)         # (N, K)
        wt = F.softmax(wt, dim=1)
        return wt.reshape(b, hh, ww, k).permute(0, 3, 1, 2).contiguous()  # (B, K, H, W)


def build_gate(kind: str, n_experts: int, n_classes: int = 19) -> nn.Module:
    kind = kind.lower()
    if kind == "cnn":
        return CNNGate(n_experts, n_classes)
    if kind == "transformer":
        return TransformerGate(n_experts, n_classes)
    raise ValueError(f"unknown gate kind '{kind}' (cnn|transformer)")
