"""Load frozen expert segmentation models and extract their logits.

Reuses src.models.build_model with each checkpoint's embedded config, exactly as
scripts/reeval_official.py does, then freezes the model so only the gate trains.
"""
import os

import torch
from omegaconf import OmegaConf

from src.models import build_model

# short name -> (subdir under repo root, filename pattern with {seed})
EXPERT_SPECS = {
    "Dp": ("checkpoints", "pilot_fullres_DM_distmap_seed{seed}/epoch_160.pth"),       # DistMap D'  (CE+DistMap)
    "Cp": ("checkpoints", "pilot_fullres_DMdice_distmap_seed{seed}/epoch_160.pth"),   # DistMap C'  (CE+Dice+DistMap)
    "G":  ("checkpoints", "pilot_fullres_G_blob_seed{seed}/epoch_160.pth"),           # Blob G     (CE+Dice+0.5·blob Kofler)
    "D":  ("reeval_slim", "pilot_fullres_D_ce_boundary_seed{seed}__epoch_160.pth"),   # Kervadec D  (CE+Boundary)
    "C":  ("reeval_slim", "pilot_fullres_C_boundary_seed{seed}__epoch_160.pth"),      # Kervadec C  (CE+Dice+Boundary)
    "B":  ("reeval_slim", "pilot_fullres_B_baseline_seed{seed}__epoch_160.pth"),      # Baseline B  (CE+Dice)
    "A":  ("reeval_slim", "pilot_fullres_A_ce_seed{seed}__epoch_160.pth"),            # CE A
}

EXPERT_LABELS = {
    "Dp": "DistMap(D')", "Cp": "DistMap(C')", "G": "Blob(G)",
    "D": "Kervadec(D)", "C": "Kervadec(C)",
    "B": "Baseline(B)", "A": "CE(A)",
}


def resolve_expert_path(name: str, seed: int, repo_root: str) -> str:
    if name not in EXPERT_SPECS:
        raise KeyError(f"unknown expert '{name}' (known: {list(EXPERT_SPECS)})")
    sub, pat = EXPERT_SPECS[name]
    return os.path.join(repo_root, sub, pat.format(seed=seed))


def load_expert(ckpt_path: str, device):
    """Rebuild the exact architecture from the checkpoint's embedded config, load the
    weights (strict), move to device, set eval, and freeze all parameters.

    Returns (model, cfg). cfg is reused to build the shared val dataloader.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"expert checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.backbone.pretrained = "none"  # never download timm weights; we overwrite everything
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])  # strict=True: embedded cfg guarantees key match
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def expert_logits(model, images):
    """Forward a frozen expert -> raw logits (B, 19, H, W). Guards the eval/train tuple path."""
    out = model(images)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out["main"]
    return out
