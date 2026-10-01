"""Mixture-of-Experts (Paper 3) — late-fusion d'experts gelés (baseline, DESIGN.md)
et MoE patch-wise INTERNE expert-initialisé (MoE-V3-CS, PLAN_TRANSFERT_BRATS_V3.md)."""
from src.moe.experts import (
    EXPERT_SPECS,
    EXPERT_LABELS,
    resolve_expert_path,
    load_expert,
    expert_logits,
)
from src.moe.gate import CNNGate, TransformerGate, build_gate
from src.moe.patch_moe2d import MOE_V3_CS, PatchMoE2D, fpn_block_expert
from src.moe.patch_tiling2d import from_patches, pad_to_grid, to_patches

__all__ = [
    "EXPERT_SPECS",
    "EXPERT_LABELS",
    "resolve_expert_path",
    "load_expert",
    "expert_logits",
    "CNNGate",
    "TransformerGate",
    "build_gate",
    "MOE_V3_CS",
    "PatchMoE2D",
    "fpn_block_expert",
    "from_patches",
    "pad_to_grid",
    "to_patches",
]
