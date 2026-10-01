#!/usr/bin/env python3
"""P3.07 — non-régression du port 2D PatchMoE2D + tiling (CPU, sans GPU).

Validations (plan `PLAN_TRANSFERT_BRATS_V3.md` §7, P3.07 et garde-fou §6
« artefact de grille 2D ») :
  [1] tiling 2D : aller-retour exact (avec et sans halo), pad_to_grid n'ajoute
      des zéros qu'à droite/en bas ;
  [2] forme de sortie préservée, y compris entrée NON divisible par la grille ;
  [3] NON-RÉGRESSION 1-expert/top-1 == bloc plein-volume (au BN près : en mode
      eval les running stats figées rendent l'égalité EXACTE) — prouve que le
      halo + recadrage n'impriment aucune grille ;
  [4] experts IDENTIQUES ⇒ sortie indépendante du routage (top-2/4, poids
      convexes, bruit actif) ;
  [5] expert par défaut = structure copiable depuis `head.fpn_convs.0` d'UPerNet
      (mêmes clés de state_dict) — condition du P3.08 ;
  [6] aux_loss + diagnostics : clés présentes, `entropy_norm` / `entropy_token`
      / `entropy_token_clean` bornées, frac sommé à top_k, balance='none' ⇒ 0 ;
  [7] set_epoch : recuit linéaire déduit du numéro d'époque (safe reprise) ;
  [8] bruit de Shazeer actif en train seulement (routing déterministe en eval) ;
  [9] gradients : backward fini, gate et experts sélectionnés reçoivent un grad ;
  [10] autocast BF16 (règle projet : mixed precision BF16) sans erreur ni NaN ;
  [11] forme réelle : forward C=512 au point de fonctionnement UPerNet.

Run: python3 tests/test_patch_moe2d.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.moe.patch_moe2d import MOE_V3_CS, PatchMoE2D, fpn_block_expert
from src.moe.patch_tiling2d import from_patches, pad_to_grid, to_patches
from src.models.heads.upernet import UPerNetHead

PASS, FAIL, results = "PASS", "FAIL", []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return cond


torch.manual_seed(42)

# ---------------------------------------------------------------------------
print("[1] tiling 2D : aller-retour exact")
x = torch.randn(2, 4, 48, 96)
grid = (3, 3)
p0, shape0 = to_patches(x, grid, halo=0)
check("to_patches(halo=0) formes", p0.shape == (2 * 9, 4, 16, 32) and shape0 == (16, 32))
rt0 = from_patches(p0, 2, grid, shape0)
check("aller-retour halo=0 exact", torch.equal(rt0, x))

hal = 1
p1, shape1 = to_patches(x, grid, halo=hal)
check("to_patches(halo=1) formes", p1.shape == (2 * 9, 4, 18, 34))
interior = p1[:, :, hal:hal + shape1[0], hal:hal + shape1[1]]
rt1 = from_patches(interior, 2, grid, shape1)
check("aller-retour halo=1 exact (intérieur des tuiles)", torch.equal(rt1, x))

xg = pad_to_grid(torch.randn(1, 4, 47, 94), grid)
check("pad_to_grid rend divisible par la grille", xg.shape[2] % 3 == 0 and xg.shape[3] % 3 == 0,
      f"47x94 -> {tuple(xg.shape[2:])}")

src = torch.randn(1, 4, 5, 7)
pad = pad_to_grid(src, (3, 3))
check("pad_to_grid : zéros uniquement à droite/en bas, données préservées",
      pad.shape == (1, 4, 6, 9) and torch.equal(pad[:, :, :5, :7], src)
      and float(pad[:, :, 5:, :].abs().sum()) == 0.0
      and float(pad[:, :, :, 7:].abs().sum()) == 0.0)

# halo : la bordure d'une tuile intérieure vient EXACTEMENT du voisin (pas de zéros).
# src (1,4,5,7) -> pad_to_grid (1,4,6,9), tuile (r=1,c=1) : intérieur lignes 2:4,
# colonnes 3:6 ; sa bordure HAUTE (ligne 0 du patch, (C, win_h, win_w) -> [:, 0, :])
# = ligne 1 de src, colonnes 2:7.
p_h, sh = to_patches(pad_to_grid(src, (3, 3)), (3, 3), halo=1)
check("halo : bordure d'une tuile intérieure == pixels EXACTS du voisin",
      torch.equal(p_h[4][:, 0, :], src[0, :, 1, 2:7]))

# ---------------------------------------------------------------------------
print("[2] forme de sortie préservée (divisible ET non divisible)")
for shape in [(2, 8, 48, 96), (1, 8, 47, 94), (3, 8, 13, 13)]:
    xs = torch.randn(*shape)
    moe = PatchMoE2D(channels=8, **MOE_V3_CS)
    moe.eval()
    with torch.no_grad():
        ys = moe(xs)
    check(f"forward {shape} -> conserve la forme", ys.shape == xs.shape,
          f"got {tuple(ys.shape)}")
    check(f"forward {shape} : résiduel = x + moe(x), pas x seul",
          not torch.allclose(ys, xs))

# ---------------------------------------------------------------------------
print("[3] NON-RÉGRESSION : 1 expert + top-1 == bloc plein-volume (eval)")
for shape, gr in [((2, 8, 48, 96), (3, 3)), ((1, 8, 50, 100), (2, 5)), ((1, 8, 47, 94), (3, 3))]:
    torch.manual_seed(0)
    moe = PatchMoE2D(channels=shape[1], n_experts=1, top_k=1, grid=gr,
                     kernel_size=3, balance='none', noise_std=0.0)
    moe.eval()
    e = moe.experts[0]
    # Des running stats BN non triviales, pour que le test ne passe pas par
    # l'identité par accident.
    with torch.no_grad():
        e[1].running_mean.copy_(torch.randn_like(e[1].running_mean))
        e[1].running_var.copy_(torch.rand_like(e[1].running_var) + 0.5)
    xs = torch.randn(*shape)
    with torch.no_grad():
        y_moe = moe(xs)
        y_ref = xs + e(xs)
    d = float((y_moe - y_ref).abs().max())
    check(f"MoE 1-expert/top-1 == x + bloc(x) plein-volume {shape} grille {gr}",
          torch.allclose(y_moe, y_ref, atol=1e-5), f"max|Δ|={d:.2e}")

# ---------------------------------------------------------------------------
print("[4] experts identiques ⇒ sortie indépendante du routage (top-2/4, bruit actif)")
torch.manual_seed(7)
chan = 8
base = fpn_block_expert(chan)
base.eval()
with torch.no_grad():
    base[1].running_mean.copy_(torch.randn_like(base[1].running_mean))
    base[1].running_var.copy_(torch.rand_like(base[1].running_var) + 0.5)
moe4 = PatchMoE2D(channels=chan, n_experts=4, top_k=2, grid=(3, 3),
                  balance='switch', noise_std=1.0, noise_anneal=40)
with torch.no_grad():  # 4 copies exactes du même bloc
    for ex in moe4.experts:
        ex.load_state_dict(base.state_dict())
    moe4.eval()
    xs = torch.randn(2, chan, 48, 96)
    y_moe = moe4(xs)
    y_ref = xs + base(xs)
check("4 experts identiques top-2 == x + E(x)",
      torch.allclose(y_moe, y_ref, atol=1e-5),
      f"max|Δ|={float((y_moe - y_ref).abs().max()):.2e}")
w4, _, _, _ = moe4._route(xs)
check("poids top-k convexes (somme = 1 par patch)",
      torch.allclose(w4.sum(dim=-1), torch.ones(w4.shape[0]), atol=1e-6),
      f"min={float(w4.sum(-1).min()):.6f} max={float(w4.sum(-1).max()):.6f}")

# ---------------------------------------------------------------------------
print("[5] expert par défaut : structure copiable depuis head.fpn_convs.0")
head = UPerNetHead(in_channels=[16, 32, 64, 128], channels=8, num_classes=19)
ref_keys = set(head.fpn_convs[0].state_dict().keys())
exp_keys = set(fpn_block_expert(8).state_dict().keys())
check("mêmes clés de state_dict que fpn_convs[0]", ref_keys == exp_keys,
      f"sym diff = {sorted(ref_keys ^ exp_keys)}")
moe5 = PatchMoE2D(channels=8, n_experts=4)
err = moe5.experts[3].load_state_dict(head.fpn_convs[0].state_dict())
check("load_state_dict du bloc UPerNet dans un expert : strict, 0 manquante/inattendue",
      len(err.missing_keys) == 0 and len(err.unexpected_keys) == 0)

# ---------------------------------------------------------------------------
print("[6] aux_loss + diagnostics de routage")
torch.manual_seed(1)
moe6 = PatchMoE2D(channels=8, **MOE_V3_CS)
moe6.train()
xs = torch.randn(2, 8, 48, 96)
y6 = moe6(xs)
st = moe6.last_stats
check("aux_loss scalaire fini, requires_grad", moe6.aux_loss.ndim == 0
      and torch.isfinite(moe6.aux_loss) and moe6.aux_loss.requires_grad,
      f"aux={float(moe6.aux_loss):.4g}")
check("defaults = recette V3 BRATS", dict(n_experts=moe6.n_experts, top_k=moe6.top_k,
      grid=moe6.grid, balance=moe6.balance, balance_weight=moe6.balance_weight,
      noise_std0=moe6.noise_std0, noise_anneal=moe6.noise_anneal)
      == dict(n_experts=4, top_k=2, grid=(3, 3), balance='switch',
              balance_weight=0.001, noise_std0=1.0, noise_anneal=40))
check("stats : les DEUX entropies loggées + clean", all(k in st for k in
      ('frac', 'entropy_norm', 'entropy_token', 'entropy_token_clean',
       'top_expert_share', 'n_tokens')))
check("stats bornées", 0.0 <= st['entropy_norm'] <= 1.0 + 1e-6
      and 0.0 <= st['entropy_token'] <= 1.0 + 1e-6
      and 0.0 <= st['entropy_token_clean'] <= 1.0 + 1e-6
      and 0.0 < st['top_expert_share'] <= 1.0,
      f"H_norm={st['entropy_norm']:.3f} H_tok={st['entropy_token']:.3f} "
      f"H_clean={st['entropy_token_clean']:.3f} part={st['top_expert_share']:.3f}")
check("n_tokens = B x grille", st['n_tokens'] == 2 * 9)
# frac = part des patchs traités par CHAQUE expert : avec top_k=2, chaque patch
# est compté pour ses 2 experts => frac somme à top_k (comportement BRATS fidèle).
check("frac sommé à top_k (tous les patchs routés)",
      abs(float(st['frac'].sum()) - 2.0) < 1e-5, f"sum={float(st['frac'].sum()):.6f}")

moe6b = PatchMoE2D(channels=8, n_experts=4, balance='none')
moe6b.train()
moe6b(xs)
check("balance='none' ⇒ aux_loss == 0 exact", float(moe6b.aux_loss) == 0.0)

# ---------------------------------------------------------------------------
print("[7] set_epoch : recuit linéaire déduit (safe reprise)")
moe7 = PatchMoE2D(channels=8, noise_std=1.0, noise_anneal=40)
moe7.set_epoch(0)
a = moe7.noise_std
moe7.set_epoch(20)
b = moe7.noise_std
moe7.set_epoch(40)
c = moe7.noise_std
moe7.set_epoch(30)  # reprise APRÈS un appel plus lointain : ne doit pas accumuler
moe7.set_epoch(10)
d = moe7.noise_std
check("recuit 1.0 → 0.5 → 0.0 puis redéduit à 0.75",
      abs(a - 1.0) < 1e-9 and abs(b - 0.5) < 1e-9 and abs(c) < 1e-9 and abs(d - 0.75) < 1e-9,
      f"{a:.3f} {b:.3f} {c:.3f} {d:.3f}")

# ---------------------------------------------------------------------------
print("[8] bruit de Shazeer actif en train seulement")
torch.manual_seed(3)
moe8 = PatchMoE2D(channels=8, **MOE_V3_CS)
xs = torch.randn(1, 8, 48, 96)
moe8.train()
_, _, p_a, _ = moe8._route(xs)
_, _, p_b, _ = moe8._route(xs)
check("train + bruit : deux routages diffèrent", not torch.equal(p_a, p_b))
moe8.eval()
_, _, p_c, pd_ = moe8._route(xs)
_, _, p_e, _ = moe8._route(xs)
check("eval : routage déterministe", torch.equal(p_c, p_e))
check("probas clean == probas en eval (pas de bruit)", torch.equal(p_c, pd_))

# ---------------------------------------------------------------------------
print("[9] gradients : backward fini, gate + experts sélectionnés mis à jour")
torch.manual_seed(5)
moe9 = PatchMoE2D(channels=8, **MOE_V3_CS)
moe9.train()
xs = torch.randn(2, 8, 48, 96, requires_grad=True)
y9 = moe9(xs)
(y9.square().mean() + moe9.aux_loss).backward()
check("grad entrée fini", xs.grad is not None and bool(torch.isfinite(xs.grad).all()))
g_gate = [p.grad for p in moe9.gate.parameters()]
check("grad gate présent et fini", all(g is not None for g in g_gate)
      and all(bool(torch.isfinite(g).all()) for g in g_gate))
touched = [e for e in moe9.experts
           if any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in e.parameters())]
check("au moins un expert reçoit du gradient", len(touched) >= 1, f"{len(touched)}/4")

# ---------------------------------------------------------------------------
print("[10] autocast BF16 (règle projet)")
try:
    moe10 = PatchMoE2D(channels=8, **MOE_V3_CS)
    moe10.train()
    xs = torch.randn(2, 8, 48, 96)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y10 = moe10(xs)
        y10.square().mean().backward()
    check("forward/backward BF16 CPU sans erreur, sortie finie",
          bool(torch.isfinite(y10.float()).all()), f"dtype={y10.dtype}")
except (RuntimeError, NotImplementedError) as exc:
    check("forward/backward BF16 CPU sans erreur, sortie finie", False, f"exc={exc}")

# ---------------------------------------------------------------------------
print("[11] point de fonctionnement UPerNet : C=512, stride 4")
moe11 = PatchMoE2D(channels=512, **MOE_V3_CS)
moe11.eval()
xs = torch.randn(1, 512, 64, 128)
with torch.no_grad():
    y11 = moe11(xs)
check("forward (1,512,64,128) grille 3x3 non divisible : forme conservée, fini",
      y11.shape == xs.shape and bool(torch.isfinite(y11).all()),
      f"n_tokens={moe11.last_stats['n_tokens']}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
n = sum(results)
print(f"RESULT: {n}/{len(results)} checks passed")
sys.exit(0 if n == len(results) else 1)
