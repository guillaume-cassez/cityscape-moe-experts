"""Pavage 2D avec halo : la géométrie du routage patch-wise, séparée du mélange.

Port 2D fidèle de `BRATS/src/models/moe_patch_tiling.py` (P3.07, plan
`PLAN_TRANSFERT_BRATS_V3.md` §3.2). Comme côté BRATS, ce module ne contient ni
expert, ni gate, ni loss — seulement le découpage d'une carte de caractéristiques
en tuiles et son recollage, qui se teste et se relit sans rien savoir du MoE.

LE HALO N'EST PAS UN DÉTAIL. Un patch extrait sans bordure verrait, sur chacun de
ses quatre côtés, le padding zéro au lieu de ses voisins : la convolution de
l'expert imprimerait alors la grille dans la segmentation. Avec `halo >= k // 2`
par convolution spatiale de l'expert, le résultat est identique à celui du même
bloc appliqué à la carte entière, aux statistiques de normalisation près — la
BatchNorm des experts UPerNet normalise sur l'étendue spatiale des tuiles
sélectionnées, donc ses statistiques d'entraînement restent LOCALES au lot de
patches (c'est intrinsèque au routage par patch, pas un défaut d'implémentation ;
en mode eval les running stats figées rendent l'égalité exacte, cf.
`tests/test_patch_moe2d.py`).
"""

import torch.nn.functional as F


def pad_to_grid(x, grid):
    """Complète à droite/en bas pour que chaque axe soit divisible par la grille."""
    pads = []
    for size, g in zip(x.shape[2:][::-1], grid[::-1]):
        rem = (-size) % g
        pads.extend([0, rem])
    if any(pads):
        x = F.pad(x, pads)
    return x


def to_patches(x, grid, halo):
    """(B,C,H,W) -> (B*N, C, ph+2h, pw+2h), avec halo pris chez les voisins.

    Renvoie aussi `(ph, pw)`, la taille utile (sans halo) d'une tuile.
    """
    b, c = x.shape[:2]
    gh, gw = grid
    ph, pw = (s // g for s, g in zip(x.shape[2:], grid))
    h = halo
    xp = F.pad(x, (h, h, h, h)) if h else x
    p = xp.unfold(2, ph + 2 * h, ph).unfold(3, pw + 2 * h, pw)
    p = p.permute(0, 2, 3, 1, 4, 5).contiguous()
    return p.view(b * gh * gw, c, ph + 2 * h, pw + 2 * h), (ph, pw)


def from_patches(y, b, grid, shape):
    """(B*N, C, ph, pw) -> (B, C, gh*ph, gw*pw). Inverse de `to_patches` sans halo."""
    gh, gw = grid
    ph, pw = shape
    c = y.shape[1]
    y = y.view(b, gh, gw, c, ph, pw).permute(0, 3, 1, 4, 2, 5)
    return y.reshape(b, c, gh * ph, gw * pw)
