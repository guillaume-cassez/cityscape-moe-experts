"""Blob loss (Kofler et al., IPMI 2023, arXiv:2205.08209) — régime MULTI-CLASSE EXCLUSIF.

Port City_Scape du bras G de BRATS (`RegionBlobLoss`, `src/training/blob_loss_regions.py`).
BRATS tourne en *region-based* (cibles WT/TC/ET chevauchantes, sigmoid) ; Cityscapes est un
problème multi-classe EXCLUSIF (target = indices de classe, softmax sur C canaux). La formule
de Kofler s'y transpose à l'identique (eq. 1 : le domaine d'une instance est l'image entière
PRIVÉE des autres instances de la MÊME classe), et l'algèbre scatter_add du port BRATS aussi —
seul le régime de probabilités change (softmax au lieu de sigmoid).

Ce que le terme corrige
-----------------------
Une loss voxel-wise (CE/Dice) pondère chaque instance par son volume : les petites instances
(classes rares, objets lointains) sont rationnellement sacrifiées. Le terme blob donne à chaque
instance GT le MÊME poids, quel que soit son nombre de pixels. Sur BRATS, ce bras seul était
derrière le baseline (−0.00376 CV) mais son `fpn_convs[0]` est un des quatre spécialistes qui
initialisent le MoE-V3 (bras GAGNANT, +0.00566) : c'est l'usage prévu ici (plan
`src/moe/PLAN_TRANSFERT_BRATS_V3.md` §2.1, décision Guillaume 2026-09-17 — miroir exact de
B/D/K/G de BRATS).

Deux chemins d'exécution — MÊME algèbre, parité verrouillée par les tests
-------------------------------------------------------------------------
1. **Pré-calculé (rapide, utilisé par l'entraînement G)** : les labels d'instances sont
   paquetés par image (`src/losses/blob_lab.py`, fichiers `.blob.npz` côté dataset, clé
   `blob_glob/blob_counts/blob_csr` du batch — motif identique au SDT `load_sdt`) et arrivés
   dans `blob=`. Le forward ne fait plus que des noyaux GPU élémentaires : le CC CPU et les
   26 petits transferts/syncs par classe disparaissent de la boucle (MESURE 2026-09-17 :
   chemin naïf = +870 s/époque full-res, inacceptable ; ce chemin ≈ +1-2 %).
2. **Fallback (calculé à la volée)** : sans `blob=`, le paquetage est recomputé par
   `compute_blob_lab` sur le target — c'est la référence de correction (tests, harnais
   d'éval, tout consumer non-dataloader). Mêmes fonctions de paquetage ⇒ parité structurelle.

Algèbre (identique aux deux chemins — réécriture EXACTE de l'eq. 1, pas une approximation).
Pour l'instance n de la classe c, p = softmax(logits)[c], t = indicatrice GT,
Ω_n = {pixels valides, non-instance de c} ∪ {pixels de n} :

    S0  = Σ p sur {valides, hors instances de c}   (seau 0 du scatter)
    s_n = Σ p sur l'instance n                    (seau glob[c,n])
    c_n = |instance n|                            (counts pré-calculés / bincount)

    soft-dice binaire sur Ω_n = 1 − (2·s_n + eps) / (S0 + s_n + c_n + eps)

car Σ_{Ω_n} p = S0 + s_n. Pixels ignorés (255) : seau poubelle, jamais comptés comme vrais
négatifs (sémantique `valid` de BRATS). Moyenne à DEUX NIVEAUX fidèle à RegionBlobLoss :
instances → classe → image → batch.

Coût CC (fallback seul — MESURÉ sur la Tour le 2026-09-17, GT réel full-res) : cc3d 2D =
22.4 ms/image (16 classes, 118 instances ; scipy : 59.7 ms).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.losses.blob_lab import compute_blob_lab


class BlobLoss(nn.Module):
    """Terme blob de Kofler pour segmentation multi-classe exclusive (softmax).

    Parameters
    ----------
    eps : float
        Lissage du soft dice. 1.0 = convention nnU-Net/BRATS, valeur par défaut gardée.
    ignore_index : int
        Label ignoré (255 = void Cityscapes). Exclu du fond ET des instances.
    min_blob_pixels : int
        Instances GT plus petites que ce seuil ignorées. 0 = désactivé (défaut, comme BRATS :
        ce sont justement les petites qu'on cherche à rattraper). Paramètre d'ablation.
    """

    def __init__(self, eps: float = 1.0, ignore_index: int = 255, min_blob_pixels: int = 0,
                 splits: int = 64):
        super().__init__()
        self.eps = float(eps)
        self.ignore_index = int(ignore_index)
        self.min_blob_pixels = int(min_blob_pixels)
        self.splits = int(splits)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, blob: dict | None = None) -> torch.Tensor:
        """pred : (B, C, H, W) logits ; target : (B, H, W) indices longs.

        `blob` (optionnel, chemin rapide) : dict de tenseurs issus du dataloader —
        `glob` (B,H,W) entier, `counts` (B,P) int64, `csr` (B,C+1) entier (peut rester CPU).
        Scalaire DIFFÉRENTIABLE ; batch sans instance GT → 0.0 différentiable (graphe connecté).

        Toutes les classes d'une image sont traitées en UN scatter_add vectorisé (espace de
        seaux C×(N+2) : seau d'instance = identifiant global off+j ≤ N+1 ; seau S0 de la
        classe c = base_c = c·(N+2) > N+1 ⇒ jamais de collision ; seau poubelle = base_c+N+1).
        MESURE 2026-09-17 (Tour, GPU idle, batch 2 full-res, GT réel) : la version à une
        boucle Python par classe coûtait 96 ms/iter (≈130 lancements de noyaux) — la cause
        n'était PAS la contention atomique (26 scatters pleins = 22.8 ms mesurés).
        """
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"pred {tuple(pred.shape)} (B,C,H,W) / target {tuple(target.shape)} (B,H,W) attendus")
        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError(f"pred {tuple(pred.shape)} incompatible avec target {tuple(target.shape)}")
        probs = torch.softmax(pred.float(), dim=1)  # float32 : scatter_add stable en bf16 sinon
        b_sz, n_cls = probs.shape[0], probs.shape[1]

        per_element = []
        for b in range(b_sz):
            if blob is not None:
                glob_b = blob["glob"][b].to(device=probs.device, dtype=torch.int64)
                csr_b = [int(x) for x in blob["csr"][b].reshape(-1).tolist()]
                counts_b = blob["counts"][b]
                if counts_b.device != probs.device:
                    counts_b = counts_b.to(probs.device)
            else:
                t_np = target[b].detach().cpu().numpy()
                glob_np, csr_np, counts_np = compute_blob_lab(t_np, n_cls, self.ignore_index)
                csr_b = csr_np.tolist()
                glob_b = torch.from_numpy(glob_np).to(device=probs.device, dtype=torch.int64)
                counts_b = torch.from_numpy(counts_np).to(probs.device)
            n_total = csr_b[n_cls]
            if n_total == 0:
                continue  # aucune instance GT dans cette image : rien à moyenner

            pb = probs[b]                              # (C,H,W)
            tb = target[b]                             # (H,W) long
            stride = n_total + 2
            classes = torch.arange(n_cls, device=probs.device)
            valid_b = (tb != self.ignore_index).unsqueeze(-1)                       # (H,W,1) bool
            mask_all = tb.unsqueeze(-1) == classes.view(1, 1, n_cls)                # (H,W,C) bool
            base = (classes * stride).view(1, 1, n_cls)                             # base_c
            # seau = base_c + glob (instance) si pixel de c valide ; base_c (S0) si valide
            # hors c ; base_c + N+1 (poubelle) si ignoré. Layout (H,W,C) : le scatter
            # arborescent découpe alors par BLOCS SPATIAUX (toutes classes présentes dans
            # chaque bloc ⇒ les instances géantes sont bien éclatées entre lignes).
            lab_all = torch.where(mask_all, glob_b.unsqueeze(-1), torch.zeros_like(base))
            lab_all = torch.where(valid_b, lab_all + base, base + n_total + 1)
            # Scatter ARBORESCENT : la contention atomique vient des instances géantes
            # (road ≈ 500 K px → 1 seul seau ; mesuré 40.7 ms/img en scatter unique le
            # 2026-09-17). SPLITS blocs, chacun avec son propre espace de seaux
            # (zéros (SPLITS, C·stride)), puis réduction — collisions par seau ÷ SPLITS,
            # algèbre exacte à l'ordre de sommation float près (≈1e-6 relatif).
            # Repli SPLITS=1 si H·W non divisible (petits tests CPU).
            hw = lab_all.shape[0] * lab_all.shape[1]
            splits = self.splits if (hw % self.splits == 0) else 1
            flat = n_cls * stride
            pb_hwc = pb.permute(1, 2, 0)               # (H,W,C) — vue; reshape copie 1×
            if splits > 1:
                sums2d = torch.zeros(splits, flat, dtype=probs.dtype, device=probs.device)
                sums2d.scatter_add_(1, lab_all.reshape(splits, -1),
                                    pb_hwc.reshape(splits, -1))
                sums = sums2d.sum(0).view(n_cls, stride)
            else:
                sums = torch.zeros(flat, dtype=probs.dtype, device=probs.device)
                sums.scatter_add_(0, lab_all.reshape(-1), pb_hwc.reshape(-1))
                sums = sums.view(n_cls, stride)

            per_class = []
            for c in range(n_cls):
                off, n_c = csr_b[c], csr_b[c + 1] - csr_b[c]
                if n_c == 0:
                    continue
                s0 = sums[c, 0]
                s_n = sums[c, off + 1 : off + n_c + 1]
                c_n = counts_b[off + 1 : off + n_c + 1].to(probs.dtype)
                if self.min_blob_pixels > 0:
                    keep = c_n >= self.min_blob_pixels
                    if not bool(keep.any()):
                        continue
                    s_n, c_n = s_n[keep], c_n[keep]
                # Σ_{Ω_n} p = S0 + s_n ; Σ p·t = s_n ; Σ t = c_n
                dice = 1.0 - (2.0 * s_n + self.eps) / (s0 + s_n + c_n + self.eps)
                per_class.append(dice.mean())
            if per_class:
                per_element.append(torch.stack(per_class).mean())
        if not per_element:
            return probs.sum() * 0.0
        return torch.stack(per_element).mean()
