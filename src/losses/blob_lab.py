"""Pré-calcul des labels d'instances « blob » (Kofler) — chemin rapide du BlobLoss.

Pourquoi ce module existe (MESURE du 2026-09-17, Tour) : calculer les composantes connexes
DANS la boucle d'entraînement coûte ≈ +870 s/époque full-res (422 ms/iter standalone dont
≈45 ms de CC — le reste = syncs/transferts par classe), soit ≈57 h/seed au lieu de 29 h.
La solution reprend le motif SDT du projet (`scripts/precompute_sdt.py`, `load_sdt`) : les
labels d'instances ne dépendent QUE du champ de cibles, dont la seule augmentation
géométrique full_res est le flip horizontal (label-préservant). On pré-calcule donc une
fois par image un paquetage compact, chargé par les workers du dataloader (parallèle,
page-cache) et retourné dans le batch — le BlobLoss n'a plus qu'un chemin GPU élémentaire.

Format paqueté (une image, cibles en trainIds 0..C-1, ignore=255) :
- `glob` (H, W) int16 : 0 = pixel SANS instance (fond, autre classe) ; pour la classe c,
  son instance j (1..n_c, ordre cc3d) porte l'identifiant GLOBAL csr[c] + j ; les pixels
  ignorés (255) portent N+1 (seau poubelle, jamais compté comme vrai négatif).
- `csr` (C+1,) int64 : csr[c] = somme des n des classes < c ; csr[C] = N (total instances).
  n_c = csr[c+1] - csr[c].
- `counts` (N+2,) int64 : counts[i] = nombre de pixels du seau global i (1..N) ;
  counts[N+1] = pixels ignorés. Le seau 0 n'a pas de sens algébrique (S0 vient du scatter).

Algèbre côté loss (IDENTIQUE au chemin fallback — les tests de parité le verrouillent) :
pour l'instance n (global g) de la classe c, avec p = softmax(logits)[c] :
    S0  = Σ p sur {valid, pas instance de c}      (scatter seau 0)
    s_n = Σ p sur {glob == g}                     (scatter seau g)
    c_n = counts[g]
    dice = 1 − (2·s_n + eps) / (S0 + s_n + c_n + eps)
moyenne instances → classe → image → batch (deux niveaux, fidèle à RegionBlobLoss BRATS).

Pré-calcul (Tour, ≈6-10 min pour 2975 images, écriture fsync) :
    /home/ser/brats-venv/bin/python scripts/precompute_blob_lab.py
Fichiers : <label>_labelIds.blob.npz (à côté du png, comme .sdt.npy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # cc3d ≈ 3× plus rapide que scipy (mesuré 2026-09-17 : 22.4 vs 59.7 ms/image réelle)
    import cc3d

    def connected_components_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
        out = cc3d.connected_components(mask.astype(np.uint8), connectivity=8)
        return out, int(out.max())
except ImportError:  # pragma: no cover - repli sûr hors Tour
    from scipy.ndimage import label as _cc_label

    def connected_components_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
        out, n = _cc_label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        return out, int(n)


def compute_blob_lab(target_np: np.ndarray, n_classes: int, ignore_index: int = 255):
    """Cibles (H,W) en trainIds → (glob int16, csr int64 (C+1), counts int64 (N+2)).

    Déterministe à l'ordre de labellisation cc3d près (fixe pour une version donnée) ; les
    deux chemins du BlobLoss consomment le MÊME paquetage, donc la parité est structurelle.
    """
    t = np.asarray(target_np)
    if t.ndim != 2:
        raise ValueError(f"cibles (H,W) attendues, reçu {t.shape}")
    h, w = t.shape
    invalid = t == ignore_index
    glob = np.zeros((h, w), dtype=np.int64)
    csr = np.zeros(n_classes + 1, dtype=np.int64)
    off = 0
    for c in range(n_classes):
        gt = t == c
        if not gt.any():
            csr[c] = off
            continue
        cc, n = connected_components_2d(gt)
        if off + n + 1 > 32766:
            raise OverflowError(
                "trop d'instances pour int16 (%d) — augmenter le dtype de glob" % (off + n))
        glob[gt] = cc[gt] + off          # instance j ∈ [1,n] → global off+j
        off += n
        csr[c] = off - n                 # csr[c] = offset de la classe c
    csr[n_classes] = off
    n_total = off
    glob[invalid] = n_total + 1          # seau poubelle des pixels ignorés
    counts = np.bincount(glob.reshape(-1), minlength=n_total + 2).astype(np.int64)
    return glob.astype(np.int16), csr, counts[: n_total + 2]


def blob_lab_path(label_path: str | Path) -> Path:
    """Chemin du paquetage pour un png de label (même convention que .sdt.npy)."""
    return Path(str(label_path)).with_suffix(".blob.npz")


def save_blob_lab(path: str | Path, glob: np.ndarray, csr: np.ndarray, counts: np.ndarray) -> None:
    """Écriture durable (tmp + rename + fsync du fichier ET du dossier — règle poste)."""
    path = Path(path)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, glob=glob, csr=csr, counts=counts)
        fh.flush()
        import os
        os.fsync(fh.fileno())
    tmp.replace(path)
    import os
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def load_blob_lab(path: str | Path, pad_counts_to: int):
    """Charge le paquetage ; `counts` est rembourré à taille FIXE pad_counts_to pour que le
    collate par défaut du DataLoader puisse empiler un batch (N variable par image)."""
    with np.load(Path(path), allow_pickle=False) as z:
        glob, csr, counts = z["glob"], z["csr"], z["counts"]
    if counts.shape[0] > pad_counts_to:
        raise ValueError("counts %d > pad %d — augmenter blob_counts_pad" % (counts.shape[0], pad_counts_to))
    padded = np.zeros(pad_counts_to, dtype=np.int64)
    padded[: counts.shape[0]] = counts
    return glob, csr.astype(np.int64), padded
