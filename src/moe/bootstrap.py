"""P3.09 — IC 95 % bootstrap APPARIÉ sur images + collecte des confusions par-image.

Module MUTUALISÉ pour TOUS les bras du paper 3 (plan §4, point 3 de la liste des
bras) et pour la late-fusion (DESIGN.md §9, « reste à implémenter ») : un seul
endroit où vivent la convention mIoU, le masquage du void et le protocole
bootstrap, pour que les chiffres de tous les bras soient strictement comparables.

Protocole FIGÉ (DESIGN.md §9 · PLAN_TRANSFERT_BRATS_V3.md §4) :
  1. Une matrice de confusion 19×19 PAR IMAGE (`per_image_cm`, void-masking
     EXACTEMENT celui de `SegmentationMetrics`), dans un ordre d'images partagé
     entre les bras — c'est ce qui rend l'appariement possible.
  2. R réplicats ; à chaque réplicat, rééchantillonnage des N images AVEC remise,
     **même index-set pour TOUS les bras** (`paired_bootstrap` tire les indices
     une fois par réplicat, avant de boucler sur les bras).
  3. Le mIoU est **dataset-level** : on **ré-agrège** la confusion sur les images
     rééchantillonnées puis on **recalcule** le mIoU. On ne moyenne JAMAIS des
     mIoU par-image. Les seeds d'un même bras sont **moyennés dans le réplicat**
     (la statistique publiée est le mIoU dataset moyen sur seeds).
  4. Contraste Δ = métrique(x) − métrique(y) par réplicat → IC **percentile**
     [2.5, 97.5] ; significatif ⟺ 0 ∉ IC ; p bilatéral = 2·min(P(Δ≤0), P(Δ≥0)) ;
     correction de **Holm** sur la famille de paires déclarée.
  5. Pour une métrique moyenne par image (Boundary F1), `paired_bootstrap_values`
     applique le même appariement sur les valeurs par-image (S, N).

La convention mIoU d'ici est **bit-identique** à la routine officielle
cityscapesScripts (prouvé par `tests/test_official_miou.py`) → cityscapesscripts
n'est PAS requis pour le bootstrap. Numpy/torch uniquement.

Provenance : ce module factorise SANS CHANGER l'algorithmique les fonctions de
`scripts/bootstrap_miou.py` (test statistique du SPEC, décidé 2026-06-18) et de
`scripts/dump_perimage_cm.py` — ces deux scripts l'importent désormais ;
l'identité de sortie est vérifiée sur données réelles (les IC des runs paper 1
existent déjà dans `results/bootstrap_consensus_*.json`).
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np

IGNORE = 255
NC = 19

# Fichier d'ordre partagé écrit par scripts/dump_perimage_cm.py (alignement des bras).
VAL_ORDER_FILE = "val_image_order.json"

# Réglages par défaut du test statistique figé (scripts/bootstrap_miou.py, SPEC 2026-06-18).
DEFAULT_B = 10000
DEFAULT_BOOTSTRAP_SEED = 20260618


# --------------------------------------------------------------------------- #
# Confusions par-image + convention mIoU (bit-exacte vs cityscapesScripts)
# --------------------------------------------------------------------------- #

def per_image_cm(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Matrice de confusion 19×19 d'UNE image, cm[gt, pred], int64.

    Void-masking EXACTEMENT celui de SegmentationMetrics (gt ∈ [0,NC) seulement) :
    la somme des cm par-image d'un run reproduit bit-à-bit la confusion dataset et
    donc le mIoU officiel (auto-vérifié par scripts/dump_perimage_cm.py).
    """
    k = (gt != IGNORE) & (gt >= 0) & (gt < NC)
    idx = NC * gt[k].astype(np.int64) + pred[k].astype(np.int64)
    return np.bincount(idx, minlength=NC * NC).reshape(NC, NC).astype(np.int64)


def ious_from_cm(cm: np.ndarray) -> np.ndarray:
    """IoU par classe (19,) depuis une confusion agrégée ; NaN pour union vide."""
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim == 3:                      # (N, NC, NC) : ré-agréger d'abord (dataset-level)
        cm = cm.sum(axis=0)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    return iou


def miou_from_cm(cm: np.ndarray) -> float:
    """mIoU dataset-level (moyenne des IoU des classes d'union non vide)."""
    iou = ious_from_cm(cm)
    valid = ~np.isnan(iou)
    return float(iou[valid].mean()) if valid.any() else 0.0


# --------------------------------------------------------------------------- #
# Chargement / collecte des confusions par-image
# --------------------------------------------------------------------------- #

def cm_path(cm_dir, arm: str, seed: int, suffix: str = "__epoch_160.cm.npy") -> Path:
    """Convention de nommage partagée (scripts/dump_perimage_cm.py)."""
    return Path(cm_dir) / f"{arm}_seed{seed}{suffix}"


def load_arm_stack(cm_dir, arm: str, seeds, suffix: str = "__epoch_160.cm.npy") -> np.ndarray:
    """(S, N, 19, 19) empilé sur les seeds d'un bras. FileNotFoundError si absent."""
    arrs = []
    for s in seeds:
        p = cm_path(cm_dir, arm, s, suffix)
        if not p.exists():
            raise FileNotFoundError(p)
        arrs.append(np.load(p))
    n = arrs[0].shape[0]
    for a in arrs:
        if a.shape != (n, NC, NC):
            raise ValueError(f"{arm}: shape {a.shape} != {(n, NC, NC)}")
    return np.stack(arrs, 0)


def load_val_order(cm_dir) -> list:
    p = Path(cm_dir) / VAL_ORDER_FILE
    return json.loads(p.read_text()) if p.exists() else []


def collect_per_image_cms(predict_fn, gt_fn, indices, keys=None, log=None) -> tuple:
    """Collecte GÉNÉRIQUE pour un bras quelconque (harness P3.10).

    Args:
        predict_fn(i) -> pred (H,W) entier OU logits (C,H,W) (tensor/numpy, argmax pris) ;
        gt_fn(i)      -> label (H,W) entier ;
        indices       -> itérable d'indices d'images, dans l'ordre du holdout partagé ;
        keys          -> clés d'images (alignement inter-bras), défaut idx{i}.
    Returns: (cms (N,19,19) int64, keys list).
    """
    indices = list(indices)
    cms, out_keys = [], []
    for j, i in enumerate(indices):
        pred = predict_fn(i)
        if hasattr(pred, "dim") and pred.dim() == 3:      # logits (C,H,W)
            pred = pred.argmax(0)
        pred = np.asarray(pred.detach().cpu() if hasattr(pred, "detach") else pred)
        gt = np.asarray(gt_fn(i))
        cms.append(per_image_cm(pred.astype(np.int64), gt.astype(np.int64)))
        out_keys.append(str(keys[j]) if keys is not None else f"idx{i}")
        if log is not None and (j + 1) % 50 == 0:
            log(f"    cm par-image : {j + 1}/{len(indices)}")
    return np.stack(cms, 0), out_keys


def save_arm_cms(arr: np.ndarray, out_dir, arm: str, seed: int,
                 suffix: str = "__epoch_160.cm.npy") -> Path:
    """Écriture durable (tmp + os.replace + fsync, règle globale JARVIS)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{arm}_seed{seed}{suffix}"
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# Bootstrap apparié — mIoU dataset-level (confusions) et valeurs par-image (BF1)
# --------------------------------------------------------------------------- #

def seedmean_miou(stack_sncc: np.ndarray, idx: np.ndarray) -> float:
    """mIoU dataset moyen sur seeds, sur le sous-échantillon d'images `idx`.

    IDENTIQUE à scripts/bootstrap_miou.py (ré-agrégation entière de la confusion
    par seed puis mIoU, moyenne des mIoU sur seeds — jamais l'inverse).
    """
    sub = stack_sncc[:, idx].sum(axis=1)  # (S, 19, 19)
    return float(np.mean([miou_from_cm(sub[s]) for s in range(sub.shape[0])]))


def holm(pvals: dict) -> dict:
    """Correction de Holm sur une famille de p (même convention qu'aggregate_official)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, run, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, pvals[k] * (m - i))
        out[k] = float(min(run, 1.0))
    return out


def _bootstrap_indices(n: int, B: int, rng_seed: int) -> list:
    """LES index-set partagés : tirés UNE fois, appliqués à tous les bras (appariement)."""
    rng = np.random.default_rng(rng_seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def _finalize(point: dict, dist: dict, pairs, n: int, B: int, statistic: str) -> dict:
    out = {
        "test": f"paired image-bootstrap ({statistic}), two-sided",
        "n_images": int(n), "B": int(B),
        "arms": list(point),
        "point": point,
        "ci95": {a: [float(np.percentile(dist[a], 2.5)),
                     float(np.percentile(dist[a], 97.5))] for a in dist},
        "pairwise": {},
    }
    raw = {}
    for x, y in pairs:
        d = dist[x] - dist[y]
        p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        raw[f"{x}_vs_{y}"] = p
        out["pairwise"][f"{x}_vs_{y}"] = {
            "delta": float(point[x] - point[y]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(p, 1.0)),
            "significant": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
        }
    for k, v in holm(raw).items():
        out["pairwise"][k]["p_holm"] = v
    return out


def paired_bootstrap(stacks: dict, B: int = DEFAULT_B, rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
                     pairs=None) -> dict:
    """Bootstrap apparié du mIoU dataset-level pour plusieurs bras.

    Args:
        stacks: {bras: (S, N, 19, 19)} — S seeds (S peut différer entre bras : les
            seeds sont moyennés DANS chaque réplicat, bras par bras) ; les N images
            sont dans le MÊME ordre pour tous les bras (appariement).
        B: réplicats. rng_seed: graine du tirage (reproductible).
        pairs: liste de (x, y) à contraster ; défaut toutes les combinaisons.
    Returns: dict JSON-sérialisable (point, ci95 marginales, pairwise Δ/IC/p/Holm).
    """
    stacks = {a: np.asarray(s) for a, s in stacks.items()}
    n = next(iter(stacks.values())).shape[1]
    for a, s in stacks.items():
        if s.ndim != 4 or s.shape[1] != n or s.shape[2:] != (NC, NC):
            raise ValueError(f"bras {a}: shape {s.shape} incompatible avec (S,{n},{NC},{NC})")
    arms = list(stacks)
    pairs = list(pairs) if pairs is not None else list(itertools.combinations(arms, 2))
    for x, y in pairs:
        if x not in stacks or y not in stacks:
            raise KeyError(f"paire ({x},{y}) : bras inconnu")

    full = np.arange(n)
    point = {a: seedmean_miou(stacks[a], full) for a in arms}
    boot_idx = _bootstrap_indices(n, B, rng_seed)
    dist = {a: np.array([seedmean_miou(stacks[a], idx) for idx in boot_idx]) for a in arms}
    return _finalize(point, dist, pairs, n, B, "dataset-level mIoU, seed-mean")


def paired_bootstrap_values(values: dict, B: int = DEFAULT_B, rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
                            pairs=None) -> dict:
    """Même appariement pour une statistique MOYENNE PAR IMAGE (Boundary F1).

    Args: values: {bras: (S, N)} valeurs par image (moyenne seeds dans le réplicat).
    """
    values = {a: np.asarray(v, dtype=np.float64) for a, v in values.items()}
    n = next(iter(values.values())).shape[1]
    for a, v in values.items():
        if v.ndim != 2 or v.shape[1] != n:
            raise ValueError(f"bras {a}: shape {v.shape} incompatible avec (S,{n})")
    arms = list(values)
    pairs = list(pairs) if pairs is not None else list(itertools.combinations(arms, 2))

    def stat(v, idx):
        return float(np.mean([v[s, idx].mean() for s in range(v.shape[0])]))

    full = np.arange(n)
    point = {a: stat(values[a], full) for a in arms}
    boot_idx = _bootstrap_indices(n, B, rng_seed)
    dist = {a: np.array([stat(values[a], idx) for idx in boot_idx]) for a in arms}
    return _finalize(point, dist, pairs, n, B, "mean-per-image value, seed-mean")


def format_report(out: dict, unit: float = 100.0) -> str:
    """Rendu texte d'une sortie paired_bootstrap (mêmes colonnes que bootstrap_miou.py)."""
    lines = [f"n_images={out['n_images']} B={out['B']} bras={out['arms']}"]
    for a in out["arms"]:
        lo, hi = out["ci95"][a]
        lines.append(f"  {a}: mIoU={out['point'][a]*unit:.2f}  CI95=[{lo*unit:.2f}, {hi*unit:.2f}]")
    for k, r in out["pairwise"].items():
        lo, hi = r["ci95"]
        sig = "*" if r.get("p_holm", r["p_two_sided"]) < 0.05 else " "
        lines.append(f"  {k}: Δ={r['delta']*unit:+.2f}  CI95=[{lo*unit:+.2f}, {hi*unit:+.2f}]  "
                     f"p={r['p_two_sided']:.3f}  Holm={r.get('p_holm', float('nan')):.3f} {sig}")
    return "\n".join(lines)
