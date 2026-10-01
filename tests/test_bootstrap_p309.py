"""Tests non-régression P3.09 — src/moe/bootstrap.py (IC bootstrap apparié, mutualisé).

Vérifient (CPU, numpy/torch uniquement, sans GPU ni données Cityscapes) :
  1. `per_image_cm` : void-masking EXACTEMENT celui de SegmentationMetrics, et la
     somme des cm par-image reproduit BIT-À-BIT la confusion dataset (donc le mIoU
     officiel de `compute()`) — la propriété dont dépend tout le bootstrap ;
  2. `miou_from_cm`/`ious_from_cm` : identiques aux implémentations historiques de
     scripts/dump_perimage_cm.py et scripts/bootstrap_miou.py (garde du refactor
     P3.09 : le code a été DÉPLACÉ, pas réécrit) ;
  3. `paired_bootstrap` : reproduit bit-à-bit l'algorithme de référence de
     scripts/bootstrap_miou.py pré-refactor (recopié ici) sur des stacks aléatoires ;
  4. sémantique statistique : bras parfaits vs bras décalés → Δ>0, IC exclut 0,
     significatif ; bras identiques → Δ=0, p=1 ; déterminisme (même rng_seed) ;
  5. `paired_bootstrap_values` : l'APPARIEMENT paie — Δ constant par image donne un
     IC du Δ réduit à un point excluant 0 alors que les IC marginaux se recouvrent
     très largement (l'exemple même du §9 DESIGN : sans appariement on conclurait
     à tort « non significatif ») ;
  6. `holm` : vecteur connu, monotonie, ≥ p bruts ;
  7. I/O : convention de nommage des .cm.npy, FileNotFoundError/ValueError,
     `collect_per_image_cms` (logits OU labels, keys, ordre).

Exécution (Tour ou toute machine avec torch+scipy) :
    python3 -m pytest tests/test_bootstrap_p309.py -q
"""

import itertools
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import SegmentationMetrics
from src.moe.bootstrap import (
    NC, IGNORE, per_image_cm, miou_from_cm, ious_from_cm, seedmean_miou,
    paired_bootstrap, paired_bootstrap_values, holm,
    load_arm_stack, cm_path, collect_per_image_cms, save_arm_cms,
)

RNG = np.random.default_rng(7)


def _scene(n_images=4, h=17, w=23, with_void=True):
    """Scène synthétique : gt avec classes rares + void, pred corrélé au gt."""
    gts, preds = [], []
    for _ in range(n_images):
        gt = RNG.integers(0, NC, size=(h, w))
        if with_void:
            gt[0:2, :] = IGNORE
            gt[:, -1] = IGNORE
        pred = gt.copy()
        flip = RNG.random((h, w)) < 0.25
        pred[flip] = RNG.integers(0, NC)[()]
        preds.append(np.where(gt == IGNORE, 0, pred))  # pred sous void : doit être ignoré
        gts.append(gt)
    return np.stack(preds), np.stack(gts)


# --------------------------------------------------------------------------- #
# 1. per_image_cm ≡ SegmentationMetrics (bit-exact)
# --------------------------------------------------------------------------- #

def test_per_image_cm_sums_to_segmentation_metrics_bit_exact():
    preds, gts = _scene()
    metrics = SegmentationMetrics(num_classes=NC)
    cms = []
    for p, g in zip(preds, gts):
        metrics.update(torch.from_numpy(p).unsqueeze(0), torch.from_numpy(g).unsqueeze(0))
        cms.append(per_image_cm(p, g))
    agg = np.stack(cms).sum(0)
    assert np.array_equal(agg, metrics.confusion_matrix)          # confusion bit-exacte
    official = metrics.compute()
    assert miou_from_cm(agg) == official["mIoU"]                  # mIoU bit-exact (==)
    ious = ious_from_cm(agg)
    for c, v in official["per_class_iou"].items():
        if np.isnan(ious[c]):
            assert v == 0.0                                       # classe absente : 0 officiel
        else:
            assert ious[c] == v


def test_per_image_cm_void_and_out_of_range_masked():
    gt = np.full((5, 5), IGNORE, dtype=np.int64)
    gt[0, 0] = 3
    gt[1, 1] = -7        # hors bornes négatif → masqué comme le void
    gt[2, 2] = NC + 4    # hors bornes haut → masqué
    pred = np.full((5, 5), 8, dtype=np.int64)
    cm = per_image_cm(pred, gt)
    assert cm.sum() == 1 and cm[3, 8] == 1


def test_miou_zero_cm_and_absent_classes():
    assert miou_from_cm(np.zeros((NC, NC), dtype=np.int64)) == 0.0
    cm = np.zeros((NC, NC), dtype=np.int64)
    cm[0, 0] = 10
    cm[1, 1] = 5
    cm[1, 2] = 5
    ious = ious_from_cm(cm)
    assert np.isnan(ious[5])                 # classe absente de gt ET pred → NaN (exclue)
    assert ious[2] == 0.0                    # présente en pred seulement → IoU 0 (officiel)
    assert miou_from_cm(cm) == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    # (N,19,19) : ré-agrégation dataset-level AVANT le mIoU (jamais moyenne par image)
    stacked = np.stack([cm, cm])
    assert miou_from_cm(stacked) == pytest.approx(miou_from_cm(cm))


# --------------------------------------------------------------------------- #
# 2. Garde du refactor : formules historiques de dump_perimage_cm.py (recopiées)
# --------------------------------------------------------------------------- #

def _legacy_dump_per_image_cm(pred, gt):
    k = (gt != 255) & (gt >= 0) & (gt < 19)
    idx = 19 * gt[k].astype(np.int64) + pred[k].astype(np.int64)
    return np.bincount(idx, minlength=19 * 19).reshape(19, 19).astype(np.int64)


def _legacy_miou_from_cm(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    valid = union > 0
    iou = np.divide(inter, union, out=np.zeros(19), where=valid)
    return float(iou[valid].mean()) if valid.any() else 0.0


def test_module_matches_legacy_dump_formulas():
    preds, gts = _scene(n_images=6)
    for p, g in zip(preds, gts):
        assert np.array_equal(per_image_cm(p, g), _legacy_dump_per_image_cm(p, g))
    agg = np.stack([per_image_cm(p, g) for p, g in zip(preds, gts)]).sum(0)
    assert miou_from_cm(agg) == _legacy_miou_from_cm(agg)


# --------------------------------------------------------------------------- #
# 3. paired_bootstrap ≡ scripts/bootstrap_miou.py pré-refactor (bit-à-bit)
# --------------------------------------------------------------------------- #

def _reference_bootstrap_miou(stacks, variants, B, rng_seed):
    """Algorithme EXACT de scripts/bootstrap_miou.py @ commit 55c8cd1 (pré-refactor)."""
    def _miou_one(cm):
        inter = np.diag(cm).astype(np.float64)
        union = cm.sum(0) + cm.sum(1) - inter
        valid = union > 0
        iou = np.divide(inter, union, out=np.zeros(19), where=valid)
        return float(iou[valid].mean()) if valid.any() else 0.0

    def _smiou(stack, idx):
        sub = stack[:, idx].sum(axis=1)                       # (S, 19, 19)
        return float(np.mean([_miou_one(sub[s]) for s in range(sub.shape[0])]))

    n = next(iter(stacks.values())).shape[1]
    rng = np.random.default_rng(rng_seed)
    full = np.arange(n)
    point = {v: _smiou(stacks[v], full) for v in variants}
    boot_idx = [rng.integers(0, n, n) for _ in range(B)]
    dist = {v: np.array([_smiou(stacks[v], idx) for idx in boot_idx]) for v in variants}

    out = {"pairwise": {}}
    rawp = {}
    for x, y in itertools.combinations(variants, 2):
        d = dist[x] - dist[y]
        p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        rawp[f"{x}_vs_{y}"] = float(min(p, 1.0))
        out["pairwise"][f"{x}_vs_{y}"] = {
            "delta_miou": float(point[x] - point[y]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(p, 1.0)),
        }
    order = sorted(rawp, key=lambda k: rawp[k])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, rawp[k] * (m - i))
        out["pairwise"][k]["p_holm"] = float(min(run, 1.0))
    out["point_miou"] = point
    out["miou_ci95"] = {v: [float(np.percentile(dist[v], 2.5)),
                            float(np.percentile(dist[v], 97.5))] for v in variants}
    return out


def _random_stack(s=2, n=40, seed=0):
    """Stack entier (S,N,19,19) : bruit de Poisson hors-diagonale + diagonale dominante
    (signal inégal par classe et par seed) — forme réelle des .cm.npy du projet."""
    g = np.random.default_rng(seed)
    stack = g.poisson(3.0, size=(s, n, NC, NC)).astype(np.int64)
    idx = np.arange(NC)
    for si in range(s):
        stack[si, :, idx, idx] += ((80 + 30 * idx) + 7 * si)[:, None]
    # un léger biais inter-seed sur quelques classes
    stack[1:, :, idx[:4], idx[:4]] += 11
    return stack


def test_paired_bootstrap_bit_identical_to_reference_script():
    variants = ["V1", "V2", "V3"]
    stacks = {v: _random_stack(seed=i) for i, v in enumerate(variants)}
    ref = _reference_bootstrap_miou(stacks, variants, B=300, rng_seed=20260618)
    got = paired_bootstrap(stacks, B=300, rng_seed=20260618)

    assert got["n_images"] == 40 and got["B"] == 300
    for v in variants:                                        # points ET IC marginaux
        assert got["point"][v] == ref["point_miou"][v]
        assert got["ci95"][v] == ref["miou_ci95"][v]
    for k in ref["pairwise"]:                                 # Δ, IC appariés, p, Holm
        assert got["pairwise"][k]["delta"] == ref["pairwise"][k]["delta_miou"]
        assert got["pairwise"][k]["ci95"] == ref["pairwise"][k]["ci95"]
        assert got["pairwise"][k]["p_two_sided"] == ref["pairwise"][k]["p_two_sided"]
        assert got["pairwise"][k]["p_holm"] == ref["pairwise"][k]["p_holm"]


def test_point_miou_equals_full_data_and_seedmean_explicit():
    stack = _random_stack(s=3, n=25, seed=9)
    out = paired_bootstrap({"X": stack}, B=50, rng_seed=1, pairs=[])
    manual = float(np.mean([miou_from_cm(stack[s]) for s in range(3)]))
    assert out["point"]["X"] == manual
    assert out["point"]["X"] == seedmean_miou(stack, np.arange(25))


def test_determinism_same_rng_seed():
    stacks = {"A": _random_stack(seed=1), "B": _random_stack(seed=2)}
    o1 = paired_bootstrap(stacks, B=120, rng_seed=99)
    o2 = paired_bootstrap(stacks, B=120, rng_seed=99)
    assert o1 == o2
    o3 = paired_bootstrap(stacks, B=120, rng_seed=42)
    assert o3["point"] == o1["point"]                         # le point ne dépend pas du tirage


# --------------------------------------------------------------------------- #
# 4. Sémantique statistique
# --------------------------------------------------------------------------- #

def _perfect_and_shifted(n=60, h=12, w=12):
    """X = gt parfait ; Y = gt décalé d'une classe sur toutes les images."""
    g = np.random.default_rng(3)
    gts = g.integers(0, NC, size=(n, h, w))
    cx = np.stack([per_image_cm(gt, gt) for gt in gts])
    cy = np.stack([per_image_cm((gt + 1) % NC, gt) for gt in gts])
    return {"X": cx[None], "Y": cy[None]}, gts


def test_clear_effect_is_significant():
    stacks, _ = _perfect_and_shifted()
    out = paired_bootstrap(stacks, B=200, rng_seed=5, pairs=[("X", "Y")])
    assert out["point"]["X"] == 1.0
    assert out["point"]["Y"] == 0.0
    d = out["pairwise"]["X_vs_Y"]
    assert d["delta"] == 1.0 and d["ci95"] == [1.0, 1.0]
    assert d["significant"] and d["p_two_sided"] == 0.0


def test_identical_arms_zero_delta_p_one():
    stacks, _ = _perfect_and_shifted(n=20)
    out = paired_bootstrap({"X": stacks["X"], "Xcopy": stacks["X"].copy()},
                           B=100, rng_seed=5, pairs=[("X", "Xcopy")])
    d = out["pairwise"]["X_vs_Xcopy"]
    assert d["delta"] == 0.0 and d["ci95"] == [0.0, 0.0]
    assert d["p_two_sided"] == 1.0 and d["p_holm"] == 1.0
    assert not d["significant"]


def test_paired_values_pairing_beats_marginal_overlap():
    """§9 DESIGN : Δ constant +0.02/image, difficulté très variable → marginales
    qui se recouvrent largement MAIS IC du Δ apparié = [+0.02,+0.02], 0 ∉ IC."""
    g = np.random.default_rng(11)
    n, B = 200, 2000
    vy = g.random((1, n))                     # difficulté par image, très variable
    vx = vy + 0.02
    out = paired_bootstrap_values({"gate": vx, "mean": vy}, B=B, rng_seed=7,
                                  pairs=[("gate", "mean")])
    lo_g, hi_g = out["ci95"]["gate"]
    lo_m, hi_m = out["ci95"]["mean"]
    assert max(lo_g, lo_m) < min(hi_g, hi_m)  # IC marginales : recouvrement total
    d = out["pairwise"]["gate_vs_mean"]
    assert d["ci95"] == pytest.approx([0.02, 0.02], abs=1e-12)
    assert d["significant"] and d["p_two_sided"] == 0.0


# --------------------------------------------------------------------------- #
# 5. holm
# --------------------------------------------------------------------------- #

def test_holm_known_vector_and_monotonic():
    raw = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
    got = holm(raw)
    assert got == {"a": pytest.approx(0.004), "b": pytest.approx(0.06),
                   "c": pytest.approx(0.06), "d": pytest.approx(0.5)}
    for k in raw:
        assert got[k] >= raw[k] - 1e-15
    ordered = sorted(raw, key=raw.get)
    assert all(got[ordered[i]] <= got[ordered[i + 1]] + 1e-15 for i in range(len(ordered) - 1))


# --------------------------------------------------------------------------- #
# 6. I/O : convention de nommage, erreurs, collecte, écriture durable
# --------------------------------------------------------------------------- #

def test_cm_path_and_load_arm_stack(tmp_path):
    seeds = [42, 123]
    for s in seeds:
        np.save(tmp_path / f"V_seed{s}__epoch_160.cm.npy",
                np.zeros((7, NC, NC), dtype=np.int64))
    stack = load_arm_stack(tmp_path, "V", seeds)
    assert stack.shape == (2, 7, NC, NC)
    assert cm_path(tmp_path, "V", 42).name == "V_seed42__epoch_160.cm.npy"
    with pytest.raises(FileNotFoundError):
        load_arm_stack(tmp_path, "V", [42, 456])              # seed absent
    np.save(tmp_path / "W_seed42__epoch_160.cm.npy", np.zeros((5, NC, NC), dtype=np.int64))
    np.save(tmp_path / "W_seed123__epoch_160.cm.npy", np.zeros((9, NC, NC), dtype=np.int64))
    with pytest.raises(ValueError):
        load_arm_stack(tmp_path, "W", seeds)                  # N incohérent entre seeds


def test_save_arm_cms_roundtrip_durable(tmp_path):
    arr = np.stack([per_image_cm(p, g) for p, g in zip(*_scene(n_images=3))])
    p = save_arm_cms(arr, tmp_path, "monbras", 42)
    assert p.name == "monbras_seed42__epoch_160.cm.npy"
    assert np.array_equal(np.load(p), arr)
    assert not list(tmp_path.glob("*.tmp"))                   # pas de résidu temporaire


def test_collect_per_image_cms_labels_logits_keys():
    preds, gts = _scene(n_images=5)
    pmap = {i: preds[i] for i in range(5)}
    gmap = {i: gts[i] for i in range(5)}

    cms, keys = collect_per_image_cms(lambda i: pmap[i], lambda i: gmap[i], range(5))
    assert cms.shape == (5, NC, NC) and cms.dtype == np.int64
    assert keys == [f"idx{i}" for i in range(5)]
    for i in range(5):
        assert np.array_equal(cms[i], per_image_cm(preds[i], gts[i]))

    # logits (C,H,W) torch → argmax pris ; keys fournies ; ordre respecté
    logits = {i: torch.from_numpy(np.eye(NC, dtype=np.float32)[preds[i]]).permute(2, 0, 1)
              for i in range(5)}
    order = [3, 0, 4]
    cms2, keys2 = collect_per_image_cms(lambda i: logits[i], lambda i: gmap[i],
                                        order, keys=["k3", "k0", "k4"])
    assert keys2 == ["k3", "k0", "k4"]
    for j, i in enumerate(order):
        assert np.array_equal(cms2[j], per_image_cm(preds[i], gts[i]))


def test_bootstrap_rejects_bad_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap({"A": np.zeros((2, 30, NC, NC)), "B": np.zeros((2, 31, NC, NC))},
                         B=10, rng_seed=1)
    with pytest.raises(KeyError):
        paired_bootstrap({"A": np.zeros((1, 8, NC, NC), dtype=np.int64)}, B=10, rng_seed=1,
                         pairs=[("A", "Z")])
    with pytest.raises(ValueError):
        paired_bootstrap_values({"A": np.zeros((1, 8, 3))}, B=10, rng_seed=1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
