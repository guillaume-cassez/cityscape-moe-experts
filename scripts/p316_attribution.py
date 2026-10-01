#!/usr/bin/env python3
"""P3.16a — Attribution PER-CLASS : ce que chaque bras améliore, classe par classe.

Question Guillaume (26/09 23h31) : « chaque expert séparément améliore quoi ? chaque
consensus améliore quoi ? le MoE de 4 experts améliore quoi ? »

Méthode — MÊME protocole apparié que P3.10/P3.14 (bootstrap.py figé) :
  * confusions par-image (500 val, ordre partagé val_image_order.json), 3 seeds/bras ;
  * réplicats bootstrap B=10000, MÊMES index pour tous les bras, seeds moyennés dans
    chaque réplicat ; IoU dataset-level recalculé depuis la confusion ré-agrégée ;
  * Δ = bras − réf par classe → IC percentile [2.5, 97.5], p bilatéral, Holm PAR BRAS
    (famille = 19 classes du bras) ;
  * deux références : `controle` (apparié MoE, table P3.14) et `B` (baseline 160 ép.,
    comparaison à budget d'entraînement identique pour les experts).
  * accélération : agrégation par produit matriciel counts·CM (mathématiquement
    identique au fancy-index sum ; exact en float64 aux tailles en jeu), sanity-check
    mIoU global vs master_table.json P3.14.

CPU-only. Sorties : results/moe_v3_cs/p316/{attribution_perclass_vs_controle.json,
attribution_perclass_vs_B.json, attribution_perclass.md, heatmap_*.png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.moe.bootstrap import (  # noqa: E402
    _bootstrap_indices, holm, DEFAULT_B, DEFAULT_BOOTSTRAP_SEED, NC, IGNORE,
)

SEEDS = (42, 123, 456)
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
           "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
           "truck", "bus", "train", "motorcycle", "bicycle"]
ARM_SOURCES = [
    (REPO / "results/perimage_cm", "__epoch_160.cm.npy"),
    (REPO / "results/moe_v3_cs/harness", ".cm.npy"),
]
OUT = REPO / "results/moe_v3_cs/p316"
ARM_ORDER = ["A", "B", "C", "Cp", "D", "Dp", "G", "fused_CvetoB", "fused_CpvetoB",
             "fused_DvetoB", "fused_DpvetoB", "moe_v3cs"]


def discover() -> dict:
    found = {}
    for d, suffix in ARM_SOURCES:
        for p in sorted(d.glob(f"*{suffix}")):
            stem = p.name[: -len(suffix)]
            arm, _, sseed = stem.rpartition("_seed")
            if arm and sseed.isdigit():
                found.setdefault(arm, {})[int(sseed)] = p
    return {a: v for a, v in found.items() if all(s in v for s in SEEDS)}


def load_flats(per_seed: dict) -> list:
    """[(500,361) float64] par seed — CM cm[gt,pred] mis à plat."""
    flats = []
    for s in SEEDS:
        a = np.load(per_seed[s])
        assert a.shape[1:] == (NC, NC), f"{per_seed[s]}: shape {a.shape}"
        flats.append(a.reshape(a.shape[0], NC * NC).astype(np.float64))
    return flats


def ious_from_agg(agg: np.ndarray) -> np.ndarray:
    """(…,19,19) agrégats → IoU par classe (…,19), NaN si union vide."""
    inter = np.diagonal(agg, axis1=-2, axis2=-1)
    union = agg.sum(-1) + agg.sum(-2) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def per_class_bootstrap(flats: dict, ref: str, B: int, rng_seed: int) -> dict:
    n = flats[ref][0].shape[0]
    arms = [a for a in ARM_ORDER if a in flats] + \
           [a for a in sorted(flats) if a not in ARM_ORDER and a != ref]
    arms = [a for a in dict.fromkeys(arms) if a != ref]

    def arm_iou_replicates(name, idx_list):
        """(R,19) : IoU par classe, moyenne des seeds DANS chaque réplicat (protocole)."""
        out = np.empty((len(idx_list), NC))
        for r, idx in enumerate(idx_list):
            counts = np.bincount(idx, minlength=n).astype(np.float64)
            per_seed = np.stack([ious_from_agg((counts @ f).reshape(NC, NC))
                                 for f in flats[name]])
            out[r] = np.nanmean(per_seed, axis=0)
        return out

    full = [np.arange(n)]
    point = {a: arm_iou_replicates(a, full)[0] for a in arms}
    point[ref] = arm_iou_replicates(ref, full)[0]
    boot_idx = _bootstrap_indices(n, B, rng_seed)
    dist = {a: arm_iou_replicates(a, boot_idx) for a in [ref] + arms}

    tables = {}
    for a in arms:
        d = dist[a] - dist[ref]                      # (R,19)
        raw_p = {}
        rows = []
        for c in range(NC):
            dm = d[np.isfinite(d[:, c]), c]          # garde NaN (classe absente d'un réplicat)
            if dm.size == 0:
                p_val, lo, hi = 1.0, float("nan"), float("nan")
            else:
                p_val = 2.0 * min(float((dm <= 0).mean()), float((dm >= 0).mean()))
                lo = float(np.percentile(dm, 2.5) * 100)
                hi = float(np.percentile(dm, 97.5) * 100)
            raw_p[CLASSES[c]] = float(min(p_val, 1.0))
            rows.append({
                "classe": CLASSES[c],
                "iou_bras": float(point[a][c]),
                "iou_ref": float(point[ref][c]),
                "delta_pt": float((point[a][c] - point[ref][c]) * 100),
                "ci_lo_pt": lo,
                "ci_hi_pt": hi,
                "p_two_sided": raw_p[CLASSES[c]],
            })
        ph = holm(raw_p)
        for row in rows:
            row["p_holm_arm"] = ph[row["classe"]]
            row["significant"] = bool(np.isfinite(row["ci_lo_pt"]) and
                                      (row["ci_lo_pt"] > 0 or row["ci_hi_pt"] < 0))
        miou_d = np.nanmean(dist[a], axis=1) - np.nanmean(dist[ref], axis=1)
        tables[a] = {
            "miou_point": float(np.nanmean(point[a])),
            "miou_delta_pt": float((np.nanmean(point[a]) - np.nanmean(point[ref])) * 100),
            "miou_delta_ci_pt": [float(np.percentile(miou_d, 2.5) * 100),
                                 float(np.percentile(miou_d, 97.5) * 100)],
            "per_class": rows,
        }
    return {"ref": ref, "B": B, "bootstrap_seed": rng_seed, "n_images": n,
            "arms": tables}


def to_md(t: dict, ref: str) -> str:
    lines = [f"# Attribution per-class — Δ IoU vs **{ref}** (pt, IC95 apparié B={t['B']}, "
             f"Holm par bras)", ""]
    for a, tab in t["arms"].items():
        sig_up = [r for r in tab["per_class"] if r["significant"] and r["delta_pt"] > 0]
        sig_dn = [r for r in tab["per_class"] if r["significant"] and r["delta_pt"] < 0]
        sig_up.sort(key=lambda r: -r["delta_pt"])
        sig_dn.sort(key=lambda r: r["delta_pt"])
        up = ", ".join(f"{r['classe']} {r['delta_pt']:+.2f}" for r in sig_up[:8]) or "—"
        dn = ", ".join(f"{r['classe']} {r['delta_pt']:+.2f}" for r in sig_dn[:8]) or "—"
        lines += [f"## {a} — mIoU Δ {tab['miou_delta_pt']:+.2f} pt "
                  f"[{tab['miou_delta_ci_pt'][0]:+.2f}, {tab['miou_delta_ci_pt'][1]:+.2f}]",
                  f"- améliorations significatives : {up}",
                  f"- dégradations significatives : {dn}", ""]
    return "\n".join(lines)


def heatmap(t: dict, ref: str, png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(t["arms"])
    mat = np.array([[r["delta_pt"] for r in t["arms"][a]["per_class"]] for a in arms])
    sig = np.array([[r["significant"] for r in t["arms"][a]["per_class"]] for a in arms])
    vmax = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=9)
    for i in range(len(arms)):
        for j in range(len(CLASSES)):
            if sig[i, j]:
                ax.text(j, i, "•", ha="center", va="center", fontsize=7,
                        color="black" if mat[i, j] > 0 else "white")
    ax.set_title(f"Δ IoU par classe (pt) vs {ref} — holdout 500, 3 seeds, "
                 f"bootstrap apparié B={t['B']} ; • = IC95 exclut 0 (Holm par bras)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="Δ IoU (pt)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"→ {png}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms_paths = discover()
    print("bras découverts :", sorted(arms_paths))
    flats = {a: load_flats(v) for a, v in arms_paths.items()}

    # sanity : mIoU global doit retomber sur la master table P3.14
    master = json.loads((REPO / "results/moe_v3_cs/p314/master_table.json").read_text())

    out = {}
    for ref in ("controle", "B"):
        if ref not in flats:
            print(f"réf {ref} absente — saut")
            continue
        t = per_class_bootstrap(flats, ref, DEFAULT_B, DEFAULT_BOOTSTRAP_SEED)
        (OUT / f"attribution_perclass_vs_{ref}.json").write_text(
            json.dumps(t, indent=1, ensure_ascii=False))
        md = to_md(t, ref)
        (OUT / f"attribution_perclass_vs_{ref}.md").write_text(md)
        heatmap(t, ref, OUT / f"heatmap_perclass_vs_{ref}.png")
        out[ref] = t
        # sanity mIoU vs master table (réf controle uniquement)
        if ref == "controle":
            print("\n[sanity] Δ mIoU vs P3.14 master table :")
            for a, tab in t["arms"].items():
                m = master["pairwise"].get(f"{a}_vs_controle")
                d14 = m["delta"] * 100 if m else float("nan")
                print(f"  {a:16s} P3.16a {tab['miou_delta_pt']:+.3f} vs P3.14 {d14:+.3f}"
                      f"  |écart|={abs(tab['miou_delta_pt']-d14):.4f}")
    print("\nterminé →", OUT)


if __name__ == "__main__":
    main()
