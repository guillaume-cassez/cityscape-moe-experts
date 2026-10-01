#!/usr/bin/env python3
"""P3.16 — Attribution MÉTIERS par expert et par consensus (paper 3, question Guillaume 26/09 23h31).

Répond à « chaque expert séparément améliore quoi ? chaque consensus améliore quoi ? » en
étendant le protocole P3.15 EXACT (même holdout first:500, mêmes seeds, bootstrap apparié
B=10000 + Holm) aux 6 experts 160-époques (B, C, Cp, D, Dp, G) et aux 4 consensus veto du
paper 1 (fused_CvetoB, fused_DvetoB, fused_CpvetoB, fused_DpvetoB), plus les fragments
(count_fragments 8-connexe) pour controle/moe_v3cs qui n'avaient jamais été mesurés.

Phases (résuumables — tout fichier présent est sauté) :
  A  forwards GPU des 6 experts (≈45 s/arm/seed) → preds uint8 sur PREDS_DIR (disque 8T,
     le / du dépôt est à 96 %)
  A2 fusion cc_veto (CPU, prim/veto int64, max_drop_size None = BRATS-strict, classes
     fines protégées — MÊME appel que evaluate_consensus.py:195) → preds fused
  B  table GT piétons : réutilisée (symlink depuis results/moe_v3_cs/metiers)
  C  métriques par image (M.metrics_task de p3_metiers_eval) pour les 30 (arm, seed)
  C2 fragments par image pour controle + moe_v3cs (preds déjà sur disque)
  D  bootstraps appariés : experts/fused vs controle, fused vs primary → JSON + MD

Usage (Tour) :
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 /home/ser/brats-venv/bin/python \
      scripts/p316_metiers_experts.py --device cuda --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/ser/Bureau/City_Scape")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import p3_metiers_eval as M  # noqa: E402
from src.moe.bootstrap import (  # noqa: E402
    paired_bootstrap, paired_bootstrap_values, holm, format_report,
    DEFAULT_B, DEFAULT_BOOTSTRAP_SEED,
)
from src.postprocessing.consensus import cc_veto, count_fragments  # noqa: E402

SEEDS = (42, 123, 456)
NC = 19
N_POS = 500
PREDS_DIR = Path("/mnt/data8t/cityscape_p316")          # preds 1 Go/arm/seed → disque 8T
OUT_DIR = REPO / "results/moe_v3_cs/metiers_experts"    # npz métriques + tables (petits)
METIERS_DIR = REPO / "results/moe_v3_cs/metiers"        # P3.15 : GT, npz controle/moe, preds

# experts 160-époques : chemins vérifiés sur disque le 26/09 (reeval_ckpt_layout + checkpoints)
EXPERTS = {
    "B":  "reeval_ckpt_layout/pilot_fullres_B_baseline_seed{seed}/epoch_160.pth",
    "C":  "reeval_ckpt_layout/pilot_fullres_C_boundary_seed{seed}/epoch_160.pth",
    "Cp": "reeval_ckpt_layout/pilot_fullres_DMdice_distmap_seed{seed}/epoch_160.pth",
    "D":  "reeval_ckpt_layout/pilot_fullres_D_ce_boundary_seed{seed}/epoch_160.pth",
    "Dp": "reeval_ckpt_layout/pilot_fullres_DM_distmap_seed{seed}/epoch_160.pth",
    "G":  "checkpoints/pilot_fullres_G_blob_seed{seed}/epoch_160.pth",
}
# consensus veto du paper 1 (noms = master table P3.14) : (primary, veto)
FUSED = {
    "fused_CvetoB": ("C", "B"),
    "fused_DvetoB": ("D", "B"),
    "fused_CpvetoB": ("Cp", "B"),
    "fused_DpvetoB": ("Dp", "B"),
}
ALL_ARMS = list(EXPERTS) + list(FUSED)


def log(msg):
    print(f"[p316 {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Phase A — forwards experts (préparation du dossier façon P3.15 puis M.phase_a)
# --------------------------------------------------------------------------- #

def phase_a_experts(device: str, overwrite: bool):
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("gt_trainids.npy", "gt_instances_ped.npz", "label_paths.json"):
        dst = PREDS_DIR / name
        if not dst.exists():
            os.symlink(METIERS_DIR / name, dst)
            log(f"  [A] symlink GT {name}")
    arms = [{"name": a, "kind": "dense", "payload": str(REPO / p)}
            for a, p in EXPERTS.items()]
    old_arms = M.ARMS
    M.ARMS = arms
    try:
        ns = argparse.Namespace(device=device, overwrite=overwrite)
        M.phase_a(ns, PREDS_DIR, list(SEEDS), N_POS)
    finally:
        M.ARMS = old_arms
    # contrôle d'alignement GT (le même que P3.15, refait explicitement ici)
    assert np.array_equal(np.load(PREDS_DIR / "gt_trainids.npy", mmap_mode="r"),
                          np.load(METIERS_DIR / "gt_trainids.npy", mmap_mode="r"))


# --------------------------------------------------------------------------- #
# Phase A2 — consensus veto (CPU, identique evaluate_consensus.py:195)
# --------------------------------------------------------------------------- #

def fused_task(task):
    fused, seed = task
    prim, veto = FUSED[fused]
    dst = PREDS_DIR / f"pred_{fused}_seed{seed}.npy"
    if dst.exists():
        return str(dst)
    p = np.load(PREDS_DIR / f"pred_{prim}_seed{seed}.npy", mmap_mode="r")
    v = np.load(PREDS_DIR / f"pred_{veto}_seed{seed}.npy", mmap_mode="r")
    out = np.empty_like(p)
    t0 = time.time()
    for i in range(N_POS):
        out[i] = cc_veto(np.asarray(p[i]).astype(np.int64),
                         np.asarray(v[i]).astype(np.int64)).astype(np.uint8)
    np.save(dst, out)
    log(f"  [A2] {fused} seed {seed} : fusion {time.time()-t0:.0f}s")
    return str(dst)


# --------------------------------------------------------------------------- #
# Phase C2 — fragments par image (controle + moe_v3cs, preds P3.15 existants)
# --------------------------------------------------------------------------- #

def fragments_task(task):
    arm, seed = task
    dst = OUT_DIR / f"frag_{arm}_seed{seed}.npy"
    if dst.exists():
        return str(dst)
    preds = np.load(METIERS_DIR / f"pred_{arm}_seed{seed}.npy", mmap_mode="r")
    fr = np.zeros(N_POS, dtype=np.int64)
    for i in range(N_POS):
        fr[i] = sum(count_fragments(np.asarray(preds[i])).values())
    np.save(dst, fr)
    log(f"  [C2] fragments {arm} seed {seed} : mean={fr.mean():.1f}")
    return str(dst)


# --------------------------------------------------------------------------- #
# Phase D — bootstraps appariés + tables
# --------------------------------------------------------------------------- #

def load_met_stack(arm: str, key: str):
    rows = []
    for s in SEEDS:
        d = np.load(OUT_DIR / f"metriques_{arm}_seed{s}.npz")
        rows.append(np.asarray(d[key], dtype=np.float64))
    return np.stack(rows)


def build_pairs():
    pairs = [(a, "controle") for a in ALL_ARMS]
    pairs += [(f, p) for f, (p, _v) in FUSED.items()]   # fused vs son primaire
    pairs += [("moe_v3cs", "controle")]
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-a", action="store_true", help="sauter forwards+fusion (npz requis)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if not args.skip_a:
        phase_a_experts(args.device, args.overwrite)
        log("  [A] forwards experts terminés")
        from multiprocessing import Pool
        with Pool(min(args.workers, 4)) as pool:
            pool.map(fused_task, [(f, s) for f in FUSED for s in SEEDS])
        log("  [A2] consensus veto terminés")

    # phase C — métriques par image (tâches manquantes seulement)
    from multiprocessing import Pool
    tasks = [(a, s, str(PREDS_DIR), N_POS) for a in ALL_ARMS for s in SEEDS
             if args.overwrite or not (OUT_DIR / f"metriques_{a}_seed{s}.npz").exists()]
    # metrics_task écrit dans son dossier `args_out` → redirige via symlink si besoin
    for name in ("gt_trainids.npy", "gt_instances_ped.npz", "label_paths.json"):
        dst = PREDS_DIR / name
        if not dst.exists():
            os.symlink(METIERS_DIR / name, dst)
    if tasks:
        log(f"  [C] {len(tasks)} tâches métriques ({args.workers} workers)")
        # les npz doivent atterrir dans OUT_DIR : metrics_task écrit dans args_out →
        # on le laisse écrire dans PREDS_DIR puis on déplace.
        with Pool(args.workers) as pool:
            done = pool.map(M.metrics_task, tasks)
        for p in done:
            src = Path(p)
            dst = OUT_DIR / src.name
            if not dst.exists():
                shutil.move(str(src), str(dst))
    else:
        log("  [C] toutes les métriques déjà calculées")

    # phase C2 — fragments controle/moe
    frag_tasks = [(a, s) for a in ("controle", "moe_v3cs") for s in SEEDS
                  if args.overwrite or not (OUT_DIR / f"frag_{a}_seed{s}.npy").exists()]
    if frag_tasks:
        log(f"  [C2] {len(frag_tasks)} tâches fragments")
        with Pool(args.workers) as pool:
            pool.map(fragments_task, frag_tasks)

    # ---- assemblage : bras = controle + moe (npz P3.15 recopiés) + experts/fused ----
    for s in SEEDS:
        for a in ("controle", "moe_v3cs"):
            dst = OUT_DIR / f"metriques_{a}_seed{s}.npz"
            if not dst.exists():
                os.symlink(METIERS_DIR / f"metriques_{a}_seed{s}.npz", dst)

    arms_table = ["controle"] + ALL_ARMS + ["moe_v3cs"]
    cms_stacks, bf1_st, bf1p_st = {}, {}, {}
    strat_rec_st = {st: {} for st in M.STRATES}
    strat_det_st = {st: {} for st in M.STRATES}
    rec_str_st, prec_st = {}, {}
    for arm in arms_table:
        cms_stacks[arm] = load_met_stack(arm, "cm").astype(np.int64)
        bf1_st[arm] = load_met_stack(arm, "bf1")
        bf1p_st[arm] = load_met_stack(arm, "bf1_ped")
        rec_str_st[arm] = load_met_stack(arm, "rec_strict")
        prec_st[arm] = load_met_stack(arm, "prec_ped")
        rows_rec = {st: [] for st in M.STRATES}
        rows_det = {st: [] for st in M.STRATES}
        for s in SEEDS:
            d = np.load(OUT_DIR / f"metriques_{arm}_seed{s}.npz")
            r_, d_ = np.asarray(d["strat_rec"]), np.asarray(d["strat_det"])
            for si, st in enumerate(M.STRATES):
                rows_rec[st].append(r_[si])
                rows_det[st].append(d_[si])
        for st in M.STRATES:
            strat_rec_st[st][arm] = np.stack(rows_rec[st])
            strat_det_st[st][arm] = np.stack(rows_det[st])

    pairs = build_pairs()
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocole": "holdout P3.10 first:500, seeds 42/123/456, bootstrap APPARIÉ "
                     f"B={args.B} + Holm ; protocoles P3.15/P3.16 identiques "
                     "(fusion = cc_veto prim/veto int64 max_drop_size None)",
        "pairs": pairs,
        "tables": {},
    }

    log("  [D] mIoU")
    result["tables"]["mIoU"] = paired_bootstrap(cms_stacks, B=args.B,
                                                rng_seed=args.bootstrap_seed, pairs=pairs)
    for classe, nom in ((11, "IoU_person"), (12, "IoU_rider")):
        log(f"  [D] {nom}")
        result["tables"][nom] = M.boot_iou_classe(cms_stacks, classe, args.B,
                                                  args.bootstrap_seed, pairs)
    log("  [D] boundary_f1")
    result["tables"]["boundary_f1_3px"] = paired_bootstrap_values(
        bf1_st, B=args.B, rng_seed=args.bootstrap_seed, pairs=pairs)
    result["tables"]["boundary_f1_3px_pieds"] = paired_bootstrap_values(
        bf1p_st, B=args.B, rng_seed=args.bootstrap_seed, pairs=pairs)
    for key, stack, nom in (("rec_strict", rec_str_st, "rappel_strict_instances"),
                            ("prec_ped", prec_st, "precision_ped_pixels")):
        log(f"  [D] {nom}")
        result["tables"][nom] = M.boot_values_masque(stack, args.B, args.bootstrap_seed,
                                                     pairs, nom)
    for st in M.STRATES:
        for genre, stack in (("rappel", strat_rec_st[st]), ("det05", strat_det_st[st])):
            finite = np.isfinite(stack["controle"]).all(0)
            if finite.sum() < 10:
                continue
            log(f"  [D] instances {st} — {genre}")
            result["tables"][f"instances_{st}_{genre}"] = M.boot_values_masque(
                stack, args.B, args.bootstrap_seed, pairs, f"{genre} instances {st}")

    # fragments : bootstrap apparié sur (S,N) — controle vs moe (+ experts via paper1)
    frag_st = {}
    have_frag = all((OUT_DIR / f"frag_{a}_seed{s}.npy").exists()
                    for a in ("controle", "moe_v3cs") for s in SEEDS)
    if have_frag:
        for a in ("controle", "moe_v3cs"):
            frag_st[a] = np.stack([np.load(OUT_DIR / f"frag_{a}_seed{s}.npy").astype(np.float64)
                                   for s in SEEDS])
        log("  [D] fragments")
        result["tables"]["fragments"] = paired_bootstrap_values(
            frag_st, B=args.B, rng_seed=args.bootstrap_seed,
            pairs=[("moe_v3cs", "controle")])
        result["fragments_point"] = {a: float(frag_st[a].mean()) for a in frag_st}

    dest = OUT_DIR / "table_metiers_experts.json"
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"→ {dest}  (total {time.time()-t_start:.0f}s)")

    # rapport texte condensé
    lines = [f"# P3.16 — métriques métiers experts & consensus (holdout 500, {args.B} réplicats)"]
    for met, t in result["tables"].items():
        lines.append(f"\n## {met}")
        for pk, pv in t["pairwise"].items():
            if not pv["significant"] and pv["p_two_sided"] > 0.2:
                continue
            lines.append(f"- {pk}: Δ={pv['delta']*100:+.3f}pt "
                         f"[{pv['ci95'][0]*100:+.3f},{pv['ci95'][1]*100:+.3f}] "
                         f"p={pv['p_two_sided']:.4g} holm={pv['p_holm']:.4g}"
                         f"{' ✅' if pv['significant'] else ''}")
        pt = t.get("point", {})
        lines.append("  points %: " + ", ".join(f"{a}={v*100:.2f}" for a, v in sorted(pt.items())))
    (OUT_DIR / "table_metiers_experts.md").write_text("\n".join(lines) + "\n")
    log("→ table_metiers_experts.md")


if __name__ == "__main__":
    main()
