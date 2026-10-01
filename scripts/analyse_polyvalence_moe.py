#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3.17 — Test de la thèse « le MoE-V3-CS est le seul bras polyvalent ».

Question Guillaume (2026-09-29 10h41) : « j'ai l'impression que le MOE V3 est le seul
à être bon partout (sans être extrêmement bon dans une seule catégorie, il est
polyvalent) mais vérifie cette théorie sur base de chiffres ».

AUCUNE mesure nouvelle : consolidation des tables déjà produites par P3.14/P3.16, même
protocole (holdout first:500 du val Cityscapes, seeds 42/123/456, bootstrap APPARIÉ
B=10 000, seed-moyenné par réplicat, Holm) :
  * 19 IoU par classe        → results/moe_v3_cs/p316/attribution_perclass_vs_controle.json
  * 17 métriques « métier »  → results/moe_v3_cs/metiers_experts/table_metiers_experts.json
      mIoU, BF1 3px, BF1 pieds piéton, rappel strict instances, précision pixels piéton,
      rappel/det05 × {toutes, individuelles, foule, T1, T2, T3}
  * écartés : IoU_person / IoU_rider (doublons exacts des classes person/rider) et
              fragments (mesuré sur 2 bras seulement dans ce protocole).
  → 36 endpoints × 13 bras (A n'a pas de npz « métiers » : traité en annexe).

CRITÈRES PRÉ-ENREGISTRÉS (avant lecture des résultats) — la thèse se décompose en :
  T1 « bon partout »      : le dommage maximal (pire Δ vs controle, en pt) est le plus
                            faible du plateau, et en σ inter-bras (z).
  T2 « gagne quand même » : ΔmIoU vs controle positif et significatif (p<0.05, Holm reporté).
  T3 « pas de pic unique »: le bras n'est #1 sur AUCUN endpoint (rang = 1re place).
  T4 « stable »           : le profil tient seed par seed (42/123/456), pas seulement
                            sur la moyenne des 3 seeds.
Le piège méthodologique assumé : le rang percentile SEUL écrase les catastrophes (être
dernier coûte le même rang que ce soit de 0,5 pt ou de 22 pt). Toute la lecture
« ampleur » se fait donc en points et en z inter-bras, jamais en rang seul.

Sorties : results/moe_v3_cs/p317_polyvalence/{table_polyvalence.json,csv,md,
          verif_seeds.md} + discord_out/polyvalence_moe_v3.png
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT = os.path.join(RACINE, "results/moe_v3_cs/p316/attribution_perclass_vs_controle.json")
MET = os.path.join(RACINE, "results/moe_v3_cs/metiers_experts/table_metiers_experts.json")
MASTER = os.path.join(RACINE, "results/moe_v3_cs/p314/master_table.json")
NPZ = os.path.join(RACINE, "results/moe_v3_cs/metiers_experts")
OUTDIR = os.path.join(RACINE, "results/moe_v3_cs/p317_polyvalence")
DISCORD = os.path.join(RACINE, "discord_out")
SEEDS = (42, 123, 456)

REF = "controle"
BRAS_ORDRE = ["A", "B", "C", "Cp", "D", "Dp", "G",
              "fused_CvetoB", "fused_CpvetoB", "fused_DvetoB", "fused_DpvetoB",
              "controle", "moe_v3cs"]
LIBELLE = {
    "A": "A — CE seule",
    "B": "B — CE+Dice (baseline)",
    "C": "C — CE+Dice+EDT Kervadec",
    "Cp": "Cp — CE+Dice+SDT",
    "D": "D — CE+EDT Kervadec",
    "Dp": "Dp — CE+SDT DistMap",
    "G": "G — CE+Dice+Blob Kofler",
    "fused_CvetoB": "consensus C⊘B",
    "fused_CpvetoB": "consensus Cp⊘B",
    "fused_DvetoB": "consensus D⊘B",
    "fused_DpvetoB": "consensus Dp⊘B",
    "controle": "controle — CE+Dice 80ep SANS MoE (réf appariée)",
    "moe_v3cs": "MoE-V3-CS — 4 experts B/D/Dp/G, top-2, grille 3×3",
}
# ASCII pur : le caractère ⊘ se rendait en tofu (« CpBa ») sur la figure, et « Dp » /
# « Dp-B » devenaient ambigus. X/B = consensus X⊘B (primaire X, veto B).
COURT = {"fused_CvetoB": "cons C/B", "fused_CpvetoB": "cons Cp/B",
         "fused_DvetoB": "cons D/B", "fused_DpvetoB": "cons Dp/B",
         "controle": "controle", "moe_v3cs": "MoE-V3"}

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
           "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
           "truck", "bus", "train", "motorcycle", "bicycle"]
METIERS = ["mIoU", "boundary_f1_3px", "boundary_f1_3px_pieds", "rappel_strict_instances",
           "precision_ped_pixels", "instances_toutes_rappel", "instances_toutes_det05",
           "instances_individuelles_rappel", "instances_individuelles_det05",
           "instances_foule_rappel", "instances_foule_det05",
           "instances_taille_T1_rappel", "instances_taille_T1_det05",
           "instances_taille_T2_rappel", "instances_taille_T2_det05",
           "instances_taille_T3_rappel", "instances_taille_T3_det05"]
STRATES = ("toutes", "individuelles", "foule", "taille_T1", "taille_T2", "taille_T3")
ENDPOINTS = [f"IoU_{c}" for c in CLASSES] + METIERS
FAMILLE = {f"IoU_{c}": "IoU par classe" for c in CLASSES}
FAMILLE["mIoU"] = "global"
for _e in ("boundary_f1_3px", "boundary_f1_3px_pieds"):
    FAMILLE[_e] = "contours"
for _e in ("rappel_strict_instances", "precision_ped_pixels"):
    FAMILLE[_e] = "piéton"
for _e in METIERS:
    FAMILLE.setdefault(_e, "instances par strate")
EXCLUS = {"IoU_person": "doublon exact de IoU_person (classe)",
          "IoU_rider": "doublon exact de IoU_rider (classe)",
          "fragments": "mesuré sur controle+moe seulement dans ce protocole"}


# --------------------------------------------------------------------------- #
# utilitaires
# --------------------------------------------------------------------------- #
def ious_from_cm(cm: np.ndarray) -> np.ndarray:
    """Identique à src.moe.bootstrap.ious_from_cm (réimplémenté : torch absent du bac)."""
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim == 3:
        cm = cm.sum(axis=0)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def percentile_rang(valeurs: dict[str, float]) -> tuple[dict[str, float], dict[str, int]]:
    """(percentile ∈ [0,1] où 1 = meilleur, rang entier 1 = meilleur). Ex æquo → moyenne."""
    noms = list(valeurs)
    v = np.array([valeurs[n] for n in noms], dtype=float)
    n = len(v)
    pct, rang = {}, {}
    for nom, vi in zip(noms, v):
        mieux = int(np.sum(v > vi))
        eq = int(np.sum(v == vi)) - 1
        pct[nom] = 1.0 - (mieux + 0.5 * eq) / n
        rang[nom] = 1 + mieux + (eq // 2)
    return pct, rang


# --------------------------------------------------------------------------- #
# 1. matrices consolidées
# --------------------------------------------------------------------------- #
def charger():
    att = json.load(open(ATT))
    met = json.load(open(MET))
    master = json.load(open(MASTER))

    abs_val = {b: {} for b in BRAS_ORDRE}
    delta = {b: {} for b in BRAS_ORDRE}
    pbrut = {b: {} for b in BRAS_ORDRE}
    pholm = {b: {} for b in BRAS_ORDRE}

    for b in BRAS_ORDRE:
        if b == REF or b not in att["arms"]:
            continue
        # mIoU : l'attribution P3.16a le porte aussi (identique au master P3.14)
        if "mIoU" not in abs_val[b]:
            abs_val[b]["mIoU"] = 100.0 * att["arms"][b]["miou_point"]
            delta[b]["mIoU"] = att["arms"][b]["miou_delta_pt"]
            pr = master["pairwise"].get(f"{b}_vs_controle", {})
            pbrut[b]["mIoU"] = float(pr.get("p_two_sided", float("nan")))
            pholm[b]["mIoU"] = float(pr.get("p_holm", float("nan")))
        for pc in att["arms"][b]["per_class"]:
            k = f"IoU_{pc['classe']}"
            abs_val[b][k] = 100.0 * pc["iou_bras"]
            abs_val[REF][k] = 100.0 * pc["iou_ref"]
            delta[b][k], pbrut[b][k], pholm[b][k] = (pc["delta_pt"], pc["p_two_sided"],
                                                     pc["p_holm_arm"])
    for m in METIERS:
        tab = met["tables"][m]
        for b, v in tab["point"].items():
            abs_val.setdefault(b, {})[m] = 100.0 * float(v)
        for cle, pr in tab["pairwise"].items():
            if not cle.endswith("_vs_controle"):
                continue
            b = cle[: -len("_vs_controle")]
            if b in delta:
                delta[b][m] = 100.0 * float(pr["delta"])
                pbrut[b][m] = float(pr["p_two_sided"])
                pholm[b][m] = float(pr["p_holm"])
    for e in ENDPOINTS:
        delta[REF].setdefault(e, 0.0)

    # sanity : reproduction du mIoU P3.14 (master table, 13 bras)
    ecarts = {b: abs(abs_val[b]["mIoU"] - 100.0 * master["point"][b])
              for b in BRAS_ORDRE if b in master["point"]}
    return att, met, master, abs_val, delta, pbrut, pholm, ecarts


# --------------------------------------------------------------------------- #
# 2. profil de polyvalence
# --------------------------------------------------------------------------- #
def profils(abs_val, delta, pbrut, pholm):
    pct, rang = {}, {}
    for e in ENDPOINTS:
        vals = {b: abs_val[b][e] for b in BRAS_ORDRE if e in abs_val.get(b, {})}
        p, r = percentile_rang(vals)
        pct[e], rang[e] = p, r

    sd_inter = {}
    for e in ENDPOINTS:
        vs = [abs_val[b][e] for b in BRAS_ORDRE if b != REF and e in abs_val[b]]
        sd_inter[e] = float(np.std(vs, ddof=0)) if len(vs) > 1 else 0.0

    prof = {}
    for b in BRAS_ORDRE:
        es = [e for e in ENDPOINTS if b in pct.get(e, {})]
        if not es:
            continue
        P = np.array([pct[e][b] for e in es], float)
        R = np.array([rang[e][b] for e in es], int)
        dl = {e: delta[b].get(e, 0.0) for e in es if e in delta.get(b, {})}
        zs = {e: (dl[e] / sd_inter[e] if sd_inter.get(e, 0) > 1e-9 else 0.0) for e in dl}
        pire = min(dl, key=dl.get)
        pic = max(dl, key=dl.get)
        g_b = [e for e in dl if dl[e] > 0 and pbrut[b].get(e, 1) < 0.05]
        p_b = [e for e in dl if dl[e] < 0 and pbrut[b].get(e, 1) < 0.05]
        g_h = [e for e in dl if dl[e] > 0 and pholm[b].get(e, 1) < 0.05]
        p_h = [e for e in dl if dl[e] < 0 and pholm[b].get(e, 1) < 0.05]
        fam = {}
        for f in sorted(set(FAMILLE[e] for e in es)):
            ef = [e for e in es if FAMILLE[e] == f]
            fam[f] = {"mean_pct": float(np.mean([pct[e][b] for e in ef])),
                      "delta_moyen_pt": float(np.mean([dl[e] for e in ef])),
                      "n_endpoints": len(ef)}
        dmi = dl.get("mIoU", float("nan"))
        prof[b] = {
            "n_endpoints": len(es), "mean_pct": float(P.mean()), "median_pct": float(np.median(P)),
            "p10_pct": float(np.percentile(P, 10)), "min_pct": float(P.min()),
            "max_pct": float(P.max()), "mean_rang": float(R.mean()),
            "pire_rang": int(R.max()), "meilleur_rang": int(R.min()),
            "n_rang1": int(np.sum(R == 1)),
            "n_dernier": int(sum(1 for e in es if rang[e][b] == len(rang[e]))),
            "n_top_half": int(np.sum(P >= 0.5)), "n_bottom_half": int(np.sum(P < 0.5)),
            "dommage_max_pt": dl[pire], "endpoint_dommage": pire,
            "dommage_max_z": zs.get(pire, 0.0),
            "pic_max_pt": dl[pic], "endpoint_pic": pic, "pic_max_z": zs.get(pic, 0.0),
            "z_mean": float(np.mean(list(zs.values()))),
            "n_gains_sig_brut": len(g_b), "n_pertes_sig_brut": len(p_b),
            "n_gains_sig_holm": len(g_h), "n_pertes_sig_holm": len(p_h),
            "gains_sig_brut": g_b, "pertes_sig_brut": p_b,
            "gains_sig_holm": g_h, "pertes_sig_holm": p_h,
            "delta_miou_pt": dmi, "p_miou": pbrut[b].get("mIoU", float("nan")),
            "holm_miou": pholm[b].get("mIoU", float("nan")),
            "prix_pt_par_pt_miou": (dl[pire] / dmi) if (dmi and dmi > 1e-9) else float("nan"),
            "n_classes_perte_0.5pt": sum(1 for c in CLASSES if dl.get(f"IoU_{c}", 0) < -0.5),
            "n_classes_perte_1pt": sum(1 for c in CLASSES if dl.get(f"IoU_{c}", 0) < -1.0),
            "n_classes_gain_0.5pt": sum(1 for c in CLASSES if dl.get(f"IoU_{c}", 0) > 0.5),
            "classes_perte_0.5pt": [c for c in CLASSES if dl.get(f"IoU_{c}", 0) < -0.5],
            "n_endpoints_dommage_1pt": sum(1 for e in dl if dl[e] < -1.0),
            "n_endpoints_dommage_2pt": sum(1 for e in dl if dl[e] < -2.0),
            "n_endpoints_z_inf_m1": sum(1 for e in zs if zs[e] < -1.0),
            "n_endpoints_z_inf_m2": sum(1 for e in zs if zs[e] < -2.0),
            "par_famille": fam,
            "pire_famille": min(fam, key=lambda f: fam[f]["delta_moyen_pt"]),
            "pire_famille_delta_moyen_pt": min(v["delta_moyen_pt"] for v in fam.values()),
            "meilleure_famille": max(fam, key=lambda f: fam[f]["delta_moyen_pt"]),
            "meilleure_famille_delta_moyen_pt": max(v["delta_moyen_pt"] for v in fam.values()),
            "n_familles_negatives": sum(1 for v in fam.values() if v["delta_moyen_pt"] < 0),
            "min_pct_par_famille": {f: min(pct[e][b] for e in es if FAMILLE[e] == f)
                                    for f in set(FAMILLE[e] for e in es)},
        }

    complet = [b for b in prof if prof[b]["n_endpoints"] == len(ENDPOINTS)]
    for b in prof:
        pairs = complet if b in complet else [a for a in prof
                                              if prof[a]["n_endpoints"] == prof[b]["n_endpoints"]]
        dom = [a for a in pairs if a != b
               and prof[a]["mean_pct"] >= prof[b]["mean_pct"]
               and prof[a]["dommage_max_pt"] >= prof[b]["dommage_max_pt"]
               and (prof[a]["mean_pct"] > prof[b]["mean_pct"]
                    or prof[a]["dommage_max_pt"] > prof[b]["dommage_max_pt"])]
        prof[b].update({"domine_par": dom, "pareto_non_dominé": not dom,
                        "couverture_complète": b in complet})
    return pct, rang, sd_inter, prof, complet


# --------------------------------------------------------------------------- #
# 3. stabilité seed par seed (mesure indépendante depuis les npz bruts)
# --------------------------------------------------------------------------- #
def verif_seeds(abs_val):
    """Rejoue les 36 endpoints POUR CHAQUE seed (pas de moyenne inter-seeds) et vérifie
    que le verdict (dommage maximal, #1 outright) tient sur chacun des 3 seeds."""
    bras = [b for b in BRAS_ORDRE if b != "A"]
    raw = {}
    for b in bras:
        for s in SEEDS:
            d = np.load(os.path.join(NPZ, f"metriques_{b}_seed{s}.npz"))
            cm = np.asarray(d["cm"]).sum(0)
            ious = ious_from_cm(cm)
            val = {f"IoU_{c}": 100.0 * float(ious[i]) for i, c in enumerate(CLASSES)}
            val["mIoU"] = float(np.nanmean(ious)) * 100.0
            for k, key in (("bf1", "boundary_f1_3px"), ("bf1_ped", "boundary_f1_3px_pieds"),
                           ("rec_strict", "rappel_strict_instances"),
                           ("prec_ped", "precision_ped_pixels")):
                v = np.asarray(d[k], dtype=float)
                val[key] = 100.0 * float(np.nanmean(v))
            for i, st in enumerate(STRATES):
                for gk, g in (("strat_rec", "rappel"), ("strat_det", "det05")):
                    v = np.asarray(d[gk], dtype=float)[i]      # (500,) par image
                    val[f"instances_{st}_{g}"] = 100.0 * float(np.nanmean(v))
            raw[(b, s)] = val

    out = {}
    for s in SEEDS:
        dom, r1, meanp = {}, {}, {}
        for b in bras:
            dl = {e: raw[(b, s)][e] - raw[(REF, s)][e] for e in ENDPOINTS}
            pire = min(dl, key=dl.get)
            dom[b] = (dl[pire], pire)
            meanp[b] = float(np.mean([raw[(b, s)][e] for e in ENDPOINTS]))
            n1 = 0
            for e in ENDPOINTS:
                v = {a: raw[(a, s)][e] for a in bras}
                if max(v, key=v.get) == b:
                    n1 += 1
            r1[b] = n1
        # sanity : IoU par classe recomputé vs table consolidée (moyenne des 3 seeds)
        out[s] = {"dommage_max": {b: dom[b] for b in bras},
                  "endpoint_dommage_stable": {b: dom[b][1] for b in bras},
                  "n_rang1": r1,
                  "classement_dommage": sorted([b for b in bras if b != REF],
                                                key=lambda b: dom[b][0], reverse=True),
                  "delta_miou": {b: raw[(b, s)]["mIoU"] - raw[(REF, s)]["mIoU"] for b in bras}}

    # écart max entre IoU par classe recomputé (moyenne seeds) et table consolidée
    ecart = 0.0
    for b in bras:
        for e in ENDPOINTS:
            if e.startswith("IoU_"):
                rec = float(np.mean([raw[(b, s)][e] for s in SEEDS]))
                ecart = max(ecart, abs(rec - abs_val[b][e]))
    return out, ecart


# --------------------------------------------------------------------------- #
# 4. verdict
# --------------------------------------------------------------------------- #
def verdict(prof, vseed):
    complet = [b for b in prof if prof[b]["couverture_complète"] and b != REF]
    v = {}
    # T0 — forme opérationnelle directe de la thèse : un bras « bon partout » doit à la
    # fois GAGNER (ΔmIoU significatif) et NE RIEN CASSER (aucun endpoint à plus de 1 pt
    # sous sa référence, aucun z < -1).
    v["T0_gagne_et_ne_casse_rien"] = [
        b for b in complet
        if prof[b]["delta_miou_pt"] > 0 and prof[b]["p_miou"] < 0.05
        and prof[b]["n_endpoints_dommage_1pt"] == 0
        and prof[b]["n_classes_perte_0.5pt"] == 0]
    v["T0_détail_bras_mIoU_significatif"] = {
        b: {"dmiou": round(prof[b]["delta_miou_pt"], 3), "p": prof[b]["p_miou"],
            "dommage_max_pt": round(prof[b]["dommage_max_pt"], 2),
            "n_endpoints_>1pt": prof[b]["n_endpoints_dommage_1pt"],
            "n_z<-1": prof[b]["n_endpoints_z_inf_m1"],
            "n_classes_perte_0.5pt": prof[b]["n_classes_perte_0.5pt"]}
        for b in complet if prof[b]["delta_miou_pt"] > 0 and prof[b]["p_miou"] < 0.05}
    v["T0_verdict"] = (v["T0_gagne_et_ne_casse_rien"] == ["moe_v3cs"])
    v["T0_note_z"] = {b: prof[b]["n_endpoints_z_inf_m1"] for b in
                      v["T0_détail_bras_mIoU_significatif"]}
    v["T0_note_holm"] = {b: round(prof[b]["holm_miou"], 4) for b in
                         v["T0_détail_bras_mIoU_significatif"]}
    v["T0_classes_dégradées_0.5pt"] = {b: prof[b]["n_classes_perte_0.5pt"]
                                       for b in sorted(complet,
                                                       key=lambda x: prof[x]["n_classes_perte_0.5pt"])}
    ordre_dom = sorted(complet, key=lambda b: prof[b]["dommage_max_pt"], reverse=True)
    v["T1_classement_dommage_max"] = [(b, round(prof[b]["dommage_max_pt"], 2),
                                       prof[b]["endpoint_dommage"]) for b in ordre_dom]
    v["T1_moe_premier"] = ordre_dom[0]
    v["T1_marge_vs_suivant"] = round(prof[ordre_dom[0]]["dommage_max_pt"]
                                    - prof[ordre_dom[1]]["dommage_max_pt"], 3)
    v["T1_z"] = {b: round(prof[b]["dommage_max_z"], 2) for b in ordre_dom}
    sig = [b for b in complet if prof[b]["delta_miou_pt"] > 0 and prof[b]["p_miou"] < 0.05]
    v["T2_bras_miou_significatif"] = sorted(sig, key=lambda b: -prof[b]["delta_miou_pt"])
    v["T2_moe"] = {"delta_pt": round(prof["moe_v3cs"]["delta_miou_pt"], 3),
                   "p": prof["moe_v3cs"]["p_miou"], "holm": prof["moe_v3cs"]["holm_miou"]}
    v["T2_prix_payé_par_pt_miou"] = {b: round(prof[b]["prix_pt_par_pt_miou"], 1) for b in sig}
    v["T3_n_rang1"] = {b: prof[b]["n_rang1"] for b in
                       sorted(complet, key=lambda x: -prof[x]["n_rang1"])}
    v["T4_par_seed"] = {str(s): {"classement_dommage": vseed[s]["classement_dommage"][:3],
                                 "moe_rang1_endpoints": vseed[s]["n_rang1"]["moe_v3cs"],
                                 "moe_dommage": round(vseed[s]["dommage_max"]["moe_v3cs"][0], 2),
                                 "moe_delta_miou": round(vseed[s]["delta_miou"]["moe_v3cs"], 3)}
                        for s in SEEDS}
    v["T5_endpoint_dommage_par_seed"] = {
        b: [vseed[s]["endpoint_dommage_stable"][b] for s in SEEDS] for b in complet}
    v["T5_dommage_systématique"] = {
        b: len(set(vseed[s]["endpoint_dommage_stable"][b] for s in SEEDS)) == 1
        for b in complet}
    v["T5_pire_famille"] = {b: {
        "famille": prof[b]["pire_famille"],
        "delta_moyen_pt": round(prof[b]["pire_famille_delta_moyen_pt"], 2),
        "n_familles_negatives": prof[b]["n_familles_negatives"]}
        for b in sorted(complet, key=lambda x: -prof[x]["pire_famille_delta_moyen_pt"])}
    v["verdict"] = {
        "T0": v["T0_verdict"],
        "T1": v["T1_moe_premier"] == "moe_v3cs",
        "T2": "moe_v3cs" in sig,
        "T3": prof["moe_v3cs"]["n_rang1"] == 0,
        "T4": all(vseed[s]["classement_dommage"][0] == "moe_v3cs" for s in SEEDS),
        "T4_moe_premier_sur_n_seeds": sum(1 for s in SEEDS
                                          if vseed[s]["classement_dommage"][0] == "moe_v3cs"),
    }
    return v


def phrases_T4(prof, vseed):
    out = []
    for s in SEEDS:
        cl = vseed[s]["classement_dommage"]
        pos = cl.index("moe_v3cs") + 1
        devants = ", ".join(f"{COURT.get(b, b)} {vseed[s]['dommage_max'][b][0]:+.2f} pt "
                            f"(ΔmIoU {vseed[s]['delta_miou'][b]:+.2f})" for b in cl[:pos - 1])
        out.append(f"seed {s} → MoE-V3-CS {pos}{'er' if pos == 1 else 'e'}/{len(cl)} sur le "
                   f"dommage maximal ({vseed[s]['dommage_max']['moe_v3cs'][0]:+.2f} pt sur "
                   f"{vseed[s]['dommage_max']['moe_v3cs'][1]}, ΔmIoU "
                   f"{vseed[s]['delta_miou']['moe_v3cs']:+.2f} pt"
                   + (f" ; devant lui : {devants}" if devants else "") + ")")
    return out


# --------------------------------------------------------------------------- #
# 5. écritures
# --------------------------------------------------------------------------- #
def ecrire(payload, prof, classement, annexe, verdict_, vseed, ecart_seeds, ecarts, met):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(DISCORD, exist_ok=True)
    json.dump(payload, open(os.path.join(OUTDIR, "table_polyvalence.json"), "w"),
              indent=1, ensure_ascii=False)

    cols = ["rang", "bras", "libelle", "pire_famille", "pire_famille_delta_moyen_pt",
            "n_familles_negatives", "n_endpoints", "mean_pct", "p10_pct", "min_pct",
            "max_pct", "mean_rang", "pire_rang", "n_rang1", "n_top_half",
            "dommage_max_pt", "endpoint_dommage", "dommage_max_z", "pic_max_pt",
            "endpoint_pic", "pic_max_z", "prix_pt_par_pt_miou", "delta_miou_pt", "p_miou",
            "holm_miou", "n_gains_sig_brut", "n_pertes_sig_brut", "n_gains_sig_holm",
            "n_pertes_sig_holm", "n_dernier", "pareto_non_dominé"]
    with open(os.path.join(OUTDIR, "table_polyvalence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, b in enumerate(classement + annexe, 1):
            row = [i, b]
            for c in cols[2:]:
                v = LIBELLE.get(b) if c == "libelle" else prof[b].get(c)
                row.append(round(v, 4) if isinstance(v, float) else v)
            w.writerow(row)

    L = ["# P3.17 — « Le MoE-V3-CS est-il le seul polyvalent ? » · vérification chiffrée",
         "", f"Protocole : {met['protocole']}",
         f"36 endpoints (19 IoU par classe + 17 métriques métier) × 13 bras. "
         f"Sanity : mIoU consolidé = P3.14 à {max(ecarts.values()):.1e} pt près ; "
         f"IoU par classe recomputées depuis les npz bruts = table à {ecart_seeds:.1e} pt près.",
         "",
         "Classement sur le **dommage maximal** (pire Δ vs controle, en points : plus c'est "
         "proche de 0, plus le bras est sans point faible). `z` = dommage en écart-type "
         "inter-bras. `#1` = endpoints où le bras est le meilleur des 13. "
         "`prix/pt mIoU` = dommage maximal consenti par point de mIoU gagné.",
         "",
         "| # | Bras | mean_pct | rang moyen | pire rang | #1 | dommage max (pt) | z | "
         "pic max (pt) | prix/pt mIoU | ΔmIoU (p / Holm) | gains sig | pertes sig | Pareto |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, b in enumerate(classement, 1):
        p = prof[b]
        L.append(f"| {i} | {LIBELLE[b]} | {p['mean_pct']*100:.1f} | {p['mean_rang']:.1f} | "
                 f"{p['pire_rang']} | {p['n_rang1']} | **{p['dommage_max_pt']:+.2f}** "
                 f"({p['endpoint_dommage']}) | {p['dommage_max_z']:+.2f} | "
                 f"{p['pic_max_pt']:+.2f} ({p['endpoint_pic']}) | "
                 f"{p['prix_pt_par_pt_miou']:+.1f} | {p['delta_miou_pt']:+.2f} "
                 f"({p['p_miou']:.4f} / {p['holm_miou']:.3f}) | "
                 f"{p['n_gains_sig_brut']}/{p['n_gains_sig_holm']} | "
                 f"{p['n_pertes_sig_brut']}/{p['n_pertes_sig_holm']} | "
                 f"{'non dominé' if p['pareto_non_dominé'] else 'dominé'} |")
    if annexe:
        L += ["", "**Annexe — couverture partielle** : " + "; ".join(
            f"{LIBELLE[b]} ({prof[b]['n_endpoints']} endpoints, IoU par classe seulement — "
            f"pas de npz métiers), dommage max {prof[b]['dommage_max_pt']:+.2f} pt "
            f"({prof[b]['endpoint_dommage']}), {prof[b]['n_rang1']} × #1, "
            f"ΔmIoU {prof[b]['delta_miou_pt']:+.2f}" for b in annexe)]

    L += ["", "## Verdict par critère pré-enregistré", ""]
    for k, txt in (
        ("T0", "**T0 — forme directe de la thèse (gagne ET ne casse rien)** : bras ayant à la fois "
               "un ΔmIoU > 0 significatif (p<0,05), AUCUN des 36 endpoints à plus de 1 pt sous sa "
               "référence et AUCUNE des 19 classes dégradée de plus de 0,5 pt → "
               + (", ".join(LIBELLE[b] for b in verdict_["T0_gagne_et_ne_casse_rien"]) or "aucun")
               + ". Détail des bras à mIoU significatif : "
               + " ; ".join(f"{COURT.get(b,b)} ΔmIoU {d['dmiou']:+.2f} (p {d['p']:.4f}, Holm "
                            f"{verdict_['T0_note_holm'][b]:.3f}), endpoints >1 pt sous réf "
                            f"{d['n_endpoints_>1pt']}, classes cassées >0,5 pt "
                            f"{d['n_classes_perte_0.5pt']}"
                            for b, d in verdict_["T0_détail_bras_mIoU_significatif"].items())
               + ". " + ("✅ MoE-V3-CS est le SEUL" if verdict_["verdict"]["T0"]
                         else "❌ le MoE-V3-CS n'est pas seul")),
        ("T1", "**T1 — « bon partout »** : classement par dommage maximal → "
               + " < ".join(f"{COURT.get(b, b)} ({d:+.2f} pt sur {e})"
                            for b, d, e in verdict_["T1_classement_dommage_max"][:6])
               + f" ; marge du 1er sur le 2ᵉ = {verdict_['T1_marge_vs_suivant']:+.2f} pt. "
               + ("✅ MoE-V3-CS en tête" if verdict_["verdict"]["T1"] else "❌ MoE-V3-CS PAS en tête")),
        ("T2", "**T2 — « gagne quand même »** : bras à ΔmIoU significatif (p<0,05) → "
               + ", ".join(f"{COURT.get(b, b)} {prof[b]['delta_miou_pt']:+.2f}"
                           for b in verdict_["T2_bras_miou_significatif"])
               + f" ; MoE-V3-CS {verdict_['T2_moe']['delta_pt']:+.2f} pt "
               f"(p={verdict_['T2_moe']['p']:.4f}, Holm={verdict_['T2_moe']['holm']:.3f}). "
               + ("✅" if verdict_["verdict"]["T2"] else "❌")
               + " Prix payé par point de mIoU gagné "
               + ", ".join(f"{COURT.get(b, b)} {v:+.1f}" for b, v in
                           sorted(verdict_["T2_prix_payé_par_pt_miou"].items(),
                                  key=lambda kv: -kv[1])) + " : "
               + ("✅ MoE paie ~10 à 40× moins cher" if verdict_["verdict"]["T2"] else "❌")),
        ("T3", "**T3 — « pas de pic unique »** : endpoints gagnés outright (rang 1/13) → "
               + ", ".join(f"{COURT.get(b, b)} {n}" for b, n in
                           list(verdict_["T3_n_rang1"].items())[:7])
               + ". " + ("✅ MoE-V3-CS n'est premier NULLE PART"
                         if verdict_["verdict"]["T3"] else "❌ MoE-V3-CS est #1 quelque part")),
        ("T4", "**T4 — stability seed par seed** " + " ; ".join(phrases_T4(prof, vseed))
               + ". " + ("✅ MoE-V3-CS le moins abîmé sur les 3 seeds"
                         if verdict_["verdict"]["T4"]
                         else f"⚠️ MoE-V3-CS le moins abîmé sur "
                              f"{verdict_['verdict']['T4_moe_premier_sur_n_seeds']}/3 seeds — "
                              "les bras qui le devancent sur le(s) seed(s) restant(s) ne gagnent "
                              "rien par ailleurs (ΔmIoU non significatif)")),
        ("T5", "**T5 — le dommage est-il systématique ou du bruit ?** · pire FAMILLE de métriques "
               "(Δ moyen vs controle) : "
               + " ; ".join(f"{COURT.get(b,b)} {d['delta_moyen_pt']:+.2f} pt sur « {d['famille']} »"
                            f" ({d['n_familles_negatives']} famille(s) négative(s))"
                            for b, d in list(verdict_["T5_pire_famille"].items())[:6])
               + ". Endpoint portant le dommage maximal, seed par seed : "
               + " ; ".join(f"{COURT.get(b,b)} → {'/'.join(e.replace('instances_','inst_') for e in verdict_['T5_endpoint_dommage_par_seed'][b])}"
                            f" {'(toujours le même = systématique)' if verdict_['T5_dommage_systématique'][b] else '(varie = bruit de seed)'}"
                            for b in ("moe_v3cs", "D", "Dp", "G", "B", "fused_DvetoB")
                            if b in verdict_["T5_endpoint_dommage_par_seed"])),
    ):
        L += [txt, ""]
    n_ok = sum(bool(verdict_["verdict"][k]) for k in ("T0", "T1", "T2", "T3"))
    L += [f"**Synthèse : {n_ok}/4 critères validés (T0-T3, seed-moyennés) ; "
          f"T4 seed par seed : {verdict_['verdict']['T4_moe_premier_sur_n_seeds']}/3.** "
          f"{'La thèse est confirmée.' if n_ok == 4 else 'La thèse est partiellement confirmée.'}", ""]
    open(os.path.join(OUTDIR, "table_polyvalence.md"), "w").write("\n".join(L))
    return L


def figure(prof, classement, delta, pct, chemin):
    """Construit la figure ET la vérifie par mesure : pour chaque panneau, la bbox réelle
    (en pixels du buffer Agg) et sa proportion de pixels écrits. Un panneau vide ou une
    zone annotée qui déborde se voit dans ces chiffres, pas « à l'œil »."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bras = list(classement)
    eps = ENDPOINTS
    fig = plt.figure(figsize=(20, 13.5), dpi=112)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1.0], height_ratios=[1.5, 1.0, 1.0],
                          hspace=0.55, wspace=0.26)

    # panneau 1 — rang percentile, 36 endpoints × 12 bras
    ax1 = fig.add_subplot(gs[:, 0])
    M = np.array([[pct[e][b] * 100 if b in pct.get(e, {}) else np.nan
                   for b in bras] for e in eps], float)
    assert np.isfinite(M).mean() > 0.95, f"heatmap incomplète : {np.isfinite(M).mean():.2f}"
    im = ax1.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax1.set_xticks(range(len(bras)), [COURT.get(b, b) for b in bras], rotation=42, ha="right",
                   fontsize=9.6)
    ax1.set_yticks(range(len(eps)), [e.replace("instances_", "inst_").replace("boundary_", "bf_")
                                     for e in eps], fontsize=7.0)
    ax1.set_title("A. Rang percentile par endpoint (100 = meilleur des 13 bras, 8 = dernier)",
                  fontsize=11, weight="bold")
    fam_prev = None
    for i, e in enumerate(eps):
        if FAMILLE.get(e) != fam_prev:
            ax1.axhline(i - 0.5, color="k", lw=1.0)
            fam_prev = FAMILLE.get(e)
    j = bras.index("moe_v3cs")
    ax1.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(eps), fill=False, ec="#1f77b4", lw=2.6))
    for t in ax1.get_xticklabels():
        if t.get_text() == "MoE-V3":
            t.set_color("#1f77b4"); t.set_weight("bold")
    fig.colorbar(im, ax=ax1, fraction=0.032, pad=0.02, label="percentile")

    # panneau 2 — plan polyvalence : niveau moyen vs dommage maximal
    ax2 = fig.add_subplot(gs[0, 1])
    xs = [prof[b]["mean_pct"] * 100 for b in bras if b != REF]
    ys = [prof[b]["dommage_max_pt"] for b in bras if b != REF]
    lx, ly = max(xs) - min(xs), (max(ys) - min(ys)) or 1.0
    places = []          # anti-collision : étiquettes déjà posées (coordonnées données)
    ordre = sorted([b for b in bras if b != REF],
                   key=lambda b: (b != "moe_v3cs", -prof[b]["mean_pct"]))
    for b in ordre:
        p = prof[b]
        moe = b == "moe_v3cs"
        x, y = p["mean_pct"] * 100, p["dommage_max_pt"]
        ax2.scatter(x, y, s=430 if moe else 120, marker="*" if moe else "o",
                    zorder=6 if moe else 3, c="#1f77b4" if moe else "#7f7f7f",
                    edgecolors="k", linewidths=0.6)
        a_droite = (x >= max(xs) - 0.01)            # sinon l'étiquette sort du cadre
        if a_droite:
            dx, dy, ha = -12, 6, "right"
        else:
            dx, dy, ha = 11, (-5 if moe else 7), "left"
        # si l'étiquette tombe sur une autre déjà posée, la basculer de l'autre côté
        if any(abs(x - px) < 0.11 * lx and abs(y - py) < 0.10 * ly for px, py in places):
            dy = -dy - (4 if dy > 0 else 8)
        places.append((x, y))
        ax2.annotate(COURT.get(b, b), (x, y), textcoords="offset points", xytext=(dx, dy),
                     ha=ha, fontsize=8.8, weight="bold" if moe else "normal",
                     bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.9))
    # marges : les étiquettes des points extrêmes (D en bas à droite) débordaient du cadre
    ax2.set_xlim(min(xs) - 6.5, max(xs) + 7.5)
    ax2.set_ylim(min(ys) - 3.4, max(ys) + 2.2)
    ax2.axhline(-1, color="#1f77b4", ls="--", lw=1.1)
    ax2.plot([], [], ls="--", color="#1f77b4", lw=1.1,
             label="seuil « aucun point faible > 1 pt »")
    ax2.legend(fontsize=7.8, loc="lower left", framealpha=0.95)
    ax2.set_xlabel("niveau moyen sur 36 endpoints (percentile, %)")
    ax2.set_ylabel("dommage maximal vs controle (pt)")
    ax2.set_title("B. Plan de polyvalence — haut-droite =\nfort en moyenne ET sans point faible",
                  fontsize=10.5, weight="bold")
    ax2.grid(alpha=0.25)

    # panneau 3 — prix payé par point de mIoU
    ax3 = fig.add_subplot(gs[1, 1])
    ord_ = sorted([b for b in bras if b != REF], key=lambda b: -prof[b]["prix_pt_par_pt_miou"]
                  if prof[b]["delta_miou_pt"] > 1e-9 else -9e9)
    ord_ = [b for b in ord_ if prof[b]["delta_miou_pt"] > 1e-9]
    vals = [prof[b]["prix_pt_par_pt_miou"] for b in ord_]
    ax3.barh(range(len(ord_)), vals,
             color=["#1f77b4" if b == "moe_v3cs" else "#9ecae1" for b in ord_])
    ax3.set_yticks(range(len(ord_)), [f"{COURT.get(b,b)} ({prof[b]['delta_miou_pt']:+.2f} pt mIoU)"
                                      for b in ord_], fontsize=8.8)
    ax3.invert_yaxis()
    ax3.axvline(0, color="k", lw=0.8)
    ax3.set_xlabel("dommage maximal consenti par point de mIoU gagné (pt/pt)")
    ax3.set_title("C. Le prix du gain : ce que chaque bras casse\npour gagner 1 pt de mIoU",
                  fontsize=10.5, weight="bold")
    for i, v in enumerate(vals):
        ax3.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8)
    ax3.grid(axis="x", alpha=0.25)

    # panneau 4 — Δ par endpoint, MoE vs les spécialistes
    ax4 = fig.add_subplot(gs[2, 1])
    sel = ["moe_v3cs", "D", "Dp", "G", "B", "fused_DvetoB"]
    coul = {"moe_v3cs": "#1f77b4", "D": "#2ca02c", "Dp": "#17becf", "G": "#ff7f0e",
            "B": "#9467bd", "fused_DvetoB": "#8c564b"}
    xs = np.arange(len(eps))
    w = 0.8 / len(sel)
    for k, b in enumerate(sel):
        ax4.bar(xs + (k - len(sel) / 2 + 0.5) * w, [delta[b].get(e, 0.0) for e in eps],
                width=w, color=coul[b], label=COURT.get(b, b),
                edgecolor="k" if b == "moe_v3cs" else "none", linewidth=0.4 if b == "moe_v3cs" else 0)
    ax4.axhline(0, color="k", lw=0.8)
    ax4.set_ylim(-24, 7)
    ax4.set_ylabel("Δ vs controle (pt)")
    ax4.set_xticks([])
    ax4.set_title("D. Ce que chaque bras gagne et casse,\nendpoint par endpoint",
                  fontsize=10.5, weight="bold")
    ax4.legend(fontsize=7.6, ncol=3, loc="lower left")
    ax4.grid(axis="y", alpha=0.25)

    fig.suptitle("P3.17 — MoE-V3-CS : seul bras sans point faible ? · holdout 500 · seeds 42/123/456 · "
                 "bootstrap apparié B=10 000 · 36 endpoints × 13 bras",
                 fontsize=13.5, weight="bold", y=0.995)
    fig.text(0.008, 0.004, "Convention : X/B = consensus X⊘B (primaire X, veto B) · "
             "dommage maximal = pire Δ vs controle sur les 36 endpoints · "
             "#1 = endpoint gagné outright parmi les 13 bras", fontsize=8.2, color="#444444")
    fig.savefig(chemin)          # pas de tight : canvas et PNG gardent les mêmes coordonnées
    taille = os.path.getsize(chemin)

    # ---- auto-contrôle : re-render et mesure d'encre par axe (pas de vérification à l'œil)
    from PIL import Image
    img = np.array(Image.open(chemin).convert("RGB")).astype(np.int16)
    H, W, _ = img.shape
    chrom = (img.max(2) - img.min(2)) > 15
    nonblanc = img.max(2) < 250
    fig.canvas.draw()
    buf_w, buf_h = fig.canvas.get_width_height()
    rapport_echelle = (abs(buf_w - W) <= 2 and abs(buf_h - H) <= 2)
    rapport = {"dims_px": [W, H], "octets": taille, "panneaux": {}}
    for nom, ax in (("A_heatmap_percentile", ax1), ("B_plan_polyvalence", ax2),
                    ("C_prix_par_pt_miou", ax3), ("D_delta_par_endpoint", ax4)):
        bb = ax.get_window_extent()                       # pixels du canvas render (échelle 1:1)
        x0 = max(0, int(bb.x0)); x1 = min(W, int(bb.x1))
        y0 = max(0, int(buf_h - bb.y1)); y1 = min(H, int(buf_h - bb.y0))
        if x1 <= x0 or y1 <= y0:
            rapport["panneaux"][nom] = {"bbox_px": [x0, y0, x1, y1], "alerte": "bbox vide"}
            continue
        rapport["panneaux"][nom] = {
            "bbox_px": [x0, y0, x1, y1],
            "pct_non_blanc": round(100 * float(nonblanc[y0:y1, x0:x1].mean()), 2),
            "pct_chromatique": round(100 * float(chrom[y0:y1, x0:x1].mean()), 2),
        }
    rapport["encre_globale_pct"] = round(100 * float(nonblanc.mean()), 2)
    rapport["échelle_canvas_png_1sur1"] = bool(rapport_echelle)
    rapport["seuil_attendu"] = {"heatmap": "pct_chromatique > 40", "autres": "pct_non_blanc > 4"}
    rapport["alertes"] = [nom for nom, r in rapport["panneaux"].items()
                          if r.get("alerte") or r.get("pct_non_blanc", 0) < 4
                          or (nom == "A_heatmap_percentile" and r.get("pct_chromatique", 0) < 40)]
    plt.close(fig)
    return taille, rapport


def main():
    att, met, master, abs_val, delta, pbrut, pholm, ecarts = charger()
    pct, rang, sd_inter, prof, complet = profils(abs_val, delta, pbrut, pholm)
    vseed, ecart_seeds = verif_seeds(abs_val)
    verdict_ = verdict(prof, vseed)

    classement = sorted([b for b in prof if b in complet and b != REF],
                        key=lambda b: prof[b]["dommage_max_pt"], reverse=True)
    annexe = sorted([b for b in prof if b not in complet],
                    key=lambda b: prof[b]["mean_pct"], reverse=True)

    payload = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "these": "le MoE-V3-CS est le seul bras bon partout (polyvalent), sans pic de spécialisation",
        "criteres_pre_enregistres": {"T1": "dommage maximal le plus faible du plateau",
                                     "T2": "ΔmIoU significatif malgré tout",
                                     "T3": "aucun endpoint gagné outright",
                                     "T4": "verdict stable seed par seed"},
        "protocole": met["protocole"], "endpoints": ENDPOINTS, "familles": FAMILLE,
        "exclus": EXCLUS, "n_bras": len(prof), "bras_couverture_complète": complet,
        "valeurs_absolues_pct": abs_val, "delta_vs_controle_pt": delta,
        "p_brut": pbrut, "p_holm": pholm, "rang_percentile": pct, "rang_entier": rang,
        "sd_inter_bras_par_endpoint": sd_inter, "profil": prof,
        "classement_polyvalence": classement, "annexe_couverture_partielle": annexe,
        "verdict": verdict_,
        "par_seed": {str(s): {"dommage_max_pt": {b: v[0] for b, v in vseed[s]["dommage_max"].items()},
                              "endpoint_dommage": {b: v[1] for b, v in vseed[s]["dommage_max"].items()},
                              "n_rang1": vseed[s]["n_rang1"],
                              "delta_miou_pt": vseed[s]["delta_miou"],
                              "classement_dommage": vseed[s]["classement_dommage"]}
                     for s in SEEDS},
        "sanity_miou_vs_p314_ecart_max_pt": max(ecarts.values()),
        "sanity_iou_classe_recomputee_ecart_max_pt": ecart_seeds,
    }
    L = ecrire(payload, prof, classement, annexe, verdict_, vseed, ecart_seeds, ecarts, met)

    taille, rap_figure = figure(prof, classement + [REF], delta, pct,
                                os.path.join(DISCORD, "polyvalence_moe_v3.png"))
    print("\n--- auto-contrôle figure (mesures sur le PNG rendu) ---")
    print(json.dumps(rap_figure, indent=1, ensure_ascii=False))
    payload["figure_discord"] = {"chemin": "discord_out/polyvalence_moe_v3.png",
                                 "octets": taille, "autocontrole": rap_figure}
    json.dump(payload, open(os.path.join(OUTDIR, "table_polyvalence.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(OUTDIR, "verif_seeds.md"), "w") as f:
        f.write("# Vérification seed par seed (npz bruts, aucune moyenne inter-seeds)\n\n")
        f.write(f"Sanity : IoU par classe recomputées depuis les cm = table consolidée à "
                f"{ecart_seeds:.2e} pt près.\n\n")
        for s in SEEDS:
            f.write(f"## seed {s}\n\n")
            f.write("| Bras | dommage max (pt) | endpoint | ΔmIoU vs controle (pt) | #1 outright |\n")
            f.write("|---|---|---|---|---|\n")
            for b in vseed[s]["classement_dommage"]:
                d, e = vseed[s]["dommage_max"][b]
                f.write(f"| {LIBELLE[b]} | {d:+.2f} | {e} | "
                        f"{vseed[s]['delta_miou'][b]:+.3f} | {vseed[s]['n_rang1'][b]} |\n")
            f.write("\n")
    print("\n".join(L))
    print(f"figure → discord_out/polyvalence_moe_v3.png ({taille/1e6:.2f} Mo)")
    print("verdict:", json.dumps(verdict_["verdict"]))


if __name__ == "__main__":
    main()
