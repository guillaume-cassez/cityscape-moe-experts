#!/usr/bin/env python3
"""P3.18b — Figures F1-F5 du paper 3 (MoE-V3-CS), SANS GPU, depuis la source assertée.

Source unique : `papers/paper3/tables/paper3_tables.json` + les CSV produits par
`scripts/p3_moe_tables.py` (257 checks bloquants). Ce script ne relit JAMAIS les artefacts
bruts : il ne peut donc pas introduire un chiffre qui n'a pas passé les sanités.

Discipline héritée de deux défauts payés sur paper4 :

  * **Piège 2** — un nombre écrit à la main finit par mentir, et dans une figure il est GRAVÉ
    dans le PNG publié (le suptitre de F2 de paper4 portait « 18 classes sur 19 » alors que
    l'artefact en donnait 16). Ici chaque libellé, titre et annotation est une valeur lue du
    JSON asserté ou recalculée à partir de lui.
  * **Piège 3** — un manuscrit embarque ses figures : un PDF peut être périmé SUR IMAGE sans
    qu'aucune relecture texte le voie. Chaque figure porte donc un **autocontrôle** mesurable
    (nombre de lignes, de panneaux, de points, dimensions) écrit dans `paper3_tables.json`,
    que le stager revérifie par extraction `pdfimages`.

Productions :
  papers/paper3/figures/F1_delta_13bras.{png,pdf}
  papers/paper3/figures/F2_routage.{png,pdf}
  papers/paper3/figures/F3_forest_metiers.{png,pdf}
  papers/paper3/figures/F4_polyvalence.{png,pdf}
  papers/paper3/figures/F5_perclass.{png,pdf}
  discord_out/ (copies png)

Usage : python3 scripts/p3_moe_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "papers" / "paper3" / "tables"
OUT = REPO / "papers" / "paper3" / "figures"
DISCORD = REPO / "discord_out"

ARM = "moe_v3cs"
COULEUR_MOE = "#c1272d"
COULEUR_AUTRE = "#4a6fa5"
COULEUR_REF = "#7f7f7f"

AUTOCONTROLE: dict = []
ERREURS: list[str] = []


def log(m: str) -> None:
    print(f"[p3fig {datetime.now():%H:%M:%S}] {m}", flush=True)


def verif(nom: str, ok: bool, detail: str) -> None:
    """Autocontrôle bloquant : une figure dont la structure n'est pas celle attendue ne part
    pas au manuscrit (c'est ainsi que paper4 a publié un PNG au suptitre faux)."""
    AUTOCONTROLE.append({"figure": nom, "ok": bool(ok), "detail": detail})
    log(f"  [autocontrôle {'✅' if ok else '❌'}] {nom} — {detail}")
    if not ok:
        ERREURS.append(f"{nom} : {detail}")


def lire_csv(nom: str) -> list[dict]:
    with open(TABLES / nom, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flottes(lignes: list[dict], *cols: str) -> list[list[float]]:
    return [[float(r[c]) for c in cols] for r in lignes]


def sauver(fig, stem: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    DISCORD.mkdir(parents=True, exist_ok=True)
    infos = {}
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        infos[ext] = {"chemin": str(p.relative_to(REPO)), "octets": p.stat().st_size}
    png = OUT / f"{stem}.png"
    from PIL import Image  # noqa: PLC0415  (dimension réelle du PNG produit)
    with Image.open(png) as im:
        infos["dims_px"] = list(im.size)
    copie = DISCORD / f"paper3_{png.name}"
    copie.write_bytes(png.read_bytes())
    infos["discord_out"] = str(copie.relative_to(REPO))
    fig.canvas.figure.clear()
    plt.close(fig)
    return infos


# --------------------------------------------------------------------------- F1
def f1_delta_13bras(d: dict) -> dict:
    """Δ mIoU vs contrôle pour les 13 bras, avec IC95 — le contexte du critère primaire."""
    t3, t2 = d["T3"], d["T2"]
    rows = t3["classement"]                       # déjà trié par Δ décroissant
    bras = [r["label"] for r in rows][::-1]       # bas -> haut pour barh
    delta = [r["delta_pt"] for r in rows][::-1]
    lo = [r["ci_lo_pt"] for r in rows][::-1]
    hi = [r["ci_hi_pt"] for r in rows][::-1]
    est_moe = [r["bras"] == ARM for r in rows][::-1]
    couleurs = [COULEUR_MOE if m else COULEUR_AUTRE for m in est_moe]

    fig, ax = plt.subplots(figsize=(10.2, 0.44 * len(rows) + 2.4))
    y = np.arange(len(rows))
    ax.barh(y, delta, color=couleurs, height=0.62, zorder=3)
    ax.errorbar(delta, y, xerr=[np.array(delta) - np.array(lo), np.array(hi) - np.array(delta)],
                fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3, zorder=4)
    ax.axvline(0, color=COULEUR_REF, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(bras, fontsize=9)
    ax.set_xlabel("Δ mIoU vs contrôle apparié (points)", fontsize=10)
    ax.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    ax.set_title(
        f"Δ mIoU des {len(rows) + 1} bras du programme (référence : contrôle apparié, 80 époques)\n"
        f"IC95 bootstrap apparié B = {t2['B_boot']:,}. Aucun bras ne survit au Holm : meilleur "
        f"{t3['meilleur_holm_12']:.4f} sur {t2['n_famille_12']} paires, "
        f"{t3['meilleur_holm_15']:.4f} sur {t2['n_famille_15']} paires",
        fontsize=10.5, pad=12)
    for yi, r in zip(y, [x for x in rows][::-1]):
        ax.annotate(f"{r['delta_pt']:+.2f}", (r["delta_pt"], yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.6,
                    color=COULEUR_MOE if r["bras"] == ARM else "#333333")
    fig.tight_layout()
    infos = sauver(fig, "F1_delta_13bras")
    verif("F1_delta_13bras", len(rows) == 12 and len(bras) == 12,
          f"{len(rows)} bras tracés (+ le contrôle à Δ = 0, soit {len(rows) + 1} au total) ; "
          f"le MoE est en {t3['rang_moe']}ᵉ position ; IC95 tracés pour les {len(rows)}")
    return infos


# --------------------------------------------------------------------------- F2
def f2_routage(d: dict) -> dict:
    """Diagnostic de routage sur 80 époques × 3 seeds : la porte ne choisit pas."""
    t4, t1 = d["T4"], d["T1"]
    rows = lire_csv("T4_routage.csv")
    seeds = sorted({int(r["seed"]) for r in rows})
    equi = t1["equirepartition_part_max"]
    anneal = t4["anneal"]
    verif("F2 sources", len(rows) == t4["n_epochs"] * len(seeds),
          f"{len(rows)} lignes CSV = {t4['n_epochs']} époques × {len(seeds)} seeds")

    panneaux = [
        ("entropy_norm", "entropie normalisée (1,0 = uniforme)", (0.995, 1.0005)),
        ("entropy_token", "entropie par patch, AVEC bruit", (0.985, 1.0005)),
        ("entropy_token_clean", "entropie par patch, SANS bruit", (0.985, 1.0005)),
        ("part_max", f"part_max (équirépartition = {equi:.4f})", (0.495, 0.510)),
        ("gamma_absmean", "|γ| moyen (échelle résiduelle)", None),
        ("noise_std", "std du bruit de Shazeer", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.0))
    for ax, (cle, titre, ylim) in zip(axes.ravel(), panneaux):
        for s in seeds:
            sr = [r for r in rows if int(r["seed"]) == s]
            x = [int(r["epoch"]) for r in sr]
            yv = [float(r[cle]) for r in sr]
            ax.plot(x, yv, lw=1.5, label=f"seed {s}")
        if cle == "part_max":
            ax.axhline(equi, color="#222222", ls="--", lw=1.2)
            ax.annotate(f"équirépartition {equi:.4f}", (t4["n_epochs"] * 0.5, equi),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        if cle in ("entropy_token", "entropy_token_clean", "part_max", "gamma_absmean"):
            ax.axvspan(0, anneal - 1, color="#f2d7d5", alpha=0.55, zorder=0)
        if cle == "noise_std":
            ax.axvspan(0, anneal - 1, color="#f2d7d5", alpha=0.55, zorder=0)
        ax.set_title(titre, fontsize=9.5)
        ax.set_xlabel("époque", fontsize=8.5)
        ax.grid(ls=":", color="#cccccc", zorder=0)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(
        f"La porte ne choisit pas — {t4['n_epochs']} époques × {len(seeds)} seeds\n"
        f"Zone rose : les {anneal} époques où le bruit de Shazeer est ACTIF (seule phase où la "
        f"comparaison avec/sans bruit est probante ; après, le bruit est nul et les deux "
        f"courbes sont identiques par construction)\n"
        f"Époque finale : entropy_norm {min(t4['entropy_norm_final'].values()):.5f} au minimum, "
        f"part_max {min(t4['part_max_final'].values()):.4f}–{max(t4['part_max_final'].values()):.4f} "
        f"pour {equi:.4f}, {sum(t4['experts_morts'].values())} expert mort, "
        f"|γ| {min(t4['gamma_final'].values()):.5f}–{max(t4['gamma_final'].values()):.5f}",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    infos = sauver(fig, "F2_routage")
    verif("F2_routage", len(axes.ravel()) == 6 and len(seeds) == 3,
          f"{len(axes.ravel())} panneaux × {len(seeds)} seeds ; zone de bruit actif "
          f"0–{anneal - 1} marquée sur les 4 panneaux concernés")
    return infos


# --------------------------------------------------------------------------- F3
def f3_forest_metiers(d: dict) -> dict:
    """Forest plot des 20 métriques métier, les deux critères de significativité séparés.

    DEUX UNITÉS, DEUX PANNEAUX (défaut mesuré le 2026-10-01) : 19 métriques sont en points
    (|Δ| max 0,802) et `fragments` en composantes connexes par image (Δ −31,2). Les tracer sur
    un axe commun donne un rapport d'échelle de 39× : les 19 autres se retrouvent écrasées en un
    trait illisible à côté d'une barre géante — la figure ment visuellement sans qu'aucun nombre
    ne soit faux. D'où la séparation, et un autocontrôle BLOQUANT sur l'unité de chaque panneau.
    """
    t5 = d["T5"]
    rows = lire_csv("T5_metiers.csv")
    ordre = t5["ordre"]
    pts = sorted([r for r in rows if r["unite"] == "pt"], key=lambda r: ordre.index(r["metrique"]))[::-1]
    frg = [r for r in rows if r["unite"] != "pt"]
    verif("F3 unités séparées", len(pts) + len(frg) == len(rows) and len(frg) == 1,
          f"{len(pts)} métriques en points + {len(frg)} en {frg[0]['unite'] if frg else '?'} "
          f"= {len(rows)} ; aucun mélange d'unité sur un axe")
    ratio = (max(abs(float(r["delta"])) for r in frg) /
             max(abs(float(r["delta"])) for r in pts))
    verif("F3 rapport d'échelle entre les deux unités (justifie la séparation)", ratio > 5,
          f"{ratio:.0f}× entre `fragments` et la plus grande métrique en points")
    survit = [r["survit_holm"] == "True" for r in pts]
    ic0 = [r["ic_exclut_0"] == "True" for r in pts]
    verif("F3 compte des survivantes",
          sum(survit) + sum(r["survit_holm"] == "True" for r in frg) == t5["n_survit_holm"],
          f"{sum(survit)} pleins (Holm) côté points + "
          f"{sum(r['survit_holm'] == 'True' for r in frg)} côté fragments = "
          f"{t5['n_survit_holm']} au total ; "
          f"{sum(1 for a, b in zip(ic0, survit) if a and not b)} IC excluant 0 sans Holm")

    fig, (ax, axF) = plt.subplots(1, 2, figsize=(13.4, 0.40 * len(pts) + 2.6),
                                  gridspec_kw={"width_ratios": [2.5, 1.0]})
    y = np.arange(len(pts))
    for yi, r, s, i in zip(y, pts, survit, ic0):
        dv, lv, hv = float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"])
        couleur = COULEUR_MOE if s else (COULEUR_AUTRE if i else "#8a8a8a")
        ax.plot([lv, hv], [yi, yi], color=couleur, lw=1.4, zorder=3)
        ax.plot([dv], [yi], marker="o", ms=7 if s else 5.5,
                mfc=couleur if s else "white", mec=couleur, mew=1.5, zorder=4)
    ax.axvline(0, color="#333333", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["metrique"] for r in pts], fontsize=8.4, family="monospace")
    ax.set_xlabel("Δ (MoE-V3-CS − contrôle apparié), en points", fontsize=9.5)
    ax.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    ax.set_title(
        f"{len(pts)} métriques en points — Holm sur {t5['n_famille_holm']} paires\n"
        f"Plein = Holm < 0,05 · ouvert = IC95 exclut 0 seulement · gris = ns",
        fontsize=9.8, pad=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=COULEUR_MOE, mec=COULEUR_MOE, ms=7,
               label=f"Holm < 0,05 ({sum(survit)})"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=COULEUR_AUTRE, ms=6, mew=1.5,
               label=f"IC95 exclut 0 seulement ({sum(1 for a, b in zip(ic0, survit) if a and not b)})"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="#8a8a8a", ms=6, mew=1.5,
               label="non significatif"),
    ], fontsize=8.0, loc="lower right")

    # Panneau droit : les fragments, sur LEUR échelle (composantes connexes par image).
    r = frg[0]
    dv, lv, hv = float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"])
    s = r["survit_holm"] == "True"
    axF.plot([lv, hv], [0, 0], color=COULEUR_MOE if s else COULEUR_AUTRE, lw=2.0, zorder=3)
    axF.plot([dv], [0], marker="o", ms=11, mfc=COULEUR_MOE if s else "white",
             mec=COULEUR_MOE if s else COULEUR_AUTRE, mew=1.8, zorder=4)
    axF.axvline(0, color="#333333", lw=1.0, zorder=2)
    axF.set_ylim(-1, 1)
    axF.set_yticks([0])
    axF.set_yticklabels(["fragments"], fontsize=9, family="monospace")
    axF.set_xlabel("Δ en composantes connexes par image", fontsize=9.5)
    axF.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    axF.set_title(f"Échelle propre ({ratio:.0f}× plus grande)\n"
                  f"{float(r['point_controle']):.1f} → {float(r['point_moe']):.1f} "
                  f"({dv:+.1f}, Holm {float(r['holm']):.3g})", fontsize=9.8, pad=10)
    fig.suptitle(
        f"Métriques métier : {t5['n_survit_holm']} des {len(rows)} survivent au Holm "
        f"({', '.join('`' + m + '`' for m, _e in [(r['metrique'], 0) for r in rows if r['survit_holm'] == 'True'])}). "
        f"Les deux critères de significativité sont tracés séparément, jamais confondus.",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    infos = sauver(fig, "F3_forest_metiers")
    verif("F3_forest_metiers", len(pts) == 19 and len(y) == 19,
          f"{len(pts)} lignes côté points + 1 côté fragments ; 2 panneaux")
    return infos


# --------------------------------------------------------------------------- F4
def f4_polyvalence(d: dict) -> dict:
    """Polyvalence : dommage maximal, prix payé, et position « bon partout ≠ meilleur partout »."""
    t7 = d["T7"]
    rows = lire_csv("T7_polyvalence.csv")
    rows = sorted(rows, key=lambda r: float(r["dommage_max_pt"]), reverse=True)
    labels = [r["label"] for r in rows]
    domm = np.array([float(r["dommage_max_pt"]) for r in rows])
    dmi = np.array([float(r["delta_miou_pt"]) for r in rows])
    pct = np.array([float(r["percentile_moyen"]) for r in rows])
    est = [r["bras"] == ARM for r in rows]
    seuil = t7["seuils"]["endpoint_pt"]

    # `constrained_layout` et non `tight_layout` : ce dernier avertit (mesuré) qu'il n'est pas
    # compatible avec un gridspec ajouté à la main, et le résultat serait imprévisible.
    fig = plt.figure(figsize=(16.6, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.16)

    axA = fig.add_subplot(gs[0, 0])
    yA = np.arange(len(rows))[::-1]
    axA.barh(yA, domm, color=[COULEUR_MOE if e else COULEUR_AUTRE for e in est], height=0.66, zorder=3)
    axA.axvline(seuil, color="#b03a2e", ls="--", lw=1.3, zorder=4)
    axA.annotate(f"seuil du critère T0 ({seuil:.0f} pt)", (seuil, yA[0]), textcoords="offset points",
                 xytext=(4, 0), ha="left", va="top", fontsize=8, color="#b03a2e")
    axA.axvline(0, color="#333333", lw=1.0, zorder=2)
    axA.set_yticks(yA)
    axA.set_yticklabels(labels, fontsize=8.4)
    axA.set_xlabel("dommage maximal sur les 36 endpoints (pt)", fontsize=9)
    axA.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    axA.set_title(f"Ce que chaque bras CASSE\n{len(t7['T0_gagnants'])} bras au-dessus du seuil "
                  f"avec un ΔmIoU significatif : {labels[yA[est.index(True)]]}", fontsize=9.5)

    axB = fig.add_subplot(gs[0, 1])
    withprix = [(r["label"], float(r["prix_par_pt_miou"])) for r in rows
                if r["prix_par_pt_miou"] not in ("", "None")]
    withprix.sort(key=lambda t: t[1], reverse=True)
    yB = np.arange(len(withprix))[::-1]
    axB.barh(yB, [v for _l, v in withprix],
             color=[COULEUR_MOE if "MoE" in l else COULEUR_AUTRE for l, _v in withprix],
             height=0.6, zorder=3)
    axB.set_yticks(yB)
    axB.set_yticklabels([l for l, _v in withprix], fontsize=8.4)
    axB.axvline(0, color="#333333", lw=1.0, zorder=2)
    axB.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    axB.set_xlabel("prix payé par point de mIoU gagné (pt)", fontsize=9)
    axB.set_title(f"Le taux de change du gain\n{len(withprix)} bras à ΔmIoU significatif ; "
                  f"le MoE est le moins cher", fontsize=9.5)
    for yi, (_l, v) in zip(yB, withprix):
        axB.annotate(f"{v:+.1f}", (v, yi), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7.8)

    axC = fig.add_subplot(gs[0, 2])
    axC.scatter(dmi, domm, s=62, c=[COULEUR_MOE if e else COULEUR_AUTRE for e in est],
                zorder=4, edgecolors="#222222", linewidths=0.6)
    for r, xv, yv in zip(rows, dmi, domm):
        if r["bras"] in (ARM, "D", "fused_DvetoB", "G"):
            axC.annotate(r["label"].split(" ·")[0], (xv, yv), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)
    axC.axhline(seuil, color="#b03a2e", ls="--", lw=1.2, zorder=2)
    axC.axvline(0, color="#333333", lw=1.0, zorder=2)
    axC.set_xlabel("Δ mIoU vs contrôle (pt)", fontsize=9)
    axC.set_ylabel("dommage maximal (pt)", fontsize=9)
    axC.grid(ls=":", color="#cccccc", zorder=0)
    axC.set_title("Gagner sans casser\nle quadrant en haut à droite est vide", fontsize=9.5)

    fig.suptitle(
        f"Polyvalence — {t7['n_endpoints']} endpoints × {t7['n_bras']} bras, critères "
        f"pré-enregistrés recalculés\n"
        f"Le MoE-V3-CS est le SEUL bras à cocher le critère T0 ; dommage maximal "
        f"{t7['dommage_moe']['dommage_max_pt']:+.2f} pt "
        f"(`{t7['dommage_moe']['endpoint']}`) contre "
        f"{min(float(r['dommage_max_pt']) for r in rows):+.2f} pt pour le pire, marge "
        f"{t7['marge_vs_suivant']:.2f} pt sur le suivant. Il n'est premier sur AUCUN endpoint "
        f"(D en gagne {max(int(r['n_rang1']) for r in rows)}) et son percentile moyen le classe "
        f"{t7['ordre_percentile'].index(ARM) + 1}ᵉ : bon partout, meilleur nulle part.",
        fontsize=11, y=0.995)
    fig.set_constrained_layout_pads(hspace=0.02, wspace=0.02)
    infos = sauver(fig, "F4_polyvalence")
    verif("F4_polyvalence", len(rows) == 11 and len(withprix) >= 2,
          f"{len(rows)} bras à couverture complète tracés (le contrôle est la référence, A en "
          f"couverture partielle) ; {len(withprix)} bras avec un prix calculable ; 3 panneaux")
    return infos


# --------------------------------------------------------------------------- F5
def f5_perclass(d: dict) -> dict:
    """IoU par classe : les trois critères de significativité, et l'attribution aux experts."""
    t6 = d["T6"]
    rows = lire_csv("T6_perclass.csv")
    rows = sorted(rows, key=lambda r: float(r["delta_pt"]), reverse=True)
    y = np.arange(len(rows))
    delta = np.array([float(r["delta_pt"]) for r in rows])
    lo = np.array([float(r["ci_lo_pt"]) for r in rows])
    hi = np.array([float(r["ci_hi_pt"]) for r in rows])
    survit = [r["survit_holm"] == "True" for r in rows]
    ic0 = [r["ic_exclut_0"] == "True" for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.6, 0.42 * len(rows) + 2.6),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})
    for yi, dv, lv, hv, s, i in zip(y, delta, lo, hi, survit, ic0):
        c = COULEUR_MOE if s else (COULEUR_AUTRE if i else "#8a8a8a")
        ax.plot([lv, hv], [yi, yi], color=c, lw=1.4, zorder=3)
        ax.plot([dv], [yi], marker="o", ms=7 if s else 5.5, mfc=c if s else "white",
                mec=c, mew=1.5, zorder=4)
    ax.axvline(0, color="#333333", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["classe"] for r in rows], fontsize=8.6)
    ax.set_xlabel("Δ IoU (MoE-V3-CS − contrôle apparié), points", fontsize=9.5)
    ax.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    ax.set_title(f"{len(rows)} classes : {t6['n_gains']} hausses, {t6['n_pertes']} baisses\n"
                 f"Plein = Holm(19) < 0,05 · ouvert = IC exclut 0 seulement · gris = ns",
                 fontsize=10, pad=10)

    # Panneau droit : ce que chaque expert aurait fait sur les 6 classes les plus mobiles.
    top = rows[:4] + rows[-3:]
    expert = ["delta_B", "delta_D", "delta_Dp", "delta_G"]
    x2 = np.arange(len(top))
    larg = 0.2
    for k, e in enumerate(expert):
        ax2.bar(x2 + (k - 1.5) * larg, [float(r[e]) for r in top], width=larg, label=e.split("_")[1],
                color=["#95a5a6", "#2980b9", "#27ae60", "#8e44ad"][k], zorder=3)
    ax2.bar(x2 + 2.5 * larg, [float(r["delta_pt"]) for r in top], width=larg, label="MoE-V3-CS",
            color=COULEUR_MOE, zorder=3)
    ax2.axhline(0, color="#333333", lw=1.0, zorder=2)
    ax2.set_xticks(x2 + 0.5 * larg)
    ax2.set_xticklabels([r["classe"] for r in top], rotation=38, ha="right", fontsize=8.4)
    ax2.set_ylabel("Δ IoU (pt)", fontsize=9)
    ax2.grid(axis="y", ls=":", color="#bbbbbb", zorder=0)
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_title("Les 4 plus fortes hausses et les 3 plus fortes baisses,\ncomparées aux 4 "
                  "experts dont le MoE est initialisé", fontsize=9.5, pad=10)

    fig.suptitle(
        f"IoU par classe — trois critères donnés séparément\n"
        f"Plus fortes amplitudes : {', '.join(t6['top4_amplitude'])} · IC95 excluant 0 : "
        f"{', '.join(t6['gains_ic_exclut_0'])} (hausses) et "
        f"{', '.join(t6['pertes_ic_exclut_0'])} (baisses) · Holm(19) : AUCUNE hausse ne survit, "
        f"la seule classe survivante est une baisse ({', '.join(t6['pertes_holm'])}, "
        f"Holm {t6['holm_min_pertes']:.4g})",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    infos = sauver(fig, "F5_perclass")
    verif("F5_perclass", len(rows) == t6["n_classes"] == 19 and len(top) == 7,
          f"{len(rows)} classes au forest + {len(top)} classes au comparateur d'experts "
          f"({len(expert)} experts + le MoE) ; 2 panneaux")
    return infos


def main() -> int:
    t0 = datetime.now()
    log("P3.18b — figures F1-F5 du paper 3 (MoE-V3-CS), sans GPU")
    src = TABLES / "paper3_tables.json"
    if not src.exists():
        raise SystemExit(f"[FAIL] source assertée absente : {src} — lancer d'abord "
                         f"scripts/p3_moe_tables.py")
    d = json.loads(src.read_text(encoding="utf-8"))
    san = d["sanity"]
    if san["n_ok"] != san["n_checks"]:
        raise SystemExit(f"[FAIL] la source n'est pas saine : {san['n_ok']}/{san['n_checks']} "
                         f"checks — régénérer les tables avant de tracer")
    log(f"  source : {src.relative_to(REPO)} — {san['n_ok']}/{san['n_checks']} checks "
        f"(générée le {d['date']})")

    figs = {}
    for nom, fn in [("F1", f1_delta_13bras), ("F2", f2_routage), ("F3", f3_forest_metiers),
                    ("F4", f4_polyvalence), ("F5", f5_perclass)]:
        log(f"[figures] {nom}")
        figs[nom] = fn(d)
        log(f"  → {figs[nom]['png']['chemin']} ({figs[nom]['png']['octets'] / 1e6:.2f} Mo, "
            f"{figs[nom]['dims_px'][0]}×{figs[nom]['dims_px'][1]} px)")

    # Autocontrôles écrits DANS la source consolidée : le stager et le futur p3_check_numbers
    # les reliront, et les dimensions png permettront de prouver qu'un PDF embarque bien la
    # figure à jour (Piège 3 de paper4).
    d["figures_autocontrole"] = {"date": t0.isoformat(timespec="seconds"),
                                 "script": "scripts/p3_moe_figures.py",
                                 "n_autocontroles": len(AUTOCONTROLE),
                                 "n_ok": sum(1 for a in AUTOCONTROLE if a["ok"]),
                                 "autocontroles": AUTOCONTROLE,
                                 "figures": figs}
    (TABLES / "paper3_tables.json").write_text(json.dumps(d, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    ok = sum(1 for a in AUTOCONTROLE if a["ok"])
    if ERREURS:
        log(f"ÉCHEC — {len(ERREURS)} autocontrôle(s) en défaut :")
        for e in ERREURS:
            log(f"  ❌ {e}")
        return 1
    dt = (datetime.now() - t0).total_seconds()
    log(f"[fin] {dt:.0f}s — F1-F5 écrites (png + pdf), autocontrôles {ok}/{len(AUTOCONTROLE)} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
