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

# --------------------------------------------------------------------------- #
# TAILLE IMPRIMÉE — récidive du 2026-10-04 (5ᵉ passage sur la mise en page)
#
# Symptôme relevé : « du texte écrit trop petit en haut de la page 11 du
# papier 3 ». Mesure sur le PDF livré (paper_fr.pdf md5 c6b4ed03…) : la figure
# du haut de la page 11 est F5_perclass_fr.png, 2635 px à 170 dpi = 15,50 in
# naturelles, placée par \includegraphics à 515,6 pt = 7,16 in → ÉCHELLE
# 0,462 ; ses polices 8,4-10,5 pt du code s'imprimaient à 3,9-4,9 pt sur un
# corps de texte de 9,96 pt. Même cause sur les 5 figures : échelles mesurées
# 0,426 (F4) à 0,710 (F1), 275 textes imprimés sous 6,5 pt, minimum 3,38 pt.
#
# CAUSE : les figures étaient dessinées à 10-17 in de large, soit ~2× la
# largeur du bloc de texte, puis réduites à \linewidth. Tous les fontsize du
# code étaient donc multipliés par 0,43-0,71 à l'impression — et aucun gate ne
# mesurait ce facteur (verifier_figure ne compare que des pixels display : une
# figure réduite uniformément ne se chevauche pas davantage, elle devient
# juste illisible).
#
# CORRECTIF (changement d'approche, pas de dosage) : TOUTES les figures sont
# désormais dessinées À LA LARGEUR IMPRIMÉE (L = 7,16 in = 515,6 pt =
# \linewidth pour letter + marges 1,7 cm). L'échelle de placement vaut alors
# 1,000 et chaque fontsize du code EST la taille réelle sur le papier. Les
# quatre tailles ci-dessous sont des points IMPRIMÉS ; le gate
# verifier_taille_imprimee() (fig_overlap_gate) re-mesure l'échelle effective
# sur la tight bbox et bloque sous POLICE_MIN_IMPRIMEE_PT.
from fig_overlap_gate import LARGEUR_IMPRIMEE_IN as L  # noqa: E402

FZ_TICK = 7.3      # ticks, légendes, annotations courtes (plancher 6,5 pt
                   # IMPRIMÉS : 6,8 laissait la marge quand l'encre + le pad de
                   # savefig font placer la figure à 0,98 plutôt qu'à 1,000 ;
                   # remonté à 7,3 le 2026-10-04 19h : « pas bien lisibles »)
FZ_LAB = 7.6       # xlabel / ylabel
FZ_TITLE = 8.2     # titres de panneaux
FZ_SUPT = 8.8      # suptitle


def label_moyen(label: str) -> str:
    """Libellé de bras sans sa parenthèse explicative : « B · CE+Dice
    (baseline) » → « B · CE+Dice », « MoE-V3-CS (4 experts) » → « MoE-V3-CS ».

    Pourquoi (mesure du 2026-10-04) : F4_polyvalence est la figure la plus
    dense du papier — 3 panneaux de ~2,1 in une fois dessinée à la largeur
    imprimée. Ses tick labels y portaient le libellé LONG (jusqu'à
    « G · CE+Dice+Blob (Kofler) », 25 caractères = 1,14 in à 6,6 pt) et
    constrained_layout s'effondrait : « axes sizes collapsed to zero »,
    panneaux réduits à 42 × 40 px, titres recoupés à un mot par ligne,
    tick labels empilés les uns sur les autres (6 collisions d'encre
    mesurées). La parenthèse est une redite du tableau T5 et de F1 (page 5,
    qui porte les 13 libellés complets) ; la retirer rend 0,3 à 0,5 in par
    panneau sans retirer aucun chiffre. Aucun autre texte n'est modifié."""
    return label.split(" (")[0].strip()

AUTOCONTROLE: list = []
ERREURS: list[str] = []

# --------------------------------------------------------------------------- i18n
# Défaut n°3 du contrôle visuel du 2026-10-01 : les figures embarquaient titres,
# suptitres, axes et légendes en FRANÇAIS y compris dans le manuscrit ANGLAIS. La
# cause est un libellé codé en dur dans chaque appel set_title/set_xlabel/… Ici le
# CHROME de la figure (tout ce qui est prose d'auteur) passe par T() qui choisit la
# langue courante ; les libellés PILOTÉS PAR LES DONNÉES (noms de classes Cityscapes,
# identifiants de métriques snake_case, labels de bras) restent tels quels : ce sont
# des identifiants que le manuscrit anglais affiche déjà verbatim dans ses tables
# (`rappel_strict_instances`, « consensus D⊘B »…), les traduire ici créerait un
# décalage figure↔table. Chaque gabarit porte les MÊMES placeholders numériques
# (passés depuis les expressions d'origine) : la valeur gravée est identique en
# FR et EN, seule la prose change — c'est la discipline anti-Piège-2 du docstring.
LANG = "fr"


def set_lang(lang: str) -> None:
    global LANG
    if lang not in ("fr", "en"):
        raise ValueError(f"langue inconnue : {lang}")
    LANG = lang


TR: dict[str, dict[str, str]] = {
    "F1.xlabel": {"fr": "Δ mIoU vs contrôle apparié (points)",
                  "en": "Δ mIoU vs matched control (points)"},
    "F1.title": {"fr": "Δ mIoU des {n} bras du programme (référence : contrôle apparié, 80 époques)\n"
                       "IC95 bootstrap apparié B = {B}. Aucun bras ne survit au Holm : meilleur "
                       "{m12} sur {nf12} paires, {m15} sur {nf15} paires",
                 "en": "Δ mIoU of the {n} program arms (reference: matched control, 80 epochs)\n"
                       "Paired bootstrap 95% CI, B = {B}. No arm survives Holm: best "
                       "{m12} over {nf12} pairs, {m15} over {nf15} pairs"},
    "F2.p.entropy_norm": {"fr": "entropie normalisée (1,0 = uniforme)",
                          "en": "normalised entropy (1.0 = uniform)"},
    "F2.p.entropy_token": {"fr": "entropie par patch, AVEC bruit",
                           "en": "per-patch entropy, WITH noise"},
    "F2.p.entropy_token_clean": {"fr": "entropie par patch, SANS bruit",
                                 "en": "per-patch entropy, WITHOUT noise"},
    "F2.p.part_max": {"fr": "part_max (équirépartition = {equi})",
                      "en": "part_max (equidistribution = {equi})"},
    "F2.p.gamma_absmean": {"fr": "|γ| moyen (échelle résiduelle)",
                           "en": "mean |γ| (residual scale)"},
    "F2.p.noise_std": {"fr": "std du bruit de Shazeer",
                       "en": "Shazeer noise std"},
    "F2.annotate_equi": {"fr": "équirépartition {equi}", "en": "equidistribution {equi}"},
    "F2.xlabel": {"fr": "époque", "en": "epoch"},
    "F2.suptitle": {"fr": "La porte ne choisit pas — {n_ep} époques × {n_seed} seeds\n"
                          "Zone rose : les {anneal} époques où le bruit de Shazeer est ACTIF "
                          "(seule phase où la comparaison avec/sans bruit est probante ; après, "
                          "le bruit est nul et les deux courbes sont identiques par construction)\n"
                          "Époque finale : entropy_norm {en_min} au minimum, part_max {pm_lo}–{pm_hi} "
                          "pour {equi}, {n_dead} expert mort, |γ| {g_lo}–{g_hi}",
                    "en": "The gate does not choose — {n_ep} epochs × {n_seed} seeds\n"
                          "Pink band: the {anneal} epochs where the Shazeer noise is ACTIVE "
                          "(the only phase where the with/without-noise comparison is probative; "
                          "afterwards the noise is zero and the two curves are identical by "
                          "construction)\n"
                          "Final epoch: entropy_norm {en_min} at the minimum, part_max {pm_lo}–{pm_hi} "
                          "for {equi}, {n_dead} dead expert, |γ| {g_lo}–{g_hi}"},
    "F3.xlabel": {"fr": "Δ (MoE-V3-CS − contrôle apparié), en points",
                  "en": "Δ (MoE-V3-CS − matched control), in points"},
    "F3.title": {"fr": "{npts} métriques en points — Holm sur {nfam} paires\n"
                       "Plein = Holm < 0,05 · ouvert = IC95 exclut 0 seulement · gris = ns",
                 "en": "{npts} metrics in points — Holm over {nfam} pairs\n"
                       "filled = Holm < 0.05 · open = 95% CI excludes 0 only · grey = ns"},
    "F3.legend_holm": {"fr": "Holm < 0,05 ({n})", "en": "Holm < 0.05 ({n})"},
    "F3.legend_ic": {"fr": "IC95 exclut 0 seulement ({n})", "en": "95% CI excludes 0 only ({n})"},
    "F3.legend_ns": {"fr": "non significatif", "en": "not significant"},
    "F3.axF.xlabel": {"fr": "Δ en composantes connexes par image",
                      "en": "Δ in connected components per image"},
    "F3.axF.title": {"fr": "Échelle propre ({ratio}× plus grande)\n{pc} → {pm} ({dv}, Holm {holm})",
                     "en": "Own scale ({ratio}× larger)\n{pc} → {pm} ({dv}, Holm {holm})"},
    "F3.suptitle": {"fr": "Métriques métier : {nsurv} des {nrows} survivent au Holm ({liste}). "
                          "Les deux critères de significativité sont tracés séparément, jamais confondus.",
                    "en": "Business metrics: {nsurv} of {nrows} survive Holm ({liste}). "
                          "The two significance criteria are plotted separately, never conflated."},
    "F4.annotate_seuil": {"fr": "seuil du critère T0 ({seuil} pt)",
                          "en": "T0-criterion threshold ({seuil} pt)"},
    "F4.axA.xlabel": {"fr": "dommage maximal sur les 36 endpoints (pt)",
                      "en": "maximal damage over the 36 endpoints (pt)"},
    "F4.axA.title": {"fr": "Ce que chaque bras CASSE\n{ng} bras au-dessus du seuil avec un ΔmIoU "
                           "significatif : {lab}",
                     "en": "What each arm BREAKS\n{ng} arms above the threshold with a significant "
                           "ΔmIoU: {lab}"},
    "F4.axB.xlabel": {"fr": "prix payé par point de mIoU gagné (pt)",
                      "en": "price paid per mIoU point gained (pt)"},
    "F4.axB.title": {"fr": "Le taux de change du gain\n{np} bras à ΔmIoU significatif ; le MoE est "
                           "le moins cher",
                     "en": "The exchange rate of the gain\n{np} arms with significant ΔmIoU; the MoE "
                           "is the cheapest"},
    "F4.axC.xlabel": {"fr": "Δ mIoU vs contrôle (pt)", "en": "Δ mIoU vs control (pt)"},
    "F4.axC.ylabel": {"fr": "dommage maximal (pt)", "en": "maximal damage (pt)"},
    "F4.axC.title": {"fr": "Gagner sans casser\nle quadrant en haut à droite est vide",
                     "en": "Winning without breaking\nthe top-right quadrant is empty"},
    "F4.suptitle": {"fr": "Polyvalence — {nep} endpoints × {nbras} bras, critères pré-enregistrés "
                          "recalculés\nLe MoE-V3-CS est le SEUL bras à cocher le critère T0 ; dommage "
                          "maximal {dmoe} pt (`{epmoe}`) contre {dpire} pt pour le pire, marge "
                          "{marge} pt sur le suivant. Il n'est premier sur AUCUN endpoint (D en gagne "
                          "{ndwin}) et son percentile moyen le classe {rang}ᵉ : bon partout, meilleur "
                          "nulle part.",
                    "en": "Versatility — {nep} endpoints × {nbras} arms, pre-registered criteria "
                          "recomputed\nMoE-V3-CS is the ONLY arm that meets the T0 criterion; maximal "
                          "damage {dmoe} pt (`{epmoe}`) against {dpire} pt for the worst, a margin of "
                          "{marge} pt over the next. It is first on NONE of the endpoints (D wins "
                          "{ndwin}) and its mean percentile ranks it {rang_en}: good everywhere, "
                          "best nowhere."},
    "F5.xlabel": {"fr": "Δ IoU (MoE-V3-CS − contrôle apparié), points",
                  "en": "Δ IoU (MoE-V3-CS − matched control), points"},
    "F5.title": {"fr": "{n} classes : {ng} hausses, {np} baisses\n"
                       "Plein = Holm(19) < 0,05 · ouvert = IC exclut 0 seulement · gris = ns",
                 "en": "{n} classes: {ng} rises, {np} falls\n"
                       "filled = Holm(19) < 0.05 · open = CI excludes 0 only · grey = ns"},
    "F5.ax2.title": {"fr": "Les 4 plus fortes hausses et les 3 plus fortes baisses,\ncomparées aux 4 "
                           "experts dont le MoE est initialisé",
                     "en": "The 4 largest rises and the 3 largest falls,\ncompared to the 4 experts "
                           "the MoE is initialised from"},
    "F5.suptitle": {"fr": "IoU par classe — trois critères donnés séparément\n"
                          "Plus fortes amplitudes : {top4} · IC95 excluant 0 : {gic} (hausses) et "
                          "{pic} (baisses) · Holm(19) : AUCUNE hausse ne survit, la seule classe "
                          "survivante est une baisse ({perd}, Holm {hmin})",
                    "en": "Per-class IoU — three criteria given separately\n"
                          "Largest amplitudes: {top4} · 95% CI excluding 0: {gic} (rises) and "
                          "{pic} (falls) · Holm(19): NO rise survives, the only surviving class is "
                          "a fall ({perd}, Holm {hmin})"},
}


def T(key: str, **kw) -> str:
    """Rend le gabarit de `key` dans la langue courante, en injectant les valeurs `kw`.
    Les placeholders numériques viennent des expressions d'origine : FR et EN gravent
    la même valeur."""
    return TR[key][LANG].format(**kw)


def ordinal(n: int) -> str:
    """Suffixe ordinal anglais (1st, 2nd, 3rd, 5th…) — pour le rang percentile de F4."""
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def log(m: str) -> None:
    print(f"[p3fig {datetime.now():%H:%M:%S}] {m}", flush=True)


def verif(nom: str, ok: bool, detail: str) -> None:
    """Autocontrôle bloquant : une figure dont la structure n'est pas celle attendue ne part
    pas au manuscrit (c'est ainsi que paper4 a publié un PNG au suptitre faux). Idempotent par
    nom : les contrôles structuraux sont indépendants de la langue, et la fonction tourne une
    fois par langue — on écrase l'entrée existante au lieu de la dupliquer."""
    global ERREURS
    entree = {"figure": nom, "ok": bool(ok), "detail": detail}
    AUTOCONTROLE[:] = [a for a in AUTOCONTROLE if a["figure"] != nom]
    AUTOCONTROLE.append(entree)
    log(f"  [autocontrôle {'✅' if ok else '❌'}] {nom} — {detail}")
    ERREURS = [e for e in ERREURS if not e.startswith(f"{nom} :")]
    if not ok:
        ERREURS.append(f"{nom} : {detail}")


def lire_csv(nom: str) -> list[dict]:
    with open(TABLES / nom, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flottes(lignes: list[dict], *cols: str) -> list[list[float]]:
    return [[float(r[c]) for c in cols] for r in lignes]


def sauver(fig, stem: str) -> dict:
    """Écrit la figure dans la langue courante : nom NU pour l'anglais (aligné sur
    paper.md), suffixe `_fr` pour le français (aligné sur paper_fr.md)."""
    OUT.mkdir(parents=True, exist_ok=True)
    DISCORD.mkdir(parents=True, exist_ok=True)
    base = stem if LANG == "en" else f"{stem}_fr"
    # GATE D'ENCRE (récidive 2026-10-03) : AVANT d'écrire, on mesure les
    # superpositions réelles (texte×texte, ligne de données×texte, hors-canvas)
    # avec le renderer — les 16 figures publiées à 19:41 portaient des
    # collisions suptitle×titres jamais mesurées à l'encre. Fail-closed :
    # le moindre défaut part dans ERREURS et fait échouer le script.
    from fig_overlap_gate import verifier_figure  # noqa: PLC0415
    problemes = verifier_figure(fig, base)
    verif(f"encre_{base}", not problemes,
          "; ".join(problemes[:6]) if problemes
          else "0 superposition d'encre mesurée (texte×texte, ligne×texte, hors-canvas)")
    infos = {"lang": LANG}
    for ext in ("png", "pdf"):
        p = OUT / f"{base}.{ext}"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        infos[ext] = {"chemin": str(p.relative_to(REPO)), "octets": p.stat().st_size}
    png = OUT / f"{base}.png"
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

    # Dessinée À LA LARGEUR IMPRIMÉE (voir le bloc « TAILLE IMPRIMÉE » en tête
    # de fichier) : hauteur = l'empreinte mesurée sur le PDF du 2026-10-03
    # (515,6 × 386,0 pt) pour ne pas déplacer la pagination.
    fig, ax = plt.subplots(figsize=(L, 0.30 * len(rows) + 1.80))
    y = np.arange(len(rows))
    ax.barh(y, delta, color=couleurs, height=0.62, zorder=3)
    ax.errorbar(delta, y, xerr=[np.array(delta) - np.array(lo), np.array(hi) - np.array(delta)],
                fmt="none", ecolor="#222222", elinewidth=0.9, capsize=2.2, zorder=4)
    ax.axvline(0, color=COULEUR_REF, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(bras, fontsize=FZ_TICK)
    ax.set_xlabel(T("F1.xlabel"), fontsize=FZ_LAB)
    ax.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    ax.set_title("__TITRE__", fontsize=FZ_TITLE, pad=7)   # réservé, posé après layout
    for yi, r in zip(y, [x for x in rows][::-1]):
        # Libellé DEHORS de l'intervalle de confiance, à droite du cap haut :
        # posé au-dessus du bout de barre (avant le 2026-10-03) il mordait sur
        # la barre et sur la ligne d'IC — mesuré sur le PNG publié.
        ax.annotate(f"{r['delta_pt']:+.2f}", (max(r['ci_hi_pt'], r['delta_pt']) + 0.05, yi),
                    ha="left", va="center", fontsize=FZ_TICK,
                    color=COULEUR_MOE if r["bras"] == ARM else "#333333")
    ax.set_xlim(min(lo) - 0.15, max(hi) + 0.85)
    fig.tight_layout()
    # Titre posé APRÈS le layout final et RECOUPÉ sur la largeur réelle autour
    # de son centre (le titre est centré sur les AXES, pas sur le canvas :
    # mesuré le 2026-10-03, le titre FR débordait de 12 px à droite alors
    # qu'il tenait « dans la limite canvas »).
    from fig_overlap_gate import cadrer_largeur  # noqa: PLC0415
    fig.canvas.draw()
    _pos = ax.get_position()
    _cx = (_pos.x0 + _pos.x1) / 2 * fig.canvas.get_width_height()[0]
    ax.set_title(
        cadrer_largeur(fig, T("F1.title", n=len(rows) + 1, B=f"{t2['B_boot']:,}",
                              m12=f"{t3['meilleur_holm_12']:.4f}", nf12=t2['n_famille_12'],
                              m15=f"{t3['meilleur_holm_15']:.4f}", nf15=t2['n_famille_15']),
                       FZ_TITLE, centre_px=_cx),
        fontsize=FZ_TITLE, pad=7)
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
        ("entropy_norm", "F2.p.entropy_norm", (0.995, 1.0005)),
        ("entropy_token", "F2.p.entropy_token", (0.985, 1.0005)),
        ("entropy_token_clean", "F2.p.entropy_token_clean", (0.985, 1.0005)),
        ("part_max", "F2.p.part_max", (0.495, 0.510)),
        ("gamma_absmean", "F2.p.gamma_absmean", None),
        ("noise_std", "F2.p.noise_std", None),
    ]
    # 2×3 panneaux À LA LARGEUR IMPRIMÉE : chaque panneau fait alors 2,3 in de
    # large sur le papier, exactement comme avant (515,6 pt / 3), mais les
    # polices ne sont plus divisées par 2,14.
    fig, axes = plt.subplots(2, 3, figsize=(L, 4.90))
    for ax, (cle, tr_key, ylim) in zip(axes.ravel(), panneaux):
        for s in seeds:
            sr = [r for r in rows if int(r["seed"]) == s]
            x = [int(r["epoch"]) for r in sr]
            yv = [float(r[cle]) for r in sr]
            ax.plot(x, yv, lw=1.0, label=f"seed {s}")
        if cle == "part_max":
            ax.axhline(equi, color="#222222", ls="--", lw=1.2)
            # Pas de libellé posé sur la courbe : la valeur d'équirépartition est
            # déjà dans le titre du panneau ; l'annotate central (avant le
            # 2026-10-03) était traversé par les trois courbes de seeds.
        if cle in ("entropy_token", "entropy_token_clean", "part_max", "gamma_absmean"):
            ax.axvspan(0, anneal - 1, color="#f2d7d5", alpha=0.55, zorder=0)
        if cle == "noise_std":
            ax.axvspan(0, anneal - 1, color="#f2d7d5", alpha=0.55, zorder=0)
        titre = T(tr_key, equi=f"{equi:.4f}") if tr_key == "F2.p.part_max" else T(tr_key)
        ax.set_title(titre, fontsize=FZ_TITLE)
        ax.set_xlabel(T("F2.xlabel"), fontsize=FZ_TICK)
        ax.grid(ls=":", color="#cccccc", zorder=0)
        if ylim:
            ax.set_ylim(*ylim)
    # Légende UNIQUE hors des axes (2026-10-04 19h, récidive « chiffres
    # superposés ») : six légendes loc="best" dans six panneaux de 2,3 in
    # tombaient SUR les courbes (mesuré à 300 dpi sur le PDF livré : les
    # tracés seed 123/456 passaient sous le cadre semi-transparent de
    # part_max, |γ| moyen et std). Une légende de figure en bas, hors de
    # toute zone de données, ne peut plus être traversée par rien.
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=FZ_TICK,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    # Suptitle placé PAR MESURE d'encre (récidive 2026-10-03 : y=0.995 +
    # rect=0.93 étaient des dosages — le suptitle multiligne descendait sous
    # la réserve et chevauchait les titres de panneaux).
    from fig_overlap_gate import placer_suptitle  # noqa: PLC0415
    rect_top = placer_suptitle(
        fig,
        T("F2.suptitle", n_ep=t4['n_epochs'], n_seed=len(seeds), anneal=anneal,
          en_min=f"{min(t4['entropy_norm_final'].values()):.5f}",
          pm_lo=f"{min(t4['part_max_final'].values()):.4f}",
          pm_hi=f"{max(t4['part_max_final'].values()):.4f}", equi=f"{equi:.4f}",
          n_dead=sum(t4['experts_morts'].values()),
          g_lo=f"{min(t4['gamma_final'].values()):.5f}",
          g_hi=f"{max(t4['gamma_final'].values()):.5f}"),
        fontsize=FZ_SUPT)
    fig.tight_layout(rect=(0, 0.045, 1, rect_top))
    # Titres de PANNEAUX recoupés sur leur largeur RÉELLE (2,3 in à l'échelle 1)
    # APRÈS le layout : un titre d'axes est centré sur ses axes, pas sur le
    # canvas, et 6 panneaux à 7,8 pt imposent des lignes plus courtes qu'à
    # 15,4 in de large. Puis re-layout pour réserver les lignes ajoutées.
    from fig_overlap_gate import recadrer_panneaux  # noqa: PLC0415
    recadrer_panneaux(fig, list(axes.ravel()), FZ_TITLE, FZ_TICK)
    fig.tight_layout(rect=(0, 0.045, 1, rect_top))
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

    fig, (ax, axF) = plt.subplots(1, 2, figsize=(L, 0.245 * len(pts) + 1.45),
                                  gridspec_kw={"width_ratios": [2.5, 1.0]})
    y = np.arange(len(pts))
    for yi, r, s, i in zip(y, pts, survit, ic0):
        dv, lv, hv = float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"])
        couleur = COULEUR_MOE if s else (COULEUR_AUTRE if i else "#8a8a8a")
        ax.plot([lv, hv], [yi, yi], color=couleur, lw=1.0, zorder=3)
        ax.plot([dv], [yi], marker="o", ms=4.6 if s else 3.6,
                mfc=couleur if s else "white", mec=couleur, mew=1.1, zorder=4)
    ax.axvline(0, color="#333333", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["metrique"] for r in pts], fontsize=FZ_TICK, family="monospace")
    # Bande vide réservée AU-DESSUS de la première ligne (2026-10-04 20h30) :
    # la légende haut-gauche, même opaque, masquait le tronçon gauche de la
    # barre d'IC de boundary_f1_3px_pieds (3ᵉ ligne). Avec 2,2 unités d'air
    # au-dessus de mIoU, le cadre de légende ne recouvre PLUS AUCUNE donnée :
    # il n'y a rien dans cette bande.
    ax.set_ylim(-0.8, len(pts) - 1 + 2.2)
    ax.set_xlabel(T("F3.xlabel"), fontsize=FZ_LAB)
    ax.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    ax.set_title(T("F3.title", npts=len(pts), nfam=t5['n_famille_holm']),
                 fontsize=FZ_TITLE, pad=6)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=COULEUR_MOE, mec=COULEUR_MOE, ms=7,
               label=T("F3.legend_holm", n=sum(survit))),
        Line2D([], [], marker="o", ls="", mfc="white", mec=COULEUR_AUTRE, ms=6, mew=1.5,
               label=T("F3.legend_ic", n=sum(1 for a, b in zip(ic0, survit) if a and not b))),
        Line2D([], [], marker="o", ls="", mfc="white", mec="#8a8a8a", ms=6, mew=1.5,
               label=T("F3.legend_ns")),
    ], fontsize=FZ_TICK, loc="upper left", framealpha=1.0,
       edgecolor="#cccccc")
    # Légende au coin haut gauche, cadre OPAQUE (2026-10-04 19h) : c'est la
    # seule zone sans donnée du panneau (les IC des 4 premières lignes
    # restent > -0,5), et le cadre opaque framealpha=1,0 interdit qu'une
    # grille ou un tracé transparaisse derrière le texte au zoom mobile.
    # Avant le 2026-10-03 elle était en bas à droite SUR la ligne d'IC de
    # IoU_rider ; une légende de figure en bas de canvas, testée à 19h54,
    # tombait sur le tick IoU_rider (49×10 px mesurés) : refusée par le gate.

    # Panneau droit : les fragments, sur LEUR échelle (composantes connexes par image).
    r = frg[0]
    dv, lv, hv = float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"])
    s = r["survit_holm"] == "True"
    axF.plot([lv, hv], [0, 0], color=COULEUR_MOE if s else COULEUR_AUTRE, lw=1.4, zorder=3)
    axF.plot([dv], [0], marker="o", ms=7.5, mfc=COULEUR_MOE if s else "white",
             mec=COULEUR_MOE if s else COULEUR_AUTRE, mew=1.3, zorder=4)
    axF.axvline(0, color="#333333", lw=1.0, zorder=2)
    axF.set_ylim(-1, 1)
    axF.set_yticks([0])
    axF.set_yticklabels(["fragments"], fontsize=FZ_TICK, family="monospace")
    axF.set_xlabel(T("F3.axF.xlabel"), fontsize=FZ_LAB)
    axF.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    axF.set_title(T("F3.axF.title", ratio=f"{ratio:.0f}", pc=f"{float(r['point_controle']):.1f}",
                    pm=f"{float(r['point_moe']):.1f}", dv=f"{dv:+.1f}",
                    holm=f"{float(r['holm']):.3g}"), fontsize=FZ_TITLE, pad=6)
    from fig_overlap_gate import placer_suptitle  # noqa: PLC0415
    rect_top = placer_suptitle(
        fig,
        T("F3.suptitle", nsurv=t5['n_survit_holm'], nrows=len(rows),
          liste=', '.join('`' + m + '`' for m, _e in
                          [(r['metrique'], 0) for r in rows if r['survit_holm'] == 'True'])),
        fontsize=FZ_SUPT)
    fig.tight_layout(rect=(0, 0, 1, rect_top))
    from fig_overlap_gate import recadrer_panneaux  # noqa: PLC0415
    recadrer_panneaux(fig, [ax, axF], FZ_TITLE, FZ_LAB, titre_pad=6)
    fig.tight_layout(rect=(0, 0, 1, rect_top))
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
    labels = [label_moyen(r["label"]) for r in rows]
    domm = np.array([float(r["dommage_max_pt"]) for r in rows])
    dmi = np.array([float(r["delta_miou_pt"]) for r in rows])
    pct = np.array([float(r["percentile_moyen"]) for r in rows])
    est = [r["bras"] == ARM for r in rows]
    seuil = t7["seuils"]["endpoint_pt"]

    # GRILLE 2 LIGNES, panneau A en pleine largeur (1 × 3 avant le 2026-10-04).
    # Mesuré : une fois la figure dessinée à la largeur imprimée, 3 panneaux ne
    # font plus que ~2,1 in de large chacun et les décorations ne tiennent plus —
    # constrained_layout s'effondrait (« axes sizes collapsed to zero », axes de
    # 44 × 19 px, 11 tick labels empilés sur 20 px, 6 collisions d'encre) et
    # tight_layout laissait les titres de panneaux remonter sous le suptitle
    # (chevauchement mesuré 187 × 41 px). Le panneau A porte 11 libellés de bras
    # de 19 caractères : il lui faut la largeur entière. B et C se partagent la
    # seconde ligne. Empreinte imprimée 7,16 × 4,70 in (2,88 in avant) — le gain
    # de hauteur est le prix de polices lisibles à l'échelle 1.
    # constrained_layout (et PAS tight_layout) : c'est le seul moteur qui gère
    # un panneau qui ENJAMBE plusieurs colonnes (gs[0, :]) — tight_layout rend
    # « Axes not compatible with tight_layout » et ne réserve alors aucune
    # marge à gauche : les 11 tick labels du panneau A sortaient du canvas
    # (mesuré x0 = −19,3 px sur 716), ce qui élargissait la tight bbox à 7,35 in
    # et faisait retomber l'échelle d'impression à 0,924.
    fig = plt.figure(figsize=(L, 5.30), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.10, 1.0], hspace=0.10, wspace=0.18)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    yA = np.arange(len(rows))[::-1]
    axA.barh(yA, domm, color=[COULEUR_MOE if e else COULEUR_AUTRE for e in est], height=0.66, zorder=3)
    axA.axvline(seuil, color="#b03a2e", ls="--", lw=1.3, zorder=4)
    # zorder=6 > la ligne (4) et fond BLANC OPAQUE : la pointillée passe
    # DERRIÈRE la boîte du libellé au lieu de le traverser (alpha 0.85 avant
    # le 2026-10-03 : la ligne se voyait par transparence sous le texte).
    axA.annotate(T("F4.annotate_seuil", seuil=f"{seuil:.0f}"), (seuil, yA[0]), textcoords="offset points",
                 xytext=(3, 0), ha="left", va="top", fontsize=FZ_TICK, color="#b03a2e", zorder=6,
                 bbox=dict(fc="white", ec="none", alpha=1.0, pad=1))
    axA.axvline(0, color="#333333", lw=1.0, zorder=2)
    axA.set_yticks(yA)
    axA.set_yticklabels(labels, fontsize=FZ_TICK)
    axA.set_xlabel(T("F4.axA.xlabel"), fontsize=FZ_LAB)
    axA.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    axA.set_title(T("F4.axA.title", ng=len(t7['T0_gagnants']),
                    lab=labels[yA[est.index(True)]]), fontsize=FZ_TITLE)

    withprix = [(label_moyen(r["label"]), float(r["prix_par_pt_miou"])) for r in rows
                if r["prix_par_pt_miou"] not in ("", "None")]
    withprix.sort(key=lambda t: t[1], reverse=True)
    yB = np.arange(len(withprix))[::-1]
    axB.barh(yB, [v for _l, v in withprix],
             color=[COULEUR_MOE if "MoE" in l else COULEUR_AUTRE for l, _v in withprix],
             height=0.6, zorder=3)
    axB.set_yticks(yB)
    axB.set_yticklabels([l for l, _v in withprix], fontsize=FZ_TICK)
    axB.axvline(0, color="#333333", lw=1.0, zorder=2)
    axB.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    axB.set_xlabel(T("F4.axB.xlabel"), fontsize=FZ_LAB)
    axB.set_title(T("F4.axB.title", np=len(withprix)), fontsize=FZ_TITLE)
    vminB = min(x for _x, x in withprix)
    for yi, (_l, v) in zip(yB, withprix):
        # Libellé à gauche du bout de barre (zone vide), SAUF pour les barres
        # dont le bout touche le bord gauche de l'axe : là le libellé sortirait
        # de l'axe et cognerait le tick y (mesuré sur le PNG du 2026-10-03,
        # « D · CE+Kervadec EDT-43.1 ») — celles-là portent leur libellé en
        # blanc DANS la barre.
        if abs(v) > 0.55 * abs(vminB):
            axB.annotate(f"{v:+.1f}", (v * 0.98, yi), ha="left", va="center",
                         fontsize=FZ_TICK, color="white")
        else:
            axB.annotate(f"{v:+.1f}", (v, yi), textcoords="offset points", xytext=(-3, 0),
                         ha="right", va="center", fontsize=FZ_TICK)
    axB.set_xlim(vminB - 5.5, 0.6)

    axC.scatter(dmi, domm, s=34, c=[COULEUR_MOE if e else COULEUR_AUTRE for e in est],
                zorder=4, edgecolors="#222222", linewidths=0.5)
    for r, xv, yv in zip(rows, dmi, domm):
        if r["bras"] in (ARM, "D", "fused_DvetoB", "G"):
            axC.annotate(label_moyen(r["label"]).split(" ·")[0], (xv, yv),
                         textcoords="offset points", xytext=(4, 3), fontsize=FZ_TICK)
    axC.axhline(seuil, color="#b03a2e", ls="--", lw=1.2, zorder=2)
    axC.axvline(0, color="#333333", lw=1.0, zorder=2)
    axC.set_xlabel(T("F4.axC.xlabel"), fontsize=FZ_LAB)
    axC.set_ylabel(T("F4.axC.ylabel"), fontsize=FZ_LAB)
    axC.grid(ls=":", color="#cccccc", zorder=0)
    axC.set_title(T("F4.axC.title"), fontsize=FZ_TITLE, pad=12)

    rang = t7['ordre_percentile'].index(ARM) + 1
    # Suptitle posé PAR MESURE d'encre (placer_suptitle rend le haut de rect à
    # donner à tight_layout) : ni y=0.995 ni rect=0.93 ne sont des dosages
    # acceptables ici, et cadrer_largeur empêche une ligne trop longue
    # d'élargir le PNG bbox-tight (±245 px mesurés le 2026-10-03).
    from fig_overlap_gate import (placer_suptitle, recadrer_avant_layout,  # noqa: PLC0415
                                  recadrer_panneaux)
    rect_top = placer_suptitle(
        fig,
        T("F4.suptitle", nep=t7['n_endpoints'], nbras=t7['n_bras'],
          dmoe=f"{t7['dommage_moe']['dommage_max_pt']:+.2f}",
          epmoe=t7['dommage_moe']['endpoint'],
          dpire=f"{min(float(r['dommage_max_pt']) for r in rows):+.2f}",
          marge=f"{t7['marge_vs_suivant']:.2f}",
          ndwin=max(int(r['n_rang1']) for r in rows),
          rang=rang, rang_en=ordinal(rang)),
        fontsize=FZ_SUPT)
    # Le suptitle est ancré au bord haut du canvas (y=1.0, va='top') : la bande
    # qu'il occupe doit être RETIRÉE de la zone de layout, sinon le panneau A
    # remonte dessous (mesuré : chevauchement suptitle × titre de 412 × 21,6 px
    # et × les tick labels). ConstrainedLayoutEngine accepte ce `rect`.
    from matplotlib.layout_engine import ConstrainedLayoutEngine  # noqa: PLC0415
    fig.set_layout_engine(ConstrainedLayoutEngine(rect=(0, 0, 1, rect_top)))
    # Recadrage en DEUX temps : sur la part de canvas estimée AVANT le premier
    # draw (sinon constrained_layout s'effondre, voir recadrer_avant_layout),
    # puis sur la largeur RÉELLE des axes une fois le layout établi.
    recadrer_avant_layout(fig, [axA, axB, axC], [1.0, 0.5, 0.5], FZ_TITLE, FZ_LAB)
    fig.canvas.draw()
    recadrer_panneaux(fig, [axA, axB, axC], FZ_TITLE, FZ_LAB, titre_pad=14)
    fig.canvas.draw()
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(L, 0.235 * len(rows) + 1.30),
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})
    for yi, dv, lv, hv, s, i in zip(y, delta, lo, hi, survit, ic0):
        c = COULEUR_MOE if s else (COULEUR_AUTRE if i else "#8a8a8a")
        ax.plot([lv, hv], [yi, yi], color=c, lw=1.0, zorder=3)
        ax.plot([dv], [yi], marker="o", ms=4.6 if s else 3.6, mfc=c if s else "white",
                mec=c, mew=1.1, zorder=4)
    ax.axvline(0, color="#333333", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["classe"] for r in rows], fontsize=FZ_TICK)
    ax.set_xlabel(T("F5.xlabel"), fontsize=FZ_LAB)
    ax.grid(axis="x", ls=":", color="#bbbbbb", zorder=0)
    ax.set_title(T("F5.title", n=len(rows), ng=t6['n_gains'], np=t6['n_pertes']),
                 fontsize=FZ_TITLE, pad=6)

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
    # rotation 68° (38° avant le 2026-10-04, 62° entre 16h et 19h) : à
    # l'échelle imprimée 1 le panneau fait 3,2 in pour 7 catégories, soit
    # 30 pt d'espacement ; un libellé de 10 caractères (« motorcycle ») à
    # 7,3 pt occupe 32,8 pt de large une fois pivoté à 45° → boîtes voisines
    # chevauchantes (mesuré : 'bicycle' × 'motorcycle', 9,9 × 29,1 px à 6,8 pt
    # puis 1,3 × 35,8 px à 7,3 pt/62°). À 68° l'emprise horizontale retombe
    # sous l'espacement de 30 pt.
    ax2.set_xticklabels([r["classe"] for r in top], rotation=68, ha="right", fontsize=FZ_TICK)
    ax2.set_ylabel("Δ IoU (pt)", fontsize=FZ_LAB)
    ax2.grid(axis="y", ls=":", color="#bbbbbb", zorder=0)
    # cadre OPAQUE (framealpha 1,0) : le panneau de droite a sa moitié haute
    # libre (aucune barre au-delà de +1,5 sous la légende, mesuré sur le PDF
    # livré) ; l'opacité garantit qu'aucune barre ne transparait derrière le
    # texte si une seed future changeait les hauteurs.
    ax2.legend(fontsize=FZ_TICK, ncol=2, framealpha=1.0, edgecolor="#cccccc")
    ax2.set_title(T("F5.ax2.title"), fontsize=FZ_TITLE, pad=6)

    from fig_overlap_gate import placer_suptitle  # noqa: PLC0415
    rect_top = placer_suptitle(
        fig,
        T("F5.suptitle", top4=', '.join(t6['top4_amplitude']),
          gic=', '.join(t6['gains_ic_exclut_0']), pic=', '.join(t6['pertes_ic_exclut_0']),
          perd=', '.join(t6['pertes_holm']), hmin=f"{t6['holm_min_pertes']:.4g}"),
        fontsize=FZ_SUPT)
    fig.tight_layout(rect=(0, 0, 1, rect_top))
    from fig_overlap_gate import recadrer_panneaux  # noqa: PLC0415
    recadrer_panneaux(fig, [ax, ax2], FZ_TITLE, FZ_LAB, titre_pad=6)
    fig.tight_layout(rect=(0, 0, 1, rect_top))
    infos = sauver(fig, "F5_perclass")
    verif("F5_perclass", len(rows) == t6["n_classes"] == 19 and len(top) == 7,
          f"{len(rows)} classes au forest + {len(top)} classes au comparateur d'experts "
          f"({len(expert)} experts + le MoE) ; 2 panneaux")
    return infos


def main() -> int:
    t0 = datetime.now()
    log("P3.18b — figures F1-F5 du paper 3 (MoE-V3-CS), sans GPU — bilingue EN (nom nu) + FR (_fr)")
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

    figs: dict = {}
    FNS = [("F1", f1_delta_13bras), ("F2", f2_routage), ("F3", f3_forest_metiers),
           ("F4", f4_polyvalence), ("F5", f5_perclass)]
    # Deux passes, une par langue : nom NU = anglais (comme paper.md), `_fr` = français
    # (comme paper_fr.md). Les autocontrôles structuraux sont indépendants de la langue et
    # sont dédupliqués par nom dans verif() — d'où 5 entrées quel que soit le nombre de passes.
    for lang in ("en", "fr"):
        set_lang(lang)
        log(f"[figures] langue {lang}")
        for nom, fn in FNS:
            infos = fn(d)
            figs.setdefault(nom, {})[lang] = infos
            log(f"  → {infos['png']['chemin']} ({infos['png']['octets'] / 1e6:.2f} Mo, "
                f"{infos['dims_px'][0]}×{infos['dims_px'][1]} px)")

    # Autocontrôles écrits DANS la source consolidée : le stager et le futur p3_check_numbers
    # les reliront, et les dimensions png permettront de prouver qu'un PDF embarque bien la
    # figure à jour (Piège 3 de paper4). Les dims sont consignées PAR LANGUE : le stager vérifie
    # que chaque PDF embarque les figures de SA langue.
    d["figures_autocontrole"] = {"date": t0.isoformat(timespec="seconds"),
                                 "script": "scripts/p3_moe_figures.py",
                                 "langues": ["en", "fr"],
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
    log(f"[fin] {dt:.0f}s — F1-F5 écrites en 2 langues (10 png + 10 pdf : nom nu=EN, _fr=FR), "
        f"autocontrôles {ok}/{len(AUTOCONTROLE)} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
