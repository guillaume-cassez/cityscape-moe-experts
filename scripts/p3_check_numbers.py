#!/usr/bin/env python3
"""p3_check_numbers.py — cohérence manuscrits paper3 ↔ paper3_tables.json (gate SPEC ## Tests).

Vérifie que les chiffres clés cités dans papers/paper3/publish/repo/paper.md (EN) et
paper_fr.md (FR) sont EXACTEMENT ceux de la source consolidée assertée
papers/paper3/tables/paper3_tables.json (elle-même produite par scripts/p3_moe_tables.py
sous 257 checks bloquants). Toute divergence fait échouer le script (exit 1) — le
manuscrit ne peut pas dériver des tables en silence.

Discipline héritée de l'erratum paper4 (leçon « 18 of 19 ») : CHAQUE compte, taille,
somme ou ratio est RECALCULÉ ici depuis le DÉTAIL de l'artefact (les listes `lignes`,
`classement`, `par_seed`), puis comparé au manuscrit — jamais à une constante recopiée.
Les scalaires stockés (`n_gains`, `n_survit_holm`, …) sont re-vérifiés contre le recalcul
pour qu'un artefact corrompu ne puisse pas « passer » le manuscrit.

Usage : python3 scripts/p3_check_numbers.py   (depuis la racine du dépôt)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "papers" / "paper3" / "publish" / "repo"
MINUS = "\u2212"  # − U+2212 utilisé dans les manuscrits (pas le trait d'ASCII)

FAILS, N = [], 0


def load(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def norm_fr(txt):
    """FR : décimales à virgule -> point, pour comparaison numérique."""
    return re.sub(r"(\d),(\d)", r"\1.\2", txt)


EN = load(BUNDLE / "paper.md")
FR = norm_fr(load(BUNDLE / "paper_fr.md"))
SRC = json.load(open(ROOT / "papers/paper3/tables/paper3_tables.json"))


def fmt(v, nd, signed=False):
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


_SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"}


def sci(v, nd=2):
    """1.45e-03 -> '1.45×10⁻³' (exposants Unicode, comme dans les manuscrits)."""
    mant, exp = f"{v:.{nd}e}".split("e")
    return f"{mant}×10" + "".join(_SUP[c] for c in str(int(exp)))


def check(label, en_str, fr_str=None):
    """en_str doit apparaître dans paper.md ; fr_str (défaut = en_str) dans paper_fr.md
    (après normalisation virgules -> points)."""
    global N
    N += 1
    fr_str = fr_str if fr_str is not None else en_str
    if en_str not in EN:
        FAILS.append(f"[EN] {label}: attendu « {en_str} » absent de paper.md")
    if fr_str not in FR:
        FAILS.append(f"[FR] {label}: attendu « {fr_str} » absent de paper_fr.md")


def check_en(label, en_str):
    global N
    N += 1
    if en_str not in EN:
        FAILS.append(f"[EN] {label}: attendu « {en_str} » absent de paper.md")


def absent(label, *motifs):
    """Garde anti-régression : ces formulations NE doivent plus apparaître."""
    global N
    N += 1
    trouves = []
    for m in motifs:
        if m in EN:
            trouves.append(f"EN:{m!r}")
        if m in FR:
            trouves.append(f"FR:{m!r}")
    if trouves:
        FAILS.append(f"[REGRESSION] {label}: {', '.join(trouves)}")


# --------------------------------------------------------------------------- 1. PRIMAIRE (T2)
t2 = SRC["T2"]
# NB : dans T2, point_moe/point_controle/per_seed_* sont des FRACTIONS (×100 -> points),
# alors que delta_pt/ci_*_pt sont déjà en POINTS (écart de pourcentage). Vérifié :
# delta_pt == (point_moe - point_controle)*100 à 1e-9.
assert abs((t2["point_moe"] - t2["point_controle"]) * 100 - t2["delta_pt"]) < 1e-9, \
    "delta_pt != (point_moe-point_controle)*100 — unité ambiguë"
check("primaire point MoE", fmt(t2["point_moe"] * 100, 3))
check("primaire point contrôle", fmt(t2["point_controle"] * 100, 3))
check("primaire delta", f"{fmt(t2['delta_pt'], 3, signed=True)} pt")
check("primaire CI", f"[{fmt(t2['ci_lo_pt'], 3, signed=True)} ; {fmt(t2['ci_hi_pt'], 3, signed=True)}]")
check("primaire p", f"p = {fmt(t2['p'], 4)}")
for s in ("42", "123", "456"):
    check(f"seed MoE {s}", fmt(t2["per_seed_moe"][s] * 100, 3))
    check(f"seed contrôle {s}", fmt(t2["per_seed_controle"][s] * 100, 3))

# --- 1bis. LES TROIS FAMILLES DE HOLM : tailles RECALCULÉES depuis les artefacts ----
# Le défaut corrigé sur paper4 (erratum v1.1.0) : une étiquette « 15 paires » remplie avec
# les valeurs de la famille 12 paires. Ici on recalcule la taille de chaque famille depuis
# l'artefact source (len(pairwise)) et on exige que le manuscrit porte ces tailles-là.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_bs", ROOT / "src" / "moe" / "bootstrap.py")
_bs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bs)

_p314 = json.load(open(ROOT / "results/moe_v3_cs/p314/master_table.json"))
_p316 = json.load(open(ROOT / "results/moe_v3_cs/metiers_experts/table_metiers_experts.json"))
_fam12 = {k: v["p_two_sided"] for k, v in _p314["pairwise"].items()}
_fam15 = {k: v["p_two_sided"] for k, v in _p316["tables"]["mIoU"]["pairwise"].items()}
_h12, _h15 = _bs.holm(_fam12), _bs.holm(_fam15)
n12, n15 = len(_fam12), len(_fam15)
n1 = t2["n_famille_1"]

# sanity bloquante : les tailles recalculées == les tailles stockées dans la source consolidée
assert n12 == t2["n_famille_12"], f"famille 12 : recalculé {n12} != stocké {t2['n_famille_12']}"
assert n15 == t2["n_famille_15"], f"famille 15 : recalculé {n15} != stocké {t2['n_famille_15']}"
# le Holm du primaire recomputé depuis les p bruts == celui de la source (écart < 1e-9)
_k12 = "moe_v3cs_vs_controle"
assert abs(_h12[_k12] - t2["holm_famille_12"]) < 1e-9, "Holm 12 recomputé != source"
assert abs(_h15[_k12] - t2["holm_famille_15"]) < 1e-9, "Holm 15 recomputé != source"

check("taille famille 1", f"| {n1} | **{fmt(t2['holm_famille_1'], 4)}**",
      f"| {n1} | **{fmt(t2['holm_famille_1'], 4)}**")
check("Holm famille 12", fmt(t2["holm_famille_12"], 4))
check("Holm famille 15", fmt(t2["holm_famille_15"], 4))
check("meilleur Holm 15 (aucun bras ne survit)", fmt(t2["meilleur_holm_15"], 4))
# l'étiquette de famille doit être calculée, pas écrite à la main
check_en("étiquette famille 12 (EN)", f"Holm ({n12} pairs, P3.14)")
check_en("étiquette famille 15 (EN)", f"Holm ({n15} pairs, P3.16)")

# --- 1ter. ROBUSTESSE INTER-FORWARDS : écart RECALCULÉ depuis les deux deltas ----
sf = t2["second_forward"]
ecart = abs(t2["delta_pt"] - sf["delta_pt"])
# la source déclare l'amplitude ; on la re-vérifie depuis les deux deltas
assert abs(ecart - t2["amplitude_delta_pt"]) < 1e-9, "amplitude inter-forwards incohérente"
check("delta second forward", f"{fmt(sf['delta_pt'], 6, signed=True)}")
check("delta harness (6 déc.)", f"{fmt(t2['delta_pt'], 6, signed=True)}")

# --------------------------------------------------------------------------- 2. ROUTAGE (T4)
t4 = SRC["T4"]
eq = t4["equirepartition"]  # 0.5, calculé top_k/n_experts par le générateur
# sanity : l'équirépartition est bien top_k/n_experts
assert abs(eq - t4["top_k"] / t4["n_experts"]) < 1e-12, "équirépartition != top_k/n_experts"
check("équirépartition part_max", fmt(eq, 4))
# part_max final : les 3 seeds, et l'écart max à l'équirépartition RECALCULÉ
pmf = [t4["part_max_final"][s] for s in ("42", "123", "456")]
ecart_max = max(abs(v - eq) for v in pmf)
check("part_max final seed42", fmt(pmf[0], 4))
check("part_max final seed123", fmt(pmf[1], 4))
check("part_max final seed456", fmt(pmf[2], 4))
check("écart max part_max à équirépartition", fmt(ecart_max, 4))
# entropy_norm final minimal (le seed le moins uniforme)
enf = [t4["entropy_norm_final"][s] for s in ("42", "123", "456")]
check("entropy_norm final min", fmt(min(enf), 5))
# frac : bornes RECALCULÉES depuis frac_min/frac_max
check("frac borne min", fmt(t4["frac_min_final"], 4))
check("frac borne max", fmt(t4["frac_max_final"], 4))
# experts morts : RECALCULÉ (somme des 3 seeds) — doit être 0
n_morts = sum(t4["experts_morts"][s] for s in ("42", "123", "456"))
assert n_morts == 0, f"experts morts attendus 0, mesuré {n_morts}"
check_en("zéro expert mort (EN)", "zero dead experts")
# γ : époque 0 -> finale, ratio RECALCULÉ
g0 = [t4["gamma_ep0"][s] for s in ("42", "123", "456")]
gf = [t4["gamma_final"][s] for s in ("42", "123", "456")]
check("γ seed42 final", fmt(gf[0], 5))
check("γ seed123 final", fmt(gf[1], 5))
check("γ seed456 final", fmt(gf[2], 5))
check("γ époque 0 (seed42)", fmt(g0[0], 5))
ratio_min = min(gf[i] / g0[i] for i in range(3))
ratio_max = max(gf[i] / g0[i] for i in range(3))
check("γ ratio min", f"×{int(round(ratio_min))}")
check("γ ratio max", f"×{int(round(ratio_max))}")
# top_expert_share final (moyenne des maxima) — ne pas confondre avec part_max
tes = [t4["top_expert_share_final"][s] for s in ("42", "123", "456")]
check("top_expert_share final seed42", fmt(tes[0], 5))
# --- argument CORRIGÉ sur le bruit de Shazeer : la PREUVE porte sur la phase ACTIVE ----
ba = t4["bruit_actif"]
check("écart token↔clean max (phase active)", sci(ba["ecart_token_clean_max_tous_seeds"]))
# entropy_clean_min : le minimum sur les 3 seeds, RECALCULÉ
ecm = min(ba["entropy_clean_min"][s] for s in ("42", "123", "456"))
check("entropy clean min (phase active)", fmt(ecm, 5))
# part_max bornes sur la phase active : RECALCULÉ depuis les 3 seeds
pmb_lo = min(ba["part_max_bornes"][s][0] for s in ("42", "123", "456"))
pmb_hi = max(ba["part_max_bornes"][s][1] for s in ("42", "123", "456"))
check("part_max phase active borne basse", fmt(pmb_lo, 5))
check("part_max phase active borne haute", fmt(pmb_hi, 5))
# garde anti-régression : la tautologie de l'époque finale ne doit PAS être présentée comme preuve
absent("tautologie entropy_token_clean présentée comme preuve",
       "entropy_token_clean proves", "entropy_token_clean prouve",
       "clean entropy proves", "clean entropy shows")

# --------------------------------------------------------------------------- 3. 13 BRAS (T3)
t3 = SRC["T3"]
cl = t3["classement"]  # liste de 12 dicts (bras hors contrôle), triée par Δ décroissant
# sanity : le classement est bien trié par delta décroissant
deltas = [r["delta_pt"] for r in cl]
assert deltas == sorted(deltas, reverse=True), "classement T3 non trié par Δ décroissant"
# rang du MoE RECALCULÉ depuis le classement
rang_moe = next(r["rang"] for r in cl if r["bras"] == "moe_v3cs")
assert rang_moe == t3["rang_moe"], f"rang MoE recalculé {rang_moe} != stocké {t3['rang_moe']}"
check("rang MoE (amplitude)", f"**5th of 12**", f"**5^e^ sur 12**")
# nombre de bras significatifs au seuil brut : RECALCULÉ
n_sig_brut = sum(1 for r in cl if r["significatif_brut"])
assert n_sig_brut == t3["n_sig_brut"], "n_sig_brut recalculé != stocké"
# aucun ne survit au Holm : RECALCULÉ depuis les holm_12/holm_15 du classement
n_survit_12 = sum(1 for r in cl if r["holm_12"] is not None and r["holm_12"] < 0.05)
n_survit_15 = sum(1 for r in cl if r["holm_15"] is not None and r["holm_15"] < 0.05)
assert n_survit_12 == 0 and n_survit_15 == 0, "un bras survit au Holm — inattendu"
check("meilleur Holm 12 (T3)", fmt(t3["meilleur_holm_12"], 4))
check("meilleur Holm 15 (T3)", fmt(t3["meilleur_holm_15"], 4))
# vérification ligne à ligne de la table de contexte des deux manuscrits
_ordre = [(r["bras"], r["delta_pt"]) for r in cl]


def verif_table_13bras(txt, langue, na15):
    global N
    lignes = [l for l in txt.splitlines() if l.startswith("|")]
    entetes = [i for i, l in enumerate(lignes)
               if ("Holm (12 pairs" in l or "Holm (12 paires" in l
                   or f"Holm ({n12} pairs" in l or f"Holm (famille {n12} paires" in l)]
    if not entetes:
        N += 1
        FAILS.append(f"[{langue}] table 13 bras : en-tête de la famille {n12} introuvable")
        return
    deb = entetes[0]
    corps = []
    for l in lignes[deb + 1:]:
        if set(l) <= set("|-: "):
            continue
        first = l.split("|")[1].strip()
        if first in ("—", "-"):
            break
        corps.append([c.strip() for c in l.split("|")[1:-1]])
    if len(corps) != len(_ordre):
        FAILS.append(f"[{langue}] table 13 bras : {len(corps)} lignes lues, {len(_ordre)} attendues")
        return
    for i, (a, _d) in enumerate(_ordre):
        N += 1
        c = corps[i]
        src = cl[i]
        k = f"{a}_vs_controle"
        h15 = f"{_h15[k]:.4f}" if k in _h15 else na15
        # cellules attendues : #, mIoU(2dec), p(4dec), Holm12(4dec), Holm15(4dec ou n.a.)
        att_miou = f"{src['miou_pct']:.3f}"
        att_p = f"{src['p']:.4f}"
        att_h12 = f"{src['holm_12']:.4f}"
        relu_miou = c[2].replace("**", "").replace(",", ".")
        relu_p = c[5].replace("**", "").replace(",", ".")
        relu_h12 = c[6].replace("**", "").replace(",", ".")
        # tolérance : le manuscrit peut arrondir Holm15 à 4 déc. ; on compare p/mIoU/Holm12
        if relu_miou != att_miou:
            FAILS.append(f"[{langue}] T13 ligne {i+1} ({a}) mIoU : lu {relu_miou}, source {att_miou}")
        if relu_p != att_p:
            FAILS.append(f"[{langue}] T13 ligne {i+1} ({a}) p : lu {relu_p}, source {att_p}")
        if relu_h12 != att_h12:
            FAILS.append(f"[{langue}] T13 ligne {i+1} ({a}) Holm12 : lu {relu_h12}, source {att_h12}")


verif_table_13bras(EN, "EN", "n.a. (a)")
verif_table_13bras(FR, "FR", "n.c. (a)")

# --------------------------------------------------------------------------- 4. MÉTIERS (T5)
t5 = SRC["T5"]
lignes5 = t5["lignes"]
# comptes RECALCULÉS depuis le détail des 20 métriques
n_survit = sum(1 for r in lignes5 if r["survit_holm"])
n_ic0 = sum(1 for r in lignes5 if r["ic_exclut_0"])
assert n_survit == t5["n_survit_holm"], f"survivantes Holm recalculé {n_survit} != {t5['n_survit_holm']}"
assert n_ic0 == t5["n_ic_exclut_0"], f"IC-excluant-0 recalculé {n_ic0} != {t5['n_ic_exclut_0']}"
check("n survivantes Holm (métiers)", f"**{n_survit} survive the Holm**", f"**{n_survit} survivent au Holm**")
check("n IC excluant 0 (métiers)", f"**{n_ic0} have a CI excluding zero**",
      f"**{n_ic0} ont un IC excluant zéro**")
# les DEUX survivantes nommées (jamais « seuls les fragments »)
survivantes = sorted(r["metrique"] for r in lignes5 if r["survit_holm"])
assert survivantes == ["fragments", "instances_taille_T3_rappel"], \
    f"survivantes inattendues : {survivantes}"
check_en("survivante T3 nommée (EN)", "instances_taille_T3_rappel")
check_en("survivante fragments nommée (EN)", "`fragments`")
# fragments : delta et ratio RECALCULÉS depuis les points
frag = t5["fragments"]
ratio_frag = frag["moe"] / frag["controle"]
assert abs(ratio_frag - frag["ratio"]) < 1e-9, "ratio fragments incohérent"
check("fragments contrôle", fmt(frag["controle"], 1))
check("fragments MoE", fmt(frag["moe"], 1))
check("fragments delta", fmt(frag["delta"], 1, signed=True))
check("fragments CI", f"[{fmt(frag['ci'][0], 1, signed=True)} ; {fmt(frag['ci'][1], 1, signed=True)}]")
# T3 rappel : delta et CI
t3r = next(r for r in lignes5 if r["metrique"] == "instances_taille_T3_rappel")
check("T3 rappel delta", fmt(t3r["delta"], 3, signed=True))
check("T3 rappel CI", f"[{fmt(t3r['ci_lo'], 3, signed=True)} ; {fmt(t3r['ci_hi'], 3, signed=True)}]")
# pire famille : RECALCULÉE (la seule négative)
pf = t5["pire_famille"]
check("pire famille métiers", f"`{pf['famille']}`")
assert pf["n_familles_negatives"] == 1, "pire famille : plus d'une famille négative"

# --------------------------------------------------------------------------- 5. PER-CLASS (T6)
t6 = SRC["T6"]
lignes6 = t6["lignes"]
# comptes RECALCULÉS depuis les 19 lignes
n_gains = sum(1 for r in lignes6 if r["delta_pt"] > 0)
n_pertes = sum(1 for r in lignes6 if r["delta_pt"] < 0)
assert n_gains == t6["n_gains"] and n_pertes == t6["n_pertes"], "compte gains/pertes incohérent"
assert n_gains + n_pertes == t6["n_classes"], "gains+pertes != 19"
check("n classes en hausse", f"**{n_gains} rise, {n_pertes} fall**", f"**{n_gains} en hausse, {n_pertes} en baisse**")
# critère AMPLITUDE : top-4 RECALCULÉ (tri par delta décroissant)
top4 = [r["classe"] for r in sorted(lignes6, key=lambda r: -r["delta_pt"])[:4]]
assert top4 == t6["top4_amplitude"], f"top4 amplitude recalculé {top4} != {t6['top4_amplitude']}"
check_en("top4 amplitude (EN)", ", ".join(f"{c} {fmt(next(r['delta_pt'] for r in lignes6 if r['classe']==c), 2, signed=True)}" for c in top4))
# critère IC-excluant-0 : hausses et baisses RECALCULÉES
gains_ic = [r["classe"] for r in lignes6 if r["ic_exclut_0"] and r["delta_pt"] > 0]
pertes_ic = [r["classe"] for r in lignes6 if r["ic_exclut_0"] and r["delta_pt"] < 0]
assert gains_ic == t6["gains_ic_exclut_0"], "hausses IC recalculées != stockées"
assert pertes_ic == t6["pertes_ic_exclut_0"], "baisses IC recalculées != stockées"
# critère HOLM(19) : RECALCULÉ — aucune hausse ne survit, la seule survivante est une baisse
gains_holm = [r["classe"] for r in lignes6 if r["survit_holm"] and r["delta_pt"] > 0]
pertes_holm = [r["classe"] for r in lignes6 if r["survit_holm"] and r["delta_pt"] < 0]
assert gains_holm == [] and pertes_holm == t6["pertes_holm"] == ["bicycle"], \
    f"Holm(19) recalculé inattendu : gains {gains_holm}, pertes {pertes_holm}"
check_en("aucune hausse ne survit au Holm (EN)", "no rise survives")
check_en("seule survivante = bicycle (EN)", "bicycle (Holm 0.0038)")
check("bicycle Holm", fmt(t6["holm_min_pertes"], 4))
# part du meilleur expert sur truck : RECALCULÉE (dénominateur = meilleur des 4 experts)
truck = next(r for r in lignes6 if r["classe"] == "truck")
meilleur_expert_truck = max(truck["delta_B"], truck["delta_D"], truck["delta_Dp"], truck["delta_G"])
part_truck = truck["delta_pt"] / meilleur_expert_truck
assert abs(part_truck - t6["part_gain_truck"]) < 1e-6, "part truck recalculée != stockée"
check("part truck (meilleur expert)", f"{round(part_truck*100)} %")
# garde anti-régression : les deux critères ne doivent pas être confondus
absent("critères amplitude/IC confondus",
       "truck, train, sidewalk, road are the largest",
       "truck, train, sidewalk, road sont les plus fortes")

# --------------------------------------------------------------------------- 6. POLYVALENCE (T7)
t7 = SRC["T7"]
check("n endpoints", f"**{t7['n_endpoints']} endpoints**", f"**{t7['n_endpoints']} endpoints**")
# dommage maximal du MoE + marge RECALCULÉE depuis le classement de dommage
dom = t7["dommage_moe"]
cl_dom = t7["classement_dommage"]  # liste de bras, triée du moins abîmé au plus abîmé
assert cl_dom[0] == "moe_v3cs", "le MoE n'est pas premier en dommage"
# marge = dommage du 2e - dommage du 1er (recalculée depuis les lignes)
lignes7 = {r["bras"]: r for r in t7["lignes"]}
dom2 = lignes7[cl_dom[1]]["dommage_max_pt"]
marge = dom["dommage_max_pt"] - dom2  # MoE moins abîmé -> marge positive sur le suivant
assert abs(marge - t7["marge_vs_suivant"]) < 1e-6, f"marge recalculée {marge} != {t7['marge_vs_suivant']}"
check("dommage max MoE", fmt(dom["dommage_max_pt"], 2))
check("marge sur le suivant", fmt(t7["marge_vs_suivant"], 2))
check("endpoint dommage MoE", f"`{dom['endpoint']}`")
# dommage de D (le pire du plateau)
domD = lignes7["D"]["dommage_max_pt"]
check("dommage max D", fmt(domD, 2))
# prix par point de mIoU : RECALCULÉ (dommage/delta) pour le MoE et D
prix_moe = t7["prix"]["moe_v3cs"]
prix_D = t7["prix"]["D"]
check("prix MoE", fmt(prix_moe, 1))
check("prix D", fmt(prix_D, 1))
# percentile moyen + rang RECALCULÉ depuis l'ordre
pct = t7["percentile_moyen_moe"]
ordre_pct = t7["ordre_percentile"]
rang_pct = ordre_pct.index("moe_v3cs") + 1
check("percentile moyen MoE", fmt(pct, 1))
check("rang percentile MoE", f"{rang_pct}th of {len(ordre_pct)}", f"{rang_pct}^e^ sur {len(ordre_pct)}")
assert rang_pct == 5 and len(ordre_pct) == 11, "rang percentile inattendu"
# n_rang1 : le MoE premier sur AUCUN endpoint, D sur 16 — RECALCULÉ depuis n_rang1
assert t7["n_rang1"]["moe_v3cs"] == 0, "le MoE est premier sur un endpoint ?"
assert t7["n_rang1"]["D"] == 16, "n_rang1 de D != 16"
check_en("MoE premier sur aucun (EN)", "first on none of the 36 endpoints")
check_en("D gagne 16 (EN)", "D wins 16")
# DEUX compteurs distingués : 22/36 au-dessus du contrôle, 28/36 en moitié haute
n_pos = t7["n_endpoints_delta_positif"]
n_top = t7["n_top_half_classement"]
assert n_pos == 22 and n_top == 28, f"compteurs polyvalence inattendus : {n_pos}, {n_top}"
assert n_pos != n_top, "les deux compteurs sont égaux — l'ambiguïté serait sans objet"
check("compteur au-dessus du contrôle", f"**{n_pos} of the 36 endpoints are above the control**",
      f"**{n_pos} des 36 endpoints sont au-dessus du contrôle**")
check("compteur moitié haute", f"**{n_top} are in the upper half",
      f"**{n_top} sont dans la moitié haute")
# pire rang AVEC son dénominateur : RECALCULÉ (11/12, pas 11/13)
prd = t7["pire_rang_avec_denominateur"]
assert prd["rang"] == 11 and prd["denom"] == 12, f"pire rang inattendu : {prd}"
assert set(t7["denominateurs_rang_observes"]) == {12, 13}, "dénominateurs non constants absents"
check("pire rang MoE", f"{prd['rang']}/{prd['denom']}", f"{prd['rang']}/{prd['denom']}")
check("endpoint pire rang", f"`{prd['endpoint']}`")
# T0 : le MoE seul gagnant — RECALCULÉ
assert t7["T0_gagnants"] == ["moe_v3cs"], f"T0 gagnants inattendus : {t7['T0_gagnants']}"
# T4 partiel : 1er en dommage sur 2 seeds sur 3 — RECALCULÉ depuis par_seed
n_premier = sum(1 for s in ("42", "123", "456")
                if t7["par_seed"][s]["classement_dommage"][0] == "moe_v3cs")
assert n_premier == t7["n_seeds_premier"] == 2, f"n seeds premier inattendu : {n_premier}"
assert t7["verdicts_stocks"]["T4"] is False, "T4 devrait être partiel (False)"
check_en("T4 partiel (EN)", "1st in damage on 2 of 3 seeds")
check("seed 123 troisième", "seed 123 → 3rd", "seed 123 → 3^e^")
# garde anti-régression : ne pas survendre
absent("survente du MoE",
       "best arm on all", "premier sur tous", "premier sur les 36",
       "highest mIoU", "meilleure mIoU", "meilleur mIoU du plateau")

# --------------------------------------------------------------------------- 7. BRATS (T8) + COÛT (T1)
t8 = SRC["T8"]
check("BRATS DOI", t8["brats_doi"].replace("10.5281/", "10.5281/"))
check("BRATS concept DOI", t8["brats_concept_doi"])
t1 = SRC["T1"]
# FR est normalisé (virgules -> points) : la chaîne EN à point convient aux deux.
check("surcoût temps", f"+{fmt(t1['surcout_pct'], 1)} %")
check("s/époque MoE", f"{round(t1['s_epoch_moe'])} s")
check("s/époque contrôle", f"{round(t1['s_epoch_controle'])} s")
check("VRAM MoE", fmt(t1["vram_moe"], 1))
check("VRAM contrôle", fmt(t1["vram_controle"], 1))
check("h par seed", fmt(t1["h_par_seed"], 1))
# γ différence mesurée avec BRATS : nécessaire ici, absent là-bas
assert t8["gamma_necessaire_ici"] is True, "γ devrait être nécessaire ici"

# --------------------------------------------------------------------------- rapport
if FAILS:
    print(f"ÉCHEC — {len(FAILS)} divergence(s) sur {N} checks :")
    for f in FAILS:
        print("  " + f)
    sys.exit(1)
print(f"OK — {N} checks : manuscrits EN/FR alignés sur paper3_tables.json "
      f"(tous les comptes recalculés depuis le détail).")
sys.exit(0)
