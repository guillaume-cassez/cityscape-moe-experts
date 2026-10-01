#!/usr/bin/env python3
"""P3.18a — Tables T1-T8 du paper 3 (MoE-V3-CS), SANS GPU, depuis les artefacts vérifiés.

Aucun GPU ni forward n'est nécessaire : toutes les statistiques existent déjà sur disque
(harness P3.10, master P3.14, métiers P3.16, attribution per-class P3.16, polyvalence P3.17,
routing 3 seeds × 80 époques, provenance des experts, logs de coût). Ce script les LIT, les
CROISE et les met en forme — il ne ré-entraîne et ne ré-évalue rien.

Régime de garde-fous hérité de l'erratum v1.1.0 du paper4 (2026-10-01), où une étiquette de
famille de multiplicité ÉCRITE À LA MAIN mentait pendant que les valeurs, elles, étaient justes :

  1. AUCUN compte, aucune taille, aucune somme, aucun ratio n'est écrit en dur. Tout est
     calculé depuis l'artefact (`len(...)`, `sum(...)`, `min(...)`) et chaque compte porte un
     `check()` bloquant.
  2. Le Holm est RECOMPUTÉ depuis les p bruts avec la fonction du programme
     (`src/moe/bootstrap.holm`) et comparé à la valeur stockée à 1e-12 — dans CHACUNE des
     trois familles de multiplicité du critère primaire (1 paire P3.10, 12 paires P3.14,
     15 paires P3.16). Une étiquette de famille ne peut plus mentir : sa taille vient de
     `len(pairwise)`.
  3. Les verdicts de polyvalence (T0-T5) sont RECALCULÉS indépendamment depuis les deltas et
     les rangs bruts, puis comparés aux verdicts stockés. La thèse « seul bras polyvalent »
     est donc re-démontrée ici, pas recopiée.
  4. Les valeurs sont croisées entre artefacts avec des tolérances MESURÉES puis déclarées
     (deux forwards GPU indépendants des mêmes checkpoints diffèrent de ~6e-4 pt, cuDNN/BF16
     non déterministe) — jamais masquées.

Usage (indifféremment TPL ou Tour, CPU) :
    python3 scripts/p3_moe_tables.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# `src/moe/bootstrap.py` est chargé DIRECTEMENT au fichier, sans passer par le paquet
# `src.moe` : son `__init__.py` importe torch, alors que ce script n'a besoin que de `holm()`
# (numpy seul). C'est exactement le même code, vérifié par les checks de recomputation.
_spec = importlib.util.spec_from_file_location("_bootstrap", REPO / "src" / "moe" / "bootstrap.py")
_bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
holm = _bs.holm

RES = REPO / "results" / "moe_v3_cs"
HARNESS = RES / "harness" / "table_controle_v3cs_P310.json"
P314 = RES / "p314" / "master_table.json"
P316_MET = RES / "metiers_experts" / "table_metiers_experts.json"
P316_ATT = RES / "p316" / "attribution_perclass_vs_controle.json"
P317 = RES / "p317_polyvalence" / "table_polyvalence.json"
P317_SEEDS = RES / "p317_polyvalence" / "verif_seeds.md"
OUT_TABLES = REPO / "papers" / "paper3" / "tables"
BRATS_P3_YAML = Path("/home/dev/mnt/tour/BRATS/papers/paper3/paper.yaml")

SEEDS = (42, 123, 456)
ARM = "moe_v3cs"
REF = "controle"

LABEL = {
    "A": "A · CE seule", "B": "B · CE+Dice (baseline)", "C": "C · CE+Dice+EDT",
    "Cp": "Cp · CE+Dice+SDT", "D": "D · CE+Kervadec EDT", "Dp": "Dp · CE+SDT (distmap)",
    "G": "G · CE+Dice+Blob (Kofler)",
    "fused_CvetoB": "consensus C⊘B", "fused_CpvetoB": "consensus Cp⊘B",
    "fused_DvetoB": "consensus D⊘B", "fused_DpvetoB": "consensus Dp⊘B",
    "moe_v3cs": "MoE-V3-CS (4 experts)", "controle": "contrôle (CE+Dice 80 ep, réf)",
}
COURT = {**LABEL, "moe_v3cs": "MoE-V3-CS", "controle": "contrôle"}

# Unité d'affichage par métrique métier. Ce n'est pas un nombre mesuré mais une déclaration
# sémantique ; elle est VERROUILLÉE par un check bloquant (toute métrique « fraction » doit
# avoir ses points dans [0,1], `fragments` doit être > 1) : une inversion d'unité ne passe pas.
FRACTION_METRICS = None  # rempli à l'exécution depuis l'artefact (voir unite())

SANITY: list[dict] = []


def log(msg: str) -> None:
    print(f"[p3 {datetime.now():%H:%M:%S}] {msg}", flush=True)


def check(nom: str, ok: bool, detail: str) -> None:
    """Sanity BLOQUANTE : échec = sortie non nulle, aucune table écrite."""
    SANITY.append({"check": nom, "ok": bool(ok), "detail": detail})
    log(f"  [sanity {'✅' if ok else '❌'}] {nom} — {detail}")
    if not ok:
        raise SystemExit(f"SANITY ÉCHOUÉE : {nom} — {detail}")


def j(x: Path) -> dict:
    return json.loads(Path(x).read_text(encoding="utf-8"))


def pt(v: float, nd: int = 3, signed: bool = False) -> str:
    """Fraction -> points de pourcentage."""
    return f"{v * 100:+.{nd}f}" if signed else f"{v * 100:.{nd}f}"


def fp(p: float, nd: int = 4) -> str:
    return f"{p:.{nd}f}"


def ic(c: list, nd: int = 3, frac: bool = True) -> str:
    f = 100.0 if frac else 1.0
    return f"[{c[0] * f:+.{nd}f} ; {c[1] * f:+.{nd}f}]"


# --------------------------------------------------------------------------- #
# Chargement + familles de multiplicité (le cœur du régime anti-erratum)
# --------------------------------------------------------------------------- #

def familles() -> dict:
    """Les TROIS familles de multiplicité du critère primaire, tailles CALCULÉES.

    Retourne aussi le Holm recomputé depuis les p bruts et l'écart maximal au Holm stocké :
    c'est ce qui rend une étiquette de famille infalsifiable.
    """
    h, p314, met, poly = j(HARNESS), j(P314), j(P316_MET), j(P317)

    # Famille 1 — la paire UNIQUE pré-enregistrée (P3.10). Holm = p par définition.
    cle1 = next(iter(h["bootstrap_miou"]["pairwise"]))
    pw1 = h["bootstrap_miou"]["pairwise"][cle1]
    f1 = {cle1: pw1["p_two_sided"]}
    h1 = holm(f1)
    check("famille 1 paire (P3.10) : taille calculée == 1", len(f1) == 1, f"len = {len(f1)}")
    check("famille 1 paire : Holm == p (recomputé)", abs(h1[cle1] - pw1["p_two_sided"]) < 1e-12,
          f"holm={h1[cle1]:.6g} p={pw1['p_two_sided']:.6g}")
    check("famille 1 paire : Holm recomputé == stocké",
          abs(h1[cle1] - pw1["p_holm"]) < 1e-12, f"{h1[cle1]:.6g} vs {pw1['p_holm']:.6g}")

    # Famille 12 paires — master table P3.14 (les 12 bras contre le contrôle).
    f12 = {k: v["p_two_sided"] for k, v in p314["pairwise"].items()}
    h12 = holm(f12)
    e12 = max(abs(h12[k] - p314["pairwise"][k]["p_holm"]) for k in f12)
    check("famille P3.14 : taille calculée == 12", len(f12) == 12, f"len(pairwise) = {len(f12)}")
    check("famille P3.14 : Holm recomputé == stocké (12 paires)", e12 < 1e-12,
          f"écart max {e12:.3g}")

    # Famille 15 paires — P3.16 (12 bras vs contrôle + 3 fusion-vs-expert... mesuré).
    pw15 = met["tables"]["mIoU"]["pairwise"]
    f15 = {k: v["p_two_sided"] for k, v in pw15.items()}
    h15 = holm(f15)
    e15 = max(abs(h15[k] - pw15[k]["p_holm"]) for k in f15)
    check("famille P3.16 : taille calculée == len(pairs) déclaré",
          len(f15) == len(met["pairs"]), f"{len(f15)} vs {len(met['pairs'])}")
    check("famille P3.16 : Holm recomputé == stocké", e15 < 1e-12, f"écart max {e15:.3g}")

    # La polyvalence P3.17 doit citer la famille 15 paires (sinon un papier qui recopie
    # P3.17 citerait une famille différente de celle annoncée — le défaut du paper4).
    ecarts = []
    for a in p314["arms"]:
        k = f"{a}_vs_{REF}"
        if a == REF or k not in pw15:
            continue
        ecarts.append(abs(poly["p_holm"][a]["mIoU"] - h15[k]))
    check("P3.17 cite la famille 15 paires (pas une autre)", max(ecarts) < 1e-9,
          f"{len(ecarts)} bras comparés, écart max {max(ecarts):.3g}")
    check("les 3 familles sont DISTINCTES (Holm moe différent partout)",
          len({round(h1[cle1], 9), round(h12[f"{ARM}_vs_{REF}"], 9),
               round(h15[f"{ARM}_vs_{REF}"], 9)}) == 3,
          f"1 paire {h1[cle1]:.4f} / 12 paires {h12[f'{ARM}_vs_{REF}']:.4f} / "
          f"15 paires {h15[f'{ARM}_vs_{REF}']:.4f}")

    # Bras effectivement comparés AU CONTRÔLE dans la famille 15 paires. Les paires
    # fusion-contre-expert (`fused_DvetoB_vs_D`, …) en font partie mais ne doivent PAS être
    # prises pour des bras : un retrait de suffixe produirait des clés fantômes.
    # Le compte de ces paires est CALCULÉ depuis `met["pairs"]` — l'écrire en dur est
    # exactement le travers que ce script interdit (mesuré : il y en a 4, pas 3).
    suffixe = f"_vs_{REF}"
    couverts = {k[: -len(suffixe)] for k in pw15 if k.endswith(suffixe)}
    n_vs_ref = sum(1 for _a, b in met["pairs"] if b == REF)
    n_hors_ref = len(met["pairs"]) - n_vs_ref
    check("famille P3.16 : bras vs contrôle == paires déclarées vs contrôle",
          len(couverts) == n_vs_ref and len(couverts) + n_hors_ref == len(f15),
          f"{len(couverts)} bras vs contrôle + {n_hors_ref} paires fusion-vs-expert "
          f"= {len(f15)} paires")
    check("le MoE est parmi les bras comparés au contrôle", ARM in couverts,
          f"{len(couverts)} bras : {sorted(couverts)}")

    return {"cle1": cle1, "h1": h1, "n1": len(f1), "p1": f1,
            "h12": h12, "n12": len(f12), "p12": f12,
            "h15": h15, "n15": len(f15), "p15": f15,
            "couverts_15": couverts,
            "harness": h, "p314": p314, "met": met, "poly": poly}


def unite(metrique: str, points: dict) -> tuple[float, int, str]:
    """(facteur, décimales, nom d'unité) — déduit puis VERROUILLÉ par un check bloquant."""
    vals = [v for v in points.values() if v is not None]
    if metrique == "fragments":
        ok = all(v > 1.0 for v in vals)
        check(f"unité de « {metrique} » = composantes/image (valeurs > 1)", ok,
              f"min {min(vals):.1f}, max {max(vals):.1f}")
        return 1.0, 1, "composantes/image"
    ok = all(0.0 <= v <= 1.0 for v in vals)
    check(f"unité de « {metrique} » = fraction dans [0,1] -> points", ok,
          f"min {min(vals):.6f}, max {max(vals):.6f}")
    return 100.0, 3, "pt"


# --------------------------------------------------------------------------- #
# T1 — architecture, recette, coût
# --------------------------------------------------------------------------- #

def parse_logs() -> dict:
    """s/époque et VRAM relevés dans les logs d'entraînement (jamais recopiés)."""
    out = {}
    for bras, motif, rx_t, rx_v in [
        (ARM, "moe_v3cs_Binit", r"\|\s*(\d+)s\s*$", r"VRAM=([\d.]+)GB"),
        (REF, "control_nomoe_Binit", r"time=(\d+)s", r"VRAM=([\d.]+)GB"),
    ]:
        per_seed = {}
        for s in SEEDS:
            f = REPO / "results" / "p3_queue" / f"{motif}_seed{s}.log"
            txt = f.read_text(encoding="utf-8", errors="replace")
            lignes = [l for l in txt.splitlines() if re.match(r"^Epoch \d+/\d+ \|", l)]
            temps = [int(re.search(rx_t, l).group(1)) for l in lignes if re.search(rx_t, l)]
            vram = [float(re.search(rx_v, l).group(1)) for l in lignes if re.search(rx_v, l)]
            check(f"log {motif} seed{s} : 80 époques lues", len(temps) == 80,
                  f"{len(temps)} lignes de temps, {len(vram)} de VRAM")
            per_seed[s] = {"s_epoch": temps, "vram": vram,
                           "s_moy": sum(temps) / len(temps), "vram_max": max(vram)}
        out[bras] = per_seed
    return out


def t1_archi(fam: dict, logs: dict) -> tuple[str, dict]:
    sums = {s: j(RES / f"train_moe_v3cs_Binit_seed{s}_summary.json") for s in SEEDS}
    prov = sums[SEEDS[0]]["provenance"]
    kw = prov["moe_kwargs"]
    n_exp, top_k = int(kw["n_experts"]), int(kw["top_k"])
    grille = tuple(kw["grid"])

    # Provenance des experts : l'initialisation bit-exacte est la thèse du papier, elle est
    # vérifiée sur CHAQUE expert de CHAQUE seed (pas seulement sur le seed montré).
    ecarts, meth = [], []
    for s in SEEDS:
        for e in sums[s]["provenance"]["experts"]:
            ecarts.append(float(e["max_ecart"]))
            meth.append((s, e["expert"], e["method"], e["seed"]))
    check("experts initialisés bit-à-bit (max_ecart = 0 partout)", max(ecarts) == 0.0,
          f"{len(ecarts)} copies, max_ecart {max(ecarts)}")
    check("nombre de copies d'experts = n_experts × n_seeds",
          len(ecarts) == n_exp * len(SEEDS), f"{len(ecarts)} vs {n_exp}×{len(SEEDS)}")
    check("chaque expert apparié au MÊME seed que le run",
          all(ms == es for ms, _e, _m, es in meth),
          f"{len(meth)} couples (seed run, seed expert) tous égaux")
    check("4 méthodes distinctes en experts", len({m for _s, _e, m, _x in meth}) == n_exp,
          str(sorted({m for _s, _e, m, _x in meth})))
    check("source_block = head.fpn_convs.0", prov["source_block"] == "head.fpn_convs.0",
          prov["source_block"])
    check("gate aléatoire (aucun pré-appariement)", prov["gate"] == "random", prov["gate"])
    check("époques = 80 pour les 3 seeds", all(sums[s]["epochs"] == 80 for s in SEEDS),
          str([sums[s]["epochs"] for s in SEEDS]))

    # Équirépartition EXACTE de part_max pour top_k parmi n_experts : calculée, pas 0,5 en dur.
    equi = top_k / n_exp
    check("équirépartition part_max = top_k/n_experts", abs(equi - 0.5) < 1e-12,
          f"{top_k}/{n_exp} = {equi}")
    check("residual_scale initial = 0 (époque 0 bit-exacte au contrôle)",
          kw["residual_scale"] == 0.0, str(kw["residual_scale"]))

    m = {s: logs[ARM][s] for s in SEEDS}
    c = {s: logs[REF][s] for s in SEEDS}
    s_moe = sum(m[s]["s_moy"] for s in SEEDS) / len(SEEDS)
    s_ctl = sum(c[s]["s_moy"] for s in SEEDS) / len(SEEDS)
    surcout = (s_moe / s_ctl - 1.0) * 100.0
    v_moe = max(m[s]["vram_max"] for s in SEEDS)
    v_ctl = max(c[s]["vram_max"] for s in SEEDS)
    check("surcoût temporel MoE vs contrôle > 0", surcout > 0, f"+{surcout:.1f} %")
    check("surcoût VRAM MoE vs contrôle > 0", v_moe > v_ctl, f"{v_moe} vs {v_ctl} Go")
    h_seed = sum(sums[s]["elapsed_s"] for s in SEEDS) / len(SEEDS) / 3600.0

    # γ final : le diagnostic direct de la thèse (γ→0 = MoE inutile).
    gam = {s: sums[s]["final_epoch_stats"]["gamma_absmean"] for s in SEEDS}
    ep0 = {s: json.loads((RES / f"routing_moe_v3cs_Binit_seed{s}.jsonl").read_text().splitlines()[0])
           for s in SEEDS}
    check("γ croît strictement de l'époque 0 à la finale (3 seeds)",
          all(gam[s] > ep0[s]["gamma_absmean"] for s in SEEDS),
          " · ".join(f"s{s} {ep0[s]['gamma_absmean']:.5f}→{gam[s]:.5f}" for s in SEEDS))
    check("γ final strictement positif (le MoE sert)", all(gam[s] > 0 for s in SEEDS),
          f"min {min(gam.values()):.5f}")

    lg = lambda k: " ".join(f"{sums[s]['final_epoch_stats'][k]:.5f}" for s in SEEDS)  # noqa: E731
    lignes = [
        ("Backbone", "ConvNeXt-V2-Base (ImageNet-22K) + UPerNet, pleine résolution 1024×2048, "
                     "19 classes Cityscapes, BF16, channels_last",
         "provenance backbone = méthode B, epoch 160"),
        ("Point d'accroche", "`head.fpn_bottleneck` — la couche MoE est le DERNIER élément du "
                             "Sequential (indices conservés 0=conv, 1=BN, 2=ReLU, 3=MoE), donc "
                             "`fused ← fused + experts(fused)`",
         "src/moe/moe_model.py:V3CS_ATTACH_SEQ + attach_patch_moe"),
        ("Experts", f"{n_exp}, initialisés depuis `{prov['source_block']}` des 4 spécialistes "
                    f"du MÊME seed (B, D, Dp, G), entraînés 160 époques",
         f"provenance experts, max_ecart = {max(ecarts)} sur {len(ecarts)} copies"),
        ("Routage", f"top-{top_k} parmi {n_exp}, grille de patches {grille[0]}×{grille[1]}, "
                    f"gate convolutif kernel {kw['kernel_size']}, porte {prov['gate']}",
         "provenance moe_kwargs"),
        ("Équilibrage", f"loss {kw['balance']} poids {kw['balance_weight']} "
                        f"(aux pondéré {lg('aux_loss_weighted')})", "moe_kwargs + logs"),
        ("Bruit de Shazeer", f"std {kw['noise_std']} × std(logits), recuit sur "
                             f"{kw['noise_anneal']} époques",
         "moe_kwargs + routing jsonl (noise_std 1,00 → 0,00)"),
        ("Garde-fou résiduel γ", f"échelle résiduelle apprenable par canal, init "
                                 f"{kw['residual_scale']} → |γ| moyen {lg('gamma_absmean')}",
         "moe_kwargs + final_epoch_stats"),
        ("Contrôle apparié", "`control_nomoe_Binit` : même initialisation B, même recette, "
                             "80 époques, SANS couche MoE, 3 seeds",
         "results/p3_queue/control_nomoe_Binit_seed*.log"),
        ("Recette", "80 époques, batch 2 × accumulation 4 (effectif 8), AdamW lr 6e-5, wd 0,01, "
                    "betas (0,9 ; 0,999), poly power 1,0, warmup 1 époque, BF16, hflip p=0,5",
         "scripts/train_moe_v3cs.py"),
        ("Coût MoE", f"{s_moe:.0f} s/époque en moyenne, VRAM {v_moe:.1f} Go, "
                     f"{h_seed:.1f} h/seed (elapsed_s moyen {sum(sums[s]['elapsed_s'] for s in SEEDS)/len(SEEDS):,.0f} s)",
         "logs 3 seeds × 80 époques + summaries"),
        ("Coût contrôle", f"{s_ctl:.0f} s/époque, VRAM {v_ctl:.1f} Go", "logs 3 seeds × 80 époques"),
        ("Surcoût du MoE", f"+{surcout:.1f} % de temps/époque, "
                           f"+{v_moe - v_ctl:.1f} Go de VRAM", "calculé depuis les logs"),
        ("Budget complet", f"les {n_exp} experts doivent exister (4 × 160 époques) + 80 époques "
                           "de MoE et 80 de contrôle", "provenance experts (epoch 160)"),
    ]
    md = ["# T1 — Objet technique, recette et coût (MoE-V3-CS)", "",
          "| Poste | Valeur mesurée | Source |", "|---|---|---|"]
    md += [f"| {a} | {b} | `{c}` |" for a, b, c in lignes]
    md += ["", f"Équirépartition exacte de `part_max` pour top-{top_k} parmi {n_exp} experts : "
               f"**{equi:.4f}** (calculé `top_k/n_experts`, jamais recopié). C'est la valeur de "
               f"référence contre laquelle T4 lit le routage.", "",
           f"Provenance vérifiée sur les {len(ecarts)} copies d'experts ({n_exp} experts × "
           f"{len(SEEDS)} seeds) : `max_ecart` maximal = **{max(ecarts)}**, et chaque expert est "
           f"apparié au même seed que son run — d'où l'appariement strict par seed."]
    data = {"moe_kwargs": kw, "n_experts": n_exp, "top_k": top_k, "grille": list(grille),
            "equirepartition_part_max": equi, "source_block": prov["source_block"],
            "gate": prov["gate"], "max_ecart_experts": max(ecarts),
            "n_copies_experts": len(ecarts), "gamma_final": {str(s): gam[s] for s in SEEDS},
            "gamma_ep0": {str(s): ep0[s]["gamma_absmean"] for s in SEEDS},
            "s_epoch_moe": s_moe, "s_epoch_controle": s_ctl, "surcout_pct": surcout,
            "vram_moe": v_moe, "vram_controle": v_ctl, "h_par_seed": h_seed,
            "elapsed_s": {str(s): sums[s]["elapsed_s"] for s in SEEDS}}
    return "\n".join(md) + "\n", data


# --------------------------------------------------------------------------- #
# T2 — critère primaire pré-enregistré
# --------------------------------------------------------------------------- #

def t2_primaire(fam: dict) -> tuple[str, dict]:
    h, p314, met = fam["harness"], fam["p314"], fam["met"]
    bm = h["bootstrap_miou"]
    cle = fam["cle1"]
    pw = bm["pairwise"][cle]
    # La clé du harness est `controle_vs_moe_v3cs` : le delta stocké est (réf − bras).
    signe = -1.0 if cle.startswith(f"{REF}_vs_") else 1.0
    d = signe * pw["delta"] * 100.0
    lo, hi = sorted([signe * pw["ci95"][0] * 100.0, signe * pw["ci95"][1] * 100.0])
    p, holm1 = pw["p_two_sided"], fam["h1"][cle]
    k12, k15 = f"{ARM}_vs_{REF}", f"{ARM}_vs_{REF}"
    holm12, holm15 = fam["h12"][k12], fam["h15"][k15]

    check("primaire : Δ(moe − contrôle) positif", d > 0, f"{d:+.4f} pt")
    check("primaire : IC95 exclut 0 (significatif au sens IC)", lo > 0, f"[{lo:+.4f} ; {hi:+.4f}]")
    check("primaire : p < 0,05", p < 0.05, fp(p))
    check("primaire : le harnais annonce la paire pré-enregistrée moe/contrôle",
          {ARM, REF} == set(cle.split("_vs_")), cle)

    seeds_m = h["arms"][ARM]["miou_dataset_par_seed"]
    seeds_c = h["arms"][REF]["miou_dataset_par_seed"]
    check("3 seeds par bras dans le harnais", len(seeds_m) == 3 and len(seeds_c) == 3,
          f"{len(seeds_m)} / {len(seeds_c)}")
    # Le point publié = moyenne des seeds dans chaque réplicat : la moyenne des mIoU
    # dataset-level par seed doit donc le redonner.
    for nom, sd, attendu in [(ARM, seeds_m, bm["point"][ARM]), (REF, seeds_c, bm["point"][REF])]:
        moy = sum(sd.values()) / len(sd)
        check(f"primaire : moyenne des mIoU par seed == point bootstrap ({nom})",
              abs(moy - attendu) < 1e-12, f"{moy * 100:.6f} vs {attendu * 100:.6f}")

    # Deux forwards GPU indépendants des mêmes checkpoints coexistent dans les artefacts.
    # L'écart est MESURÉ puis déclaré, jamais masqué.
    alt = met["tables"]["mIoU"]["point"][ARM]
    altc = met["tables"]["mIoU"]["point"][REF]
    ecart = abs(alt - bm["point"][ARM]) * 100.0
    ecartc = abs(altc - bm["point"][REF]) * 100.0
    check("écart harness ↔ métiers P3.16 (2ᵉ forward) < 0,01 pt", ecart < 0.01,
          f"moe {ecart:.2e} pt, contrôle {ecartc:.2e} pt")
    d_alt = met["tables"]["mIoU"]["pairwise"][k15]["delta"] * 100.0
    check("verdict identique dans les deux forwards (significatif partout)",
          (d > 0) == (d_alt > 0) and p < 0.05,
          f"harness {d:+.4f} pt p={fp(p)} · métiers {d_alt:+.4f} pt")

    # Cohérence du Δ entre les 5 artefacts qui le portent.
    srcs = {"harness P3.10": d, "master P3.14": p314["pairwise"][k12]["delta"] * 100.0,
            "métiers P3.16": d_alt,
            "attribution P3.16": j(P316_ATT)["arms"][ARM]["miou_delta_pt"],
            "polyvalence P3.17": fam["poly"]["delta_vs_controle_pt"][ARM]["mIoU"]}
    amplitude = max(srcs.values()) - min(srcs.values())
    check("Δ primaire cohérent entre 5 artefacts (< 0,01 pt)", amplitude < 0.01,
          f"amplitude {amplitude:.2e} pt sur {len(srcs)} sources")

    ci_m = bm["ci95"][ARM]
    ci_c = bm["ci95"][REF]
    md = ["# T2 — Critère primaire pré-enregistré : MoE-V3-CS vs son contrôle apparié", "",
          f"**Primary metric / métrique primaire** : mIoU dataset-level officiel "
          f"(cityscapesScripts, 19 classes) sur le holdout partagé `first:{h['holdout']['n']}`, "
          f"bras `{ARM}` contre son **contrôle apparié** `{REF}` (même initialisation B, même "
          f"recette, 80 époques, sans couche MoE). Bootstrap **apparié par image** "
          f"B = {bm['B']:,} (seed {h['bootstrap_seed']}), seeds {SEEDS[0]}/{SEEDS[1]}/{SEEDS[2]} "
          f"**moyennés dans chaque réplicat**, p bilatéral. Critère déclaré avant les runs "
          f"d'analyse (protocole P3.10).", "",
          "| Quantité | MoE-V3-CS | contrôle apparié |", "|---|---|---|",
          f"| mIoU point | **{pt(bm['point'][ARM])}** | **{pt(bm['point'][REF])}** |",
          f"| IC95 | [{ci_m[0] * 100:.3f} ; {ci_m[1] * 100:.3f}] | "
          f"[{ci_c[0] * 100:.3f} ; {ci_c[1] * 100:.3f}] |"]
    for s in SEEDS:
        md.append(f"| mIoU seed {s} | {seeds_m[str(s)] * 100:.3f} | {seeds_c[str(s)] * 100:.3f} |")
    md += ["",
           f"**Δ(MoE − contrôle) = {d:+.3f} pt · IC95 [{lo:+.3f} ; {hi:+.3f}] · p = {fp(p)} → "
           f"SIGNIFICATIF.**", "",
           "## Les trois familles de multiplicité, données ensemble", "",
           "Le même contraste appartient à trois familles de Holm différentes selon le protocole "
           "qui le déclare. Les trois sont rapportées, aucune n'est choisie parce qu'elle "
           "arrange — et leurs tailles sont **calculées** depuis les artefacts, le Holm étant "
           "**recomputé** depuis les p bruts puis comparé à la valeur stockée (écart < 1e-12).", "",
           "| Famille | Taille | Holm du critère primaire | Artefact source |",
           "|---|---|---|---|",
           f"| paire unique pré-enregistrée (P3.10) | {fam['n1']} | **{holm1:.4f}** (= p) | "
           f"`results/moe_v3_cs/harness/table_controle_v3cs_P310.json` |",
           f"| bras contre contrôle (P3.14) | {fam['n12']} | {holm12:.4f} | "
           f"`results/moe_v3_cs/p314/master_table.json` |",
           f"| exploratoire du programme (P3.16) | {fam['n15']} | {holm15:.4f} | "
           f"`results/moe_v3_cs/metiers_experts/table_metiers_experts.json` |", "",
           f"Lecture honnête : le critère primaire est significatif au seuil brut "
           f"(p = {fp(p)}) et **dans la famille pré-enregistrée à {fam['n1']} paire** "
           f"(Holm = {holm1:.4f}). Il **ne survit pas** à la famille exploratoire à "
           f"{fam['n15']} paires (Holm = {holm15:.4f} > 0,05) — et aucun bras du plateau n'y "
           f"survit non plus (meilleur Holm {min(fam['h15'].values()):.4f}). C'est écrit tel "
           f"quel : sur le mIoU, les conclusions du programme reposent sur les **ampleurs et "
           f"leurs IC**, et la contribution de ce papier porte sur ce que chaque bras **casse** "
           f"(T7), pas sur la significativité post-Holm du mIoU.", "",
           f"## Robustesse : deux forwards GPU indépendants des mêmes checkpoints", "",
           f"| Source | mIoU MoE | mIoU contrôle | Δ | p |", "|---|---|---|---|---|",
           f"| harnais P3.10 (primaire cité ci-dessus) | {bm['point'][ARM] * 100:.6f} | "
           f"{bm['point'][REF] * 100:.6f} | {d:+.6f} pt | {fp(p)} |",
           f"| régénération métiers P3.16 | {alt * 100:.6f} | {altc * 100:.6f} | "
           f"{d_alt:+.6f} pt | {fp(met['tables']['mIoU']['pairwise'][k15]['p_two_sided'])} |", "",
           f"Écart mesuré **{ecart:.2e} pt** sur le mIoU du MoE et **{ecartc:.2e} pt** sur celui "
           f"du contrôle : non-déterminisme cuDNN/BF16 (`deterministic: false`), du même ordre "
           f"que la sanity déclarée par P3.17 "
           f"({fam['poly']['sanity_miou_vs_p314_ecart_max_pt']:.2e} pt). Le verdict est "
           f"identique dans les deux sources.", "",
           f"Provenance : `{HARNESS.relative_to(REPO)}`, `{P314.relative_to(REPO)}`, "
           f"`{P316_MET.relative_to(REPO)}`."]

    data = {"point_moe": bm["point"][ARM], "point_controle": bm["point"][REF],
            "ci_moe": bm["ci95"][ARM], "ci_controle": bm["ci95"][REF],
            "delta_pt": d, "ci_lo_pt": lo, "ci_hi_pt": hi, "p": p,
            "holm_famille_1": holm1, "holm_famille_12": holm12, "holm_famille_15": holm15,
            "n_famille_1": fam["n1"], "n_famille_12": fam["n12"], "n_famille_15": fam["n15"],
            "meilleur_holm_15": min(fam["h15"].values()),
            "per_seed_moe": seeds_m, "per_seed_controle": seeds_c,
            "second_forward": {"moe": alt, "controle": altc, "delta_pt": d_alt,
                               "ecart_moe_pt": ecart, "ecart_controle_pt": ecartc},
            "delta_par_source": srcs, "amplitude_delta_pt": amplitude,
            "B_boot": bm["B"], "bootstrap_seed": h["bootstrap_seed"],
            "n_images": bm["n_images"], "holdout": h["holdout"]["description"]}
    return "\n".join(md) + "\n", data


# --------------------------------------------------------------------------- #
# T3 — position parmi les 13 bras
# --------------------------------------------------------------------------- #

def t3_master(fam: dict) -> tuple[str, str, dict]:
    p314 = fam["p314"]
    bras = [a for a in p314["arms"] if a != REF]
    check("master P3.14 : 13 bras dont le contrôle", len(p314["arms"]) == 13 and len(bras) == 12,
          f"{len(p314['arms'])} bras, {len(bras)} hors contrôle")
    rows = sorted(((a, p314["pairwise"][f"{a}_vs_{REF}"]["delta"]) for a in bras),
                  key=lambda x: -x[1])
    rows13 = sorted(list(rows) + [(REF, 0.0)], key=lambda x: -x[1])
    check("classement trié par Δ décroissant",
          all(rows13[i][1] >= rows13[i + 1][1] for i in range(len(rows13) - 1)),
          f"{len(rows13)} bras")
    rang = [a for a, _ in rows].index(ARM) + 1
    rang13 = [a for a, _ in rows13].index(ARM) + 1
    rang_ctl = [a for a, _ in rows13].index(REF) + 1
    check("rang du MoE cohérent avec la position du contrôle",
          rang13 == rang + (1 if rang_ctl <= rang else 0),
          f"rang {rang}/12, rang13 {rang13}/13, contrôle {rang_ctl}/13")

    md = ["# T3 — Position du MoE-V3-CS parmi les 13 bras du programme", "",
          "mIoU dataset-level, holdout `first:500`, bootstrap apparié B = 10 000 "
          "(seed 20260618), seeds moyennés dans chaque réplicat, référence = contrôle apparié. "
          "Les **deux** familles de Holm sont données, avec leur taille calculée depuis "
          "l'artefact (leçon de l'erratum v1.1.0 du paper4 : une étiquette de famille écrite à "
          "la main finit par mentir).", "",
          f"| # | Bras | mIoU | Δ vs contrôle | IC95 | p | Holm ({fam['n12']} paires, P3.14) | "
          f"Holm ({fam['n15']} paires, P3.16) |", "|---|---|---|---|---|---|---|---|"]
    for i, (a, _d) in enumerate(rows, 1):
        e = p314["pairwise"][f"{a}_vs_{REF}"]
        h15 = (f"{fam['h15'][f'{a}_vs_{REF}']:.4f}" if a in fam["couverts_15"] else "n.c. (a)")
        # Le gras marque la significativité au seuil brut. Le marqueur « ce papier » est placé
        # HORS de la portion grasée : l'imbrication `**A ← **B****` casse le rendu markdown
        # (défaut vu sur la ligne du MoE à la première passe).
        gras = ("**", "**") if e["p_two_sided"] < 0.05 else ("", "")
        star = " ← ce papier" if a == ARM else ""
        md.append(f"| {i} | {gras[0]}{LABEL[a]}{gras[1]}{star} | {p314['point'][a] * 100:.3f} | "
                  f"{e['delta'] * 100:+.3f} | [{e['ci95'][0] * 100:+.3f} ; "
                  f"{e['ci95'][1] * 100:+.3f}] | {fp(e['p_two_sided'])} | "
                  f"{e['p_holm']:.4f} | {h15} |")
    md.append(f"| — | {LABEL[REF]} | {p314['point'][REF] * 100:.3f} | 0 (réf) | — | — | — | — |")

    n_sig_brut = sum(1 for a, _ in rows if p314["pairwise"][f"{a}_vs_{REF}"]["p_two_sided"] < 0.05)
    n_sig_h12 = sum(1 for a, _ in rows if fam["h12"][f"{a}_vs_{REF}"] < 0.05)
    n_sig_h15 = sum(1 for a in fam["couverts_15"] if fam["h15"][f"{a}_vs_{REF}"] < 0.05)
    check("aucun bras significatif après Holm (les deux familles)",
          n_sig_h12 == 0 and n_sig_h15 == 0,
          f"{n_sig_brut} bras à p brut < 0,05 ; {n_sig_h12} après Holm {fam['n12']} ; "
          f"{n_sig_h15} après Holm {fam['n15']}")
    hors = sorted(set(p314["arms"]) - {REF} - fam["couverts_15"])
    md += ["", f"(a) hors de la famille {fam['n15']} paires (couverture partielle) : "
               f"{', '.join(hors) if hors else 'aucun bras'}.", "",
           f"Le MoE-V3-CS est **{rang}ᵉ sur {len(rows)}** par amplitude "
           f"({rang13}ᵉ sur {len(rows13)} contrôle inclus — le contrôle, Δ = 0, se classe "
           f"{rang_ctl}ᵉ), derrière D, les deux consensus D⊘B et Dp⊘B, et Dp. "
           f"{n_sig_brut} bras ont un p brut < 0,05 ; **aucun** ne survit au Holm, ni sur "
           f"{fam['n12']} paires (meilleur {min(fam['h12'].values()):.4f}) ni sur "
           f"{fam['n15']} (meilleur {min(fam['h15'].values()):.4f}).", "",
           "C'est le point de bascule du papier : puisque l'amplitude de mIoU ne sépare pas les "
           "bras de façon significative après correction, la conclusion porte sur **ce que "
           "chaque bras casse** (T7), où le MoE-V3-CS est premier avec une marge mesurée."]

    lignes = []
    for i, (a, _d) in enumerate(rows, 1):
        e = p314["pairwise"][f"{a}_vs_{REF}"]
        lignes.append({"rang": i, "bras": a, "label": LABEL[a],
                       "miou_pct": p314["point"][a] * 100.0, "delta_pt": e["delta"] * 100.0,
                       "ci_lo_pt": e["ci95"][0] * 100.0, "ci_hi_pt": e["ci95"][1] * 100.0,
                       "p": e["p_two_sided"], "holm_12": e["p_holm"],
                       "holm_15": fam["h15"].get(f"{a}_vs_{REF}"),
                       "significatif_brut": bool(e["p_two_sided"] < 0.05)})
    csv_txt = _csv(["rang", "bras", "label", "miou_pct", "delta_pt", "ci_lo_pt", "ci_hi_pt",
                    "p", "holm_12", "holm_15", "significatif_brut"], lignes)
    data = {"classement": lignes, "rang_moe": rang, "rang_moe_13": rang13,
            "rang_controle_13": rang_ctl, "n_bras": len(p314["arms"]),
            "n_sig_brut": n_sig_brut, "n_sig_holm_12": n_sig_h12, "n_sig_holm_15": n_sig_h15,
            "meilleur_holm_12": min(fam["h12"].values()),
            "meilleur_holm_15": min(fam["h15"].values()),
            "bras_hors_famille_15": hors, "points": p314["point"]}
    return "\n".join(md) + "\n", csv_txt, data


def _csv(colonnes: list[str], lignes: list[dict]) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=colonnes, extrasaction="ignore")
    w.writeheader()
    for l in lignes:
        w.writerow(l)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# T4 — diagnostic de routage : « la porte ne choisit pas »
# --------------------------------------------------------------------------- #

def t4_routage(fam: dict) -> tuple[str, str, dict]:
    kw = j(RES / f"train_moe_v3cs_Binit_seed{SEEDS[0]}_summary.json")["provenance"]["moe_kwargs"]
    n_exp, top_k, anneal = int(kw["n_experts"]), int(kw["top_k"]), int(kw["noise_anneal"])
    equi = top_k / n_exp
    rout, sums = {}, {}
    for s in SEEDS:
        lignes = [json.loads(l) for l in
                  (RES / f"routing_moe_v3cs_Binit_seed{s}.jsonl").read_text().splitlines() if l.strip()]
        lignes.sort(key=lambda r: r["epoch"])
        rout[s] = lignes
        sums[s] = j(RES / f"train_moe_v3cs_Binit_seed{s}_summary.json")
        check(f"routing seed{s} : 80 époques, epochs 0..79 sans trou",
              [r["epoch"] for r in lignes] == list(range(80)), f"{len(lignes)} entrées")
        check(f"routing seed{s} : n_experts cohérent avec moe_kwargs",
              all(int(r["n_experts"]) == n_exp for r in lignes), str(n_exp))
        for r in lignes:
            # Invariant lu dans le code, pas supposé : `frac` est la part des PATCHS traités par
            # chaque expert (`onehot.clamp_max(1).mean(0)`, src/moe/patch_moe2d.py:197-198).
            # Avec top_k experts actifs par patch, la somme vaut top_k — pas 1. Mesuré : 2,0000000
            # avec un excès de 2,8e-8 dû à l'accumulation float32, d'où la tolérance 1e-6.
            if r["epoch"] in (0, len(lignes) - 1):
                check(f"routing seed{s} ep{r['epoch']} : somme des frac == top_k",
                      abs(sum(r["frac"]) - top_k) < 1e-6,
                      f"{sum(r['frac']):.9f} pour top_k = {top_k}")
                check(f"routing seed{s} ep{r['epoch']} : part_max == max(frac)",
                      abs(r["part_max"] - max(r["frac"])) < 1e-12, f"{r['part_max']:.9f}")
                # `part_max` = maximum des MOYENNES par batch (max(out["frac"] moyenné)),
                # `top_expert_share` = moyenne des MAXIMA par batch. Par Jensen, moyenne des
                # maxima ≥ maximum des moyennes : l'égalité que j'avais d'abord supposée est
                # fausse (mesuré 0,667 vs 0,501 à l'époque 0). C'est l'invariant réel, lu dans
                # scripts/train_moe_v3cs.py:_epoch_mean.
                check(f"routing seed{s} ep{r['epoch']} : top_expert_share ≥ part_max (Jensen)",
                      r["top_expert_share"] >= r["part_max"] - 1e-12,
                      f"moyenne des maxima {r['top_expert_share']:.6f} ≥ maximum des moyennes "
                      f"{r['part_max']:.6f}")
        check(f"routing seed{s} : entropies dans [0,1] (normalisées)",
              all(0.0 <= r[k] <= 1.0 for r in lignes
                  for k in ("entropy_norm", "entropy_token", "entropy_token_clean")),
              "entropy_norm / entropy_token / entropy_token_clean")
        # la dernière ligne du jsonl DOIT être égale au final_epoch_stats du summary
        f = sums[s]["final_epoch_stats"]
        for k in ("entropy_norm", "entropy_token", "entropy_token_clean", "gamma_absmean",
                  "part_max", "top_expert_share", "aux_loss_weighted", "noise_std"):
            check(f"routing seed{s} : jsonl ep79 == summary final_epoch_stats ({k})",
                  abs(rout[s][-1][k] - f[k]) < 1e-12, f"{rout[s][-1][k]:.9g} vs {f[k]:.9g}")
        # recuit du bruit : strictement décroissant, nul à partir de `anneal`
        ns = [r["noise_std"] for r in lignes]
        check(f"routing seed{s} : bruit recuit sur {anneal} époques puis nul",
              all(ns[i] >= ns[i + 1] for i in range(len(ns) - 1)) and ns[0] > 0
              and all(v == 0.0 for v in ns[anneal:]),
              f"noise_std {ns[0]:.2f} → {ns[anneal - 1]:.3f} → {ns[-1]:.2f}")

    finals = {s: rout[s][-1] for s in SEEDS}
    morts = {s: sum(1 for f in finals[s]["frac"] if f == 0.0) for s in SEEDS}
    check("aucun expert mort à l'époque finale (3 seeds)", sum(morts.values()) == 0,
          str(morts))
    frac_min = min(min(finals[s]["frac"]) for s in SEEDS)
    frac_max = max(max(finals[s]["frac"]) for s in SEEDS)
    part_max = {s: finals[s]["part_max"] for s in SEEDS}
    check(f"part_max final dans un voisinage de l'équirépartition {equi:.4f}",
          all(abs(v - equi) < 0.01 for v in part_max.values()),
          " · ".join(f"s{s} {part_max[s]:.4f}" for s in SEEDS))
    check(f"frac des {n_exp} experts ∈ [{frac_min:.4f} ; {frac_max:.4f}] autour de {equi:.4f}",
          frac_max - equi < 0.01 and equi - frac_min < 0.01,
          f"étendue {frac_max - frac_min:.4f} pt")
    ent = {k: {s: finals[s][k] for s in SEEDS}
           for k in ("entropy_norm", "entropy_token", "entropy_token_clean")}
    check("entropy_norm ≥ 0,98 partout (porte quasi uniforme)",
          min(ent["entropy_norm"].values()) >= 0.98, f"min {min(ent['entropy_norm'].values()):.5f}")

    # PIÈGE ÉVITÉ (mesuré le 2026-10-01) : à l'époque FINALE le bruit est recuit à 0, donc
    # `entropy_token == entropy_token_clean` PAR CONSTRUCTION — citer cette égalité comme preuve
    # que « le bruit de Shazeer n'explique rien » est une tautologie, pas un argument. La phase
    # probante est celle où le bruit est ACTIF (époques < noise_anneal) : c'est là qu'on mesure
    # l'écart entre les deux entropies et la platitude de la distribution SANS bruit.
    actif, ecart_actif, clean_min, tok_min, pm_actif = {}, {}, {}, {}, {}
    for s in SEEDS:
        ra = [r for r in rout[s] if r["noise_std"] > 0]
        check(f"seed{s} : le bruit est actif sur exactement noise_anneal époques",
              len(ra) == anneal and [r["epoch"] for r in ra] == list(range(anneal)),
              f"{len(ra)} époques à noise_std > 0 (0..{ra[-1]['epoch']})")
        check(f"seed{s} : après recuit, token == clean par construction (tautologie, non preuve)",
              all(abs(r["entropy_token"] - r["entropy_token_clean"]) < 1e-12
                  for r in rout[s] if r["noise_std"] == 0),
              f"{sum(1 for r in rout[s] if r['noise_std'] == 0)} époques à bruit nul")
        actif[s] = len(ra)
        ecart_actif[s] = max(abs(r["entropy_token"] - r["entropy_token_clean"]) for r in ra)
        clean_min[s] = min(r["entropy_token_clean"] for r in ra)
        tok_min[s] = min(r["entropy_token"] for r in ra)
        pm_actif[s] = (min(r["part_max"] for r in ra), max(r["part_max"] for r in ra))
    check("phase à bruit ACTIF : l'écart token↔clean reste négligeable devant la platitude",
          max(ecart_actif.values()) < 0.01,
          f"écart max {max(ecart_actif.values()):.2e} sur {anneal} époques × {len(SEEDS)} seeds")
    check("phase à bruit ACTIF : la distribution SANS bruit est déjà plate (clean ≥ 0,99)",
          min(clean_min.values()) >= 0.99, f"min {min(clean_min.values()):.5f}")
    check("phase à bruit ACTIF : part_max reste collé à l'équirépartition",
          all(abs(v - equi) < 0.01 for s in SEEDS for v in pm_actif[s]),
          f"bornes {min(pm_actif[s][0] for s in SEEDS):.5f}–"
          f"{max(pm_actif[s][1] for s in SEEDS):.5f} pour {equi:.4f}")
    gam0 = {s: rout[s][0]["gamma_absmean"] for s in SEEDS}
    gamF = {s: finals[s]["gamma_absmean"] for s in SEEDS}
    check("γ final ≫ γ époque 0 (le MoE sert, mais comme moyenne)",
          all(gamF[s] > 10 * gam0[s] for s in SEEDS),
          " · ".join(f"s{s} ×{gamF[s] / gam0[s]:.0f}" for s in SEEDS))

    md = ["# T4 — Diagnostic de routage : la porte ne choisit pas", "",
          f"Relevé dans `routing_moe_v3cs_Binit_seed*.jsonl` ({len(rout[SEEDS[0]])} époques × "
          f"{len(SEEDS)} seeds) et croisé avec `final_epoch_stats` des summaries (identité "
          f"vérifiée à 1e-12 sur 8 grandeurs × 3 seeds). Équirépartition exacte de `part_max` "
          f"pour top-{top_k} parmi {n_exp} experts = **{equi:.4f}** (calculée `top_k/n_experts`).", "",
          "| Grandeur | époque 0 (seed 42/123/456) | époque finale (seed 42/123/456) | "
          "référence | lecture |", "|---|---|---|---|---|"]
    def triple(d, nd=5):
        return " / ".join(f"{d[s]:.{nd}f}" for s in SEEDS)
    md += [
        f"| `entropy_norm` | {triple({s: rout[s][0]['entropy_norm'] for s in SEEDS})} | "
        f"**{triple(ent['entropy_norm'])}** | 1,0 = uniforme | porte quasi uniforme |",
        f"| `entropy_token` | {triple({s: rout[s][0]['entropy_token'] for s in SEEDS})} | "
        f"**{triple(ent['entropy_token'])}** | 1,0 = uniforme | idem, bruit inclus |",
        f"| `entropy_token_clean` | {triple({s: rout[s][0]['entropy_token_clean'] for s in SEEDS})} | "
        f"**{triple(ent['entropy_token_clean'])}** | 1,0 = uniforme | égale à `entropy_token` à "
        f"l'époque finale **par construction** (bruit recuit à 0) : la comparaison probante est "
        f"celle des {anneal} époques à bruit actif, ci-dessous |",
        f"| `part_max` | {triple({s: rout[s][0]['part_max'] for s in SEEDS})} | "
        f"**{triple(part_max, 4)}** | {equi:.4f} = équirépartition | écart max "
        f"{max(abs(v - equi) for v in part_max.values()):.4f} |",
        f"| `frac` des {n_exp} experts | — | [{frac_min:.4f} ; {frac_max:.4f}] | {equi:.4f} | "
        f"aucun expert favorisé |",
        f"| experts morts (`frac` = 0) | — | **{sum(morts.values())}** | 0 | aucun effondrement |",
        f"| `top_expert_share` | {triple({s: rout[s][0]['top_expert_share'] for s in SEEDS})} | "
        f"{triple({s: finals[s]['top_expert_share'] for s in SEEDS})} | 1,0 = un seul expert | "
        f"part du premier expert, quasi stable |",
        f"| `noise_std` | {triple({s: rout[s][0]['noise_std'] for s in SEEDS}, 2)} | "
        f"{triple({s: finals[s]['noise_std'] for s in SEEDS}, 2)} | recuit sur {anneal} époques | "
        f"exploration éteinte, la platitude subsiste |",
        f"| `gamma_absmean` (γ) | {triple(gam0)} | **{triple(gamF)}** | 0 = MoE inutile | "
        f"strictement positif et multiplié par ×{min(gamF[s] / gam0[s] for s in SEEDS):.0f} à "
        f"×{max(gamF[s] / gam0[s] for s in SEEDS):.0f} |",
        f"| `train_loss` | {triple({s: rout[s][0]['train_loss'] for s in SEEDS}, 4)} | "
        f"{triple({s: finals[s]['train_loss'] for s in SEEDS}, 4)} | — | convergence normale |",
    ]
    md += ["", "**Lecture.** À l'époque finale, la distribution de routage est plate à "
              f"{min(ent['entropy_norm'].values()) * 100:.3f} % d'entropie normalisée minimale, "
              f"`part_max` vaut {triple(part_max, 4)} contre {equi:.4f} pour l'équirépartition "
              f"exacte, les {n_exp} experts reçoivent entre {frac_min * 100:.2f} % et "
              f"{frac_max * 100:.2f} % des patches, et **aucun expert n'est mort**. La porte ne "
              f"choisit donc pas : elle moyenne.", "",
              f"**Ce n'est pas un artefact du bruit de Shazeer — et la preuve n'est pas celle "
              f"qu'on croit.** À l'époque finale le bruit est recuit à 0, donc "
              f"`entropy_token_clean` est *identique par construction* à `entropy_token` : cette "
              f"égalité-là ne prouve rien. La comparaison probante porte sur les **{anneal} "
              f"époques où le bruit est actif** (std 1,0 × std des logits) : l'écart maximal "
              f"entre les deux entropies y est de **{max(ecart_actif.values()):.2e}** "
              f"({', '.join(f'seed {s} {ecart_actif[s]:.2e}' for s in SEEDS)}), tandis que la "
              f"distribution **sans bruit** y reste plate à "
              f"**≥ {min(clean_min.values()):.5f}** d'entropie normalisée et `part_max` y reste "
              f"entre {min(pm_actif[s][0] for s in SEEDS):.5f} et "
              f"{max(pm_actif[s][1] for s in SEEDS):.5f} pour une équirépartition à {equi:.4f}. "
              f"Autrement dit : la platitude du routage est une propriété de la porte, pas du "
              f"bruit qu'on lui injecte.", "",
              f"Pourtant le MoE **sert** : γ, l'échelle résiduelle apprenable par canal "
              f"(initialisée à {kw['residual_scale']} pour rendre l'époque 0 bit-exacte au "
              f"contrôle), vaut {triple(gamF)} en fin de run contre {triple(gam0)} à l'époque 0. "
              f"Le gain vient donc de la **moyenne d'experts**, pas de la sélection — c'est la "
              f"réplication sur Cityscapes du résultat BRATS, avec une autre architecture et un "
              f"autre point d'accroche.", ""]
    lim = min(ent["entropy_token"].values())
    if lim < 0.995:
        s_low = min(SEEDS, key=lambda s: ent["entropy_token"][s])
        autres = ", ".join(f"{ent['entropy_token'][s]:.5f}" for s in SEEDS if s != s_low)
        md += [f"**Limite écrite telle quelle.** `entropy_token` du seed {s_low} "
               f"({ent['entropy_token'][s_low]:.5f}) est très légèrement sous les deux autres "
               f"({autres}) : la thèse tient sur les 3 seeds mais n'est pas d'une uniformité "
               f"parfaite sur celui-ci.", ""]
    md += [f"Provenance : `results/moe_v3_cs/routing_moe_v3cs_Binit_seed*.jsonl` "
           f"({len(SEEDS)} seeds × {len(rout[SEEDS[0]])} époques), "
           f"`results/moe_v3_cs/train_moe_v3cs_Binit_seed*_summary.json`."]

    lignes = []
    for s in SEEDS:
        for r in rout[s]:
            lignes.append({"seed": s, "epoch": r["epoch"],
                           "entropy_norm": r["entropy_norm"], "entropy_token": r["entropy_token"],
                           "entropy_token_clean": r["entropy_token_clean"],
                           "part_max": r["part_max"], "frac_min": min(r["frac"]),
                           "frac_max": max(r["frac"]), "gamma_absmean": r["gamma_absmean"],
                           "noise_std": r["noise_std"], "top_expert_share": r["top_expert_share"],
                           "train_loss": r["train_loss"],
                           "aux_loss_weighted": r["aux_loss_weighted"],
                           **{f"frac_{i}": r["frac"][i] for i in range(n_exp)}})
    csv_txt = _csv(list(lignes[0].keys()), lignes)
    data = {"n_experts": n_exp, "top_k": top_k, "equirepartition": equi, "anneal": anneal,
            "n_epochs": len(rout[SEEDS[0]]), "n_seeds": len(SEEDS),
            "entropy_norm_final": {str(s): ent["entropy_norm"][s] for s in SEEDS},
            "entropy_token_final": {str(s): ent["entropy_token"][s] for s in SEEDS},
            "entropy_token_clean_final": {str(s): ent["entropy_token_clean"][s] for s in SEEDS},
            "part_max_final": {str(s): part_max[s] for s in SEEDS},
            "part_max_ep0": {str(s): rout[s][0]["part_max"] for s in SEEDS},
            "frac_min_final": frac_min, "frac_max_final": frac_max,
            "experts_morts": morts, "gamma_ep0": {str(s): gam0[s] for s in SEEDS},
            "gamma_final": {str(s): gamF[s] for s in SEEDS},
            "gamma_ratio_min": min(gamF[s] / gam0[s] for s in SEEDS),
            "gamma_ratio_max": max(gamF[s] / gam0[s] for s in SEEDS),
            "train_loss_final": {str(s): finals[s]["train_loss"] for s in SEEDS},
            "top_expert_share_final": {str(s): finals[s]["top_expert_share"] for s in SEEDS},
            # L'écart à l'époque FINALE est nul par construction (bruit recuit à 0) : il est
            # conservé pour trace, mais la grandeur probante est celle de la phase ACTIVE.
            "ecart_entropie_bruit_final": max(
                abs(ent["entropy_token"][s] - ent["entropy_token_clean"][s]) for s in SEEDS),
            "bruit_actif": {"n_epochs": {str(s): actif[s] for s in SEEDS},
                            "ecart_token_clean_max": {str(s): ecart_actif[s] for s in SEEDS},
                            "ecart_token_clean_max_tous_seeds": max(ecart_actif.values()),
                            "entropy_clean_min": {str(s): clean_min[s] for s in SEEDS},
                            "entropy_token_min": {str(s): tok_min[s] for s in SEEDS},
                            "part_max_bornes": {str(s): list(pm_actif[s]) for s in SEEDS}},
            "seed_moins_uniforme": min(SEEDS, key=lambda s: ent["entropy_token"][s])}
    return "\n".join(md) + "\n", csv_txt, data


# --------------------------------------------------------------------------- #
# T5 — métriques métier
# --------------------------------------------------------------------------- #

def t5_metiers(fam: dict) -> tuple[str, str, dict]:
    met, poly = fam["met"], fam["poly"]
    tables = met["tables"]
    check("métiers P3.16 : 20 métriques", len(tables) == 20, f"{len(tables)} tables")
    check("métiers P3.16 : 15 paires", len(met["pairs"]) == 15, f"{len(met['pairs'])} paires")
    bras_met = tables["mIoU"]["arms"]
    check("métiers P3.16 : 12 bras (couverture complète, A exclu)", len(bras_met) == 12,
          f"{len(bras_met)} bras : {bras_met}")
    check("le MoE et le contrôle sont dans les 12 bras", {ARM, REF} <= set(bras_met),
          f"{ARM}, {REF}")
    # Le contexte du papier : ce que le MoE capte des experts sans hériter de leurs pertes.
    CONTEXTE = ["D", "Dp", "G", "B"]
    check("bras de contexte présents dans la table métier", set(CONTEXTE) <= set(bras_met),
          str(CONTEXTE))

    ordre = ["mIoU", "boundary_f1_3px", "boundary_f1_3px_pieds", "rappel_strict_instances",
             "precision_ped_pixels", "instances_toutes_rappel", "instances_toutes_det05",
             "instances_individuelles_rappel", "instances_individuelles_det05",
             "instances_foule_rappel", "instances_foule_det05",
             "instances_taille_T1_rappel", "instances_taille_T1_det05",
             "instances_taille_T2_rappel", "instances_taille_T2_det05",
             "instances_taille_T3_rappel", "instances_taille_T3_det05",
             "IoU_person", "IoU_rider", "fragments"]
    check("ordre d'affichage == ensemble des métriques de l'artefact",
          set(ordre) == set(tables), f"{len(set(ordre) ^ set(tables))} différence(s)")

    md = ["# T5 — Métriques métier : ce que le MoE capte, ce qu'il paie", "",
          f"Holdout `first:{tables['mIoU']['n_images']}`, seeds moyennés dans chaque réplicat, "
          f"bootstrap apparié B = {tables['mIoU']['B']:,} (seed {met.get('bootstrap_seed', fam['harness']['bootstrap_seed'])}), "
          f"Holm sur la famille à **{fam['n15']} paires**. `n` = nombre d'images contribuant à "
          f"la métrique (il varie : les strates d'instances et la précision piéton ne sont pas "
          f"définies sur toutes les images). Unité : points (×100), sauf `fragments` = "
          f"composantes connexes par image.", "",
          "**Gras** = Holm < 0,05. Les deux critères (Holm et IC excluant 0) sont séparés, "
          "jamais confondus.", "",
          f"| Métrique | n | contrôle | MoE-V3-CS | Δ(MoE−ctl) | IC95 | p | Holm({fam['n15']}) | "
          "verdict |", "|---|---|---|---|---|---|---|---|---|"]
    lignes, n_holm, n_ic = [], 0, 0
    for m in ordre:
        t = tables[m]
        f, nd, _u = unite(m, t["point"])
        e = t["pairwise"][f"{ARM}_vs_{REF}"]
        d = e["delta"] * f
        lo, hi = e["ci95"][0] * f, e["ci95"][1] * f
        ph = e["p_holm"]
        survit = ph < 0.05
        ic0 = (lo > 0) or (hi < 0)
        n_holm += survit
        n_ic += ic0
        verdict = ("**Holm-significatif**" if survit else
                   ("IC exclut 0, ne survit pas au Holm" if ic0 else "ns"))
        g = ("**", "**") if survit else ("", "")
        md.append(f"| `{m}` | {t['n_images']} | {g[0]}{t['point'][REF] * f:.{nd}f}{g[1]} | "
                  f"{g[0]}{t['point'][ARM] * f:.{nd}f}{g[1]} | {g[0]}{d:+.{nd}f}{g[1]} | "
                  f"[{lo:+.{nd}f} ; {hi:+.{nd}f}] | {fp(e['p_two_sided'])} | {ph:.4g} | {verdict} |")
        lignes.append({"metrique": m, "n_images": t["n_images"], "unite": _u,
                       "point_controle": t["point"][REF] * f, "point_moe": t["point"][ARM] * f,
                       "delta": d, "ci_lo": lo, "ci_hi": hi, "p": e["p_two_sided"],
                       "holm": ph, "survit_holm": survit, "ic_exclut_0": ic0,
                       **{f"delta_{b}": t["pairwise"][f"{b}_vs_{REF}"]["delta"] * f
                          for b in CONTEXTE if f"{b}_vs_{REF}" in t["pairwise"]}})
    check("comptes de significativité métier cohérents", n_holm <= n_ic,
          f"{n_holm} Holm-significatifs, {n_ic} IC excluant 0 (Holm ⊆ IC)")

    # Fragments : seule métrique à 2 bras dans cet artefact — le dire, ne pas le masquer.
    fr = tables["fragments"]
    check("fragments : artefact à 2 bras (contrôle + MoE seulement)", len(fr["point"]) == 2,
          f"{len(fr['point'])} bras : {list(fr['point'])}")
    fp_ = met["fragments_point"]
    check("fragments : points de la table == fragments_point de l'artefact",
          abs(fr["point"][ARM] - fp_[ARM]) < 1e-9 and abs(fr["point"][REF] - fp_[REF]) < 1e-9,
          f"{fr['point'][REF]:.3f} → {fr['point'][ARM]:.3f}")
    dfr = fr["point"][ARM] - fr["point"][REF]
    ratio = fr["point"][ARM] / fr["point"][REF]
    check("fragments : baisse mesurée", dfr < 0, f"{dfr:+.3f} composantes/image ({ratio:.4f}×)")

    md += ["", f"## Fragments (topologie des masques), mesurés sur {len(fr['point'])} bras seulement", "",
           f"| Quantité | contrôle | MoE-V3-CS | Δ | IC95 | p | Holm |", "|---|---|---|---|---|---|---|",
           f"| composantes connexes par image | {fr['point'][REF]:.1f} | **{fr['point'][ARM]:.1f}** | "
           f"**{dfr:+.1f}** ({ratio * 100 - 100:+.1f} %) | "
           f"[{fr['pairwise'][f'{ARM}_vs_{REF}']['ci95'][0]:+.1f} ; "
           f"{fr['pairwise'][f'{ARM}_vs_{REF}']['ci95'][1]:+.1f}] | "
           f"{fp(fr['pairwise'][f'{ARM}_vs_{REF}']['p_two_sided'])} | "
           f"{fr['pairwise'][f'{ARM}_vs_{REF}']['p_holm']:.4g} |"]

    # PIÈGE 2 évité de justesse (mesuré le 2026-10-01) : la prose disait « la SEULE métrique
    # métier qui survive au Holm », or l'artefact en donne DEUX (`fragments` ET
    # `instances_taille_T3_rappel`, toutes deux p = 0, Holm = 0). Le CADRAGE §4.3 portait la même
    # affirmation fautive. Le compte est donc CALCULÉ ici, et les survivantes sont nommées.
    survivantes = [(m, tables[m]["pairwise"][f"{ARM}_vs_{REF}"])
                   for m in ordre if tables[m]["pairwise"][f"{ARM}_vs_{REF}"]["p_holm"] < 0.05]
    check("compte des survivantes au Holm == n_holm calculé plus haut",
          len(survivantes) == n_holm, f"{len(survivantes)} vs {n_holm}")
    check("les fragments sont parmi les survivantes au Holm",
          any(m == "fragments" for m, _ in survivantes), str([m for m, _ in survivantes]))
    check("il y a PLUS d'une survivante (ne pas écrire « la seule »)",
          len(survivantes) > 1, " · ".join(f"{m} (Holm {e['p_holm']:.4g})" for m, e in survivantes))
    noms_surv = ", ".join(f"`{m}`" for m, _ in survivantes)
    md += ["", f"Sur les {len(ordre)} métriques métier, **{len(survivantes)}** survivent au Holm "
              f"sur la famille {fam['n15']} paires : {noms_surv} (toutes deux avec p = 0). Toutes "
              f"les autres sont non significatives une fois la multiplicité payée — y compris le "
              f"mIoU lui-même (Holm {tables['mIoU']['pairwise'][f'{ARM}_vs_{REF}']['p_holm']:.4g}).", "",
              "`fragments` est par ailleurs la seule métrique pour laquelle cet artefact ne couvre "
              "que le contrôle et le MoE (les autres bras ont leurs fragments dans "
              "`results/moe_v3_cs/p316/synthese_attribution.md`, hors bootstrap apparié) — "
              "d'où le tableau séparé ci-dessus.", ""]

    # Ce que le MoE capte des experts sans hériter de leurs pertes — calculé, pas raconté.
    md += ["## Ce que le MoE capte des experts, et ce qu'il n'hérite pas", "",
           f"| Métrique | Δ MoE | Δ D | Δ Dp | Δ G | Δ B | lecture calculée |",
           "|---|---|---|---|---|---|---|"]
    capte = []
    for m in ["instances_taille_T1_rappel", "rappel_strict_instances",
              "instances_toutes_rappel", "boundary_f1_3px", "precision_ped_pixels",
              "instances_foule_rappel", "IoU_person"]:
        t = tables[m]
        f, nd, _u = unite(m, t["point"])
        row = {b: t["pairwise"][f"{b}_vs_{REF}"]["delta"] * f for b in [ARM] + CONTEXTE}
        # lecture : le MoE prend-il le gain des experts sans leur perte ?
        gains = [b for b in CONTEXTE if row[b] > 0]
        pertes = [b for b in CONTEXTE if row[b] < -1.0]
        if row[ARM] > 0 and gains:
            lec = f"capte le gain de {'/'.join(gains)}"
        elif row[ARM] < 0 and pertes:
            lec = f"perte, mais {abs(row[ARM]):.2f} ≪ {min(row[b] for b in pertes):.2f} ({'/'.join(pertes)})"
        else:
            lec = "neutre"
        capte.append((m, row, lec))
        md.append(f"| `{m}` | {row[ARM]:+.{nd}f} | {row['D']:+.{nd}f} | {row['Dp']:+.{nd}f} | "
                  f"{row['G']:+.{nd}f} | {row['B']:+.{nd}f} | {lec} |")
    t1 = tables["instances_taille_T1_rappel"]
    f1, nd1, _ = unite("instances_taille_T1_rappel", t1["point"])
    dT1 = {b: t1["pairwise"][f"{b}_vs_{REF}"]["delta"] * f1 for b in [ARM, "D", "Dp", "G"]}
    check("le MoE capte une partie du gain T1 de D/Dp, sans l'ampleur de la perte de G",
          0 < dT1[ARM] < dT1["D"] and dT1["G"] < 0,
          f"MoE {dT1[ARM]:+.2f} · D {dT1['D']:+.2f} · Dp {dT1['Dp']:+.2f} · G {dT1['G']:+.2f}")
    rs = tables["rappel_strict_instances"]
    fr_, nd2, _ = unite("rappel_strict_instances", rs["point"])
    dRS = {b: rs["pairwise"][f"{b}_vs_{REF}"]["delta"] * fr_ for b in [ARM, "D", "Dp", "G"]}
    check("rappel piéton strict : le MoE ne hérite PAS de la perte de G",
          dRS[ARM] > -1.0 and dRS["G"] < -1.0,
          f"MoE {dRS[ARM]:+.2f} vs G {dRS['G']:+.2f}")
    tl = j(P316_ATT)["arms"]
    d_tl = {b: next(r["delta_pt"] for r in tl[b]["per_class"] if r["classe"] == "traffic light")
            for b in [ARM, "D", "Dp"]}
    check("traffic light : le MoE n'hérite pas de la perte des bras boundary/distmap",
          abs(d_tl[ARM]) < abs(d_tl["D"]) and d_tl["D"] < -1.0,
          f"MoE {d_tl[ARM]:+.2f} · D {d_tl['D']:+.2f} · Dp {d_tl['Dp']:+.2f}")

    md += ["", f"Lecture calculée : le MoE-V3-CS capte une partie du gain d'instances de D/Dp "
              f"(T1 {dT1[ARM]:+.2f} contre {dT1['D']:+.2f} et {dT1['Dp']:+.2f}) **sans** hériter "
              f"ni de leur perte `traffic light` ({d_tl[ARM]:+.2f} contre {d_tl['D']:+.2f} et "
              f"{d_tl['Dp']:+.2f}) ni du rappel piéton massacré par G ({dRS[ARM]:+.2f} contre "
              f"{dRS['G']:+.2f}). Sa contrepartie mesurée : **aucun gain de contours** "
              f"(`boundary_f1_3px` {tables['boundary_f1_3px']['pairwise'][f'{ARM}_vs_{REF}']['delta'] * 100:+.3f}, ns) — "
              f"cohérent avec T4, une moyenne d'experts lisse les spécialités de contour. "
              f"Sur {len(ordre)} métriques, {n_holm} surviv{'t' if n_holm == 1 else 'ent'} au "
              f"Holm et {n_ic} {'a' if n_ic == 1 else 'ont'} un IC excluant 0.", ""]
    pire = poly["verdict"]["T5_pire_famille"][ARM]
    md += [f"La polyvalence (T7) confirme : la pire famille du MoE est `{pire['famille']}` "
           f"({pire['delta_moyen_pt']:+.2f} pt en moyenne) et c'est sa **seule** famille négative "
           f"({pire['n_familles_negatives']} sur {len(set(poly['familles'].values()))}).", "",
           f"Provenance : `{P316_MET.relative_to(REPO)}`."]

    csv_txt = _csv(list(lignes[0].keys()), lignes)
    data = {"n_metriques": len(ordre), "ordre": ordre, "lignes": lignes,
            "n_survit_holm": n_holm, "n_ic_exclut_0": n_ic,
            "n_famille_holm": fam["n15"], "n_bras": len(bras_met), "n_paires": len(met["pairs"]),
            "fragments": {"controle": fr["point"][REF], "moe": fr["point"][ARM],
                          "delta": dfr, "ratio": ratio,
                          "p": fr["pairwise"][f"{ARM}_vs_{REF}"]["p_two_sided"],
                          "holm": fr["pairwise"][f"{ARM}_vs_{REF}"]["p_holm"],
                          "ci": fr["pairwise"][f"{ARM}_vs_{REF}"]["ci95"],
                          "n_bras_couverts": len(fr["point"])},
            "delta_T1_rappel": dT1, "delta_rappel_strict": dRS, "delta_traffic_light": d_tl,
            "pire_famille": pire}
    return "\n".join(md) + "\n", csv_txt, data


# --------------------------------------------------------------------------- #
# T6 — IoU par classe
# --------------------------------------------------------------------------- #

def t6_perclass(fam: dict) -> tuple[str, str, dict]:
    att = j(P316_ATT)
    check("attribution P3.16 : référence = contrôle", att["ref"] == REF, att["ref"])
    check("attribution P3.16 : 12 bras", len(att["arms"]) == 12, f"{len(att['arms'])} bras")
    pc = att["arms"][ARM]["per_class"]
    classes = [r["classe"] for r in pc]
    check("attribution P3.16 : 19 classes Cityscapes", len(classes) == 19, f"{len(classes)}")
    check("classes toutes distinctes", len(set(classes)) == len(classes), f"{len(set(classes))}")
    # Le mIoU DOIT être la moyenne des 19 IoU : c'est la convention officielle.
    for a in [ARM, "D", "G"]:
        moy = sum(r["iou_bras"] for r in att["arms"][a]["per_class"]) / len(classes)
        check(f"mIoU == moyenne des 19 IoU par classe ({a})",
              abs(moy - att["arms"][a]["miou_point"]) < 1e-9,
              f"{moy * 100:.6f} vs {att['arms'][a]['miou_point'] * 100:.6f}")
    moy_ref = sum(r["iou_ref"] for r in pc) / len(classes)
    ctl_h = fam["harness"]["bootstrap_miou"]["point"][REF]
    check("IoU de référence (contrôle) : moyenne des 19 == point du harnais",
          abs(moy_ref - ctl_h) < 1e-9, f"{moy_ref * 100:.6f} vs {ctl_h * 100:.6f}")
    check("Δ mIoU de l'attribution == somme des Δ par classe / 19",
          abs(sum(r["delta_pt"] for r in pc) / len(classes) - att["arms"][ARM]["miou_delta_pt"]) < 1e-9,
          f"{sum(r['delta_pt'] for r in pc) / len(classes):+.6f} vs "
          f"{att['arms'][ARM]['miou_delta_pt']:+.6f}")

    # Holm par bras sur la famille des 19 classes : recomputé depuis les p bruts.
    h19 = holm({r["classe"]: r["p_two_sided"] for r in pc})
    e19 = max(abs(h19[r["classe"]] - r["p_holm_arm"]) for r in pc)
    check("Holm 19 classes recomputé == p_holm_arm stocké", e19 < 1e-9, f"écart max {e19:.3g}")

    EXPERTS = ["B", "D", "Dp", "G"]
    ordre = sorted(pc, key=lambda r: -r["delta_pt"])
    gains = [r for r in ordre if r["delta_pt"] > 0]
    pertes = [r for r in ordre if r["delta_pt"] < 0]
    n_holm_gain = sum(1 for r in gains if r["p_holm_arm"] < 0.05)
    n_ic_gain = sum(1 for r in gains if r["ci_lo_pt"] > 0)
    n_ic_perte = sum(1 for r in pertes if r["ci_hi_pt"] < 0)
    check("gains + pertes + classes nulles == 19",
          len(gains) + len(pertes) + (len(classes) - len(gains) - len(pertes)) == len(classes),
          f"{len(gains)} gains, {len(pertes)} pertes")

    md = ["# T6 — IoU par classe : attribution des gains et des coûts", "",
          f"19 classes Cityscapes, référence = contrôle apparié, holdout `first:{att['n_images']}`, "
          f"bootstrap apparié B = {att['B']:,} (seed {att['bootstrap_seed']}), Holm sur la "
          f"**famille des 19 classes** (recomputé depuis les p bruts, écart max {e19:.1e}). "
          f"Colonnes D/Dp/G/B = Δ du spécialiste correspondant, pour lire ce que le MoE capte "
          f"sans en hériter les pertes.", "",
          "| Classe | IoU MoE | IoU ctl | Δ(MoE−ctl) | IC95 | p | Holm(19) | ΔD | ΔDp | ΔG | ΔB |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    lignes = []
    for r in ordre:
        cl = r["classe"]
        survit = r["p_holm_arm"] < 0.05
        g = ("**", "**") if survit else ("", "")
        dexp = {}
        for b in EXPERTS:
            dexp[b] = next(x["delta_pt"] for x in att["arms"][b]["per_class"] if x["classe"] == cl)
        md.append(f"| {cl} | {r['iou_bras'] * 100:.3f} | {r['iou_ref'] * 100:.3f} | "
                  f"{g[0]}{r['delta_pt']:+.3f}{g[1]} | [{r['ci_lo_pt']:+.3f} ; "
                  f"{r['ci_hi_pt']:+.3f}] | {fp(r['p_two_sided'])} | {r['p_holm_arm']:.4g} | "
                  f"{dexp['D']:+.2f} | {dexp['Dp']:+.2f} | {dexp['G']:+.2f} | {dexp['B']:+.2f} |")
        lignes.append({"classe": cl, "iou_moe": r["iou_bras"] * 100.0,
                       "iou_controle": r["iou_ref"] * 100.0, "delta_pt": r["delta_pt"],
                       "ci_lo_pt": r["ci_lo_pt"], "ci_hi_pt": r["ci_hi_pt"],
                       "p": r["p_two_sided"], "holm_19": r["p_holm_arm"],
                       "survit_holm": survit, "ic_exclut_0": bool(r["ci_lo_pt"] > 0 or r["ci_hi_pt"] < 0),
                       **{f"delta_{b}": dexp[b] for b in EXPERTS}})
    # DEUX critères distincts, jamais confondus (mesure du 2026-10-01 : le CADRAGE §4.4 citait
    # « truck, train, sidewalk, road » sans dire lequel des deux il employait — c'est l'ensemble
    # des classes dont l'IC exclut 0, PAS les 4 plus fortes amplitudes, qui sont truck, wall,
    # train, fence. L'ambiguïté est levée ici en calculant et en nommant les deux.)
    top4_amplitude = [r["classe"] for r in ordre[:4]]
    gains_ic = [r["classe"] for r in ordre if r["delta_pt"] > 0 and r["ci_lo_pt"] > 0]
    pertes_ic = [r["classe"] for r in ordre if r["delta_pt"] < 0 and r["ci_hi_pt"] < 0]
    gains_holm = [r["classe"] for r in ordre if r["delta_pt"] > 0 and r["p_holm_arm"] < 0.05]
    pertes_holm = [r["classe"] for r in ordre if r["delta_pt"] < 0 and r["p_holm_arm"] < 0.05]
    check("critère AMPLITUDE : les 4 plus fortes hausses sont truck, wall, train, fence",
          top4_amplitude == ["truck", "wall", "train", "fence"], str(top4_amplitude))
    check("critère IC-excluant-0 : hausses = truck, train, sidewalk, road",
          sorted(gains_ic) == sorted(["truck", "train", "sidewalk", "road"]), str(gains_ic))
    check("critère IC-excluant-0 : baisses = person, bicycle",
          sorted(pertes_ic) == sorted(["person", "bicycle"]), str(pertes_ic))
    check("critère HOLM(19) : aucune HAUSSE ne survit", len(gains_holm) == 0,
          f"{len(gains_holm)} hausse(s) Holm-significative(s) : {gains_holm}")
    check("critère HOLM(19) : la seule classe survivante est une BAISSE",
          len(pertes_holm) == 1, f"{pertes_holm}")
    check("les deux critères ne sont PAS réductibles l'un à l'autre",
          set(top4_amplitude) != set(gains_ic),
          f"amplitude {top4_amplitude} vs IC {gains_ic}")
    pire = [r["classe"] for r in ordre[-2:]]
    txt_amp = ", ".join(f"{r['classe']} {r['delta_pt']:+.2f}" for r in ordre[:4])
    txt_ic = ", ".join(f"{r['classe']} {r['delta_pt']:+.2f}"
                       for r in ordre if r["classe"] in gains_ic)
    txt_couts = ", ".join(f"{r['classe']} {r['delta_pt']:+.2f}" for r in ordre[-3:])
    txt_pertes_ic = ", ".join(f"{r['classe']} {r['delta_pt']:+.2f}"
                              for r in ordre if r["classe"] in pertes_ic)
    truck = next(r for r in pc if r["classe"] == "truck")
    holm_perte = min(r["p_holm_arm"] for r in ordre if r["classe"] in pertes_holm)
    md += ["", f"**Gras** = Holm(19) < 0,05. Sur {len(classes)} classes : {len(gains)} en hausse, "
              f"{len(pertes)} en baisse.", "",
              "## Trois critères de significativité, donnés séparément", "",
              "Le classement par **amplitude** et le classement par **significativité** ne "
              "coïncident pas ; les confondre produirait une affirmation fausse. Les trois sont "
              "donnés tels quels :", "",
              f"1. **Plus fortes amplitudes** (ordre décroissant du Δ) : {txt_amp}.",
              f"2. **IC95 excluant 0** (critère « significant » de P3.16) : hausses "
              f"{txt_ic} ; baisses {txt_pertes_ic}.",
              f"3. **Holm sur la famille des {len(classes)} classes** : **aucune hausse ne "
              f"survit** ; la seule classe qui survive est une **baisse** — "
              f"{', '.join(pertes_holm)} (Holm {holm_perte:.4g}).", "",
              f"C'est le point d'honnêteté central de cette table : le gain le plus spectaculaire "
              f"(`truck` {truck['delta_pt']:+.2f} pt) a un IC qui exclut 0 mais **ne survit pas** "
              f"au Holm des 19 classes ({truck['p_holm_arm']:.4g}), parce que son IC est large "
              f"([{truck['ci_lo_pt']:+.2f} ; {truck['ci_hi_pt']:+.2f}]) sur une classe rare. Le "
              f"seul effet par classe qui résiste à la correction multiple est un **coût** "
              f"({', '.join(pertes_holm)}), pas un gain. Les coûts sont faibles et distribués : "
              f"{txt_couts}.", "",
              "## Ce que le MoE garde des spécialistes, et ce qu'il évite", ""]
    for cl in ["truck", "train", "traffic light", "person", "bicycle", "rider"]:
        r = next(x for x in pc if x["classe"] == cl)
        dexp = {b: next(x["delta_pt"] for x in att["arms"][b]["per_class"] if x["classe"] == cl)
                for b in EXPERTS}
        # Le dénominateur est le meilleur des QUATRE experts (B en fait partie : c'est
        # l'expert 0), pas seulement D/Dp — mesuré sur `train` : B vaut +1,50 et dépasse D
        # (+0,40) et Dp (+0,66), si bien qu'un dénominateur D/Dp faisait dire « le MoE
        # conserve 186 % du meilleur gain spécialiste », ce qui ne veut rien dire.
        meilleur = max(dexp.values())
        meilleur_b = max(dexp, key=lambda b: dexp[b])
        if meilleur > 0 and r["delta_pt"] > 0:
            ratio = r["delta_pt"] / meilleur
            lec = (f"le MoE retrouve {ratio * 100:.0f} % du meilleur gain spécialiste "
                   f"({meilleur_b} {meilleur:+.2f})" if ratio <= 1.0 else
                   f"le MoE DÉPASSE le meilleur expert ({meilleur_b} {meilleur:+.2f}) de "
                   f"{(ratio - 1) * 100:.0f} %")
        elif r["delta_pt"] < 0 and min(dexp.values()) < 0:
            pire = min(dexp, key=lambda b: dexp[b])
            lec = (f"perte {abs(r['delta_pt']):.2f} pt, soit {abs(r['delta_pt'] / dexp[pire]) * 100:.0f} % "
                   f"de la pire perte spécialiste ({pire} {dexp[pire]:+.2f})")
        else:
            lec = "aucun expert ne gagne sur cette classe : pas de référence de gain"
        md.append(f"- **{cl}** : MoE {r['delta_pt']:+.2f} contre D {dexp['D']:+.2f}, "
                  f"Dp {dexp['Dp']:+.2f}, G {dexp['G']:+.2f}, B {dexp['B']:+.2f} — {lec}.")
    tr = next(x for x in pc if x["classe"] == "truck")
    d_tr = {b: next(x["delta_pt"] for x in att["arms"][b]["per_class"] if x["classe"] == "truck")
            for b in EXPERTS}
    part_truck = tr["delta_pt"] / max(d_tr.values())
    check("truck : le MoE retrouve une part majoritaire du MEILLEUR des 4 experts",
          0.5 < part_truck <= 1.0,
          f"{part_truck * 100:.1f} % ({tr['delta_pt']:+.2f} vs meilleur expert "
          f"{max(d_tr.values()):+.2f} = {max(d_tr, key=lambda b: d_tr[b])})")
    train_r = next(x for x in pc if x["classe"] == "train")
    d_trn = {b: next(x["delta_pt"] for x in att["arms"][b]["per_class"] if x["classe"] == "train")
             for b in EXPERTS}
    check("train : le meilleur expert est bien B, pas D ni Dp (le dénominateur D/Dp mentait)",
          max(d_trn, key=lambda b: d_trn[b]) == "B",
          " · ".join(f"{b} {d_trn[b]:+.2f}" for b in EXPERTS))
    d_per = next(x for x in pc if x["classe"] == "person")
    check("coût piéton faible et non Holm-significatif",
          -0.5 < d_per["delta_pt"] < 0 and d_per["p_holm_arm"] >= 0.05,
          f"{d_per['delta_pt']:+.3f} pt, Holm {d_per['p_holm_arm']:.4g}, "
          f"IC [{d_per['ci_lo_pt']:+.3f} ; {d_per['ci_hi_pt']:+.3f}]")
    md += ["", f"Provenance : `{P316_ATT.relative_to(REPO)}`."]
    csv_txt = _csv(list(lignes[0].keys()), lignes)
    data = {"n_classes": len(classes), "classes": classes, "lignes": lignes,
            "n_gains": len(gains), "n_pertes": len(pertes),
            "n_holm_significatifs": n_holm_gain, "n_ic_exclut_0_gains": n_ic_gain,
            "n_ic_exclut_0_pertes": n_ic_perte,
            "top4_amplitude": top4_amplitude, "gains_ic_exclut_0": gains_ic,
            "pertes_ic_exclut_0": pertes_ic, "gains_holm": gains_holm,
            "pertes_holm": pertes_holm, "holm_min_pertes": holm_perte,
            "miou_moe": att["arms"][ARM]["miou_point"] * 100.0,
            "miou_controle": moy_ref * 100.0,
            "delta_miou_pt": att["arms"][ARM]["miou_delta_pt"],
            "part_gain_truck": part_truck,
            "person_delta_pt": d_per["delta_pt"], "person_holm": d_per["p_holm_arm"],
            "traffic_light_delta": next(x["delta_pt"] for x in pc if x["classe"] == "traffic light"),
            "ecart_holm_recompute": e19}
    return "\n".join(md) + "\n", csv_txt, data


# --------------------------------------------------------------------------- #
# T7 — polyvalence : les critères pré-enregistrés RECALCULÉS
# --------------------------------------------------------------------------- #

def t7_polyvalence(fam: dict) -> tuple[str, str, dict]:
    p = fam["poly"]
    end, fam_end = p["endpoints"], p["familles"]
    bras = p["bras_couverture_complète"]
    check("polyvalence : 36 endpoints", len(end) == 36, f"{len(end)}")
    check("polyvalence : 13 bras", p["n_bras"] == 13, str(p["n_bras"]))
    check("polyvalence : 12 bras à couverture complète (A en partielle)", len(bras) == 12,
          f"{len(bras)} ; annexe = {p['annexe_couverture_partielle']}")
    check("familles : somme des effectifs == 36",
          sum(1 for _ in fam_end) == len(end), f"{len(fam_end)} affectations")
    classes_end = [e for e in end if fam_end[e] == "IoU par classe"]
    check("19 endpoints « IoU par classe »", len(classes_end) == 19, f"{len(classes_end)}")
    check("chaque bras complet a ses 36 deltas",
          all(len(p["delta_vs_controle_pt"][a]) == len(end) for a in bras),
          f"{len(bras)} bras × {len(end)} endpoints")
    for a in bras:
        s = p["sd_inter_bras_par_endpoint"]
        check(f"sd inter-bras présent pour les 36 endpoints ({a})", len(s) == len(end),
              f"{len(s)}") if a == bras[0] else None

    # ---- RECALCUL INDÉPENDANT des verdicts (c'est la thèse du papier) ----
    comp = [a for a in bras if a != REF]
    dom = {}
    for a in comp:
        d = {e: p["delta_vs_controle_pt"][a][e] for e in end}
        pire = min(d, key=lambda e: d[e])
        dom[a] = {"dommage_max_pt": d[pire], "endpoint": pire,
                  "n_dommage_1pt": sum(1 for e in end if d[e] < -1.0),
                  "n_dommage_2pt": sum(1 for e in end if d[e] < -2.0)}
    for a in comp:
        v = p["verdict"]["T1_classement_dommage_max"]
        stocke = next((x for x in v if x[0] == a), None)
        check(f"dommage max recalculé == stocké ({a})",
              stocke is not None and abs(dom[a]["dommage_max_pt"] - stocke[1]) < 0.01
              and dom[a]["endpoint"] == stocke[2],
              f"{dom[a]['dommage_max_pt']:+.4f} ({dom[a]['endpoint']}) vs {stocke}")
    classement = sorted(comp, key=lambda a: -dom[a]["dommage_max_pt"])
    check("classement par dommage recalculé == classement stocké",
          classement == [x[0] for x in p["verdict"]["T1_classement_dommage_max"]],
          f"{classement[:3]}… == {[x[0] for x in p['verdict']['T1_classement_dommage_max']][:3]}…")
    check("le MoE est premier en dommage maximal", classement[0] == ARM, classement[0])
    # Marge = dommage du MoE MOINS celui du suivant : positif car le MoE est le moins abîmé
    # (−0,53 contre −1,90). Le signe inverse donnerait une marge négative, mesuré à la 1ʳᵉ passe.
    marge = dom[ARM]["dommage_max_pt"] - dom[classement[1]]["dommage_max_pt"]
    check("marge du MoE sur le suivant == stockée", abs(marge - p["verdict"]["T1_marge_vs_suivant"]) < 0.01,
          f"{marge:.3f} vs {p['verdict']['T1_marge_vs_suivant']}")
    check("marge positive (le MoE est bien le moins abîmé)", marge > 0, f"{marge:+.3f} pt")

    # `rang_entier` n'a pas les mêmes bras sur tous les endpoints : A est en couverture
    # partielle (`annexe_couverture_partielle`), il est absent des endpoints métier. Toute
    # lecture directe par bras lèverait donc un KeyError — d'où le filtre, et le check qui
    # mesure l'asymétrie au lieu de la supposer nulle.
    bras_rang = {}
    for e in end:
        for a in p["rang_entier"][e]:
            bras_rang.setdefault(a, set()).add(e)
    couverture = {a: len(v) for a, v in bras_rang.items()}
    n_complets = sum(1 for v in couverture.values() if v == len(end))
    check("couverture de rang_entier : bras complets + bras partiels == n_bras",
          len(couverture) == p["n_bras"],
          f"{len(couverture)} bras dont {n_complets} à couverture complète sur {len(end)} "
          f"endpoints ; partiels : "
          f"{ {a: c for a, c in couverture.items() if c != len(end)} }")
    check("le MoE est à couverture complète sur les 36 endpoints",
          couverture[ARM] == len(end), f"{couverture[ARM]}/{len(end)}")
    comp_rang = [a for a in couverture if couverture[a] == len(end)]
    n_rang1 = {a: sum(1 for e in end if p["rang_entier"][e].get(a) == 1) for a in bras_rang}
    for a in comp:
        check(f"n_rang1 recalculé == profil stocké ({a})",
              n_rang1[a] == p["profil"][a]["n_rang1"], f"{n_rang1[a]} vs {p['profil'][a]['n_rang1']}")
    check("le MoE n'est premier sur AUCUN endpoint", n_rang1[ARM] == 0, str(n_rang1[ARM]))
    check("D est premier sur le plus grand nombre d'endpoints",
          max(n_rang1, key=lambda a: n_rang1[a]) == "D", f"D {n_rang1['D']}")
    n_dernier = {a: sum(1 for e in end
                        if p["rang_entier"][e][a] == len(p["rang_entier"][e])) for a in comp}
    check("le MoE n'est jamais dernier", n_dernier[ARM] == 0, str(n_dernier[ARM]))
    # Le dénominateur du rang N'EST PAS constant : A (couverture partielle) est classé sur 20
    # endpoints seulement, donc 16 endpoints ne classent que 12 bras. Écrire « pire rang 11/13 »
    # (comme le faisait le cadrage §1(c)) est impropre : mesuré, le pire rang du MoE est 11 sur
    # des endpoints qui n'en classent que 12. Le dénominateur est donc calculé endpoint par
    # endpoint, et le pire rang est rapporté AVEC son dénominateur réel.
    pire_rang, pire_denom, pire_end = 0, 0, None
    for e in end:
        r = p["rang_entier"][e].get(ARM)
        if r is not None and r > pire_rang:
            pire_rang, pire_denom, pire_end = r, len(p["rang_entier"][e]), e
    tailles_rang = {len(p["rang_entier"][e]) for e in end}
    check("le dénominateur des rangs varie selon les endpoints (A en couverture partielle)",
          len(tailles_rang) > 1, f"tailles observées {sorted(tailles_rang)}")
    check("pire rang du MoE recalculé == profil", pire_rang == p["profil"][ARM]["pire_rang"],
          f"{pire_rang} vs {p['profil'][ARM]['pire_rang']}")
    check("le pire rang du MoE n'est PAS une dernière place", pire_rang < pire_denom,
          f"{pire_rang}/{pire_denom} sur `{pire_end}` (dernière place = {pire_denom})")
    # Le pire rang doit être rapporté AVEC le dénominateur de l'endpoint où il est atteint :
    # le MoE est 11ᵉ sur `instances_foule_rappel`, qui ne classe que 12 bras (A absent).
    # Écrire 11/13 mélangerait un rang et un dénominateur d'endpoints différents — défaut
    # mesuré et corrigé ici après avoir été introduit une première fois dans cette même table.
    pire_rang_denom = {}
    for a in bras_rang:
        pr = max(((p["rang_entier"][e][a], len(p["rang_entier"][e]), e)
                  for e in end if a in p["rang_entier"][e]), key=lambda t: t[0])
        pire_rang_denom[a] = {"rang": pr[0], "denom": pr[1], "endpoint": pr[2]}
    check("pire rang + son dénominateur cohérents avec le profil (MoE)",
          pire_rang_denom[ARM]["rang"] == p["profil"][ARM]["pire_rang"]
          and pire_rang_denom[ARM]["rang"] < pire_rang_denom[ARM]["denom"],
          f"{pire_rang_denom[ARM]['rang']}/{pire_rang_denom[ARM]['denom']} sur "
          f"`{pire_rang_denom[ARM]['endpoint']}` — jamais dernier")
    check("le dénominateur du pire rang n'est PAS toujours 13 (A en couverture partielle)",
          len({v["denom"] for v in pire_rang_denom.values()}) > 1,
          f"dénominateurs observés : {sorted({v['denom'] for v in pire_rang_denom.values()})}")
    # Cohérence structurelle : le mIoU EST la moyenne des 19 IoU, donc la moyenne des Δ par
    # classe doit redonner le ΔmIoU. MESURE du 2026-10-01 : cette identité est EXACTE à
    # l'intérieur d'un même artefact (vérifiée dans T6 sur l'attribution P3.16, écart 0), mais
    # la table de polyvalence **mélange deux forwards GPU** — ses colonnes IoU par classe
    # viennent de l'attribution (Δ +0,449360) et sa colonne mIoU de la régénération métiers
    # (Δ +0,450218). L'identité n'y tient donc qu'à ~1e-3 pt. Tolérance mesurée et déclarée,
    # jamais masquée : c'est le même écart que celui documenté par T2 entre les deux forwards.
    moy_cls = sum(p["delta_vs_controle_pt"][ARM][e] for e in classes_end) / len(classes_end)
    ecart_identite = abs(moy_cls - p["delta_vs_controle_pt"][ARM]["mIoU"])
    check("moyenne des 19 Δ par classe ≈ ΔmIoU (identité exacte within-artefact, "
          "la polyvalence mélange 2 forwards)",
          ecart_identite < 1e-2,
          f"{moy_cls:+.6f} vs {p['delta_vs_controle_pt'][ARM]['mIoU']:+.6f} — écart "
          f"{ecart_identite:.2e} pt = l'écart inter-forwards mesuré en T2")
    check("l'écart d'identité de la polyvalence est du même ordre que son écart inter-forwards",
          ecart_identite < p["sanity_miou_vs_p314_ecart_max_pt"] * 3,
          f"{ecart_identite:.2e} pt vs sanity déclarée "
          f"{p['sanity_miou_vs_p314_ecart_max_pt']:.2e} pt")
    pct_moy = sum(p["rang_percentile"][e][ARM] for e in end) / len(end)
    check("percentile moyen du MoE recalculé == profil", abs(pct_moy - p["profil"][ARM]["mean_pct"]) < 1e-9,
          f"{pct_moy:.6f} vs {p['profil'][ARM]['mean_pct']:.6f}")

    # Critère T0 pré-enregistré : ΔmIoU significatif ET aucun endpoint à plus de 1 pt sous la
    # référence ET aucune des 19 classes dégradée de plus de 0,5 pt. RECALCULÉ.
    seuil_end, seuil_classe = -1.0, -0.5
    t0 = []
    for a in comp:
        sig = p["p_brut"][a]["mIoU"] < 0.05
        end_ok = dom[a]["dommage_max_pt"] > seuil_end
        n_cls = sum(1 for e in classes_end if p["delta_vs_controle_pt"][a][e] < seuil_classe)
        if sig and end_ok and n_cls == 0:
            t0.append(a)
        stocke = p["verdict"]["T0_détail_bras_mIoU_significatif"].get(a)
        if sig and stocke:
            check(f"T0 détail recalculé == stocké ({a})",
                  abs(dom[a]["dommage_max_pt"] - stocke["dommage_max_pt"]) < 0.01
                  and dom[a]["n_dommage_1pt"] == stocke["n_endpoints_>1pt"]
                  and n_cls == stocke["n_classes_perte_0.5pt"],
                  f"dommage {dom[a]['dommage_max_pt']:+.3f} vs {stocke['dommage_max_pt']}, "
                  f"end>1pt {dom[a]['n_dommage_1pt']} vs {stocke['n_endpoints_>1pt']}, "
                  f"classes {n_cls} vs {stocke['n_classes_perte_0.5pt']}")
    check("T0 recalculé : le MoE est le SEUL bras à cocher les trois critères",
          t0 == [ARM] and p["verdict"]["T0_gagne_et_ne_casse_rien"] == [ARM],
          f"recalculé {t0}, stocké {p['verdict']['T0_gagne_et_ne_casse_rien']}")
    sig_brut = [a for a in comp if p["p_brut"][a]["mIoU"] < 0.05]
    check("bras à mIoU significative recalculés == stockés",
          sorted(sig_brut) == sorted(p["verdict"]["T2_bras_miou_significatif"]),
          f"{sorted(sig_brut)}")

    # Prix consenti par point de mIoU gagné — ratio calculé, division par zéro gardée.
    prix = {}
    for a in sig_brut:
        dm = p["delta_vs_controle_pt"][a]["mIoU"]
        prix[a] = (dom[a]["dommage_max_pt"] / dm) if dm > 0 else None
    for a in prix:
        st = p["verdict"]["T2_prix_payé_par_pt_miou"].get(a)
        check(f"prix par point de mIoU recalculé == stocké ({a})",
              st is not None and abs(prix[a] - st) < 0.05, f"{prix[a]:.2f} vs {st}")
    check("le MoE paie le prix le plus faible par point de mIoU",
          min(prix, key=lambda a: abs(prix[a])) == ARM,
          " · ".join(f"{a} {prix[a]:+.1f}" for a in sorted(prix, key=lambda x: -prix[x])))

    # z-score du dommage maximal (normalisation inter-bras par endpoint)
    z = {}
    for a in comp:
        e = dom[a]["endpoint"]
        sd = p["sd_inter_bras_par_endpoint"][e]
        z[a] = dom[a]["dommage_max_pt"] / sd if sd else 0.0
    check("z du dommage maximal recalculé == stocké (MoE)",
          abs(z[ARM] - p["profil"][ARM]["dommage_max_z"]) < 0.02,
          f"{z[ARM]:+.3f} vs {p['profil'][ARM]['dommage_max_z']:+.3f}")
    check("z du MoE ≫ meilleur z des spécialistes en dommage",
          abs(z[ARM]) < abs(z["D"]), f"MoE {z[ARM]:+.2f} vs D {z['D']:+.2f}")

    # Pic de spécialisation : le MoE a-t-il un pic ?
    pic = {a: max(p["delta_vs_controle_pt"][a][e] for e in end) for a in comp}
    check("pic maximal du MoE recalculé == profil", abs(pic[ARM] - p["profil"][ARM]["pic_max_pt"]) < 0.01,
          f"{pic[ARM]:+.3f} vs {p['profil'][ARM]['pic_max_pt']:+.3f}")

    # Sanity internes déclarés par P3.17
    check("sanity P3.17 : mIoU consolidé == P3.14 à < 0,01 pt",
          p["sanity_miou_vs_p314_ecart_max_pt"] < 0.01,
          f"{p['sanity_miou_vs_p314_ecart_max_pt']:.2e} pt")
    check("sanity P3.17 : IoU par classe recomputées depuis npz à < 0,1 pt",
          p["sanity_iou_classe_recomputee_ecart_max_pt"] < 0.1,
          f"{p['sanity_iou_classe_recomputee_ecart_max_pt']:.2e} pt")
    v = p["verdict"]["verdict"]
    check("verdicts stockés T0-T3 vrais, T4 partiel (2 seeds sur 3)",
          v["T0"] and v["T1"] and v["T2"] and v["T3"] and not v["T4"]
          and v["T4_moe_premier_sur_n_seeds"] == 2,
          json.dumps(v, ensure_ascii=False))
    n_seeds_premier = sum(1 for s in SEEDS
                          if p["par_seed"][str(s)]["classement_dommage"][0] == ARM)
    check("T4 recalculé depuis par_seed : le MoE est 1er en dommage sur 2 seeds sur 3",
          n_seeds_premier == v["T4_moe_premier_sur_n_seeds"], f"{n_seeds_premier}/3")

    # ---- rendu ----
    prof = {a: p["profil"][a] for a in comp}
    ordre_prof = sorted(comp, key=lambda a: -prof[a]["mean_pct"])
    md = ["# T7 — Polyvalence : 36 endpoints × 13 bras, critères pré-enregistrés RECALCULÉS", "",
          f"Les critères ont été écrits **avant** lecture des résultats "
          f"(`criteres_pre_enregistres` de l'artefact) : "
          + " · ".join(f"**{k}** = {v}" for k, v in p["criteres_pre_enregistres"].items()) + ".", "",
          f"Protocole : {p['protocole']}. Chaque verdict du tableau ci-dessous est **recalculé "
          f"ici depuis les deltas et les rangs bruts**, puis comparé au verdict stocké — la "
          f"thèse « seul bras polyvalent » est re-démontrée, pas recopiée.", "",
          "| Critère | Énoncé | Résultat recalculé | Verdict |", "|---|---|---|---|"]
    md += [
        f"| **T0** | ΔmIoU significatif **et** aucun des {len(end)} endpoints à plus de 1 pt sous "
        f"la référence **et** aucune des {len(classes_end)} classes dégradée de plus de 0,5 pt | "
        f"{len(t0)} bras sur {len(comp)} : **{LABEL[t0[0]]}** | ✅ |",
        f"| **T1** | dommage maximal le plus faible du plateau | {dom[ARM]['dommage_max_pt']:+.2f} pt "
        f"(`{dom[ARM]['endpoint']}`) contre {dom[classement[1]]['dommage_max_pt']:+.2f} "
        f"({classement[1]}) et {dom['D']['dommage_max_pt']:+.2f} (D) — marge "
        f"{marge:.2f} pt | ✅ |",
        f"| **T2** | ΔmIoU significatif malgré tout | Δ {p['delta_vs_controle_pt'][ARM]['mIoU']:+.2f} pt, "
        f"p = {fp(p['p_brut'][ARM]['mIoU'])}, Holm({fam['n12']}) {fam['h12'][f'{ARM}_vs_{REF}']:.4f}, "
        f"Holm({fam['n15']}) {fam['h15'][f'{ARM}_vs_{REF}']:.4f} | ✅ au seuil brut, ❌ après Holm "
        f"exploratoire — écrit tel quel |",
        f"| **T3** | aucun endpoint gagné outright | {n_rang1[ARM]} rang 1 sur {len(end)} "
        f"(D en a {n_rang1['D']}), pire rang {pire_rang}/{pire_denom} sur `{pire_end}`, "
        f"jamais dernier ({n_dernier[ARM]} dernière place) | ✅ |",
        f"| **T4** | verdict stable seed par seed | 1ᵉʳ en dommage sur "
        f"{n_seeds_premier}/{len(SEEDS)} seeds (42, 456 ; seed 123 → 3ᵉ) | ❌ partiel |",
    ]
    md += ["", "## Dommage maximal par bras — le classement qui fonde la thèse", "",
           "| Rang | Bras | dommage max (pt) | endpoint | z | ΔmIoU (pt) | p | prix payé par pt de "
           "mIoU | endpoints < −1 pt | classes < −0,5 pt | percentile moyen | pire rang | #1 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(classement, 1):
        pr = prof[a]
        n_cls = sum(1 for e in classes_end if p["delta_vs_controle_pt"][a][e] < -0.5)
        gras = ("**", "**") if a == ARM else ("", "")
        md.append(f"| {i} | {gras[0]}{LABEL[a]}{gras[1]} | {gras[0]}{dom[a]['dommage_max_pt']:+.2f}{gras[1]} | "
                  f"`{dom[a]['endpoint']}` | {z[a]:+.2f} | "
                  f"{p['delta_vs_controle_pt'][a]['mIoU']:+.2f} | {fp(p['p_brut'][a]['mIoU'])} | "
                  f"{(f'{prix[a]:+.1f}' if prix.get(a) is not None else '—')} | "
                  f"{dom[a]['n_dommage_1pt']} | {n_cls} | {pr['mean_pct'] * 100:.1f} | "
                  f"{pire_rang_denom[a]['rang']}/{pire_rang_denom[a]['denom']} | {n_rang1[a]} |")
    md += ["", "« Prix payé par point de mIoU » = dommage maximal divisé par le ΔmIoU, pour les "
              "bras dont le ΔmIoU est significatif au seuil brut : c'est le taux de change entre "
              "le gain global et ce que le bras casse. Le MoE-V3-CS est le moins cher du plateau "
              "d'un facteur mesuré.", "",
              "## Niveau moyen : « bon partout » ≠ « meilleur partout »", "",
              "| Rang | Bras | percentile moyen | médiane | p10 | min | max | #1 | pire rang |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(ordre_prof, 1):
        pr = prof[a]
        gras = ("**", "**") if a == ARM else ("", "")
        md.append(f"| {i} | {gras[0]}{LABEL[a]}{gras[1]} | {gras[0]}{pr['mean_pct'] * 100:.1f}{gras[1]} | "
                  f"{pr['median_pct'] * 100:.1f} | {pr['p10_pct'] * 100:.1f} | "
                  f"{pr['min_pct'] * 100:.1f} | {pr['max_pct'] * 100:.1f} | {n_rang1[a]} | "
                  f"{pire_rang_denom[a]['rang']}/{pire_rang_denom[a]['denom']} |")
    check("le MoE n'est PAS premier en niveau moyen (ne pas survendre)",
          ordre_prof[0] != ARM, f"1er = {ordre_prof[0]} ({prof[ordre_prof[0]]['mean_pct'] * 100:.1f} %), "
                                f"MoE {ordre_prof.index(ARM) + 1}ᵉ ({prof[ARM]['mean_pct'] * 100:.1f} %)")
    # DEUX compteurs que tout lecteur confondrait : « endpoints au-dessus du contrôle » (Δ > 0,
    # mesuré ici) et « endpoints dans la moitié haute du CLASSEMENT » (`n_top_half` du profil,
    # qui compte la position parmi les 13 bras, indépendamment du signe). Le cadrage citait 22/36
    # pour le premier, le profil en donne 28 pour le second. Les deux sont calculés, nommés et
    # distingués — jamais employés l'un pour l'autre.
    n_pos = sum(1 for e in end if p["delta_vs_controle_pt"][ARM][e] > 0)
    n_neg = sum(1 for e in end if p["delta_vs_controle_pt"][ARM][e] < 0)
    check("Δ > 0 et Δ < 0 partitionnent les 36 endpoints (aucun Δ nul)",
          n_pos + n_neg == len(end), f"{n_pos} positifs + {n_neg} négatifs = {len(end)}")
    check("endpoints au-dessus du contrôle == 22 (valeur citée par le cadrage)", n_pos == 22,
          f"{n_pos}/{len(end)}")
    check("les deux compteurs ne sont PAS égaux (sinon l'ambiguïté serait sans objet)",
          n_pos != prof[ARM]["n_top_half"],
          f"Δ > 0 : {n_pos} · moitié haute du classement : {prof[ARM]['n_top_half']}")
    md += ["", f"Écrit tel quel : en **niveau moyen**, D ({prof['D']['mean_pct'] * 100:.1f} %) > "
              f"Dp ({prof['Dp']['mean_pct'] * 100:.1f} %) > consensus D⊘B "
              f"({prof['fused_DvetoB']['mean_pct'] * 100:.1f} %) > **MoE-V3-CS "
              f"({prof[ARM]['mean_pct'] * 100:.1f} %, {ordre_prof.index(ARM) + 1}ᵉ)**. Sur une "
              f"cible unique, le spécialiste reste devant. Le MoE n'est premier sur **aucun** des "
              f"{len(end)} endpoints. Ce qu'il est le seul à faire, c'est n'être **nulle part "
              f"cassé** : dommage maximal de {dom[ARM]['dommage_max_pt']:+.2f} pt là où D descend "
              f"à {dom['D']['dommage_max_pt']:+.2f} pt.", "",
              f"Deux compteurs à ne pas confondre, tous deux calculés : **{n_pos} des {len(end)} "
              f"endpoints sont au-dessus du contrôle** (Δ > 0 ; {n_neg} en dessous, aucun nul) et "
              f"**{prof[ARM]['n_top_half']} sont dans la moitié haute du classement** entre les "
              f"{len(prof) + 1} bras ({prof[ARM]['n_bottom_half']} dans la moitié basse). Le "
              f"second ne mesure pas un signe mais une **position relative** : un bras peut être "
              f"dans la moitié haute tout en restant sous le contrôle, et inversement.", "",
              "## Stabilité seed par seed", "",
              "| Seed | classement en dommage max (3 premiers) | dommage du MoE | ΔmIoU du MoE | "
              "endpoints en rang 1 |", "|---|---|---|---|---|"]
    for s in SEEDS:
        ps = p["par_seed"][str(s)]
        # Tous les champs de `par_seed` sont des dicts PAR BRAS (mesuré) : indexer par bras,
        # jamais formater le dict entier. Et ne pas dérouler les 11 bras dans une cellule.
        top3 = ps["classement_dommage"][:3]
        check(f"par_seed {s} : le MoE est classé et son dommage est du bon ordre",
              ARM in ps["classement_dommage"]
              and abs(ps["dommage_max_pt"][ARM]) < abs(dom[ARM]["dommage_max_pt"]) * 10,
              f"rang {ps['classement_dommage'].index(ARM) + 1}, dommage seed "
              f"{ps['dommage_max_pt'][ARM]:+.2f} (moyenne 3 seeds {dom[ARM]['dommage_max_pt']:+.2f})")
        md.append(f"| {s} | {' → '.join(LABEL[b] for b in top3)} | "
                  f"{ps['dommage_max_pt'][ARM]:+.2f} | {ps['delta_miou_pt'][ARM]:+.3f} | "
                  f"{ps['n_rang1'][ARM]} |")
    md += ["", f"Le seed {123} fait passer le MoE en 3ᵉ position derrière consensus C⊘B et Cp⊘B — "
              f"mais ces deux bras ont un ΔmIoU de "
              f"{p['delta_vs_controle_pt']['fused_CvetoB']['mIoU']:+.2f} et "
              f"{p['delta_vs_controle_pt']['fused_CpvetoB']['mIoU']:+.2f} pt (non significatifs) : "
              f"ils ne cassent rien parce qu'ils ne font rien. C'est la raison pour laquelle T4 "
              f"est déclaré **partiel** plutôt que passé.", "",
              "## Par famille de métriques", "",
              "| Bras | " + " | ".join(sorted(set(fam_end.values()))) + " | pire famille |",
              "|---|" + "---|" * (len(set(fam_end.values())) + 1)]
    fams = sorted(set(fam_end.values()))
    for a in [ARM] + [x for x in classement if x != ARM][:5]:
        cells = []
        for f in fams:
            es = [e for e in end if fam_end[e] == f]
            cells.append(f"{sum(p['delta_vs_controle_pt'][a][e] for e in es) / len(es):+.2f}")
        gras = ("**", "**") if a == ARM else ("", "")
        # Dans `profil`, `pire_famille` est le NOM (chaîne) et le delta vit dans
        # `pire_famille_delta_moyen_pt` ; la forme dict n'existe que dans
        # `verdict["T5_pire_famille"]`. Mesuré avant d'écrire, pas supposé.
        md.append(f"| {gras[0]}{LABEL[a]}{gras[1]} | " + " | ".join(cells) +
                  f" | {prof[a]['pire_famille']} "
                  f"({prof[a]['pire_famille_delta_moyen_pt']:+.2f}) |")
    check("une seule famille négative pour le MoE", prof[ARM]["n_familles_negatives"] == 1,
          f"{prof[ARM]['n_familles_negatives']} : {prof[ARM]['pire_famille']}")
    md += ["", f"La colonne « IoU par classe » est la moyenne des 19 Δ par classe : elle est donc "
              f"**presque identique** à la colonne « global » (le mIoU), puisque le mIoU *est* "
              f"cette moyenne. Presque seulement — mesuré à {ecart_identite:.2e} pt d'écart, parce "
              f"que cet artefact **mélange deux forwards GPU** des mêmes checkpoints : ses "
              f"colonnes IoU par classe viennent de l'attribution P3.16 (Δ {moy_cls:+.4f} pt) et "
              f"sa colonne mIoU de la régénération métiers (Δ "
              f"{p['delta_vs_controle_pt'][ARM]['mIoU']:+.4f} pt). L'identité est exacte à "
              f"l'intérieur d'un même artefact (vérifié en T6, écart 0). C'est le même écart "
              f"inter-forwards que celui déclaré en T2, pas une incohérence.", "",
              f"Sanités internes de l'artefact, relues : mIoU consolidé == P3.14 à "
              f"{p['sanity_miou_vs_p314_ecart_max_pt']:.2e} pt ; IoU par classe recomputées "
              f"depuis les npz bruts == table à {p['sanity_iou_classe_recomputee_ecart_max_pt']:.2e} pt.", "",
              f"Provenance : `{P317.relative_to(REPO)}`, `{P317_SEEDS.relative_to(REPO)}`."]

    lignes = [{"bras": a, "label": LABEL[a], "dommage_max_pt": dom[a]["dommage_max_pt"],
               "endpoint_dommage": dom[a]["endpoint"], "z": z[a],
               "delta_miou_pt": p["delta_vs_controle_pt"][a]["mIoU"],
               "p_miou": p["p_brut"][a]["mIoU"], "holm12": fam["h12"].get(f"{a}_vs_{REF}"),
               "holm15": fam["h15"].get(f"{a}_vs_{REF}"),
               "prix_par_pt_miou": prix.get(a), "n_endpoints_dommage_1pt": dom[a]["n_dommage_1pt"],
               "n_classes_perte_0.5pt": sum(1 for e in classes_end
                                            if p["delta_vs_controle_pt"][a][e] < -0.5),
               "percentile_moyen": prof[a]["mean_pct"] * 100.0,
               "pire_rang": prof[a]["pire_rang"], "n_rang1": n_rang1[a],
               "n_dernier": n_dernier[a], "pic_max_pt": pic[a],
               "rang_dommage": classement.index(a) + 1,
               "T0_verdict": a in t0} for a in comp]
    csv_txt = _csv(list(lignes[0].keys()), lignes)
    data = {"n_endpoints": len(end), "n_classes_endpoints": len(classes_end),
            "n_endpoints_delta_positif": n_pos, "n_endpoints_delta_negatif": n_neg,
            "n_top_half_classement": prof[ARM]["n_top_half"],
            "n_bottom_half_classement": prof[ARM]["n_bottom_half"],
            "pire_rang_avec_denominateur": pire_rang_denom[ARM],
            "denominateurs_rang_observes": sorted({v["denom"] for v in pire_rang_denom.values()}),
            "ecart_identite_classe_miou_pt": ecart_identite,
            "n_bras": p["n_bras"], "n_bras_complets": len(bras),
            "criteres": p["criteres_pre_enregistres"], "lignes": lignes,
            "classement_dommage": classement, "T0_gagnants": t0,
            "dommage_moe": dom[ARM], "marge_vs_suivant": marge,
            "prix": prix, "z": z, "pic": pic, "n_rang1": n_rang1, "n_dernier": n_dernier,
            "pire_rang_moe": pire_rang, "percentile_moyen_moe": pct_moy * 100.0,
            "ordre_percentile": ordre_prof, "par_seed": p["par_seed"],
            "n_seeds_premier": n_seeds_premier, "verdicts_stocks": v,
            "familles": {f: sum(1 for e in end if fam_end[e] == f) for f in fams},
            "sanity_miou_vs_p314": p["sanity_miou_vs_p314_ecart_max_pt"],
            "sanity_iou_recompute": p["sanity_iou_classe_recomputee_ecart_max_pt"],
            "seuils": {"endpoint_pt": seuil_end, "classe_pt": seuil_classe}}
    return "\n".join(md) + "\n", csv_txt, data


# --------------------------------------------------------------------------- #
# T8 — réplication BRATS
# --------------------------------------------------------------------------- #

def t8_brats(fam: dict, d4: dict) -> tuple[str, dict]:
    if not BRATS_P3_YAML.exists():
        raise SystemExit(f"[FAIL] paper.yaml BRATS illisible : {BRATS_P3_YAML}")
    txt = BRATS_P3_YAML.read_text(encoding="utf-8")

    def champ(k):
        m = re.search(rf'^{k}:\s*"?([^"\n#]+)"?', txt, re.M)
        return m.group(1).strip().strip('"') if m else None
    doi, cdoi = champ("zenodo_doi"), champ("zenodo_concept_doi")
    rec, pub, meth = champ("zenodo_record_id"), champ("zenodo_published"), champ("method")
    for k, v in [("zenodo_doi", doi), ("zenodo_concept_doi", cdoi), ("zenodo_record_id", rec),
                 ("zenodo_published", pub)]:
        check(f"BRATS paper3 : champ {k} lu dans le dépôt voisin", bool(v), str(v))
    check("BRATS paper3 publié", champ("status") == "published", str(champ("status")))
    check("le DOI BRATS diffère de celui du paper4 Cityscapes (pas de confusion)",
          doi != "10.5281/zenodo.23085640", doi)

    kw = j(RES / f"train_moe_v3cs_Binit_seed{SEEDS[0]}_summary.json")["provenance"]["moe_kwargs"]
    h = fam["harness"]
    d = -h["bootstrap_miou"]["pairwise"][fam["cle1"]]["delta"] * 100.0
    md = ["# T8 — Antériorité : la réplication BRATS, et ce qui diffère ici", "",
          f"Ce papier est une **réplication sur un second dataset**, pas une republication. "
          f"L'antériorité est le paper 3 du programme BraTS, *The Gate Does Not Choose*, "
          f"publié sous DOI `{doi}` (concept `{cdoi}`, record {rec}, {pub}). Aucun de ses "
          f"chiffres n'est repris ici ; il est cité comme antériorité et comme point de "
          f"comparaison qualitative.", "",
          "| Dimension | BraTS (antériorité) | Cityscapes (ce papier) |", "|---|---|---|"]
    md += [
        "| Dataset | BraTS-2023, gliome, 3D, 1 196 cas | Cityscapes, conduite urbaine, 2D, "
        f"holdout `first:{h['holdout']['n']}` de val |",
        "| Backbone | MedNeXt | ConvNeXt-V2-Base (ImageNet-22K) + UPerNet, 1024×2048 |",
        "| Point d'accroche | `dec_block_0` | `head.fpn_bottleneck` (dernier élément du Sequential) |",
        "| Métriques | Dice, HD95, lésion-wise | mIoU dataset-level officiel, Boundary F1, "
        "rappel d'instances par strate, fragments |",
        "| Garde-fou résiduel γ | absent (recette brute suffisante) | **nécessaire** : sans γ, la "
        "recette brute coûte −28,1 pt de mIoU à l'époque 0 (mesuré P3.08) |",
        "| Normalisation | GroupNorm, blocs MedNeXt identiques partout | BatchNorm + latéraux "
        "UPerNet : les experts voient au point d'accroche une distribution qu'ils n'ont jamais "
        "traitée |",
        f"| Routage | top-{kw['top_k']} parmi {kw['n_experts']}, grille {kw['grid'][0]}×{kw['grid'][1]} "
        f"| identique (recette reprise à l'identique) |",
        "| Résultat commun | la porte ne choisit pas ; le gain vient de l'initialisation par "
        f"experts | la porte ne choisit pas non plus (T4 : `entropy_norm` minimal "
        f"{min(d4['entropy_norm_final'].values()):.5f} sur 3 seeds, `part_max` "
        f"{max(abs(v - d4['equirepartition']) for v in d4['part_max_final'].values()):.4f} "
        f"au plus loin de l'équirépartition {d4['equirepartition']:.4f}, "
        f"{sum(d4['experts_morts'].values())} expert mort) |",
        f"| Critère primaire | — | Δ {d:+.2f} pt de mIoU vs contrôle apparié, "
        f"p = {fp(h['bootstrap_miou']['pairwise'][fam['cle1']]['p_two_sided'])} |",
    ]
    md += ["", "La différence qui compte est le **garde-fou γ** : il n'était pas nécessaire côté "
              "BraTS et il est indispensable ici. C'est une propriété mesurée de l'architecture "
              "d'accueil (distribution au point d'accroche, absence de renormalisation avant le "
              "classifier), pas un réglage choisi après coup : avec γ initialisé à "
              f"{kw['residual_scale']}, l'époque 0 est bit-exacte au contrôle, et |γ| devient un "
              "diagnostic direct de la thèse.", "",
              f"Provenance : `{BRATS_P3_YAML}` (lu, pas recopié) pour les identifiants de "
              f"l'antériorité ; artefacts Cityscapes pour tout le reste."]
    data = {"brats_doi": doi, "brats_concept_doi": cdoi, "brats_record_id": rec,
            "brats_published": pub, "brats_method": meth,
            "gamma_necessaire_ici": True, "delta_primaire_pt": d}
    return "\n".join(md) + "\n", data


# --------------------------------------------------------------------------- #
# Écriture
# --------------------------------------------------------------------------- #

def ecrire(p: Path, contenu: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contenu, encoding="utf-8")


def main() -> int:
    t0 = datetime.now()
    log("P3.18a — tables T1-T8 du paper 3 (MoE-V3-CS), sans GPU")
    for f in [HARNESS, P314, P316_MET, P316_ATT, P317, P317_SEEDS, BRATS_P3_YAML]:
        if not Path(f).exists():
            raise SystemExit(f"[FAIL] artefact absent : {f}")
        log(f"  artefact présent : {Path(f).relative_to(REPO) if REPO in Path(f).parents else f} "
            f"({Path(f).stat().st_size:,} o)")

    fam = familles()
    logs = parse_logs()

    t1, d1 = t1_archi(fam, logs)
    t2, d2 = t2_primaire(fam)
    t3, t3c, d3 = t3_master(fam)
    t4, t4c, d4 = t4_routage(fam)
    t5, t5c, d5 = t5_metiers(fam)
    t6, t6c, d6 = t6_perclass(fam)
    t7, t7c, d7 = t7_polyvalence(fam)
    t8, d8 = t8_brats(fam, d4)

    ecrire(OUT_TABLES / "T1_architecture_recette_cout.md", t1)
    ecrire(OUT_TABLES / "T2_primaire.md", t2)
    ecrire(OUT_TABLES / "T3_master_13bras.md", t3)
    ecrire(OUT_TABLES / "T3_master_13bras.csv", t3c)
    ecrire(OUT_TABLES / "T4_routage.md", t4)
    ecrire(OUT_TABLES / "T4_routage.csv", t4c)
    ecrire(OUT_TABLES / "T5_metiers.md", t5)
    ecrire(OUT_TABLES / "T5_metiers.csv", t5c)
    ecrire(OUT_TABLES / "T6_perclass.md", t6)
    ecrire(OUT_TABLES / "T6_perclass.csv", t6c)
    ecrire(OUT_TABLES / "T7_polyvalence.md", t7)
    ecrire(OUT_TABLES / "T7_polyvalence.csv", t7c)
    ecrire(OUT_TABLES / "T8_replication_brats.md", t8)
    for f in ["T1_architecture_recette_cout.md", "T2_primaire.md", "T3_master_13bras.md",
              "T4_routage.md", "T5_metiers.md", "T6_perclass.md", "T7_polyvalence.md",
              "T8_replication_brats.md"]:
        log(f"  → papers/paper3/tables/{f}")

    consol = {"date": t0.isoformat(timespec="seconds"), "script": "scripts/p3_moe_tables.py",
              "gpu_requis": False, "B": fam["harness"]["bootstrap_miou"]["B"],
              "bootstrap_seed": fam["harness"]["bootstrap_seed"],
              "holdout": fam["harness"]["holdout"], "seeds": list(SEEDS),
              "bras": ARM, "reference": REF,
              "provenance": {k: str(v.relative_to(REPO)) for k, v in
                            {"harness_P310": HARNESS, "master_P314": P314,
                             "metiers_P316": P316_MET, "attribution_P316": P316_ATT,
                             "polyvalence_P317": P317}.items()},
              "T1": d1, "T2": d2, "T3": d3, "T4": d4, "T5": d5, "T6": d6, "T7": d7, "T8": d8,
              "sanity": {"n_checks": len(SANITY), "n_ok": sum(1 for s in SANITY if s["ok"]),
                         "checks": SANITY}}
    ecrire(OUT_TABLES / "paper3_tables.json", json.dumps(consol, indent=1, ensure_ascii=False))

    ok = sum(1 for s in SANITY if s["ok"])
    smd = ["# SANITY — P3.18a tables du paper 3 (MoE-V3-CS)", "",
           f"Généré le {consol['date']} par `scripts/p3_moe_tables.py`, sans GPU.", "",
           f"**{ok}/{len(SANITY)} checks bloquants passés.** Tout échec arrête le script avant "
           f"l'écriture des tables (`SystemExit`) : aucune table ne peut être produite à partir "
           f"d'un artefact incohérent.", "",
           "## Régime (hérité de l'erratum v1.1.0 du paper4)", "",
           "1. Aucun compte, taille, somme ou ratio n'est écrit en dur : tout est calculé depuis "
           "l'artefact et vérifié par un `check()`.",
           "2. Le Holm est **recomputé** depuis les p bruts (`src/moe/bootstrap.holm`) et comparé "
           "au stocké à 1e-12, dans les **trois** familles de multiplicité du critère primaire "
           f"({d2['n_famille_1']} paire P3.10, {d2['n_famille_12']} paires P3.14, "
           f"{d2['n_famille_15']} paires P3.16). Une étiquette de famille ne peut plus mentir : "
           "sa taille vient de `len(pairwise)`.",
           "3. Les verdicts de polyvalence T0-T5 sont **recalculés** depuis les deltas et rangs "
           "bruts puis comparés aux verdicts stockés.",
           "4. Les valeurs sont croisées entre artefacts avec des tolérances **mesurées** puis "
           "déclarées (deux forwards GPU indépendants diffèrent de ~6e-4 pt).", "",
           "## Checks", "", "| # | check | détail |", "|---|---|---|"]
    smd += [f"| {i} | {'✅' if s['ok'] else '❌'} {s['check']} | {s['detail']} |"
            for i, s in enumerate(SANITY, 1)]
    ecrire(OUT_TABLES / "SANITY.md", "\n".join(smd) + "\n")
    ecrire(OUT_TABLES / "SANITY.json", json.dumps(SANITY, indent=1, ensure_ascii=False))

    dt = (datetime.now() - t0).total_seconds()
    log(f"[fin] {dt:.0f}s — T1-T8 + paper3_tables.json écrits, sanités {ok}/{len(SANITY)} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
