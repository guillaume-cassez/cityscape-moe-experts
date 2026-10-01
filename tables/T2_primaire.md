# T2 — Critère primaire pré-enregistré : MoE-V3-CS vs son contrôle apparié

**Primary metric / métrique primaire** : mIoU dataset-level officiel (cityscapesScripts, 19 classes) sur le holdout partagé `first:500`, bras `moe_v3cs` contre son **contrôle apparié** `controle` (même initialisation B, même recette, 80 époques, sans couche MoE). Bootstrap **apparié par image** B = 10,000 (seed 20260618), seeds 42/123/456 **moyennés dans chaque réplicat**, p bilatéral. Critère déclaré avant les runs d'analyse (protocole P3.10).

| Quantité | MoE-V3-CS | contrôle apparié |
|---|---|---|
| mIoU point | **81.618** | **81.168** |
| IC95 | [80.264 ; 82.709] | [79.802 ; 82.251] |
| mIoU seed 42 | 81.814 | 81.561 |
| mIoU seed 123 | 81.339 | 80.556 |
| mIoU seed 456 | 81.701 | 81.388 |

**Δ(MoE − contrôle) = +0.449 pt · IC95 [+0.109 ; +0.820] · p = 0.0066 → SIGNIFICATIF.**

## Les trois familles de multiplicité, données ensemble

Le même contraste appartient à trois familles de Holm différentes selon le protocole qui le déclare. Les trois sont rapportées, aucune n'est choisie parce qu'elle arrange — et leurs tailles sont **calculées** depuis les artefacts, le Holm étant **recomputé** depuis les p bruts puis comparé à la valeur stockée (écart < 1e-12).

| Famille | Taille | Holm du critère primaire | Artefact source |
|---|---|---|---|
| paire unique pré-enregistrée (P3.10) | 1 | **0.0066** (= p) | `results/moe_v3_cs/harness/table_controle_v3cs_P310.json` |
| bras contre contrôle (P3.14) | 12 | 0.0726 | `results/moe_v3_cs/p314/master_table.json` |
| exploratoire du programme (P3.16) | 15 | 0.0924 | `results/moe_v3_cs/metiers_experts/table_metiers_experts.json` |

Lecture honnête : le critère primaire est significatif au seuil brut (p = 0.0066) et **dans la famille pré-enregistrée à 1 paire** (Holm = 0.0066). Il **ne survit pas** à la famille exploratoire à 15 paires (Holm = 0.0924 > 0,05) — et aucun bras du plateau n'y survit non plus (meilleur Holm 0.0750). C'est écrit tel quel : sur le mIoU, les conclusions du programme reposent sur les **ampleurs et leurs IC**, et la contribution de ce papier porte sur ce que chaque bras **casse** (T7), pas sur la significativité post-Holm du mIoU.

## Robustesse : deux forwards GPU indépendants des mêmes checkpoints

| Source | mIoU MoE | mIoU contrôle | Δ | p |
|---|---|---|---|---|
| harnais P3.10 (primaire cité ci-dessus) | 81.617575 | 81.168215 | +0.449360 pt | 0.0066 |
| régénération métiers P3.16 | 81.618182 | 81.167964 | +0.450218 pt | 0.0066 |

Écart mesuré **6.07e-04 pt** sur le mIoU du MoE et **2.51e-04 pt** sur celui du contrôle : non-déterminisme cuDNN/BF16 (`deterministic: false`), du même ordre que la sanity déclarée par P3.17 (4.13e-03 pt). Le verdict est identique dans les deux sources.

Provenance : `results/moe_v3_cs/harness/table_controle_v3cs_P310.json`, `results/moe_v3_cs/p314/master_table.json`, `results/moe_v3_cs/metiers_experts/table_metiers_experts.json`.
