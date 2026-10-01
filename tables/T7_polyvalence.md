# T7 — Polyvalence : 36 endpoints × 13 bras, critères pré-enregistrés RECALCULÉS

Les critères ont été écrits **avant** lecture des résultats (`criteres_pre_enregistres` de l'artefact) : **T1** = dommage maximal le plus faible du plateau · **T2** = ΔmIoU significatif malgré tout · **T3** = aucun endpoint gagné outright · **T4** = verdict stable seed par seed.

Protocole : holdout P3.10 first:500, seeds 42/123/456, bootstrap APPARIÉ B=10000 + Holm ; protocoles P3.15/P3.16 identiques (fusion = cc_veto prim/veto int64 max_drop_size None). Chaque verdict du tableau ci-dessous est **recalculé ici depuis les deltas et les rangs bruts**, puis comparé au verdict stocké — la thèse « seul bras polyvalent » est re-démontrée, pas recopiée.

| Critère | Énoncé | Résultat recalculé | Verdict |
|---|---|---|---|
| **T0** | ΔmIoU significatif **et** aucun des 36 endpoints à plus de 1 pt sous la référence **et** aucune des 19 classes dégradée de plus de 0,5 pt | 1 bras sur 11 : **MoE-V3-CS (4 experts)** | ✅ |
| **T1** | dommage maximal le plus faible du plateau | -0.53 pt (`instances_foule_det05`) contre -1.90 (fused_CvetoB) et -22.20 (D) — marge 1.37 pt | ✅ |
| **T2** | ΔmIoU significatif malgré tout | Δ +0.45 pt, p = 0.0066, Holm(12) 0.0726, Holm(15) 0.0924 | ✅ au seuil brut, ❌ après Holm exploratoire — écrit tel quel |
| **T3** | aucun endpoint gagné outright | 0 rang 1 sur 36 (D en a 16), pire rang 11/12 sur `instances_foule_rappel`, jamais dernier (0 dernière place) | ✅ |
| **T4** | verdict stable seed par seed | 1ᵉʳ en dommage sur 2/3 seeds (42, 456 ; seed 123 → 3ᵉ) | ❌ partiel |

## Dommage maximal par bras — le classement qui fonde la thèse

| Rang | Bras | dommage max (pt) | endpoint | z | ΔmIoU (pt) | p | prix payé par pt de mIoU | endpoints < −1 pt | classes < −0,5 pt | percentile moyen | pire rang | #1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MoE-V3-CS (4 experts)** | **-0.53** | `instances_foule_det05` | -0.35 | +0.45 | 0.0066 | -1.2 | 0 | 0 | 60.4 | 11/12 | 0 |
| 2 | consensus C⊘B | -1.90 | `IoU_bus` | -1.47 | +0.07 | 0.7866 | — | 1 | 2 | 44.0 | 13/13 | 0 |
| 3 | consensus Cp⊘B | -2.78 | `IoU_bus` | -2.16 | -0.15 | 0.3532 | — | 2 | 4 | 35.7 | 13/13 | 0 |
| 4 | C · CE+Dice+EDT | -3.21 | `precision_ped_pixels` | -0.50 | +0.06 | 0.8182 | — | 2 | 2 | 47.5 | 13/13 | 0 |
| 5 | B · CE+Dice (baseline) | -4.06 | `precision_ped_pixels` | -0.63 | -0.08 | 0.6592 | — | 2 | 3 | 51.2 | 13/13 | 2 |
| 6 | consensus Dp⊘B | -4.77 | `precision_ped_pixels` | -0.74 | +0.48 | 0.0376 | -9.9 | 2 | 2 | 69.0 | 12/12 | 4 |
| 7 | Cp · CE+Dice+SDT | -6.05 | `precision_ped_pixels` | -0.94 | -0.28 | 0.0618 | — | 3 | 4 | 34.5 | 13/13 | 0 |
| 8 | G · CE+Dice+Blob (Kofler) | -6.76 | `precision_ped_pixels` | -1.05 | +0.09 | 0.6592 | — | 16 | 3 | 41.9 | 13/13 | 7 |
| 9 | consensus D⊘B | -8.70 | `precision_ped_pixels` | -1.35 | +0.49 | 0.0082 | -17.9 | 3 | 3 | 71.9 | 12/13 | 3 |
| 10 | Dp · CE+SDT (distmap) | -15.02 | `precision_ped_pixels` | -2.32 | +0.47 | 0.0330 | -31.9 | 3 | 3 | 72.7 | 13/13 | 3 |
| 11 | D · CE+Kervadec EDT | -22.20 | `precision_ped_pixels` | -3.44 | +0.52 | 0.0050 | -43.1 | 3 | 4 | 77.3 | 13/13 | 16 |

« Prix payé par point de mIoU » = dommage maximal divisé par le ΔmIoU, pour les bras dont le ΔmIoU est significatif au seuil brut : c'est le taux de change entre le gain global et ce que le bras casse. Le MoE-V3-CS est le moins cher du plateau d'un facteur mesuré.

## Niveau moyen : « bon partout » ≠ « meilleur partout »

| Rang | Bras | percentile moyen | médiane | p10 | min | max | #1 | pire rang |
|---|---|---|---|---|---|---|---|---|
| 1 | D · CE+Kervadec EDT | 77.3 | 88.1 | 23.1 | 7.7 | 100.0 | 16 | 13/13 |
| 2 | Dp · CE+SDT (distmap) | 72.7 | 83.3 | 26.9 | 7.7 | 100.0 | 3 | 13/13 |
| 3 | consensus D⊘B | 71.9 | 83.3 | 24.0 | 15.4 | 100.0 | 3 | 12/13 |
| 4 | consensus Dp⊘B | 69.0 | 72.1 | 36.2 | 8.3 | 100.0 | 4 | 12/12 |
| 5 | **MoE-V3-CS (4 experts)** | **60.4** | 58.3 | 35.9 | 16.7 | 92.3 | 0 | 11/12 |
| 6 | B · CE+Dice (baseline) | 51.2 | 56.1 | 11.9 | 7.7 | 100.0 | 2 | 13/13 |
| 7 | C · CE+Dice+EDT | 47.5 | 48.1 | 30.8 | 7.7 | 84.6 | 0 | 13/13 |
| 8 | consensus C⊘B | 44.0 | 41.7 | 15.4 | 7.7 | 92.3 | 0 | 13/13 |
| 9 | G · CE+Dice+Blob (Kofler) | 41.9 | 32.1 | 8.3 | 7.7 | 100.0 | 7 | 13/13 |
| 10 | consensus Cp⊘B | 35.7 | 27.9 | 16.7 | 7.7 | 84.6 | 0 | 13/13 |
| 11 | Cp · CE+Dice+SDT | 34.5 | 25.0 | 15.4 | 7.7 | 76.9 | 0 | 13/13 |

Écrit tel quel : en **niveau moyen**, D (77.3 %) > Dp (72.7 %) > consensus D⊘B (71.9 %) > **MoE-V3-CS (60.4 %, 5ᵉ)**. Sur une cible unique, le spécialiste reste devant. Le MoE n'est premier sur **aucun** des 36 endpoints. Ce qu'il est le seul à faire, c'est n'être **nulle part cassé** : dommage maximal de -0.53 pt là où D descend à -22.20 pt.

Deux compteurs à ne pas confondre, tous deux calculés : **22 des 36 endpoints sont au-dessus du contrôle** (Δ > 0 ; 14 en dessous, aucun nul) et **28 sont dans la moitié haute du classement** entre les 12 bras (8 dans la moitié basse). Le second ne mesure pas un signe mais une **position relative** : un bras peut être dans la moitié haute tout en restant sous le contrôle, et inversement.

## Stabilité seed par seed

| Seed | classement en dommage max (3 premiers) | dommage du MoE | ΔmIoU du MoE | endpoints en rang 1 |
|---|---|---|---|---|
| 42 | MoE-V3-CS (4 experts) → consensus C⊘B → consensus Cp⊘B | -2.98 | +0.252 | 3 |
| 123 | consensus C⊘B → consensus Cp⊘B → MoE-V3-CS (4 experts) | -3.17 | +0.783 | 0 |
| 456 | MoE-V3-CS (4 experts) → consensus Dp⊘B → consensus C⊘B | -2.76 | +0.315 | 4 |

Le seed 123 fait passer le MoE en 3ᵉ position derrière consensus C⊘B et Cp⊘B — mais ces deux bras ont un ΔmIoU de +0.07 et -0.15 pt (non significatifs) : ils ne cassent rien parce qu'ils ne font rien. C'est la raison pour laquelle T4 est déclaré **partiel** plutôt que passé.

## Par famille de métriques

| Bras | IoU par classe | contours | global | instances par strate | piéton | pire famille |
|---|---|---|---|---|---|---|
| **MoE-V3-CS (4 experts)** | +0.45 | -0.06 | +0.45 | +0.19 | +0.24 | contours (-0.06) |
| consensus C⊘B | +0.07 | -0.41 | +0.07 | +0.26 | -0.26 | contours (-0.41) |
| consensus Cp⊘B | -0.15 | -0.35 | -0.15 | -0.09 | -0.54 | piéton (-0.54) |
| C · CE+Dice+EDT | +0.06 | -0.18 | +0.06 | +0.49 | -1.39 | piéton (-1.39) |
| B · CE+Dice (baseline) | -0.08 | -0.22 | -0.08 | +1.02 | -1.57 | piéton (-1.57) |
| consensus Dp⊘B | +0.48 | -0.46 | +0.48 | +0.82 | -1.79 | piéton (-1.79) |

La colonne « IoU par classe » est la moyenne des 19 Δ par classe : elle est donc **presque identique** à la colonne « global » (le mIoU), puisque le mIoU *est* cette moyenne. Presque seulement — mesuré à 8.58e-04 pt d'écart, parce que cet artefact **mélange deux forwards GPU** des mêmes checkpoints : ses colonnes IoU par classe viennent de l'attribution P3.16 (Δ +0.4494 pt) et sa colonne mIoU de la régénération métiers (Δ +0.4502 pt). L'identité est exacte à l'intérieur d'un même artefact (vérifié en T6, écart 0). C'est le même écart inter-forwards que celui déclaré en T2, pas une incohérence.

Sanités internes de l'artefact, relues : mIoU consolidé == P3.14 à 4.13e-03 pt ; IoU par classe recomputées depuis les npz bruts == table à 6.33e-02 pt.

Provenance : `results/moe_v3_cs/p317_polyvalence/table_polyvalence.json`, `results/moe_v3_cs/p317_polyvalence/verif_seeds.md`.
