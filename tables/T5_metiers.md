# T5 — Métriques métier : ce que le MoE capte, ce qu'il paie

Holdout `first:500`, seeds moyennés dans chaque réplicat, bootstrap apparié B = 10,000 (seed 20260618), Holm sur la famille à **15 paires**. `n` = nombre d'images contribuant à la métrique (il varie : les strates d'instances et la précision piéton ne sont pas définies sur toutes les images). Unité : points (×100), sauf `fragments` = composantes connexes par image.

**Gras** = Holm < 0,05. Les deux critères (Holm et IC excluant 0) sont séparés, jamais confondus.

| Métrique | n | contrôle | MoE-V3-CS | Δ(MoE−ctl) | IC95 | p | Holm(15) | verdict |
|---|---|---|---|---|---|---|---|---|
| `mIoU` | 500 | 81.168 | 81.618 | +0.450 | [+0.110 ; +0.818] | 0.0066 | 0.0924 | IC exclut 0, ne survit pas au Holm |
| `boundary_f1_3px` | 500 | 76.627 | 76.567 | -0.061 | [-0.178 ; +0.055] | 0.3004 | 1 | ns |
| `boundary_f1_3px_pieds` | 500 | 67.486 | 67.427 | -0.059 | [-0.412 ; +0.309] | 0.7238 | 1 | ns |
| `rappel_strict_instances` | 441 | 76.253 | 76.728 | +0.475 | [-0.004 ; +0.935] | 0.0516 | 0.2064 | ns |
| `precision_ped_pixels` | 473 | 70.028 | 70.024 | -0.004 | [-0.377 ; +0.376] | 0.9896 | 0.9896 | ns |
| `instances_toutes_rappel` | 441 | 79.975 | 80.372 | +0.397 | [-0.028 ; +0.766] | 0.0694 | 0.2776 | ns |
| `instances_toutes_det05` | 441 | 87.111 | 87.256 | +0.145 | [-0.385 ; +0.651] | 0.5620 | 1 | ns |
| `instances_individuelles_rappel` | 440 | 80.096 | 80.504 | +0.407 | [-0.030 ; +0.783] | 0.0662 | 0.2648 | ns |
| `instances_individuelles_det05` | 440 | 87.175 | 87.317 | +0.141 | [-0.396 ; +0.653] | 0.5710 | 1 | ns |
| `instances_foule_rappel` | 63 | 74.584 | 74.199 | -0.384 | [-1.536 ; +0.493] | 0.4822 | 1 | ns |
| `instances_foule_det05` | 63 | 83.598 | 83.069 | -0.529 | [-2.646 ; +1.587] | 0.7946 | 1 | ns |
| `instances_taille_T1_rappel` | 315 | 64.994 | 65.796 | +0.802 | [+0.073 ; +1.510] | 0.0330 | 0.198 | IC exclut 0, ne survit pas au Holm |
| `instances_taille_T1_det05` | 315 | 71.728 | 72.042 | +0.314 | [-0.844 ; +1.455] | 0.5908 | 1 | ns |
| `instances_taille_T2_rappel` | 320 | 84.534 | 84.929 | +0.395 | [-0.031 ; +0.810] | 0.0684 | 0.6156 | ns |
| `instances_taille_T2_det05` | 320 | 93.147 | 93.432 | +0.285 | [-0.363 ; +1.020] | 0.4046 | 1 | ns |
| `instances_taille_T3_rappel` | 335 | **93.474** | **93.733** | **+0.258** | [+0.142 ; +0.384] | 0.0000 | 0 | **Holm-significatif** |
| `instances_taille_T3_det05` | 335 | 98.960 | 98.957 | -0.003 | [-0.114 ; +0.135] | 0.9104 | 1 | ns |
| `IoU_person` | 500 | 85.152 | 84.910 | -0.242 | [-0.429 ; -0.062] | 0.0080 | 0.088 | IC exclut 0, ne survit pas au Holm |
| `IoU_rider` | 500 | 68.920 | 68.555 | -0.365 | [-1.302 ; +0.524] | 0.4298 | 1 | ns |
| `fragments` | 500 | **519.3** | **488.0** | **-31.2** | [-35.7 ; -26.7] | 0.0000 | 0 | **Holm-significatif** |

## Fragments (topologie des masques), mesurés sur 2 bras seulement

| Quantité | contrôle | MoE-V3-CS | Δ | IC95 | p | Holm |
|---|---|---|---|---|---|---|
| composantes connexes par image | 519.3 | **488.0** | **-31.2** (-6.0 %) | [-35.7 ; -26.7] | 0.0000 | 0 |

Sur les 20 métriques métier, **2** survivent au Holm sur la famille 15 paires : `instances_taille_T3_rappel`, `fragments` (toutes deux avec p = 0). Toutes les autres sont non significatives une fois la multiplicité payée — y compris le mIoU lui-même (Holm 0.0924).

`fragments` est par ailleurs la seule métrique pour laquelle cet artefact ne couvre que le contrôle et le MoE (les autres bras ont leurs fragments dans `results/moe_v3_cs/p316/synthese_attribution.md`, hors bootstrap apparié) — d'où le tableau séparé ci-dessus.

## Ce que le MoE capte des experts, et ce qu'il n'hérite pas

| Métrique | Δ MoE | Δ D | Δ Dp | Δ G | Δ B | lecture calculée |
|---|---|---|---|---|---|---|
| `instances_taille_T1_rappel` | +0.802 | +2.816 | +2.016 | -4.080 | +1.565 | capte le gain de D/Dp/B |
| `rappel_strict_instances` | +0.475 | +2.183 | +1.719 | -3.417 | +0.918 | capte le gain de D/Dp/B |
| `instances_toutes_rappel` | +0.397 | +2.322 | +1.879 | -3.423 | +1.016 | capte le gain de D/Dp/B |
| `boundary_f1_3px` | -0.061 | +0.677 | +0.545 | +0.376 | -0.254 | neutre |
| `precision_ped_pixels` | -0.004 | -22.195 | -15.020 | -6.763 | -4.064 | perte, mais 0.00 ≪ -22.20 (D/Dp/G/B) |
| `instances_foule_rappel` | -0.384 | +1.511 | +1.319 | -4.612 | +1.124 | perte, mais 0.38 ≪ -4.61 (G) |
| `IoU_person` | -0.242 | -0.207 | -0.269 | +0.067 | -0.277 | neutre |

Lecture calculée : le MoE-V3-CS capte une partie du gain d'instances de D/Dp (T1 +0.80 contre +2.82 et +2.02) **sans** hériter ni de leur perte `traffic light` (-0.08 contre -2.60 et -2.34) ni du rappel piéton massacré par G (+0.47 contre -3.42). Sa contrepartie mesurée : **aucun gain de contours** (`boundary_f1_3px` -0.061, ns) — cohérent avec T4, une moyenne d'experts lisse les spécialités de contour. Sur 20 métriques, 2 survivent au Holm et 5 ont un IC excluant 0.

La polyvalence (T7) confirme : la pire famille du MoE est `contours` (-0.06 pt en moyenne) et c'est sa **seule** famille négative (1 sur 5).

Provenance : `results/moe_v3_cs/metiers_experts/table_metiers_experts.json`.
