# SANITY — P3.18a tables du paper 3 (MoE-V3-CS)

Généré le 2026-10-01T20:30:06 par `scripts/p3_moe_tables.py`, sans GPU.

**257/257 checks bloquants passés.** Tout échec arrête le script avant l'écriture des tables (`SystemExit`) : aucune table ne peut être produite à partir d'un artefact incohérent.

## Régime (hérité de l'erratum v1.1.0 du paper4)

1. Aucun compte, taille, somme ou ratio n'est écrit en dur : tout est calculé depuis l'artefact et vérifié par un `check()`.
2. Le Holm est **recomputé** depuis les p bruts (`src/moe/bootstrap.holm`) et comparé au stocké à 1e-12, dans les **trois** familles de multiplicité du critère primaire (1 paire P3.10, 12 paires P3.14, 15 paires P3.16). Une étiquette de famille ne peut plus mentir : sa taille vient de `len(pairwise)`.
3. Les verdicts de polyvalence T0-T5 sont **recalculés** depuis les deltas et rangs bruts puis comparés aux verdicts stockés.
4. Les valeurs sont croisées entre artefacts avec des tolérances **mesurées** puis déclarées (deux forwards GPU indépendants diffèrent de ~6e-4 pt).

## Checks

| # | check | détail |
|---|---|---|
| 1 | ✅ famille 1 paire (P3.10) : taille calculée == 1 | len = 1 |
| 2 | ✅ famille 1 paire : Holm == p (recomputé) | holm=0.0066 p=0.0066 |
| 3 | ✅ famille 1 paire : Holm recomputé == stocké | 0.0066 vs 0.0066 |
| 4 | ✅ famille P3.14 : taille calculée == 12 | len(pairwise) = 12 |
| 5 | ✅ famille P3.14 : Holm recomputé == stocké (12 paires) | écart max 0 |
| 6 | ✅ famille P3.16 : taille calculée == len(pairs) déclaré | 15 vs 15 |
| 7 | ✅ famille P3.16 : Holm recomputé == stocké | écart max 0 |
| 8 | ✅ P3.17 cite la famille 15 paires (pas une autre) | 11 bras comparés, écart max 0 |
| 9 | ✅ les 3 familles sont DISTINCTES (Holm moe différent partout) | 1 paire 0.0066 / 12 paires 0.0726 / 15 paires 0.0924 |
| 10 | ✅ famille P3.16 : bras vs contrôle == paires déclarées vs contrôle | 11 bras vs contrôle + 4 paires fusion-vs-expert = 15 paires |
| 11 | ✅ le MoE est parmi les bras comparés au contrôle | 11 bras : ['B', 'C', 'Cp', 'D', 'Dp', 'G', 'fused_CpvetoB', 'fused_CvetoB', 'fused_DpvetoB', 'fused_DvetoB', 'moe_v3cs'] |
| 12 | ✅ log moe_v3cs_Binit seed42 : 80 époques lues | 80 lignes de temps, 80 de VRAM |
| 13 | ✅ log moe_v3cs_Binit seed123 : 80 époques lues | 80 lignes de temps, 80 de VRAM |
| 14 | ✅ log moe_v3cs_Binit seed456 : 80 époques lues | 80 lignes de temps, 80 de VRAM |
| 15 | ✅ log control_nomoe_Binit seed42 : 80 époques lues | 80 lignes de temps, 80 de VRAM |
| 16 | ✅ log control_nomoe_Binit seed123 : 80 époques lues | 80 lignes de temps, 80 de VRAM |
| 17 | ✅ log control_nomoe_Binit seed456 : 80 époques lues | 80 lignes de temps, 80 de VRAM |
| 18 | ✅ experts initialisés bit-à-bit (max_ecart = 0 partout) | 12 copies, max_ecart 0.0 |
| 19 | ✅ nombre de copies d'experts = n_experts × n_seeds | 12 vs 4×3 |
| 20 | ✅ chaque expert apparié au MÊME seed que le run | 12 couples (seed run, seed expert) tous égaux |
| 21 | ✅ 4 méthodes distinctes en experts | ['B', 'D', 'Dp', 'G'] |
| 22 | ✅ source_block = head.fpn_convs.0 | head.fpn_convs.0 |
| 23 | ✅ gate aléatoire (aucun pré-appariement) | random |
| 24 | ✅ époques = 80 pour les 3 seeds | [80, 80, 80] |
| 25 | ✅ équirépartition part_max = top_k/n_experts | 2/4 = 0.5 |
| 26 | ✅ residual_scale initial = 0 (époque 0 bit-exacte au contrôle) | 0.0 |
| 27 | ✅ surcoût temporel MoE vs contrôle > 0 | +24.7 % |
| 28 | ✅ surcoût VRAM MoE vs contrôle > 0 | 35.2 vs 31.5 Go |
| 29 | ✅ γ croît strictement de l'époque 0 à la finale (3 seeds) | s42 0.00024→0.02026 · s123 0.00023→0.01912 · s456 0.00023→0.01815 |
| 30 | ✅ γ final strictement positif (le MoE sert) | min 0.01815 |
| 31 | ✅ primaire : Δ(moe − contrôle) positif | +0.4494 pt |
| 32 | ✅ primaire : IC95 exclut 0 (significatif au sens IC) | [+0.1086 ; +0.8195] |
| 33 | ✅ primaire : p < 0,05 | 0.0066 |
| 34 | ✅ primaire : le harnais annonce la paire pré-enregistrée moe/contrôle | controle_vs_moe_v3cs |
| 35 | ✅ 3 seeds par bras dans le harnais | 3 / 3 |
| 36 | ✅ primaire : moyenne des mIoU par seed == point bootstrap (moe_v3cs) | 81.617575 vs 81.617575 |
| 37 | ✅ primaire : moyenne des mIoU par seed == point bootstrap (controle) | 81.168215 vs 81.168215 |
| 38 | ✅ écart harness ↔ métiers P3.16 (2ᵉ forward) < 0,01 pt | moe 6.07e-04 pt, contrôle 2.51e-04 pt |
| 39 | ✅ verdict identique dans les deux forwards (significatif partout) | harness +0.4494 pt p=0.0066 · métiers +0.4502 pt |
| 40 | ✅ Δ primaire cohérent entre 5 artefacts (< 0,01 pt) | amplitude 8.58e-04 pt sur 5 sources |
| 41 | ✅ master P3.14 : 13 bras dont le contrôle | 13 bras, 12 hors contrôle |
| 42 | ✅ classement trié par Δ décroissant | 13 bras |
| 43 | ✅ rang du MoE cohérent avec la position du contrôle | rang 5/12, rang13 5/13, contrôle 10/13 |
| 44 | ✅ aucun bras significatif après Holm (les deux familles) | 5 bras à p brut < 0,05 ; 0 après Holm 12 ; 0 après Holm 15 |
| 45 | ✅ routing seed42 : 80 époques, epochs 0..79 sans trou | 80 entrées |
| 46 | ✅ routing seed42 : n_experts cohérent avec moe_kwargs | 4 |
| 47 | ✅ routing seed42 ep0 : somme des frac == top_k | 2.000000028 pour top_k = 2 |
| 48 | ✅ routing seed42 ep0 : part_max == max(frac) | 0.500560420 |
| 49 | ✅ routing seed42 ep0 : top_expert_share ≥ part_max (Jensen) | moyenne des maxima 0.667003 ≥ maximum des moyennes 0.500560 |
| 50 | ✅ routing seed42 ep79 : somme des frac == top_k | 2.000000030 pour top_k = 2 |
| 51 | ✅ routing seed42 ep79 : part_max == max(frac) | 0.503026235 |
| 52 | ✅ routing seed42 ep79 : top_expert_share ≥ part_max (Jensen) | moyenne des maxima 0.597549 ≥ maximum des moyennes 0.503026 |
| 53 | ✅ routing seed42 : entropies dans [0,1] (normalisées) | entropy_norm / entropy_token / entropy_token_clean |
| 54 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (entropy_norm) | 0.999998095 vs 0.999998095 |
| 55 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (entropy_token) | 0.999958298 vs 0.999958298 |
| 56 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (entropy_token_clean) | 0.999958298 vs 0.999958298 |
| 57 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (gamma_absmean) | 0.0202606407 vs 0.0202606407 |
| 58 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (part_max) | 0.503026235 vs 0.503026235 |
| 59 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (top_expert_share) | 0.59754914 vs 0.59754914 |
| 60 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (aux_loss_weighted) | 0.0020000299 vs 0.0020000299 |
| 61 | ✅ routing seed42 : jsonl ep79 == summary final_epoch_stats (noise_std) | 0 vs 0 |
| 62 | ✅ routing seed42 : bruit recuit sur 40 époques puis nul | noise_std 1.00 → 0.025 → 0.00 |
| 63 | ✅ routing seed123 : 80 époques, epochs 0..79 sans trou | 80 entrées |
| 64 | ✅ routing seed123 : n_experts cohérent avec moe_kwargs | 4 |
| 65 | ✅ routing seed123 ep0 : somme des frac == top_k | 2.000000027 pour top_k = 2 |
| 66 | ✅ routing seed123 ep0 : part_max == max(frac) | 0.501494440 |
| 67 | ✅ routing seed123 ep0 : top_expert_share ≥ part_max (Jensen) | moyenne des maxima 0.668983 ≥ maximum des moyennes 0.501494 |
| 68 | ✅ routing seed123 ep79 : somme des frac == top_k | 2.000000029 pour top_k = 2 |
| 69 | ✅ routing seed123 ep79 : part_max == max(frac) | 0.501868049 |
| 70 | ✅ routing seed123 ep79 : top_expert_share ≥ part_max (Jensen) | moyenne des maxima 0.595457 ≥ maximum des moyennes 0.501868 |
| 71 | ✅ routing seed123 : entropies dans [0,1] (normalisées) | entropy_norm / entropy_token / entropy_token_clean |
| 72 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (entropy_norm) | 0.997707056 vs 0.997707056 |
| 73 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (entropy_token) | 0.987453097 vs 0.987453097 |
| 74 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (entropy_token_clean) | 0.987453097 vs 0.987453097 |
| 75 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (gamma_absmean) | 0.0191158445 vs 0.0191158445 |
| 76 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (part_max) | 0.501868049 vs 0.501868049 |
| 77 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (top_expert_share) | 0.595456934 vs 0.595456934 |
| 78 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (aux_loss_weighted) | 0.00199941921 vs 0.00199941921 |
| 79 | ✅ routing seed123 : jsonl ep79 == summary final_epoch_stats (noise_std) | 0 vs 0 |
| 80 | ✅ routing seed123 : bruit recuit sur 40 époques puis nul | noise_std 1.00 → 0.025 → 0.00 |
| 81 | ✅ routing seed456 : 80 époques, epochs 0..79 sans trou | 80 entrées |
| 82 | ✅ routing seed456 : n_experts cohérent avec moe_kwargs | 4 |
| 83 | ✅ routing seed456 ep0 : somme des frac == top_k | 2.000000029 pour top_k = 2 |
| 84 | ✅ routing seed456 ep0 : part_max == max(frac) | 0.501494440 |
| 85 | ✅ routing seed456 ep0 : top_expert_share ≥ part_max (Jensen) | moyenne des maxima 0.659568 ≥ maximum des moyennes 0.501494 |
| 86 | ✅ routing seed456 ep79 : somme des frac == top_k | 2.000000029 pour top_k = 2 |
| 87 | ✅ routing seed456 ep79 : part_max == max(frac) | 0.501868049 |
| 88 | ✅ routing seed456 ep79 : top_expert_share ≥ part_max (Jensen) | moyenne des maxima 0.596951 ≥ maximum des moyennes 0.501868 |
| 89 | ✅ routing seed456 : entropies dans [0,1] (normalisées) | entropy_norm / entropy_token / entropy_token_clean |
| 90 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (entropy_norm) | 0.999966414 vs 0.999966414 |
| 91 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (entropy_token) | 0.999821044 vs 0.999821044 |
| 92 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (entropy_token_clean) | 0.999821044 vs 0.999821044 |
| 93 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (gamma_absmean) | 0.0181530798 vs 0.0181530798 |
| 94 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (part_max) | 0.501868049 vs 0.501868049 |
| 95 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (top_expert_share) | 0.596951367 vs 0.596951367 |
| 96 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (aux_loss_weighted) | 0.00199997181 vs 0.00199997181 |
| 97 | ✅ routing seed456 : jsonl ep79 == summary final_epoch_stats (noise_std) | 0 vs 0 |
| 98 | ✅ routing seed456 : bruit recuit sur 40 époques puis nul | noise_std 1.00 → 0.025 → 0.00 |
| 99 | ✅ aucun expert mort à l'époque finale (3 seeds) | {42: 0, 123: 0, 456: 0} |
| 100 | ✅ part_max final dans un voisinage de l'équirépartition 0.5000 | s42 0.5030 · s123 0.5019 · s456 0.5019 |
| 101 | ✅ frac des 4 experts ∈ [0.4979 ; 0.5030] autour de 0.5000 | étendue 0.0051 pt |
| 102 | ✅ entropy_norm ≥ 0,98 partout (porte quasi uniforme) | min 0.99771 |
| 103 | ✅ seed42 : le bruit est actif sur exactement noise_anneal époques | 40 époques à noise_std > 0 (0..39) |
| 104 | ✅ seed42 : après recuit, token == clean par construction (tautologie, non preuve) | 40 époques à bruit nul |
| 105 | ✅ seed123 : le bruit est actif sur exactement noise_anneal époques | 40 époques à noise_std > 0 (0..39) |
| 106 | ✅ seed123 : après recuit, token == clean par construction (tautologie, non preuve) | 40 époques à bruit nul |
| 107 | ✅ seed456 : le bruit est actif sur exactement noise_anneal époques | 40 époques à noise_std > 0 (0..39) |
| 108 | ✅ seed456 : après recuit, token == clean par construction (tautologie, non preuve) | 40 époques à bruit nul |
| 109 | ✅ phase à bruit ACTIF : l'écart token↔clean reste négligeable devant la platitude | écart max 1.45e-03 sur 40 époques × 3 seeds |
| 110 | ✅ phase à bruit ACTIF : la distribution SANS bruit est déjà plate (clean ≥ 0,99) | min 0.99821 |
| 111 | ✅ phase à bruit ACTIF : part_max reste collé à l'équirépartition | bornes 0.50011–0.50217 pour 0.5000 |
| 112 | ✅ γ final ≫ γ époque 0 (le MoE sert, mais comme moyenne) | s42 ×83 · s123 ×84 · s456 ×77 |
| 113 | ✅ métiers P3.16 : 20 métriques | 20 tables |
| 114 | ✅ métiers P3.16 : 15 paires | 15 paires |
| 115 | ✅ métiers P3.16 : 12 bras (couverture complète, A exclu) | 12 bras : ['controle', 'B', 'C', 'Cp', 'D', 'Dp', 'G', 'fused_CvetoB', 'fused_DvetoB', 'fused_CpvetoB', 'fused_DpvetoB', 'moe_v3cs'] |
| 116 | ✅ le MoE et le contrôle sont dans les 12 bras | moe_v3cs, controle |
| 117 | ✅ bras de contexte présents dans la table métier | ['D', 'Dp', 'G', 'B'] |
| 118 | ✅ ordre d'affichage == ensemble des métriques de l'artefact | 0 différence(s) |
| 119 | ✅ unité de « mIoU » = fraction dans [0,1] -> points | min 0.808888, max 0.816834 |
| 120 | ✅ unité de « boundary_f1_3px » = fraction dans [0,1] -> points | min 0.763734, max 0.773043 |
| 121 | ✅ unité de « boundary_f1_3px_pieds » = fraction dans [0,1] -> points | min 0.666386, max 0.677327 |
| 122 | ✅ unité de « rappel_strict_instances » = fraction dans [0,1] -> points | min 0.728367, max 0.784363 |
| 123 | ✅ unité de « precision_ped_pixels » = fraction dans [0,1] -> points | min 0.478331, max 0.700283 |
| 124 | ✅ unité de « instances_toutes_rappel » = fraction dans [0,1] -> points | min 0.765520, max 0.822974 |
| 125 | ✅ unité de « instances_toutes_det05 » = fraction dans [0,1] -> points | min 0.851170, max 0.884189 |
| 126 | ✅ unité de « instances_individuelles_rappel » = fraction dans [0,1] -> points | min 0.767350, max 0.824072 |
| 127 | ✅ unité de « instances_individuelles_det05 » = fraction dans [0,1] -> points | min 0.851735, max 0.884265 |
| 128 | ✅ unité de « instances_foule_rappel » = fraction dans [0,1] -> points | min 0.699721, max 0.764346 |
| 129 | ✅ unité de « instances_foule_det05 » = fraction dans [0,1] -> points | min 0.820106, max 0.867725 |
| 130 | ✅ unité de « instances_taille_T1_rappel » = fraction dans [0,1] -> points | min 0.609137, max 0.678103 |
| 131 | ✅ unité de « instances_taille_T1_det05 » = fraction dans [0,1] -> points | min 0.682616, max 0.738569 |
| 132 | ✅ unité de « instances_taille_T2_rappel » = fraction dans [0,1] -> points | min 0.808776, max 0.862793 |
| 133 | ✅ unité de « instances_taille_T2_det05 » = fraction dans [0,1] -> points | min 0.910834, max 0.937787 |
| 134 | ✅ unité de « instances_taille_T3_rappel » = fraction dans [0,1] -> points | min 0.916760, max 0.943435 |
| 135 | ✅ unité de « instances_taille_T3_det05 » = fraction dans [0,1] -> points | min 0.987827, max 0.991156 |
| 136 | ✅ unité de « IoU_person » = fraction dans [0,1] -> points | min 0.848751, max 0.852193 |
| 137 | ✅ unité de « IoU_rider » = fraction dans [0,1] -> points | min 0.675814, max 0.692278 |
| 138 | ✅ unité de « fragments » = composantes/image (valeurs > 1) | min 488.0, max 519.3 |
| 139 | ✅ comptes de significativité métier cohérents | 2 Holm-significatifs, 5 IC excluant 0 (Holm ⊆ IC) |
| 140 | ✅ fragments : artefact à 2 bras (contrôle + MoE seulement) | 2 bras : ['controle', 'moe_v3cs'] |
| 141 | ✅ fragments : points de la table == fragments_point de l'artefact | 519.264 → 488.047 |
| 142 | ✅ fragments : baisse mesurée | -31.217 composantes/image (0.9399×) |
| 143 | ✅ compte des survivantes au Holm == n_holm calculé plus haut | 2 vs 2 |
| 144 | ✅ les fragments sont parmi les survivantes au Holm | ['instances_taille_T3_rappel', 'fragments'] |
| 145 | ✅ il y a PLUS d'une survivante (ne pas écrire « la seule ») | instances_taille_T3_rappel (Holm 0) · fragments (Holm 0) |
| 146 | ✅ unité de « instances_taille_T1_rappel » = fraction dans [0,1] -> points | min 0.609137, max 0.678103 |
| 147 | ✅ unité de « rappel_strict_instances » = fraction dans [0,1] -> points | min 0.728367, max 0.784363 |
| 148 | ✅ unité de « instances_toutes_rappel » = fraction dans [0,1] -> points | min 0.765520, max 0.822974 |
| 149 | ✅ unité de « boundary_f1_3px » = fraction dans [0,1] -> points | min 0.763734, max 0.773043 |
| 150 | ✅ unité de « precision_ped_pixels » = fraction dans [0,1] -> points | min 0.478331, max 0.700283 |
| 151 | ✅ unité de « instances_foule_rappel » = fraction dans [0,1] -> points | min 0.699721, max 0.764346 |
| 152 | ✅ unité de « IoU_person » = fraction dans [0,1] -> points | min 0.848751, max 0.852193 |
| 153 | ✅ unité de « instances_taille_T1_rappel » = fraction dans [0,1] -> points | min 0.609137, max 0.678103 |
| 154 | ✅ le MoE capte une partie du gain T1 de D/Dp, sans l'ampleur de la perte de G | MoE +0.80 · D +2.82 · Dp +2.02 · G -4.08 |
| 155 | ✅ unité de « rappel_strict_instances » = fraction dans [0,1] -> points | min 0.728367, max 0.784363 |
| 156 | ✅ rappel piéton strict : le MoE ne hérite PAS de la perte de G | MoE +0.47 vs G -3.42 |
| 157 | ✅ traffic light : le MoE n'hérite pas de la perte des bras boundary/distmap | MoE -0.08 · D -2.60 · Dp -2.34 |
| 158 | ✅ attribution P3.16 : référence = contrôle | controle |
| 159 | ✅ attribution P3.16 : 12 bras | 12 bras |
| 160 | ✅ attribution P3.16 : 19 classes Cityscapes | 19 |
| 161 | ✅ classes toutes distinctes | 19 |
| 162 | ✅ mIoU == moyenne des 19 IoU par classe (moe_v3cs) | 81.617575 vs 81.617575 |
| 163 | ✅ mIoU == moyenne des 19 IoU par classe (D) | 81.685781 vs 81.685781 |
| 164 | ✅ mIoU == moyenne des 19 IoU par classe (G) | 81.263264 vs 81.263264 |
| 165 | ✅ IoU de référence (contrôle) : moyenne des 19 == point du harnais | 81.168215 vs 81.168215 |
| 166 | ✅ Δ mIoU de l'attribution == somme des Δ par classe / 19 | +0.449360 vs +0.449360 |
| 167 | ✅ Holm 19 classes recomputé == p_holm_arm stocké | écart max 0 |
| 168 | ✅ gains + pertes + classes nulles == 19 | 11 gains, 8 pertes |
| 169 | ✅ critère AMPLITUDE : les 4 plus fortes hausses sont truck, wall, train, fence | ['truck', 'wall', 'train', 'fence'] |
| 170 | ✅ critère IC-excluant-0 : hausses = truck, train, sidewalk, road | ['truck', 'train', 'sidewalk', 'road'] |
| 171 | ✅ critère IC-excluant-0 : baisses = person, bicycle | ['person', 'bicycle'] |
| 172 | ✅ critère HOLM(19) : aucune HAUSSE ne survit | 0 hausse(s) Holm-significative(s) : [] |
| 173 | ✅ critère HOLM(19) : la seule classe survivante est une BAISSE | ['bicycle'] |
| 174 | ✅ les deux critères ne sont PAS réductibles l'un à l'autre | amplitude ['truck', 'wall', 'train', 'fence'] vs IC ['truck', 'train', 'sidewalk', 'road'] |
| 175 | ✅ truck : le MoE retrouve une part majoritaire du MEILLEUR des 4 experts | 81.7 % (+4.24 vs meilleur expert +5.19 = Dp) |
| 176 | ✅ train : le meilleur expert est bien B, pas D ni Dp (le dénominateur D/Dp mentait) | B +1.50 · D +0.40 · Dp +0.66 · G -1.04 |
| 177 | ✅ coût piéton faible et non Holm-significatif | -0.244 pt, Holm 0.1326, IC [-0.433 ; -0.062] |
| 178 | ✅ polyvalence : 36 endpoints | 36 |
| 179 | ✅ polyvalence : 13 bras | 13 |
| 180 | ✅ polyvalence : 12 bras à couverture complète (A en partielle) | 12 ; annexe = ['A'] |
| 181 | ✅ familles : somme des effectifs == 36 | 36 affectations |
| 182 | ✅ 19 endpoints « IoU par classe » | 19 |
| 183 | ✅ chaque bras complet a ses 36 deltas | 12 bras × 36 endpoints |
| 184 | ✅ sd inter-bras présent pour les 36 endpoints (B) | 36 |
| 185 | ✅ dommage max recalculé == stocké (B) | -4.0638 (precision_ped_pixels) vs ['B', -4.06, 'precision_ped_pixels'] |
| 186 | ✅ dommage max recalculé == stocké (C) | -3.2060 (precision_ped_pixels) vs ['C', -3.21, 'precision_ped_pixels'] |
| 187 | ✅ dommage max recalculé == stocké (Cp) | -6.0509 (precision_ped_pixels) vs ['Cp', -6.05, 'precision_ped_pixels'] |
| 188 | ✅ dommage max recalculé == stocké (D) | -22.1952 (precision_ped_pixels) vs ['D', -22.2, 'precision_ped_pixels'] |
| 189 | ✅ dommage max recalculé == stocké (Dp) | -15.0202 (precision_ped_pixels) vs ['Dp', -15.02, 'precision_ped_pixels'] |
| 190 | ✅ dommage max recalculé == stocké (G) | -6.7626 (precision_ped_pixels) vs ['G', -6.76, 'precision_ped_pixels'] |
| 191 | ✅ dommage max recalculé == stocké (fused_CvetoB) | -1.8952 (IoU_bus) vs ['fused_CvetoB', -1.9, 'IoU_bus'] |
| 192 | ✅ dommage max recalculé == stocké (fused_CpvetoB) | -2.7797 (IoU_bus) vs ['fused_CpvetoB', -2.78, 'IoU_bus'] |
| 193 | ✅ dommage max recalculé == stocké (fused_DvetoB) | -8.7018 (precision_ped_pixels) vs ['fused_DvetoB', -8.7, 'precision_ped_pixels'] |
| 194 | ✅ dommage max recalculé == stocké (fused_DpvetoB) | -4.7678 (precision_ped_pixels) vs ['fused_DpvetoB', -4.77, 'precision_ped_pixels'] |
| 195 | ✅ dommage max recalculé == stocké (moe_v3cs) | -0.5291 (instances_foule_det05) vs ['moe_v3cs', -0.53, 'instances_foule_det05'] |
| 196 | ✅ classement par dommage recalculé == classement stocké | ['moe_v3cs', 'fused_CvetoB', 'fused_CpvetoB']… == ['moe_v3cs', 'fused_CvetoB', 'fused_CpvetoB']… |
| 197 | ✅ le MoE est premier en dommage maximal | moe_v3cs |
| 198 | ✅ marge du MoE sur le suivant == stockée | 1.366 vs 1.366 |
| 199 | ✅ marge positive (le MoE est bien le moins abîmé) | +1.366 pt |
| 200 | ✅ couverture de rang_entier : bras complets + bras partiels == n_bras | 13 bras dont 12 à couverture complète sur 36 endpoints ; partiels : {'A': 20} |
| 201 | ✅ le MoE est à couverture complète sur les 36 endpoints | 36/36 |
| 202 | ✅ n_rang1 recalculé == profil stocké (B) | 2 vs 2 |
| 203 | ✅ n_rang1 recalculé == profil stocké (C) | 0 vs 0 |
| 204 | ✅ n_rang1 recalculé == profil stocké (Cp) | 0 vs 0 |
| 205 | ✅ n_rang1 recalculé == profil stocké (D) | 16 vs 16 |
| 206 | ✅ n_rang1 recalculé == profil stocké (Dp) | 3 vs 3 |
| 207 | ✅ n_rang1 recalculé == profil stocké (G) | 7 vs 7 |
| 208 | ✅ n_rang1 recalculé == profil stocké (fused_CvetoB) | 0 vs 0 |
| 209 | ✅ n_rang1 recalculé == profil stocké (fused_CpvetoB) | 0 vs 0 |
| 210 | ✅ n_rang1 recalculé == profil stocké (fused_DvetoB) | 3 vs 3 |
| 211 | ✅ n_rang1 recalculé == profil stocké (fused_DpvetoB) | 4 vs 4 |
| 212 | ✅ n_rang1 recalculé == profil stocké (moe_v3cs) | 0 vs 0 |
| 213 | ✅ le MoE n'est premier sur AUCUN endpoint | 0 |
| 214 | ✅ D est premier sur le plus grand nombre d'endpoints | D 16 |
| 215 | ✅ le MoE n'est jamais dernier | 0 |
| 216 | ✅ le dénominateur des rangs varie selon les endpoints (A en couverture partielle) | tailles observées [12, 13] |
| 217 | ✅ pire rang du MoE recalculé == profil | 11 vs 11 |
| 218 | ✅ le pire rang du MoE n'est PAS une dernière place | 11/12 sur `instances_foule_rappel` (dernière place = 12) |
| 219 | ✅ pire rang + son dénominateur cohérents avec le profil (MoE) | 11/12 sur `instances_foule_rappel` — jamais dernier |
| 220 | ✅ le dénominateur du pire rang n'est PAS toujours 13 (A en couverture partielle) | dénominateurs observés : [12, 13] |
| 221 | ✅ moyenne des 19 Δ par classe ≈ ΔmIoU (identité exacte within-artefact, la polyvalence mélange 2 forwards) | +0.449360 vs +0.450218 — écart 8.58e-04 pt = l'écart inter-forwards mesuré en T2 |
| 222 | ✅ l'écart d'identité de la polyvalence est du même ordre que son écart inter-forwards | 8.58e-04 pt vs sanity déclarée 4.13e-03 pt |
| 223 | ✅ percentile moyen du MoE recalculé == profil | 0.604345 vs 0.604345 |
| 224 | ✅ T0 détail recalculé == stocké (D) | dommage -22.195 vs -22.2, end>1pt 3 vs 3, classes 4 vs 4 |
| 225 | ✅ T0 détail recalculé == stocké (Dp) | dommage -15.020 vs -15.02, end>1pt 3 vs 3, classes 3 vs 3 |
| 226 | ✅ T0 détail recalculé == stocké (fused_DvetoB) | dommage -8.702 vs -8.7, end>1pt 3 vs 3, classes 3 vs 3 |
| 227 | ✅ T0 détail recalculé == stocké (fused_DpvetoB) | dommage -4.768 vs -4.77, end>1pt 2 vs 2, classes 2 vs 2 |
| 228 | ✅ T0 détail recalculé == stocké (moe_v3cs) | dommage -0.529 vs -0.53, end>1pt 0 vs 0, classes 0 vs 0 |
| 229 | ✅ T0 recalculé : le MoE est le SEUL bras à cocher les trois critères | recalculé ['moe_v3cs'], stocké ['moe_v3cs'] |
| 230 | ✅ bras à mIoU significative recalculés == stockés | ['D', 'Dp', 'fused_DpvetoB', 'fused_DvetoB', 'moe_v3cs'] |
| 231 | ✅ prix par point de mIoU recalculé == stocké (D) | -43.06 vs -43.1 |
| 232 | ✅ prix par point de mIoU recalculé == stocké (Dp) | -31.86 vs -31.9 |
| 233 | ✅ prix par point de mIoU recalculé == stocké (fused_DvetoB) | -17.93 vs -17.9 |
| 234 | ✅ prix par point de mIoU recalculé == stocké (fused_DpvetoB) | -9.90 vs -9.9 |
| 235 | ✅ prix par point de mIoU recalculé == stocké (moe_v3cs) | -1.18 vs -1.2 |
| 236 | ✅ le MoE paie le prix le plus faible par point de mIoU | moe_v3cs -1.2 · fused_DpvetoB -9.9 · fused_DvetoB -17.9 · Dp -31.9 · D -43.1 |
| 237 | ✅ z du dommage maximal recalculé == stocké (MoE) | -0.354 vs -0.354 |
| 238 | ✅ z du MoE ≫ meilleur z des spécialistes en dommage | MoE -0.35 vs D -3.44 |
| 239 | ✅ pic maximal du MoE recalculé == profil | +4.241 vs +4.241 |
| 240 | ✅ sanity P3.17 : mIoU consolidé == P3.14 à < 0,01 pt | 4.13e-03 pt |
| 241 | ✅ sanity P3.17 : IoU par classe recomputées depuis npz à < 0,1 pt | 6.33e-02 pt |
| 242 | ✅ verdicts stockés T0-T3 vrais, T4 partiel (2 seeds sur 3) | {"T0": true, "T1": true, "T2": true, "T3": true, "T4": false, "T4_moe_premier_sur_n_seeds": 2} |
| 243 | ✅ T4 recalculé depuis par_seed : le MoE est 1er en dommage sur 2 seeds sur 3 | 2/3 |
| 244 | ✅ le MoE n'est PAS premier en niveau moyen (ne pas survendre) | 1er = D (77.3 %), MoE 5ᵉ (60.4 %) |
| 245 | ✅ Δ > 0 et Δ < 0 partitionnent les 36 endpoints (aucun Δ nul) | 22 positifs + 14 négatifs = 36 |
| 246 | ✅ endpoints au-dessus du contrôle == 22 (valeur citée par le cadrage) | 22/36 |
| 247 | ✅ les deux compteurs ne sont PAS égaux (sinon l'ambiguïté serait sans objet) | Δ > 0 : 22 · moitié haute du classement : 28 |
| 248 | ✅ par_seed 42 : le MoE est classé et son dommage est du bon ordre | rang 1, dommage seed -2.98 (moyenne 3 seeds -0.53) |
| 249 | ✅ par_seed 123 : le MoE est classé et son dommage est du bon ordre | rang 3, dommage seed -3.17 (moyenne 3 seeds -0.53) |
| 250 | ✅ par_seed 456 : le MoE est classé et son dommage est du bon ordre | rang 1, dommage seed -2.76 (moyenne 3 seeds -0.53) |
| 251 | ✅ une seule famille négative pour le MoE | 1 : contours |
| 252 | ✅ BRATS paper3 : champ zenodo_doi lu dans le dépôt voisin | 10.5281/zenodo.22903668 |
| 253 | ✅ BRATS paper3 : champ zenodo_concept_doi lu dans le dépôt voisin | 10.5281/zenodo.22776410 |
| 254 | ✅ BRATS paper3 : champ zenodo_record_id lu dans le dépôt voisin | 22903668 |
| 255 | ✅ BRATS paper3 : champ zenodo_published lu dans le dépôt voisin | 2026-09-22 |
| 256 | ✅ BRATS paper3 publié | published |
| 257 | ✅ le DOI BRATS diffère de celui du paper4 Cityscapes (pas de confusion) | 10.5281/zenodo.22903668 |
