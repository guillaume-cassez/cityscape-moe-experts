# T1 — Objet technique, recette et coût (MoE-V3-CS)

| Poste | Valeur mesurée | Source |
|---|---|---|
| Backbone | ConvNeXt-V2-Base (ImageNet-22K) + UPerNet, pleine résolution 1024×2048, 19 classes Cityscapes, BF16, channels_last | `provenance backbone = méthode B, epoch 160` |
| Point d'accroche | `head.fpn_bottleneck` — la couche MoE est le DERNIER élément du Sequential (indices conservés 0=conv, 1=BN, 2=ReLU, 3=MoE), donc `fused ← fused + experts(fused)` | `src/moe/moe_model.py:V3CS_ATTACH_SEQ + attach_patch_moe` |
| Experts | 4, initialisés depuis `head.fpn_convs.0` des 4 spécialistes du MÊME seed (B, D, Dp, G), entraînés 160 époques | `provenance experts, max_ecart = 0.0 sur 12 copies` |
| Routage | top-2 parmi 4, grille de patches 3×3, gate convolutif kernel 3, porte random | `provenance moe_kwargs` |
| Équilibrage | loss switch poids 0.001 (aux pondéré 0.00200 0.00200 0.00200) | `moe_kwargs + logs` |
| Bruit de Shazeer | std 1.0 × std(logits), recuit sur 40 époques | `moe_kwargs + routing jsonl (noise_std 1,00 → 0,00)` |
| Garde-fou résiduel γ | échelle résiduelle apprenable par canal, init 0.0 → |γ| moyen 0.02026 0.01912 0.01815 | `moe_kwargs + final_epoch_stats` |
| Contrôle apparié | `control_nomoe_Binit` : même initialisation B, même recette, 80 époques, SANS couche MoE, 3 seeds | `results/p3_queue/control_nomoe_Binit_seed*.log` |
| Recette | 80 époques, batch 2 × accumulation 4 (effectif 8), AdamW lr 6e-5, wd 0,01, betas (0,9 ; 0,999), poly power 1,0, warmup 1 époque, BF16, hflip p=0,5 | `scripts/train_moe_v3cs.py` |
| Coût MoE | 762 s/époque en moyenne, VRAM 35.2 Go, 17.0 h/seed (elapsed_s moyen 61,210 s) | `logs 3 seeds × 80 époques + summaries` |
| Coût contrôle | 611 s/époque, VRAM 31.5 Go | `logs 3 seeds × 80 époques` |
| Surcoût du MoE | +24.7 % de temps/époque, +3.7 Go de VRAM | `calculé depuis les logs` |
| Budget complet | les 4 experts doivent exister (4 × 160 époques) + 80 époques de MoE et 80 de contrôle | `provenance experts (epoch 160)` |

Équirépartition exacte de `part_max` pour top-2 parmi 4 experts : **0.5000** (calculé `top_k/n_experts`, jamais recopié). C'est la valeur de référence contre laquelle T4 lit le routage.

Provenance vérifiée sur les 12 copies d'experts (4 experts × 3 seeds) : `max_ecart` maximal = **0.0**, et chaque expert est apparié au même seed que son run — d'où l'appariement strict par seed.
