# T8 — Antériorité : la réplication BRATS, et ce qui diffère ici

Ce papier est une **réplication sur un second dataset**, pas une republication. L'antériorité est le paper 3 du programme BraTS, *The Gate Does Not Choose*, publié sous DOI `10.5281/zenodo.22903668` (concept `10.5281/zenodo.22776410`, record 22903668, 2026-09-22). Aucun de ses chiffres n'est repris ici ; il est cité comme antériorité et comme point de comparaison qualitative.

| Dimension | BraTS (antériorité) | Cityscapes (ce papier) |
|---|---|---|
| Dataset | BraTS-2023, gliome, 3D, 1 196 cas | Cityscapes, conduite urbaine, 2D, holdout `first:500` de val |
| Backbone | MedNeXt | ConvNeXt-V2-Base (ImageNet-22K) + UPerNet, 1024×2048 |
| Point d'accroche | `dec_block_0` | `head.fpn_bottleneck` (dernier élément du Sequential) |
| Métriques | Dice, HD95, lésion-wise | mIoU dataset-level officiel, Boundary F1, rappel d'instances par strate, fragments |
| Garde-fou résiduel γ | absent (recette brute suffisante) | **nécessaire** : sans γ, la recette brute coûte −28,1 pt de mIoU à l'époque 0 (mesuré P3.08) |
| Normalisation | GroupNorm, blocs MedNeXt identiques partout | BatchNorm + latéraux UPerNet : les experts voient au point d'accroche une distribution qu'ils n'ont jamais traitée |
| Routage | top-2 parmi 4, grille 3×3 | identique (recette reprise à l'identique) |
| Résultat commun | la porte ne choisit pas ; le gain vient de l'initialisation par experts | la porte ne choisit pas non plus (T4 : `entropy_norm` minimal 0.99771 sur 3 seeds, `part_max` 0.0030 au plus loin de l'équirépartition 0.5000, 0 expert mort) |
| Critère primaire | — | Δ +0.45 pt de mIoU vs contrôle apparié, p = 0.0066 |

La différence qui compte est le **garde-fou γ** : il n'était pas nécessaire côté BraTS et il est indispensable ici. C'est une propriété mesurée de l'architecture d'accueil (distribution au point d'accroche, absence de renormalisation avant le classifier), pas un réglage choisi après coup : avec γ initialisé à 0.0, l'époque 0 est bit-exacte au contrôle, et |γ| devient un diagnostic direct de la thèse.

Provenance : `/home/dev/mnt/tour/BRATS/papers/paper3/paper.yaml` (lu, pas recopié) pour les identifiants de l'antériorité ; artefacts Cityscapes pour tout le reste.
