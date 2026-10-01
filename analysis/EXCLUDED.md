# Artefacts volontairement NON embarqués

Le manuscrit le déclare (§7, limites) : les artefacts lourds ne sont pas distribués.
Rien n'est tu — chaque exclusion est nommée, avec sa raison et son chemin de
régénération. Tout ce qui est nécessaire pour re-dériver les statistiques publiées
(matrices de confusion, tables de bootstrap, routage par époque, métriques par image
du MoE et de son contrôle) EST embarqué dans `analysis/`.

| Artefact | Nature | Pourquoi il n'est pas ici | Comment le régénérer |
|---|---|---|---|
| `checkpoints/moe_v3cs_Binit_seed{42,123,456}/epoch_080.pth` | poids d'entraînement du mélange (1,59 Go chacun, 35,2 Go de VRAM, ≈ 17 h/seed) | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `checkpoints/control_nomoe_Binit_seed{42,123,456}/epoch_080.pth` | poids du contrôle apparié (1,47 Go chacun) | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `checkpoints/{B,D,Dp,G}_seed*/epoch_160.pth (les 4 experts)` | poids des spécialistes sources (4 × 160 époques chacun) — fournis par les bras compagnons | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers/gt_trainids.npy` | ground-truth train ids du holdout (1,0 Go) | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers/gt_instances_ped.npz` | instances piétonnes officielles (19 Mo) | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers/pred_*_seed*.npy` | cartes de prédiction plein cadre 500×1024×2048 uint8 (≥ 1 Go par carte) | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers_experts/metriques_*_seed*.npz (10 autres bras)` | métriques métier par image des 10 bras hors MoE/contrôle — le MoE et son contrôle sont embarqués | volume (≥ 1 Go par élément) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |

## Ce qui EST embarqué et suffit à re-dériver chaque nombre publié

- `analysis/harness/table_controle_v3cs_P310.json` — le critère primaire pré-enregistré
  (Δ(MoE−contrôle), IC95, p bilatéral, bootstrap apparié par image).
- `analysis/harness/*.cm.npy` — matrices de confusion par seed (MoE et contrôle), d'où la
  mIoU officielle est re-calculable.
- `analysis/routing/routing_moe_v3cs_Binit_seed*.jsonl` — le routage des 80 époques × 3
  seeds (frac, entropies, part_max, γ) d'où vient tout le diagnostic « la porte ne
  choisit pas » (§5.2).
- `analysis/p314/master_table.json` — la table maîtresse 13 bras (position du MoE, les
  deux familles de Holm).
- `analysis/p316/attribution_perclass_vs_controle.json` — attribution IoU par classe.
- `analysis/p317_polyvalence/table_polyvalence.json` — le classement de polyvalence
  36 endpoints × 13 bras (dommage maximal, percentile, rangs).
- `analysis/metiers_experts/` — métriques métier par image du MoE et de son contrôle
  (+ la table des 12 bras).
- `tables/paper3_tables.json` — la table de consolidation T1-T8 du papier (257 checks).

Le test de cohérence `scripts/p3_check_numbers.py` (111 checks) relie
chaque nombre du manuscrit à ces artefacts (comptes RECALCULÉS depuis le détail) et
échoue à la moindre dérive.

## Licence

Les manuscrits (`paper.md`, `paper_fr.md`, `paper.pdf`, `paper_fr.pdf`), les tableaux
et les figures sont sous **CC-BY-4.0** (c'est la licence du dépôt Zenodo). Le code
(`src/`, `scripts/`, `tests/`, `configs/`) est sous **MIT** (`LICENSE`). Les données
d'entraînement appartiennent au jeu Cityscapes et restent soumises à sa propre licence.
