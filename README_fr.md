# La porte ne choisit toujours pas

**Un mélange d'experts initialisé depuis des spécialistes de loss bat son contrôle apparié et
reste le seul bras polyvalent sous les métriques Cityscapes pleine résolution.**

**Guillaume Cassez** · **Stanislas Larnier** — recherche indépendante · ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) (G.C.) · [HAL stanislas-larnier](https://cv.hal.science/stanislas-larnier) (S.L.) · [guillaume-cassez.fr](https://guillaume-cassez.fr/voiture-autonome/)

Preprint, CC-BY-4.0. À lire : [`paper_fr.pdf`](paper_fr.pdf) (FR, 16 p.) ou
[`paper.pdf`](paper.pdf) (EN, 16 p.).

---

## De quoi il s'agit

Évaluation contrôlée et pré-enregistrée de **MoE-V3-CS**, un **mélange d'experts** par patch
inséré dans un segmenteur Cityscapes pleine résolution (ConvNeXt-V2-Base + UPerNet, 1024×2048,
19 classes, BF16). Ses quatre experts sont **initialisés bit à bit depuis le bloc
`head.fpn_convs.0` de quatre spécialistes de loss entraînés indépendamment, du même seed** — B
(CE+Dice), D (CE+Kervadec EDT), Dp (CE+SDT carte de distance), G (CE+Dice+0,5·blob) — puis une
porte top-2 sur des patches 3×3 est entraînée 80 époques depuis une échelle résiduelle γ
initialisée à zéro, de sorte que l'époque 0 est bit-exacte au contrôle. C'est une **réplication
sur un second dataset** du résultat BRATS *The Gate Does Not Choose*
([DOI 10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668)) — aucun chiffre BRATS
n'est repris.

## Le primaire est positif au seuil brut, et la porte ne choisit pas

| | valeur |
|---|---|
| **Critère primaire** (pré-enregistré) | mIoU officielle au niveau dataset, MoE vs contrôle apparié en recette |
| Δ(MoE − contrôle) | **+0.449 pt** |
| IC95 | [+0.109 ; +0.820] |
| p bilatéral, bootstrap apparié par image | **0.0066** |
| bootstrap | B = 10,000 réplicats, seed 20260618, holdout first:500 |
| mIoU (MoE / contrôle) | 81.618 / 81.168 |
| **trois familles de Holm** | 0.0066 (1 paire) · 0.0726 (12 paires) · 0.0924 (15 paires) |

Positif au seuil brut et dans la famille pré-enregistrée à une paire, il **ne survit pas à la
famille exploratoire** (0.0924 sur 15 paires) — où **aucun bras du plateau
ne survit non plus** (meilleur 0.0750). Les trois familles sont rapportées
ensemble, aucune n'est choisie parce qu'elle arrange. Robustesse : un second forward GPU
indépendant donne Δ 0.450218 pt, à 8.58e-04 pt d'écart — même verdict
(non-déterminisme cuDNN/BF16, déclaré).

**La porte ne choisit pas.** À l'époque finale le routage est plat — entropie normalisée minimale
0.99771, `part_max` 0.5030 / 0.5019 contre 0.5000 pour
l'équirépartition exacte, **zéro expert mort** — pourtant γ croît de 0.00024 à 0.02026
(×83). Le mélange sert comme **moyenne d'experts**, pas comme sélection.
Et ce n'est **pas** un artefact du bruit de Shazeer recuit : sur les 40 époques à bruit actif la
distribution sans bruit est déjà plate (≥ 0.99821)
et l'écart d'entropie token↔clean reste ≤ 1.5e-03.

**Le seul polyvalent, premier sur rien.** Sur le critère de polyvalence pré-enregistré du programme
(36 endpoints × 13 bras), MoE-V3-CS a le **dommage maximal le plus faible** du plateau
(-0.53 pt, marge de 1.37 pt sur le
suivant, −22,20 pt pour D) et le plateau le moins cher à -1.2 pt de dommage
par point de mIoU contre -43.1 pour D. Pourtant il n'est **premier sur aucun des 36
endpoints** (D en gagne 16) et se classe 5ᵉ en percentile moyen
(60.4) ; son pire rang est 11/12 sur
`instances_foule_rappel`. *Bon partout n'est pas meilleur partout.* La stabilité seed par seed (T4) est
**partielle** (1ᵉʳ en dommage sur 2 seeds sur 3).

## Résumé

> We report a pre-registered, controlled evaluation of **MoE-V3-CS**, a patch-wise
> **mixture-of-experts** inserted into a full-resolution Cityscapes segmenter (ConvNeXt-V2-Base +
> UPerNet, 1024×2048, 19 classes, BF16). Its four experts are **initialised bit-for-bit from the
> `head.fpn_convs.0` block of four independently trained loss specialists of the same seed** — B
> (CE+Dice), D (CE+Kervadec EDT), Dp (CE+SDT distance map), G (CE+Dice+0.5·blob) — and a top-2
> gate over 3×3 patches is then trained for 80 epochs from a residual scale γ initialised to zero,
> so that epoch 0 is bit-exact to the control. The **pre-registered primary endpoint** is the
> dataset-level official mIoU (cityscapesScripts, 19 classes) on the shared 500-image val holdout,
> MoE-V3-CS against its **recipe-matched control** (same B initialisation, same recipe, 80 epochs,
> no MoE layer), paired image-bootstrap B = 10 000, three seeds averaged within each replicate.
> The primary endpoint is **positive at the raw threshold and inside the pre-registered
> single-pair family**: Δ = **+0.449 pt** (MoE 81.618 vs control 81.168), 95 % CI [+0.109 ;
> +0.820], two-sided p = **0.0066**, Holm = 0.0066 over the 1-pair family. It **does not survive
> the exploratory multiplicity families**: Holm = 0.0726 over 12 pairs and Holm = 0.0924 over 15
> pairs — and **no arm of the plateau survives either** (best 0.0750). All three families are
> reported together, none is chosen because it flatters. The central result is a **routing
> diagnostic**: at the final epoch the gate distribution is flat (minimal normalised entropy
> 0.99771 over the three seeds, `part_max` 0.5030 / 0.5019 / 0.5019 against 0.5000 for exact
> equidistribution, every expert between 49.79 % and 50.30 % of the patches, **zero dead
> experts**), yet γ grows from 0.00024 to ≈ 0.020 (×77 to ×84) — the mixture **helps**, but as an
> **average of experts**, not as a selection. The gain does not come from routing; it comes from
> the expert initialisation. This **replicates on a second dataset, a second architecture and a
> second attach point** the BRATS result *The Gate Does Not Choose* [DOI 10.5281/zenodo.22903668],
> of which this paper is a replication, not a republication: no BRATS number is reused. On the
> program's pre-registered versatility criterion (36 endpoints × 13 arms), MoE-V3-CS is the **only
> all-rounder**: maximal damage **−0.53 pt** (crowd-group instances det@0.5) against −1.90
> (consensus C⊘B) and −22.20 (D, pedestrian pixel precision 70.0 → 47.8), a margin of 1.37 pt over
> the next arm, and the cheapest plateau of the field at **−1.2 pt of damage per mIoU point**
> against −43.1 for D. It is nonetheless **first on none of the 36 endpoints** (D wins 16) and its
> mean percentile is 60.4 (5th of 11): *good everywhere is not best everywhere*. Two business
> metrics survive the 15-pair Holm — `fragments` (−31.2 connected components per image) and
> `instances_taille_T3_rappel` (+0.26 pt, largest-instance recall) — while no per-class IoU rise
> survives the 19-class Holm (the only survivor is a fall, bicycle 0.0038). **Contributions.** (1)
> A pre-registered **positive primary** for an expert-initialised mixture against its
> recipe-matched control, reported with its **three** multiplicity families together and the
> explicit statement that it does not survive the exploratory one. (2) A **routing diagnostic that
> separates a tautology from a proof**: the equality `entropy_token == entropy_token_clean` at the
> final epoch holds *by construction* (the Shazeer noise is annealed to zero) and proves nothing;
> the probative comparison runs over the 40 active-noise epochs, where the clean distribution is
> already flat (≥ 0.99821) and `part_max` stays within [0.50011 ; 0.50217] of the 0.5000
> equidistribution. (3) The correct naming of two routinely conflated routing quantities —
> `sum(frac) == top_k` (= 2), not 1, and `top_expert_share` (mean of per-batch maxima, 0.667 at
> epoch 0) ≠ `part_max` (maximum of per-batch means, 0.501). (4) A **versatility verdict
> recomputed from raw deltas and ranks**, with every rank carrying the denominator of its own
> endpoint (the MoE's worst rank is 11/12 on `instances_foule_rappel`, because arm A is in partial
> coverage and the denominator is therefore not constant). (5) A measured **architectural
> difference with BRATS**: the residual guard γ is indispensable here (without it the raw recipe
> costs −28.1 mIoU points at epoch 0) and absent there. (6) Public release of code, configs,
> tables, figures and regeneration scripts. ---

## Contenu du dépôt

| chemin | contenu |
|---|---|
| `paper_fr.md` / `paper_fr.pdf` | manuscrit français (16 p.) |
| `paper.md` / `paper.pdf` | manuscrit anglais (16 p.) |
| `build.sh` + `header.tex` | la recette exacte qui reconstruit les deux PDF (pandoc → XeLaTeX, letter, marges 1,7 cm, Liberation Serif) |
| `tables/` | T1-T8 (md + csv) et `paper3_tables.json`, la table de consolidation (257 checks) |
| `figures/` | F1 Δ 13 bras, F2 routage, F3 forest métier, F4 polyvalence, F5 heatmap per-class (png + pdf) |
| `src/moe/` | `PatchMoE2D`, la logique d'accroche, l'initialisation des experts, le bootstrap, l'audit de diversité |
| `src/losses/`, `src/metrics/`, `src/postprocessing/` | les loss des spécialistes, les métriques officielles, le compte de fragments |
| `configs/` | configs Hydra du mélange et de son contrôle apparié |
| `scripts/p3_moe_tables.py` / `p3_moe_figures.py` | régénèrent toutes les tables et figures **sans GPU** depuis la source assertée |
| `scripts/p3_check_numbers.py` | **111 checks** reliant chaque nombre du manuscrit à son artefact (comptes recalculés) |
| `scripts/p3_check_manuscrit_gate.py` | rejoue les regex du gate DOI sur les manuscrits publiés |
| `scripts/train_moe_v3cs.py` / `moe_eval_harness.py` | entraînent le mélange et lancent le bootstrap primaire |
| `tests/` | accroche MoE, routage par patch, appariement par seed, bootstrap, mIoU officielle |
| `analysis/` | la chaîne de provenance complète des nombres publiés |
| `analysis/EXCLUDED.md` | les artefacts lourds **non** embarqués, nommés un par un avec leur chemin de régénération |

## Reproduire

```bash
bash build.sh                               # reconstruit les deux PDF
python3 scripts/p3_moe_tables.py            # régénère T1-T8 (sans GPU, 257 checks)
python3 scripts/p3_moe_figures.py           # régénère F1-F5 (sans GPU, 9 autocontrôles)
python3 scripts/p3_check_numbers.py         # 111 checks manuscrit<->artefacts
python3 scripts/p3_check_manuscrit_gate.py papers/paper3   # regex du gate DOI sur les manuscrits
```

Régénérer les tables et figures ne demande **aucun GPU** (elles lisent les artefacts assertés
d'`analysis/`). Régénérer *le bras lui-même* demande les quatre experts (4 × 160 époques) plus 80
époques de mélange en 1024×2048 sur une RTX PRO 6000 96 Go (≈ 17 h/seed) — voir
`tables/T1_architecture_recette_cout.md`.

**Déclaration de reproductibilité (déclarée, pas sur-vendue).** cuDNN benchmark reste activé
(`deterministic: false`, BF16) : la reproductibilité bit-à-bit sur GPU n'est **pas** revendiquée.
Ce qui est vérifié : mêmes artefacts → mêmes statistiques bit-à-bit (257 checks bloquants), et
forwards indépendants des mêmes checkpoints → accord à 8.58e-04 pt près sur le Δ
primaire.

## Papiers compagnons du programme

| papier | dataset | DOI |
|---|---|---|
| Régression auxiliaire par carte de distance | Cityscapes | [10.5281/zenodo.21006236](https://doi.org/10.5281/zenodo.21006236) |
| Ablation boundary loss (source du bras B apparié) | Cityscapes | [10.5281/zenodo.21006393](https://doi.org/10.5281/zenodo.21006393) |
| Blob loss seul (bras G, expert 3 ici) | Cityscapes | [10.5281/zenodo.23083560](https://doi.org/10.5281/zenodo.23083560) |
| **Ce papier** — mélange initialisé par experts (MoE-V3-CS) | Cityscapes | voir `CITATION.cff` |
| Mélange d'experts initialisé par experts (antériorité) | BRATS 2023 | [10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668) |
| Consensus de composantes connexes | BRATS 2023 | [10.5281/zenodo.22904810](https://doi.org/10.5281/zenodo.22904810) |

## Licences

Manuscrits, tables et figures : **CC-BY-4.0** (la licence du dépôt Zenodo). Code (`src/`,
`scripts/`, `tests/`, `configs/`) : **MIT** ([`LICENSE`](LICENSE)). Les données Cityscapes
restent soumises à leur propre licence.

## Citer

Voir [`CITATION.cff`](CITATION.cff). Auteurs : Guillaume Cassez, Stanislas Larnier, chercheurs indépendants (ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) pour Guillaume Cassez).
