# AUDIT — Diversité des experts (P3.06, fait le 2026-09-15)

> **Objet** : verrouiller par MESURE le point d'initialisation des experts du MoE-V3-CS
> ([`PLAN_TRANSFERT_BRATS_V3.md`](./PLAN_TRANSFERT_BRATS_V3.md) §3.3, risque §6 « blocs quasi
> identiques »). **Sans GPU** : lecture seule des `model_state_dict` sur CPU.
> **Script** : [`src/moe/audit_expert_diversity.py`](./audit_expert_diversity.py) —
> JSON brut : `results/moe_v3_cs/audit_diversite_20260915_220449.json` (sur la Tour).
> **Données** : checkpoints epoch-160 des experts **B/D/Dp/C** (+Cp/A en informatif),
> seeds **42/123/456** (18 fichiers, 13 s de chargement).

## Méthode

Pour chaque bloc candidat : paramètres **entraînés** aplatis-concaténés (BN running stats
exclues), puis **Frobenius relatif symétrique** `d(a,b) = ||a−b||₂ / ((||a||₂+||b||₂)/2)` :

- **between** = inter-méthodes **à seed fixé** (les 6 checkpoints d'un seed partagent
  init aléatoire ET ordre de données → le delta isole l'effet de la **loss**) ;
- **within** = inter-seeds d'une même méthode (le bruit d'entraînement complet).

## Résultats

| Bloc candidat | #params | within méd. | between méd. | between min–max | cosdist between |
|---|---|---|---|---|---|
| **`head.fpn_convs.0`** ← **retenu** | 2,36 M | 0.862 | **0.538** | **0.422–0.568** | 0.145 |
| `head.fpn_convs.1` | 2,36 M | 0.878 | 0.549 | — | — |
| `head.fpn_convs.2` | 2,36 M | 0.883 | 0.528 | — | — |
| `head.fpn_bottleneck` (accroche MoE) | 9,44 M | 1.018 | 0.711 | 0.569–0.769 | 0.253 |
| `backbone.stages_3.blocks.2` (stage 4) | 8,46 M | 0.122 | **0.122** | 0.111–0.126 | **0.0075** |
| `backbone.stages_3` (stage entier) | 27,5 M | 0.127 | 0.128 | — | — |
| `head.classifier` | 9,7 k | 1.273 | 0.430 | — | — |
| `head` (tête entière) | 33,2 M | 0.897 | 0.552 | — | — |

**Toutes les 15 paires de méthodes × 3 seeds** (A/B/C/D/Dp/Cp pris 2 à 2) sont ≥ 0.42 sur
`head.fpn_convs.0` — aucune paire ne s'effondre, les 4 experts du V3-CS (B/D/Dp/C) sont
systématiquement distincts (B–D : 0.49–0.56 ; B–Dp : 0.54–0.57 ; B–C : 0.45–0.55).

### Test de systématie du drift (métrique complémentaire)

Cosinus des vecteurs de différence entre seeds (ex. `Δ(B−D)|seed42` vs `Δ(B−D)|seed123`) :
**≈ +0.001 à +0.008 sur la tête** (plancher de bruit `1/√N ≈ 0.0007` → drift loss **non
reproductible** d'un seed à l'autre, la spécialisation vit dans le bassin de chaque seed).
Petite exception informative : `backbone.stages_3.blocks.2` B–D/B–Dp ~ +0.055 (≈ 80× le
plancher — une trace systématique faible, noyée dans un backbone qui ne bouge presque pas).

## Verdicts (VERROUILLENT le plan §3.3 & §6)

1. **Le risque §6 « `fpn_convs[0]` quasi identiques » est RÉFUTÉ** : la différenciation
   inter-loss à seed fixé est massive (~53 % de distance relative médiane, cosdist 0.145).
   → **La source d'init des experts MoE reste `head.fpn_convs.0`** : même forme (Conv3×3
   512→512 + BN + ReLU) et même résolution (stride-4) que le point d'accroche
   `fpn_bottleneck`, analogie directe avec le `dec_block_0` de BRATS.
2. **La mitigation §6 « remonter l'accroche au dernier ConvNeXtBlock stage 4 » est devenue
   INUTILE — et serait contre-productive** : le backbone profond ne porte AUCUNE
   différenciation loss (between = within = 0.122 ; directions quasi identiques,
   cosdist 0.0075). La diversité mesurable vit dans la **tête**, pas dans le backbone.
   → L'accroche du MoE reste **après `fpn_bottleneck`** dans `UPerNetHead.forward`.
3. **Protocole d'appariement des checkpoints — NOUVEAU, imposé par la mesure** : le drift
   loss n'étant pas reproductible entre seeds (verdict « test de systématie » ci-dessus),
   un run MoE-V3-CS de seed *s* DOIT être initialisé depuis les experts **du même seed s**
   (backbone ← B_s ; expert 0 ← B_s, 1 ← D_s, 2 ← Dp_s, 3 ← C_s — `head.fpn_convs.0`).
   Jamais de mélange inter-seeds : c'est l'analogue exact des folds appariés de BRATS.
4. Corollaire narratif pour la thèse « la porte ne choisit pas » : l'orthogonalité des
   drifts prédit qu'aucune « direction loss » universelle n'est routable — cohérent avec le
   fait qu'en BRATS le gate ne se spécialise pas (part_max ≈ 1/E). À confirmer par les logs
   de routage (P3.12).

## Impact sur la suite (P3.07/P3.08)

- `PatchMoE2D` : experts = `Conv3×3(512→512) + BN + ReLU` (forme `fpn_convs`), résiduel
  `x + moe(x)` après `fpn_bottleneck`, gate 2Conv-GAP, top-2/4, grille 3×3, halo ≥ kernel//2.
- `initialize_from_experts(model, seed)` : backbone+head ← B_seed ; pour i ∈ {B, D, Dp, C},
  expert_i ← `{0.weight, 0.bias?, 1.weight, 1.bias}` de `head.fpn_convs.0` du checkpoint
  i_seed ; gate ← aléatoire ; tracer la correspondance, **stop si un checkpoint manque** (§3.3.4).
- Note : les convs UPerNet sont `bias=False` — l'expert MoE doit reprendre la même forme
  (conv sans bias + BN) pour un copier-coller exact des poids.

---

## RE-AUDIT 2026-09-22 — l'expert G (Blob) à la place de C (P3.15 clos 3/3)

> **Contexte** : décision Guillaume du 17/09 — les 4 experts du MoE-V3-CS sont
> **B/D/Dp/G** (miroir B/D/K/G de BRATS), G = BlobLoss de Kofler entraîné en P3.15
> (3/3 seeds epoch-160 écrits les 18/20/21-09). Le présent re-audit rejoue la mesure
> P3.06 avec G, comme le prescrivait le « RESTE » de P3.15. Le verdict 3 ci-dessus
> (appariement seed-à-seed) s'applique désormais avec **expert 3 ← G_s**.
> **Méthode identique** : même script, `--methods B D Dp G C --extra` (7 méthodes ×
> 3 seeds = 21 checkpoints, 16.2 s de chargement, CPU/E-cores 16-27 — training P3.11
> en cours sur GPU, non touché). JSON brut :
> `results/moe_v3_cs/audit_diversite_20260922_113904.json` (sur la Tour).

### Résultats — `head.fpn_convs.0` (la source d'init retenue)

| Paire avec G (seed fixé) | s42 | s123 | s456 |
|---|---|---|---|
| B–G | 0.559 | 0.561 | **0.466** |
| D–G | 0.532 | 0.536 | 0.521 |
| Dp–G | 0.560 | 0.566 | 0.561 |

- **Les 18 paires between des 4 experts V3-CS (B/D/Dp/G)** : min **0.466**, médiane
  **0.546**, max 0.568 — même amplitude que l'audit d'origine avec C (0.422–0.568).
  Aucune paire ne s'effondre : **G est aussi différencié que C** l'était.
- **Within G** (inter-seeds) : 0.8614–0.8618 — exactement le niveau de bruit
  d'entraînement des autres méthodes (within V3-CS : 0.849–0.866). La loss blob
  n'a PAS « gelé » G près d'une autre méthode.
- `head.fpn_bottleneck` (accroche MoE) : paires avec G 0.592–0.751, cohérent avec
  l'audit d'origine (0.569–0.769).
- Le plus faible écart G (B_s456–G_s456 = 0.466) reste largement dans la plage
  d'origine et ≈ 6× au-dessus de la plus petite distance jamais mesurée sur le
  backbone profond (0.12) — sans commune mesure avec un effondrement.

### Verdict (re-verrouille §3.3 pour le jeu final B/D/Dp/G)

1. **La source d'init `head.fpn_convs.0` reste VERROUILLÉE** pour les 4 experts
   B/D/Dp/G : la différenciation loss à seed fixé est massive pour G comme pour
   les autres (≈ 0.47–0.57 de distance relative, cosdist 0.11–0.16).
2. **L'appariement seed-à-seed reste obligatoire** (within G ≈ 0.86 ≫ between ≈ 0.55 :
   le bassin de chaque seed domine) — expert 0 ← B_s, 1 ← D_s, 2 ← Dp_s, **3 ← G_s**.
3. Corroboré indépendamment par le **gate époque-0 joué par la file le 22/09 11:15:54**
   (`moe_epoch0_check.py --seed 42 --n-images 100`, expert 3 = G) : mIoU MoE ép.0 =
   0.8245 = contrôle B, Δ **+0.000 pt**, accord pixel **100.00 %** — l'init G ne casse
   rien (γ ReZero = 0). P3.12 peut s'enchaîner sans re-vérification bloquante.

### Complément — mIoU de G seul (22/09 16:29, mesuré)

La diversité ci-dessus porte sur les POIDS d'init (`fpn_convs.0`) ; la question
« G est-il un bon segmenteur tout seul ? » est distincte. Éval bras `dense` du
harness P3.10, holdout apparié 500 img × 3 seeds, GPU0 partagé avec le contrôle
P3.11 (training intact) :

| bras | mIoU dataset (3 seeds) | par seed 42 / 123 / 456 |
|------|------------------------|--------------------------|
| **G seul** (blob Kofler, ep160) | **0.8126** IC95 [0.7981, 0.8243] | 0.8130 / 0.8129 / 0.8120 |
| **B baseline** (npy, ep160) | **0.8109** IC95 [0.7969, 0.8220] | 0.8143 / 0.8086 / 0.8099 |

Bootstrap apparié (mêmes positions, n=500, B=10000) : Δ **G−B = +0.17 pt**,
IC95 [−0.55, +0.23], **p = 0.403, Holm = 0.403 → NON significatif**. G seul est
**indistinguable de la baseline**, miroir exact de BRATS (G seul −0.00376, dans le
bruit). Conclusion cohérente avec le verdict de diversité : **la valeur de G pour
le MoE n'est PAS son mIoU propre mais la différenciation de son init** — exactement
le rôle d'un expert-initialisé (transfert V3). Table : `results/moe_v3_cs/harness/
table_Gseul_P310.json`.
