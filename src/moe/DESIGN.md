# DESIGN — Cadrage MoE appris (Paper 3) — **FIGÉ**

> Livrable de la tâche board **P3.01 / #55**. Ce document **fige** le design du Mixture-of-Experts
> *appris* qui remplace le consensus fixe `C⊘B` par un gating **appris, per-pixel, conditionné à
> l'entrée**. Le « comment » d'implémentation vit dans [`SPEC.md`](./SPEC.md) ; ce fichier tranche le
> « quoi / pourquoi » et sert de référence de vérité pour P3.02→P3.05.
>
> Nommage : le code (`SPEC.md`, `train_moe.py`) dit **Paper 3** ; l'issue #55 emploie l'alias legacy
> **Paper 5** (slug mémoire `project_paper5_moe_design`). Nom canonique retenu : **Paper 3 (MoE)**.

## 1. Hypothèse de recherche

Les experts publiés de Cityscape sont **complémentaires par classe** (finding paper Kervadec, DEVLOG
2026-06-05) : `D` (Kervadec, CE+Boundary) domine les grandes classes structurées (truck, wall, bus),
`B` (baseline CE+Dice) garde les classes fines riches en signal (traffic light, train, traffic sign),
`Dp` (DistMap, CE+DistMap) apporte une 3ᵉ géométrie de frontière.

Le **consensus fixe `C⊘B`** (`src/postprocessing/consensus.py`, P3.04) est une règle **sans paramètres**
(agreement/intersection) : il ne sait que *retirer* des fragments là où deux experts se contredisent.
**Hypothèse P3** : un gating **appris** per-pixel est plus expressif — il peut **récupérer** une région
où *un seul* expert a raison (bord → Kervadec, intérieur → baseline), pas seulement soustraire. Gain
attendu : mIoU et surtout Boundary F1 au-dessus du meilleur expert seul **et** des ensembles naïfs.

**Signal préliminaire à ne pas ignorer** (run aperçu `results/moe_apercu/`, 2 configs/4, gate CNN,
logits seuls, 8 epochs, 1 split) : le gate **n'a pas battu** la moyenne — `Dp+D` gate 0.8131 ≈ mean
0.8130 ; `Dp+B` gate 0.8116 **<** mean 0.8146. Lecture : *logits-seuls + peu d'epochs + petit fit + un
seul split* ne donne pas au gate de signal exploitable au-delà de la moyenne. Le design ci-dessous
**intègre ce constat** via des entrées de gate plus riches, du routage top-1, plus de capacité, et une
validation par IC qui seule permet de statuer (un écart de 1e-4 mIoU est du bruit).

## 2. Experts gelés (epoch-160, seed principal 42)

| Clé | Variante | Origine | Best mIoU | Checkpoint (repo root, Tour) |
|---|---|---|---|---|
| `Dp` | DistMap (CE+DistMap) | paper1 | 81.64 | `checkpoints/pilot_fullres_DM_distmap_seed{S}/epoch_160.pth` |
| `D`  | Kervadec (CE+Boundary) | paper2 | 81.69 | `reeval_slim/pilot_fullres_D_ce_boundary_seed{S}__epoch_160.pth` |
| `B`  | Baseline (CE+Dice) | baseline | 81.09 | `reeval_slim/pilot_fullres_B_baseline_seed{S}__epoch_160.pth` |

Chargement (implémenté `src/moe/experts.py`) : `build_model(cfg_du_ckpt)` + `load_state_dict` strict +
`eval()` + `requires_grad_(False)`. Forward eval → **logits bruts** `(B,19,1024,2048)`, jamais de
softmax intra-expert. Les experts ne sont **jamais** ré-entraînés (mémoire, reproductibilité, honnêteté :
on mesure l'apport du *gate*, pas d'un ré-apprentissage déguisé).

## 3. Fusion late — **FIGÉ**

Empilement des logits des `K` experts → gate → poids simplex per-pixel → **combinaison convexe des
logits** :

```
stacked  ∈ ℝ^(B,K,19,H,W)              # K experts gelés, logits bruts
g = softmax_K( Gate(entrées) )  ∈ Δ^K   # (B,K,H,W), Σ_k g_k = 1 par pixel
fused = Σ_k  g_k · logits_k      ∈ ℝ^(B,19,H,W)
pred  = argmax_c fused                   # prédiction finale
```

Loss d'apprentissage du gate : **CE(fused, GT)**, `ignore_index=255`. On mixe les **logits** (pas les
probabilités) — cohérent avec LoMix (mixe des logits multi-échelle) et avec l'implémentation actuelle ;
mixer les probas softmax est une variante secondaire non retenue par défaut. BF16 autocast.

## 4. Familles de gate — **FIGÉ** (les deux comparées)

| Famille | Archi (implémentée `src/moe/gate.py`) | Rôle |
|---|---|---|
| **CNN** | `Conv3×3(K·19→h) → GN → GELU → Conv3×3 → GN → GELU → Conv1×1(h→K) → softmax_K` | **Principal** : full-res, coût borné, champ réceptif local suffisant pour un routage frontière/intérieur |
| **Transformer** | attention per-pixel sur `K` tokens-experts (token = vecteur logit 19-d) → softmax_K | **Variante** : O(H·W) séquences → entraînement **sur crops**, teste si l'attention inter-experts bat les convs |

Décision : **CNN = défaut** (rapport perf/coût), Transformer = arm de comparaison exigé par la note Keep
(CNN vs transformer). Les deux partagent exactement le protocole (splits, métriques, IC).

## 5. Grille d'expériences — **FIGÉ** (4 configs)

`{Dp,D}` · `{Dp,B}` · `{D,B}` · `{Dp,D,B}`. Les 3 **paires** isolent les complémentarités 2-à-2 ; le
**triple** teste si un 3ᵉ expert ajoute du signal ou seulement du bruit corrélé. × 2 familles de gate
= 8 runs par split-seed.

## 6. Routage : soft vs top-1 — **FIGÉ**

- **Soft** (défaut, entraînement) : `fused = Σ_k g_k·logits_k`. Gradient dense sur tous les experts →
  stable, c'est le régime d'apprentissage.
- **Top-1 sparse** (Shazeer et al., ICLR 2017) : à l'inférence, `k* = argmax_k g_k` par pixel → on
  prend l'expert dominant. Obtenu par **durcissement post-hoc** du gate soft entraîné (**soft-to-hard**,
  arXiv:2605.02124), option d'un **annealing de température** `softmax(logits_gate / τ)` avec `τ↓` en fin
  d'entraînement pour rapprocher soft de one-hot sans casser le gradient tôt.
- **Load-balancing** : avec K∈{2,3} et soft, le collapse (un expert monopolise) est peu probable mais
  mesuré (entropie moyenne du gate loggée). Pour le régime top-1, régularisation d'importance/charge
  optionnelle (Shazeer §4) si collapse observé. On rapporte **soft ET top-1** sur le même holdout.

## 7. Entrées du gate — ablation ordonnée — **FIGÉ**

Le run aperçu (logits seuls) plafonne à la moyenne → on **enrichit** l'entrée par paliers, hypothèse :
le gate a besoin de signal *que les logits ne portent pas*.

| Niveau | Entrée du gate | Canaux | Intuition |
|---|---|---|---|
| **(a)** | logits experts concaténés | `K·19` | baseline (implémenté) |
| **(b)** | (a) + **image RGB** (down-samplée au besoin) | `K·19 + 3` | contexte visuel : texture/luminance que les logits ont déjà « collapsé » |
| **(c)** | (b) + **cartes d'incertitude par expert** = entropie de Shannon `H(softmax(logits_k))` | `+ K` | héritage P8 (6 mesures d'incertitude) ; l'entropie localise *où* un expert doute → router vers l'autre |

Ablation menée `a → b → c` (chaque palier ajoute au précédent). Appui littérature : l'incertitude est un
signal naturel d'un MoE de segmentation (Extracting Uncertainty Estimates from MoE for Semantic
Segmentation, arXiv:2509.04816). **Reste à implémenter** : (b) et (c) — `gate.py` ne prend aujourd'hui
que (a) ; il faut passer l'image + les entropies au forward et adapter `in_ch`.

## 8. Protocole anti-fuite + splits — **FIGÉ**

| Split | Taille | Usage |
|---|---|---|
| **train** Cityscapes fine | 2975 | entraînement des experts **uniquement** (déjà fait, gelé) |
| **gate-fit** (⊂ val 500) | 300 | entraînement du gate |
| **gate-test** (⊂ val 500) | 200 | évaluation gate + toutes baselines, **jamais vu au fit** |

Règles : (1) fit ∩ test = ∅, découpage à **seed fixe** (`split_val_indices`, implémenté) ; (2) les
experts n'ont **jamais** vu le val → aucune fuite expert→gate ; (3) **aucune sélection de modèle sur
gate-test** — hyperparamètres du gate fixés *a priori* (ou tunés par CV *interne au gate-fit*), sinon le
test fuit ; (4) pour ne pas dépendre d'un découpage, **répéter sur ≥3 split-seeds** (42, 123, 456) et
agréger. ⚠️ `gate-test` ⊂ val ≠ les 500 standard ni le test caché du serveur → les chiffres **ne sont pas
comparables** aux mIoU publiés ; **seule la comparaison gate-vs-baselines, à holdout identique, est
valide** (apples-to-apples). Le protocole final de report (P3.05) tranchera val-complet vs test-serveur.

## 9. Métriques + validation — **FIGÉ** (règle `/metrics-validation`)

- **mIoU officiel bit-exact** (`SegmentationMetrics`, prouvé `tests/test_official_miou.py`) + **Boundary
  F1** moyen par image (`compute_boundary_f1`). Loggés pour gate / mean / argmax / consensus / chaque
  single. Per-class IoU conservé.
- **IC 95% par bootstrap apparié sur images** (best-practice confirmée) :
  1. `R = 1000` réplicats ; à chaque réplicat, rééchantillonner les `n=200` images du **gate-test** avec
     remise → **même index-set appliqué à TOUTES les méthodes** (préserve l'appariement).
  2. Le mIoU est **dataset-level** (IoU sur matrice de confusion agrégée) → **ré-agréger la confusion**
     sur les images rééchantillonnées puis **recalculer** le mIoU par réplicat. **Ne jamais** moyenner
     des mIoU par-image.
  3. Contraste `Δ = métrique(gate) − métrique(baseline)` par réplicat → IC **percentile** [2.5, 97.5].
     Différence **significative si 0 ∉ IC**.
- L'appariement est ce qui rend le test informatif : deux IC marginaux peuvent se recouvrir alors que le
  Δ apparié exclut 0 (la variance de difficulté par image s'annule). Sur le run aperçu, gate−mean = 1e-4
  → très probablement IC contenant 0 = **non significatif** : sans IC on sur-interpréterait le bruit.
- **Reste à implémenter** : le bootstrap apparié (nouveau `src/moe/bootstrap.py` + collecte des
  confusions par-image dans `metrics_eval.py`).

## 10. Baselines à battre — **FIGÉ**

Le gate doit dominer **toutes** ces baselines (Δ avec IC excluant 0) pour justifier sa complexité :

1. **best-single** — meilleur expert seul sur le gate-test.
2. **mean-ensemble** — moyenne des logits (soft-vote). *(implémenté)*
3. **argmax / majority-vote ensemble** — vote par pixel sur les argmax des experts (tie-break déterministe).
4. **consensus fixe `C⊘B`** — `src/postprocessing/consensus.py` (P3.04), la règle sans paramètres qu'on cherche à dépasser.

**Reste à implémenter** : baselines 3 et 4 dans le harness d'éval (2 = déjà là, 1 = déjà là via `singles`).

## 11. Matériel — **FIGÉ**

- **Tour** `serveur3090` (192.168.1.75), **RTX PRO 6000 96 Go**. **Un seul job lourd à la fois** →
  jobs **séquentiels**, jamais empilés (contrainte board). Dev/cadrage sur TPL (pas de GPU) ; run via
  `ssh ser@192.168.1.75 "cd /home/ser/Bureau/City_Scape && taskset -c 0-15 python scripts/train_moe.py …"`.
- **Coût dominant = forward des experts**, pas le gate (CNN ~0.5 M params, négligeable). Experts gelés
  déterministes → **cache disque des logits par expert** (fp16), calculé **une fois** et réutilisé entre
  les 4 configs (`src/moe/cache.py`). Budget VRAM cible **< 38 Go** (coexistence possible avec le
  llama-server 59 Go sur la même carte ; sinon tout le GPU est dispo).
- **État au cadrage (2026-07-11)** : RTX 6000 saturée par un job **BRATS** (`tune_moe`, 88/97 Go) →
  l'**exécution** P3.02+ attend la libération du GPU. Le **cadrage (#55) n'exige aucun GPU** → livré ici.

## 12. Décisions figées (récap) & risques

**Figé** : fusion late soft sur logits (§3) ; gate CNN principal + transformer variante (§4) ; grille 4
configs (§5) ; soft entraîné + top-1 durci post-hoc (§6) ; entrées a→b→c (§7) ; splits train / gate-fit
300 / gate-test 200, ≥3 split-seeds, zéro sélection sur test (§8) ; mIoU officiel + BF1 + **IC bootstrap
apparié** (§9) ; baselines best-single / mean / argmax / consensus (§10) ; jobs séquentiels RTX 6000,
cache logits (§11).

**Risques** :
- *Le gate ne bat pas la moyenne* (déjà observé sur logits-seuls). Mitigations cadrées : entrées (b)(c),
  capacité/epochs, top-1, multi-split + IC. **Résultat négatif assumé & publiable** : « un gate appris
  per-pixel ne dépasse pas la moyenne sur des experts full-res corrélés » est une conclusion honnête, en
  contraste avec LoMix (qui co-entraîne le mixage *bout-en-bout* avec le réseau, nous gelons les experts).
- *Sur-apprentissage du gate* sur 300 images → IC larges. Mitigation : gate léger, weight-decay, ≥3 splits.
- *Corrélation des experts* (même backbone ConvNeXt-V2, même train) → peu de désaccord exploitable ;
  l'ablation d'entrée et le triple config mesurent si de la diversité résiduelle existe.

## 13. Reste à implémenter (P3.02 → P3.05) — pointeurs

- **P3.02** (#56, gate) : gate CNN — **fait** (`gate.py`,`train_moe.py`). À compléter : entrées (b)(c) §7,
  routage top-1 §6, transformer sur crops.
- **P3.03** : IC bootstrap apparié §9 (`src/moe/bootstrap.py` + confusions par-image).
- **P3.04** : baselines argmax + consensus `C⊘B` §10 dans le harness.
- **P3.05** : benchmark protocole final (val-complet vs test-serveur) §8, multi-split §8, report + figures.

## 14. Littérature (vérifiée — WebSearch 2026-07-11)

- **Shazeer et al., ICLR 2017** — *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer*, arXiv:1701.06538 (gating appris + top-k sparse). ✔
- **model-level MoE pour segmentation** — *Towards Adversarial Robustness of Model-Level MoE
  Architectures for Semantic Segmentation*, arXiv:2412.11608, 2024 (experts = modèles, gating pondère
  leurs sorties — le setup le plus proche du nôtre). ✔ *(remplace l'ID erroné « 2604.18256 » de la note
  Keep, introuvable.)*
- **Pavlitskaya et al., CVPRW 2020** — *Using Mixture of Expert Models to Gain Insights into Semantic
  Segmentation*. ✔
- **LoMix, NeurIPS 2025** — *Learnable Weighted Multi-Scale Logits Mixing for Medical Image
  Segmentation*, arXiv:2510.22995 (mixage de logits appris > moyenne équipondérée, +1.5 % DICE). ✔
- **Soft-to-Hard Routing in Sparse MoE**, arXiv:2605.02124 (durcir un gate soft en top-1). ✔
- **Extracting Uncertainty Estimates from MoE for Semantic Segmentation**, arXiv:2509.04816, 2025
  (l'incertitude par expert comme signal de gating — appui du §7 palier (c)). ✔
- **Bootstrap apparié** : 1000 resamples, même index-set pour toutes les méthodes, IC percentile 95 %,
  significatif si 0 ∉ IC (méthodo standard de comparaison de modèles, confirmée). ✔
