# T3 — Position du MoE-V3-CS parmi les 13 bras du programme

mIoU dataset-level, holdout `first:500`, bootstrap apparié B = 10 000 (seed 20260618), seeds moyennés dans chaque réplicat, référence = contrôle apparié. Les **deux** familles de Holm sont données, avec leur taille calculée depuis l'artefact (leçon de l'erratum v1.1.0 du paper4 : une étiquette de famille écrite à la main finit par mentir).

| # | Bras | mIoU | Δ vs contrôle | IC95 | p | Holm (12 paires, P3.14) | Holm (15 paires, P3.16) |
|---|---|---|---|---|---|---|---|
| 1 | **D · CE+Kervadec EDT** | 81.686 | +0.518 | [+0.142 ; +0.910] | 0.0048 | 0.0576 | 0.0750 |
| 2 | **consensus D⊘B** | 81.652 | +0.484 | [+0.111 ; +0.871] | 0.0082 | 0.0820 | 0.1066 |
| 3 | **consensus Dp⊘B** | 81.647 | +0.479 | [+0.019 ; +0.957] | 0.0396 | 0.3168 | 0.4136 |
| 4 | **Dp · CE+SDT (distmap)** | 81.643 | +0.475 | [+0.038 ; +0.927] | 0.0312 | 0.2808 | 0.3960 |
| 5 | **MoE-V3-CS (4 experts)** ← ce papier | 81.618 | +0.449 | [+0.109 ; +0.820] | 0.0066 | 0.0726 | 0.0924 |
| 6 | A · CE seule | 81.283 | +0.115 | [-0.271 ; +0.525] | 0.5722 | 1.0000 | n.c. (a) |
| 7 | G · CE+Dice+Blob (Kofler) | 81.263 | +0.095 | [-0.279 ; +0.467] | 0.6552 | 1.0000 | 1.0000 |
| 8 | consensus C⊘B | 81.236 | +0.068 | [-0.338 ; +0.472] | 0.7758 | 1.0000 | 1.0000 |
| 9 | C · CE+Dice+EDT | 81.227 | +0.059 | [-0.344 ; +0.468] | 0.8096 | 1.0000 | 1.0000 |
| 10 | B · CE+Dice (baseline) | 81.093 | -0.075 | [-0.447 ; +0.281] | 0.6748 | 1.0000 | 1.0000 |
| 11 | consensus Cp⊘B | 81.014 | -0.154 | [-0.516 ; +0.170] | 0.3498 | 1.0000 | 1.0000 |
| 12 | Cp · CE+Dice+SDT | 80.890 | -0.278 | [-0.597 ; +0.016] | 0.0628 | 0.4396 | 0.6180 |
| — | contrôle (CE+Dice 80 ep, réf) | 81.168 | 0 (réf) | — | — | — | — |

(a) hors de la famille 15 paires (couverture partielle) : A.

Le MoE-V3-CS est **5ᵉ sur 12** par amplitude (5ᵉ sur 13 contrôle inclus — le contrôle, Δ = 0, se classe 10ᵉ), derrière D, les deux consensus D⊘B et Dp⊘B, et Dp. 5 bras ont un p brut < 0,05 ; **aucun** ne survit au Holm, ni sur 12 paires (meilleur 0.0576) ni sur 15 (meilleur 0.0750).

C'est le point de bascule du papier : puisque l'amplitude de mIoU ne sépare pas les bras de façon significative après correction, la conclusion porte sur **ce que chaque bras casse** (T7), où le MoE-V3-CS est premier avec une marge mesurée.
