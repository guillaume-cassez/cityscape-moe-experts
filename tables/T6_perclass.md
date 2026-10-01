# T6 — IoU par classe : attribution des gains et des coûts

19 classes Cityscapes, référence = contrôle apparié, holdout `first:500`, bootstrap apparié B = 10,000 (seed 20260618), Holm sur la **famille des 19 classes** (recomputé depuis les p bruts, écart max 0.0e+00). Colonnes D/Dp/G/B = Δ du spécialiste correspondant, pour lire ce que le MoE capte sans en hériter les pertes.

| Classe | IoU MoE | IoU ctl | Δ(MoE−ctl) | IC95 | p | Holm(19) | ΔD | ΔDp | ΔG | ΔB |
|---|---|---|---|---|---|---|---|---|---|---|
| truck | 84.834 | 80.593 | +4.241 | [+0.438 ; +8.752] | 0.0066 | 0.1188 | +4.94 | +5.19 | +3.43 | -0.35 |
| wall | 58.698 | 56.147 | +2.551 | [-1.085 ; +6.897] | 0.2000 | 1 | +5.49 | +3.08 | +1.07 | +1.77 |
| train | 83.463 | 82.234 | +1.229 | [+0.054 ; +2.820] | 0.0408 | 0.5712 | +0.40 | +0.66 | -1.04 | +1.50 |
| fence | 66.683 | 65.807 | +0.875 | [-0.271 ; +2.298] | 0.1586 | 1 | +1.21 | +1.06 | -0.46 | +0.26 |
| sidewalk | 87.296 | 86.975 | +0.321 | [+0.040 ; +0.660] | 0.0224 | 0.336 | +0.62 | +0.48 | +0.06 | +0.09 |
| terrain | 66.599 | 66.396 | +0.203 | [-0.252 ; +0.673] | 0.3980 | 1 | +1.21 | +0.33 | +0.01 | -0.30 |
| car | 95.698 | 95.563 | +0.134 | [-0.065 ; +0.398] | 0.2622 | 1 | +0.24 | +0.32 | +0.14 | -0.11 |
| traffic sign | 83.834 | 83.703 | +0.131 | [-0.136 ; +0.505] | 0.4942 | 1 | -1.05 | -1.01 | +0.21 | -0.18 |
| bus | 91.653 | 91.550 | +0.103 | [-0.793 ; +0.947] | 0.8056 | 1 | +0.57 | +0.79 | -1.48 | -1.80 |
| building | 93.347 | 93.251 | +0.097 | [-0.040 ; +0.258] | 0.1778 | 1 | +0.29 | +0.23 | +0.07 | +0.02 |
| road | 98.468 | 98.411 | +0.057 | [+0.013 ; +0.111] | 0.0078 | 0.1326 | +0.05 | +0.07 | +0.04 | +0.03 |
| vegetation | 93.124 | 93.134 | -0.010 | [-0.056 ; +0.033] | 0.6456 | 1 | +0.03 | +0.04 | +0.02 | -0.04 |
| sky | 95.515 | 95.530 | -0.015 | [-0.068 ; +0.031] | 0.5840 | 1 | -0.09 | -0.12 | +0.11 | +0.07 |
| traffic light | 77.279 | 77.360 | -0.081 | [-0.235 ; +0.082] | 0.3230 | 1 | -2.60 | -2.34 | +0.47 | -0.51 |
| pole | 69.935 | 70.046 | -0.111 | [-0.324 ; +0.094] | 0.2828 | 1 | +0.11 | +0.08 | +0.29 | -0.29 |
| person | 84.910 | 85.153 | -0.244 | [-0.433 ; -0.062] | 0.0078 | 0.1326 | -0.19 | -0.25 | +0.07 | -0.26 |
| bicycle | 80.068 | 80.333 | **-0.265** | [-0.408 ; -0.121] | 0.0002 | 0.0038 | -0.70 | -0.53 | +0.05 | -0.52 |
| motorcycle | 70.775 | 71.084 | -0.309 | [-1.485 ; +0.727] | 0.5610 | 1 | -0.13 | +1.34 | +0.07 | -0.39 |
| rider | 68.556 | 68.925 | -0.369 | [-1.317 ; +0.525] | 0.4308 | 1 | -0.58 | -0.39 | -1.34 | -0.44 |

**Gras** = Holm(19) < 0,05. Sur 19 classes : 11 en hausse, 8 en baisse.

## Trois critères de significativité, donnés séparément

Le classement par **amplitude** et le classement par **significativité** ne coïncident pas ; les confondre produirait une affirmation fausse. Les trois sont donnés tels quels :

1. **Plus fortes amplitudes** (ordre décroissant du Δ) : truck +4.24, wall +2.55, train +1.23, fence +0.88.
2. **IC95 excluant 0** (critère « significant » de P3.16) : hausses truck +4.24, train +1.23, sidewalk +0.32, road +0.06 ; baisses person -0.24, bicycle -0.26.
3. **Holm sur la famille des 19 classes** : **aucune hausse ne survit** ; la seule classe qui survive est une **baisse** — bicycle (Holm 0.0038).

C'est le point d'honnêteté central de cette table : le gain le plus spectaculaire (`truck` +4.24 pt) a un IC qui exclut 0 mais **ne survit pas** au Holm des 19 classes (0.1188), parce que son IC est large ([+0.44 ; +8.75]) sur une classe rare. Le seul effet par classe qui résiste à la correction multiple est un **coût** (bicycle), pas un gain. Les coûts sont faibles et distribués : bicycle -0.26, motorcycle -0.31, rider -0.37.

## Ce que le MoE garde des spécialistes, et ce qu'il évite

- **truck** : MoE +4.24 contre D +4.94, Dp +5.19, G +3.43, B -0.35 — le MoE retrouve 82 % du meilleur gain spécialiste (Dp +5.19).
- **train** : MoE +1.23 contre D +0.40, Dp +0.66, G -1.04, B +1.50 — le MoE retrouve 82 % du meilleur gain spécialiste (B +1.50).
- **traffic light** : MoE -0.08 contre D -2.60, Dp -2.34, G +0.47, B -0.51 — perte 0.08 pt, soit 3 % de la pire perte spécialiste (D -2.60).
- **person** : MoE -0.24 contre D -0.19, Dp -0.25, G +0.07, B -0.26 — perte 0.24 pt, soit 95 % de la pire perte spécialiste (B -0.26).
- **bicycle** : MoE -0.26 contre D -0.70, Dp -0.53, G +0.05, B -0.52 — perte 0.26 pt, soit 38 % de la pire perte spécialiste (D -0.70).
- **rider** : MoE -0.37 contre D -0.58, Dp -0.39, G -1.34, B -0.44 — perte 0.37 pt, soit 28 % de la pire perte spécialiste (G -1.34).

Provenance : `results/moe_v3_cs/p316/attribution_perclass_vs_controle.json`.
