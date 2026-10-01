# T4 — Diagnostic de routage : la porte ne choisit pas

Relevé dans `routing_moe_v3cs_Binit_seed*.jsonl` (80 époques × 3 seeds) et croisé avec `final_epoch_stats` des summaries (identité vérifiée à 1e-12 sur 8 grandeurs × 3 seeds). Équirépartition exacte de `part_max` pour top-2 parmi 4 experts = **0.5000** (calculée `top_k/n_experts`).

| Grandeur | époque 0 (seed 42/123/456) | époque finale (seed 42/123/456) | référence | lecture |
|---|---|---|---|---|
| `entropy_norm` | 0.99970 / 0.99967 / 0.99955 | **1.00000 / 0.99771 / 0.99997** | 1,0 = uniforme | porte quasi uniforme |
| `entropy_token` | 0.99801 / 0.99786 / 0.99676 | **0.99996 / 0.98745 / 0.99982** | 1,0 = uniforme | idem, bruit inclus |
| `entropy_token_clean` | 0.99896 / 0.99886 / 0.99821 | **0.99996 / 0.98745 / 0.99982** | 1,0 = uniforme | égale à `entropy_token` à l'époque finale **par construction** (bruit recuit à 0) : la comparaison probante est celle des 40 époques à bruit actif, ci-dessous |
| `part_max` | 0.50056 / 0.50149 / 0.50149 | **0.5030 / 0.5019 / 0.5019** | 0.5000 = équirépartition | écart max 0.0030 |
| `frac` des 4 experts | — | [0.4979 ; 0.5030] | 0.5000 | aucun expert favorisé |
| experts morts (`frac` = 0) | — | **0** | 0 | aucun effondrement |
| `top_expert_share` | 0.66700 / 0.66898 / 0.65957 | 0.59755 / 0.59546 / 0.59695 | 1,0 = un seul expert | part du premier expert, quasi stable |
| `noise_std` | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | recuit sur 40 époques | exploration éteinte, la platitude subsiste |
| `gamma_absmean` (γ) | 0.00024 / 0.00023 / 0.00023 | **0.02026 / 0.01912 / 0.01815** | 0 = MoE inutile | strictement positif et multiplié par ×77 à ×84 |
| `train_loss` | 0.1060 / 0.1015 / 0.1008 | 0.0748 / 0.0742 / 0.0740 | — | convergence normale |

**Lecture.** À l'époque finale, la distribution de routage est plate à 99.771 % d'entropie normalisée minimale, `part_max` vaut 0.5030 / 0.5019 / 0.5019 contre 0.5000 pour l'équirépartition exacte, les 4 experts reçoivent entre 49.79 % et 50.30 % des patches, et **aucun expert n'est mort**. La porte ne choisit donc pas : elle moyenne.

**Ce n'est pas un artefact du bruit de Shazeer — et la preuve n'est pas celle qu'on croit.** À l'époque finale le bruit est recuit à 0, donc `entropy_token_clean` est *identique par construction* à `entropy_token` : cette égalité-là ne prouve rien. La comparaison probante porte sur les **40 époques où le bruit est actif** (std 1,0 × std des logits) : l'écart maximal entre les deux entropies y est de **1.45e-03** (seed 42 9.54e-04, seed 123 9.95e-04, seed 456 1.45e-03), tandis que la distribution **sans bruit** y reste plate à **≥ 0.99821** d'entropie normalisée et `part_max` y reste entre 0.50011 et 0.50217 pour une équirépartition à 0.5000. Autrement dit : la platitude du routage est une propriété de la porte, pas du bruit qu'on lui injecte.

Pourtant le MoE **sert** : γ, l'échelle résiduelle apprenable par canal (initialisée à 0.0 pour rendre l'époque 0 bit-exacte au contrôle), vaut 0.02026 / 0.01912 / 0.01815 en fin de run contre 0.00024 / 0.00023 / 0.00023 à l'époque 0. Le gain vient donc de la **moyenne d'experts**, pas de la sélection — c'est la réplication sur Cityscapes du résultat BRATS, avec une autre architecture et un autre point d'accroche.

**Limite écrite telle quelle.** `entropy_token` du seed 123 (0.98745) est très légèrement sous les deux autres (0.99996, 0.99982) : la thèse tient sur les 3 seeds mais n'est pas d'une uniformité parfaite sur celui-ci.

Provenance : `results/moe_v3_cs/routing_moe_v3cs_Binit_seed*.jsonl` (3 seeds × 80 époques), `results/moe_v3_cs/train_moe_v3cs_Binit_seed*_summary.json`.
