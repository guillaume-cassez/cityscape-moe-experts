# The Gate Still Does Not Choose

**An expert-initialised mixture-of-experts beats its matched control and is the only all-rounder
arm under full-resolution Cityscapes metrics.**

**Guillaume Cassez** · **Stanislas Larnier** — independent research · ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) (G.C.) · [HAL stanislas-larnier](https://cv.hal.science/stanislas-larnier) (S.L.) · [guillaume-cassez.fr](https://guillaume-cassez.fr/voiture-autonome/)

Preprint, CC-BY-4.0. Read [`paper.pdf`](paper.pdf) (EN, 16 p.) or
[`paper_fr.pdf`](paper_fr.pdf) (FR, 16 p.).

---

## What this is

A pre-registered, controlled evaluation of **MoE-V3-CS**, a patch-wise **mixture-of-experts**
inserted into a full-resolution Cityscapes segmenter (ConvNeXt-V2-Base + UPerNet, 1024×2048,
19 classes, BF16). Its four experts are **initialised bit-for-bit from the `head.fpn_convs.0`
block of four independently trained loss specialists of the same seed** — B (CE+Dice), D
(CE+Kervadec EDT), Dp (CE+SDT distance map), G (CE+Dice+0.5·blob) — and a top-2 gate over 3×3
patches is trained 80 epochs from a residual scale γ initialised to zero, so epoch 0 is bit-exact
to the control. This is a **replication on a second dataset** of the BRATS result *The Gate Does
Not Choose* ([DOI 10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668)) — no BRATS
number is reused.

## The primary endpoint is positive raw, and the gate does not choose

| | value |
|---|---|
| **Primary endpoint** (pre-registered) | dataset-level official mIoU, MoE vs recipe-matched control |
| Δ(MoE − control) | **+0.449 pt** |
| 95 % CI | [+0.109 ; +0.820] |
| two-sided paired image-bootstrap p | **0.0066** |
| bootstrap | B = 10,000 replicates, seed 20260618, holdout first:500 |
| mIoU (MoE / control) | 81.618 / 81.168 |
| **three Holm families** | 0.0066 (1 pair) · 0.0726 (12 pairs) · 0.0924 (15 pairs) |

Positive at the raw threshold and inside the pre-registered single-pair family, it **does not
survive the exploratory family** (0.0924 over 15 pairs) — where **no arm
of the plateau survives either** (best 0.0750). All three families are
reported together, none is chosen because it flatters. Robustness: a second independent GPU
forward gives Δ 0.450218 pt, 8.58e-04 pt apart — same verdict
(cuDNN/BF16 non-determinism, declared).

**The gate does not choose.** At the final epoch the routing is flat — minimal normalised entropy
0.99771, `part_max` 0.5030 / 0.5019 against 0.5000 for
exact equidistribution, **zero dead experts** — yet γ grows from 0.00024 to 0.02026
(×83). The mixture helps as an **average of experts**, not as a
selection. And this is *not* an artifact of the annealed Shazeer noise: over the 40 active-noise
epochs the noise-free distribution is already flat (≥ 0.99821)
and the token↔clean entropy gap stays ≤ 1.5e-03.

**The only all-rounder, first on nothing.** On the program's pre-registered versatility criterion
(36 endpoints × 13 arms), MoE-V3-CS has the **lowest maximal damage** of the plateau
(-0.53 pt, a 1.37 pt margin over the
next arm, −22.20 pt for D) and the cheapest plateau at -1.2 pt of damage
per mIoU point against -43.1 for D. Yet it is **first on none of the 36 endpoints**
(D wins 16) and ranks 5th in mean percentile (60.4);
its worst rank is 11/12 on `instances_foule_rappel`. *Good everywhere is not best
everywhere.* Seed-by-seed stability (T4) is **partial** (1st in damage on 2 of
3 seeds).

## Abstract

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

## Repository layout

| path | content |
|---|---|
| `paper.md` / `paper.pdf` | manuscript, English (16 p.) |
| `paper_fr.md` / `paper_fr.pdf` | manuscript, French (16 p.) |
| `build.sh` + `header.tex` | the exact recipe that rebuilds both PDFs (pandoc → XeLaTeX, letter, 1.7 cm margins, Liberation Serif) |
| `tables/` | T1-T8 (md + csv) and `paper3_tables.json`, the consolidation table (257 checks) |
| `figures/` | F1 13-arm Δ, F2 routing, F3 business forest, F4 versatility, F5 per-class heatmap (png + pdf) |
| `src/moe/` | `PatchMoE2D`, the attach logic, the expert initialisation, the bootstrap, the diversity audit |
| `src/losses/`, `src/metrics/`, `src/postprocessing/` | the specialists' losses, official metrics, fragment count |
| `configs/` | Hydra configs of the mixture and its matched control |
| `scripts/p3_moe_tables.py` / `p3_moe_figures.py` | regenerate every table and figure **without GPU** from the asserted source |
| `scripts/p3_check_numbers.py` | **111 checks** binding each manuscript number to its artifact (counts recomputed) |
| `scripts/p3_check_manuscrit_gate.py` | re-runs the DOI gate's regexes on the published manuscripts |
| `scripts/train_moe_v3cs.py` / `moe_eval_harness.py` | train the mixture and run the primary bootstrap |
| `tests/` | MoE attach, patch routing, seed pairing, bootstrap, official mIoU |
| `analysis/` | the full provenance chain of the published numbers |
| `analysis/EXCLUDED.md` | heavy artifacts **not** shipped, named one by one with their regeneration path |

## Reproducing

```bash
bash build.sh                               # rebuild both PDFs
python3 scripts/p3_moe_tables.py            # regenerate T1-T8 (no GPU, 257 checks)
python3 scripts/p3_moe_figures.py           # regenerate F1-F5 (no GPU, 9 self-checks)
python3 scripts/p3_check_numbers.py         # 111 manuscript<->artifact checks
python3 scripts/p3_check_manuscrit_gate.py papers/paper3   # DOI-gate regexes on the manuscripts
```

Regenerating the tables and figures needs **no GPU** (they read the asserted artifacts of
`analysis/`). Regenerating the *arm itself* needs the four experts (4 × 160 epochs) plus 80 epochs
of mixture at 1024×2048 on one RTX PRO 6000 96 GB (≈ 17 h/seed) — see
`tables/T1_architecture_recette_cout.md`.

**Reproducibility statement (declared, not overclaimed).** cuDNN benchmark is left on
(`deterministic: false`, BF16), so bit-exact GPU reruns are *not* claimed. What is verified: same
artifacts → same statistics bit-for-bit (257 blocking checks), and independent forwards of the same
checkpoints → agreement within 8.58e-04 pt on the primary Δ.

## Companion papers of the program

| paper | dataset | DOI |
|---|---|---|
| Distance-map auxiliary regression | Cityscapes | [10.5281/zenodo.21006236](https://doi.org/10.5281/zenodo.21006236) |
| Boundary-loss ablation (source of the paired arm B) | Cityscapes | [10.5281/zenodo.21006393](https://doi.org/10.5281/zenodo.21006393) |
| Blob loss alone (arm G, expert 3 here) | Cityscapes | [10.5281/zenodo.23083560](https://doi.org/10.5281/zenodo.23083560) |
| **This paper** — expert-initialised mixture (MoE-V3-CS) | Cityscapes | see `CITATION.cff` |
| Expert-initialised mixture-of-experts (anteriority) | BRATS 2023 | [10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668) |
| Connected-component consensus | BRATS 2023 | [10.5281/zenodo.22904810](https://doi.org/10.5281/zenodo.22904810) |

## Licences

Manuscripts, tables and figures: **CC-BY-4.0** (the licence of the Zenodo deposit). Code
(`src/`, `scripts/`, `tests/`, `configs/`): **MIT** ([`LICENSE`](LICENSE)). Cityscapes data
remains under its own licence.

## Citation

See [`CITATION.cff`](CITATION.cff). Authors: Guillaume Cassez, Stanislas Larnier, independent researchers (ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) for Guillaume Cassez).
