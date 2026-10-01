# CHANGELOG

## v1.0.0 — 2026-10-01

Premier dépôt public du paper 3 du programme Cityscapes (mélange d'experts MoE-V3-CS,
initialisé depuis quatre spécialistes de loss). **Réplication sur un second dataset** du
résultat BRATS *The Gate Does Not Choose* ([DOI 10.5281/zenodo.22903668]) — aucun chiffre
BRATS n'est repris.

- Manuscrits EN (`paper.md` / `paper.pdf`, 16 pages) et FR (`paper_fr.md` / `paper_fr.pdf`,
  16 pages), 0 glyphe manquant (≥ ≪ ✅ ❌ ⊘ ↔ ⁻ mappés dans `header.tex`).
- Critère primaire pré-enregistré **positif au seuil brut et dans la famille à 1 paire** :
  Δ(MoE−contrôle) = +0.449 pt de mIoU, IC95 [+0.109 ;
  +0.820], p = 0.0066 (bootstrap apparié par image, B = 10,000,
  seed 20260618, holdout first:500). Les **trois** familles de Holm sont
  données ensemble : 0.0066 (1 paire), 0.0726 (12 paires),
  0.0924 (15 paires) — il ne survit pas à la famille exploratoire, où
  **aucun** bras du plateau ne survit (meilleur 0.0750).
- Diagnostic de routage : la porte **ne choisit pas** (entropie normalisée minimale 0,99771,
  `part_max` à 0,0030 de l'équirépartition 0,5000, 0 expert mort), et ce n'est **pas** un
  artefact du bruit de Shazeer — la preuve porte sur les 40 époques à bruit actif (écart
  token↔clean max 1,45e-03, distribution sans bruit plate à ≥ 0,99821), pas sur la tautologie
  de l'époque finale. γ croît de 0,00024 à ≈ 0,020 : le gain vient de la **moyenne d'experts**.
- Polyvalence (36 endpoints × 13 bras, critères pré-enregistrés RECALCULÉS) : **seul bras
  polyvalent** (dommage maximal −0,53 pt, marge 1,37 pt, prix −1,2 pt par point de mIoU contre
  −43,1 pour D), mais premier sur **0** des 36 endpoints (D en gagne 16) et 5ᵉ en percentile
  moyen (60,4) — « bon partout ≠ meilleur partout ». T4 (stabilité seed par seed) déclaré
  **partiel** (1ᵉʳ en dommage sur 2 seeds sur 3).
- Garde-fous : `scripts/p3_check_numbers.py` (111 checks, chaque compte
  RECALCULÉ depuis le détail de la source) et `scripts/p3_check_manuscrit_gate.py` (regex du
  gate DOI + 18 regex Forbidden du SPEC rejouées sur les manuscrits publiés), tous deux validés
  par contrôles négatifs probants.
- Artefacts lourds non embarqués, listés nommément dans `analysis/EXCLUDED.md`.
