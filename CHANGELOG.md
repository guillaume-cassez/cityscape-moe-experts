# CHANGELOG

## v1.1.1 — 2026-10-05

**Métadonnée seule : la description du record passe en HTML. Aucun fichier, aucun chiffre,
aucun auteur ne change par rapport à la v1.1.0.**

- **Défaut, mesuré sur l'API publique avant de corriger.** Zenodo rend le champ
  `description` comme du **HTML**. Le payload envoyait l'abstract Markdown du manuscrit tel
  quel : `GET /api/records/23146586` rendait donc **4 499 caractères contenant 50 « ** »
  littéraux et 0 balise**, et la page publique affichait `**MoE-V3-CS**`,
  `**mixture-of-experts**`, `**initialised bit-for-bit**` — le premier texte que lit un
  visiteur du record, avant les fichiers. Le record BRATS de référence que ces mêmes
  scripts citent comme modèle d'un dépôt correct (22903668, v0.13) est, lui, balisé :
  **0 « ** », 2 `<p>`**. Les deux records Cityscapes de ce programme (23090082 v1.0.0 et
  23085640/23146634 pour le papier 4) portent le même défaut : il vient de `abstract()`,
  qui renvoie le texte brut du manuscrit, posé tel quel dans `payload()`.
- **Second défaut du même champ, mesuré en même temps.** L'extraction de l'abstract laisse
  traîner le séparateur `---` qui ferme la section dans le manuscrit : la description
  publique se terminait par « …and regeneration scripts. **---** ». Retiré à la source.
- **Correctif à la cause, pas au symptôme.** Nouveau module `zenodo_description.py` :
  conversion par **pandoc** (déjà l'outil de la chaîne de build de ces manuscrits), avec
  `-smart` désactivé — mesuré, l'extension smart réécrivait les apostrophes ASCII en
  U+2019 (« program's » → « program’s »), une substitution de texte non demandée dans un
  record public. Repli sans dépendance si pandoc manque. Puis **assertion du résultat
  AVANT tout appel réseau** : 0 « ** » brut, 0 astérisque d'italique brut, 0 backtick, au
  moins une balise `<p>`, pas de `---` final, et **la liste des mots du texte rendu est
  identique à celle de la source** (713 mots pour cet abstract) — une conversion qui
  perdrait du contenu échoue au lieu de publier. Trois contrôles négatifs probants joués
  (Markdown non converti, séparateur conservé, mot perdu : chacun détecté, exit 1 ; cas
  sain accepté).
- **Pourquoi une nouvelle version plutôt que laisser courir.** Un record Zenodo publié est
  immuable : sa description ne se corrige qu'en déposant une version. Le précédent de ce
  dépôt va dans ce sens — l'erratum v1.1.0 du papier 4 a été frappé pour une simple
  ÉTIQUETTE de famille de Holm, bien moins visible que 50 astérisques dans le résumé. Le
  DOI de concept **10.5281/zenodo.23090081 ne change pas** et résout désormais sur cette
  version : aucune citation n'est cassée. La v1.1.0 reste en ligne, immuable, avec son
  défaut — cette entrée le dit plutôt que de le laisser découvrir.
- Rendu mesuré après conversion : **4 958 caractères** d'HTML, 1 `<p>`, **25 `<strong>`**,
  3 `<em>`, 10 `<code>`, **0 « ** »**, les 713 mots de la source tous présents.
  Pagination inchangée et relue par `pdfinfo` : **16 pages EN, 16 pages FR**.
  Les 3 fichiers déposés sont **les mêmes octets** qu'en v1.1.0 (paper.pdf 749 211 o
  md5 `9af4450e…`, paper_fr.pdf 760 058 o md5 `a06f79a2…`).

## v1.1.0 — 2026-10-04

**Premier dépôt public depuis la v1.0.0** (record Zenodo 23090082, 2026-10-01). Cette
version rassemble les deux étapes correctives qui n'avaient été que stagées localement
(v1.0.1 du 2026-10-03, v1.0.2 du 2026-10-04) et le correctif de légendes du 2026-10-04
20h34. **Aucun chiffre du papier ne change** : critère primaire, trois familles de Holm,
diagnostic de routage et verdict de polyvalence sont identiques à la v1.0.0 — la seule
modification de MÉTADONNÉE est la liste des auteurs.

- **Auteurs — Stanislas Larnier en 2ᵉ position dans le record Zenodo lui-même.** Les
  versions publiées avant le 2026-10-04 (v1.0.0, record 23090082) portent **Guillaume
  Cassez seul** : mesuré sur l'API publique le 2026-10-04 (`creators = ['Cassez,
  Guillaume']`), alors que `paper.yaml` en portait déjà deux. Cause : la liste était
  écrite **en dur** dans le stager et dans le script de dépôt. Elle est désormais lue
  depuis `papers/paper3/paper.yaml` par le module `paper_authors.py` du dépôt de
  recherche, source unique du payload Zenodo, de `CITATION.cff`, de `.zenodo.json` et des
  deux README — plus aucune seconde copie ne peut diverger. Ce module n'est PAS embarqué
  ici : il lit `paper.yaml`, qui contient une adresse e-mail personnelle et ne part donc
  pas dans un dépôt public. L'adresse e-mail du 2ᵉ auteur, présente
  dans `paper.yaml`, n'est rendue nulle part (la dataclass `Auteur` n'a pas de champ
  `email`) : une donnée personnelle n'a pas à partir dans un record indexé par OpenAIRE.
  Un record publié étant immuable, les versions antérieures restent en ligne à un seul
  auteur et cette entrée le dit explicitement plutôt que de le laisser découvrir.
- **Légendes des figures HORS des données** (récidive, 6ᵉ passage sur ce sujet). Mesure
  AVANT sur le PDF livré (`paper_fr.pdf` md5 `58550fd0…`, zoom 300 dpi) : F2 portait
  **six** légendes `loc="best"` à l'intérieur de six panneaux de 2,3 in, les courbes des
  seeds 123/456 passant **sous** le cadre semi-transparent ; F3 portait une légende
  haut-gauche masquant le tronçon gauche de la barre d'IC de `boundary_f1_3px_pieds`.
  Correctif de cause, pas de dosage : F2 passe à une légende UNIQUE de figure en bas,
  hors de tout axe (`fig.legend`, `frameon=False`, bande réservée par
  `tight_layout(rect=(0, 0.045, …))`) ; F3 réserve une bande vide de 2,2 unités
  AU-DESSUS de `mIoU` (`ax.set_ylim(-0.8, len+1.2)`) où le cadre opaque ne recouvre plus
  aucune donnée ; F5 passe son cadre de légende à `framealpha=1.0`.
- **Polices imprimées remontées** : 6,8 → 7,3 pt (ticks et légendes), 7,2 → 7,6,
  7,8 → 8,2, 8,4 → 8,8. Comme les figures sont dessinées à la largeur imprimée depuis
  v1.0.2 (échelle 1,000), ces valeurs SONT les tailles sur le papier : le minimum imprimé
  passe de 6,69 pt à **7,3 pt**, toujours au-dessus du seuil du gate (6,5 pt).
- **Fin de la légende orpheline en page 9.** Le paragraphe italique descriptif de F3 est
  FONDUE dans la légende de la figure : figure et légende complète voyagent ensemble au
  lieu que le flotteur parte page 8 en laissant son texte page 9. Les backticks y sont
  remplacés par de l'italique — un `\texttt` dans l'argument mobile de `\caption` cassait
  XeLaTeX (slash actif de `header.tex`).
- **Pagination mesurée, plus jamais recopiée.** Les PDF de cette version font
  **16 pages (EN)** et **16 pages (FR)**, lus par `pdfinfo`. Deux claims
  faux ont été trouvés en mesurant : le CHANGELOG v1.0.2 annonçait « 16 → 15 pages (EN
  comme FR) » alors que les deux PDF font 16 pages, et les README du paper4
  voisin annonçaient 15 p. pour un `paper_fr.pdf` qui en fait 16. Cause : les comptes
  étaient écrits en dur dans la prose du stager alors que la pagination dépend du moteur
  XeLaTeX, de la largeur des colonnes de tableaux et de la hauteur des figures.
  `scripts/sync_bundle_pages.py` réécrit désormais chaque mention depuis `pdfinfo`, est
  appelé par `build.sh` après le build, et son mode `--check` sort 1 au premier écart
  (contrôle négatif probant : un « 12 p. » injecté dans le README EN est détecté, puis
  la restauration vérifiée par md5).
- Gates rejoués sur les fichiers de CETTE version : layout **exit 0** sur les 4 PDF
  (paper3 EN 16 p. 7211 mots, FR 16 p. 7818 mots ; 0 débordement à droite,
  0 hors papier en bas), figures embarquées **5/5 pixel-égales au canon** dans chaque
  langue, nombres **111/111**, manuscrits
  **PASS** (0 marqueur, 0 regex Forbidden, auteurs 2 ✓), encre matplotlib **19/19**.

## v1.0.2 — 2026-10-04

**Figures à taille réelle + deux auteurs : aucun chiffre changé.**
Statut : **rendu public par la v1.1.0 du 2026-10-04** — cette étape corrective
n'avait été que stagée localement, en attente du feu vert de Guillaume pour un acte
public engageant un tiers ; ce feu vert est venu le 2026-10-04 23h43. Le dépôt v1.0.0
(record 23090082) porte **Guillaume Cassez seul** : un record publié est immuable, il
reste donc en ligne tel quel, et la liste à deux auteurs ci-dessous n'est publique
qu'à partir de la v1.1.0.

- **Taille du texte des figures — cause racine corrigée (récidive, 5ᵉ passage).**
  Guillaume : « du texte écrit trop petit en haut de la page 11 du papier 3 ».
  Mesure AVANT sur le PDF livré (`paper_fr.pdf` md5 `c6b4ed03…`) : la figure du haut
  de la page 11 est `F5_perclass_fr.png`, 2635 px à 170 dpi = 15,50 in naturelles,
  placée par `\includegraphics` à 515,6 pt = 7,16 in → **échelle 0,462** ; ses
  polices 8,4-10,5 pt du code s'imprimaient à **3,9-4,9 pt** sur un corps de texte
  de 9,96 pt. Même cause sur les 5 figures : échelles mesurées **0,426 (F4) à 0,710
  (F1)**, **275 textes imprimés sous 6,5 pt**, minimum **3,38 pt**. Pourquoi cinq
  relectures ne l'ont pas vu : les gates comparaient des pixels *display* — une
  figure réduite uniformément ne se chevauche pas davantage, elle devient illisible.
  Aucune mesure ne reliait le `fontsize` du code à sa taille **sur le papier**.
  Correctif : les 5 figures sont désormais dessinées **À LA LARGEUR IMPRIMÉE**
  (7,16 in = `\linewidth`), donc à l'échelle **1,000** — chaque `fontsize` du code
  est la taille réelle imprimée. Mesure APRÈS sur les PDF reconstruits : échelle
  **1,000 sur 4 figures sur 5** (F4 à 0,984, son encre + le pad de `savefig` faisant
  7,28 in), taille imprimée minimale **6,69 pt** (6,8 × 0,984), **0 texte sous
  6,5 pt**. Gate ajouté dans `scripts/fig_overlap_gate.py` :
  `verifier_taille_imprimee()` re-mesure l'échelle de placement effective (tight
  bbox **+ 2 × `pad_inches`**, ce que le PNG embarque vraiment) et bloque sous
  `POLICE_MIN_IMPRIMEE_PT` = 6,5 ; `verifier_emprise_imprimee()` bloque une figure
  plus haute que le bloc de texte.
- **F4 (polyvalence) restructurée en 2 lignes** — panneau A en pleine largeur, B et C
  dessous. Mesuré : à 3 panneaux de 2,1 in, `constrained_layout` s'effondrait
  (« axes sizes collapsed to zero », axes de 44 × 19 px, 11 tick labels empilés sur
  20 px, 6 collisions d'encre) ; `tight_layout` ne gère pas un panneau qui enjambe
  des colonnes et laissait les tick labels sortir du canvas (x0 = −19,3 px sur 716).
  Les libellés de bras perdent leur parenthèse explicative (`label_moyen`) : elle
  était une redite du tableau T5 et de F1 (page 5, qui porte les 13 libellés
  complets). Empreinte imprimée 2,88 in → 4,78 in de haut.
- **Recadrage des titres et libellés d'axes sur la largeur de LEURS axes**
  (`titre_cadre` corrigé, `etiquette_cadree` et `recadrer_panneaux` ajoutés). La
  limite précédente était la plus grande largeur centrée tenant dans le *canvas* :
  sur un panneau du MILIEU elle valait la largeur du canvas entier (696 px sur 716)
  et le titre n'était jamais recoupé — il débordait sur les panneaux voisins
  (chevauchement mesuré 112 × 21 px).
- **Auteurs — Stanislas Larnier en 2ᵉ position**, sur instruction de Guillaume du
  2026-10-04, aligné sur les quatre papiers BRATS du programme (DOI
  10.5281/zenodo.22903668, .22904810, .22906447 et le rapport distmap) dont il est
  déjà co-auteur et dont ce papier est une réplication déclarée. La décision inverse
  (« papier mono-auteur, pas de gift authorship », 2026-09-30) est **rapportée et
  datée** dans `paper.yaml`, pas effacée. Contreparties d'intégrité : section
  **Contributions des auteurs** ajoutée après §8 (mêmes rôles que sur les papiers
  BRATS — G.C. conception, entraînements, évaluations/analyses, rédaction ; S.L.
  questions de recherche, conseil méthodologique, relectures), mention en page de
  titre de la DATE d'établissement de la liste et du fait que les dépôts Zenodo
  antérieurs portent Guillaume seul, et contrôle **positif** dans
  `scripts/p3_check_manuscrit_gate.py` (le bloc d'auteurs, la section de
  contributions et l'identifiant HAL doivent être présents dans les deux langues).
  La regex Forbidden `Auteurs?\s*:.*Larnier` est retirée des deux SPEC avec la
  décision datée à sa place. `CITATION.cff`, `.zenodo.json`, `README.md` et
  `README_fr.md` du bundle portent les deux auteurs.
- Pagination : **16 pages EN et 16 pages FR**, mesurées par `pdfinfo`. Cette entrée
  annonçait à l'origine « 16 → 15 pages (EN comme FR) » : c'était **faux**, et corrigé
  ici le 2026-10-04 par la mesure — la v1.0.0 publique (record 23090082, re-téléchargée
  sans jeton, md5 `bca2905d…` EN / `ee1f4b92…` FR) fait déjà 16 + 16 pages, et les PDF
  de cette version aussi. Le compte était écrit **en dur** dans la prose du stager ;
  il est désormais lu sur le PDF par `scripts/sync_bundle_pages.py` (voir v1.1.0).
  Les tableaux respirent sans replis inutiles pendant que les figures gagnent en
  hauteur : le solde est **nul** sur ce papier.
- Gates rejoués sur les PDF finis : layout **exit 0** (7819 mots FR, 7212 EN, 0 à
  droite, 0 en bas), tableaux **exit 0** (0 longtable, 11 `table[H]`, 0 en-tête
  répété, 0 cellule IC wrappée, 0 ligne de tableau dédoublée bloquante), figures
  embarquées **5/5 pixel-égales au canon** dans les deux langues, nombres
  **111/111**, manuscrits **PASS** (0 marqueur, 0 regex Forbidden, auteurs 2 ✓).

## v1.0.1 — 2026-10-03

**Mise en page et figures : aucun chiffre changé.** Relecture visuelle du PDF par
Guillaume (tableaux coupés, colonnes mal réparties, libellés sur les courbes).
Statut : **rendu public par la v1.1.0 du 2026-10-04** — étape corrective stagée
localement le 2026-10-03, jamais déposée seule.

- Correctif de build (21:35) : les corrections de figures ci-dessous n'étaient PAS
  dans les PDF construits à 19:59. Les manuscrits référencent `figures/…` en chemin
  relatif, donc le build lit les copies de `publish/repo/figures/` — restées au
  2026-10-02, soit 4 figures FR sur 5 périmées (mesure : raster p12 embarqué
  3847×1143 contre 3668×1141 au canon ; F1/F2/F3 avaient des dimensions IDENTIQUES
  entre copie périmée et canon, seul le pixel les sépare). Désormais `build.sh`
  synchronise le canon (`scripts/sync_paper_figures.py`, copie vérifiée md5,
  fail-closed) puis le gate `scripts/check_figures_embedded.py` compare chaque
  raster du PDF FINI à la figure canonique — avant : 4/5 périmées, après : 5/5
  pixel-égales (EN comme FR). Le contrôle de dimensions du stager ne pouvait pas le
  voir ; un contrôle canon ↔ bundle y a été ajouté.

- Tableaux : le build passe par `scripts/tables_latex_postprocess.py` (pandoc → .tex
  post-traité → xelatex). Mesuré avant : la table maîtresse 13 bras était coupée
  p5→p6, la table de routage p6→p7 (ligne γ orpheline), « ce que le mélange capte »
  p9→p10 (ligne orpheline), et pandoc répartissait les colonnes à fractions égales
  (minipages 0.10\columnwidth pour 8 colonnes). Après : chaque table tient sur une
  page (`table[H]` + `\needspace`), largeurs calculées sur le contenu (water-filling
  sur chasse estimée), gate layout toujours 0 débordement (7 708 mots FR lus).
- Figures : libellés de F1 déplacés hors des intervalles de confiance (à droite du
  cap haut) ; annotation « équirépartition » de F2 retirée du tracé (valeur déjà au
  titre du panneau) ; légende de F3 passée en haut à gauche (elle recouvrait l'IC de
  IoU_rider) ; F4 reconstruit (suptitle sans `y=` forcé sous constrained_layout,
  libellés du panneau prix à gauche des bouts de barre, blancs dans les barres
  longues) ; alt-text F4 « 4 panneaux » → 3 (la figure en a 3).
- Gates rejoués : layout 0 débordement, `p3_check_numbers` 111/111, gate manuscrit PASS.

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
