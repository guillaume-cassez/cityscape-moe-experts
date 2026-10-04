#!/usr/bin/env bash
# Build the paper PDFs (EN + FR). Format matches the companion BRATS/Cityscapes
# reports (house style for this series): XeLaTeX via pandoc, US letter, 1.7 cm
# margins, Liberation Serif, with header.tex mapping inline Unicode superscripts
# and the ⊘ / ↔ glyphs to LaTeX (the main font lacks U+207B, U+2298, U+2194).
# Requires: pandoc + TeX Live with xelatex.
#
# 2026-10-03 : le build passe par un .tex intermédiaire post-traité
# (tables_latex_postprocess.py) : pandoc émettait chaque tableau en longtable
# (coupé d'une page à l'autre, lignes orphelines mesurées p5->p6, p6->p7, p9->p10)
# avec des colonnes à fractions égales (0.10\columnwidth pour 8 colonnes).
set -e
cd "$(dirname "$0")"
OPTS=(-s -V papersize=letter -V geometry:margin=1.7cm
      -V mainfont="Liberation Serif" -H header.tex)
POST=../../../../scripts/tables_latex_postprocess.py
if [ ! -f "$POST" ]; then
  echo "ERREUR: post-processeur de tableaux introuvable ($POST) — build refusé (fail-closed)" >&2
  exit 1
fi
# 2026-10-03 (récidive) : les manuscrits référencent leurs figures en chemin RELATIF
# (figures/…), donc le build lit les COPIES de ce dossier. Le canon papers/paperX/figures
# avait été corrigé à 19:43 (libellés hors des lignes d'IC, annotation retirée du tracé,
# légende déplacée) sans que ces copies — datées du 2026-10-02 — soient resynchronisées :
# 14 figures sur 16 étaient périmées et les PDF livrés à 21:11 embarquaient les vieilles
# versions (superpositions signalées par Guillaume). Synchronisation vérifiée md5.
SYNC=../../../../scripts/sync_paper_figures.py
FIGCANON=../../figures
if [ ! -f "$SYNC" ]; then
  echo "ERREUR: synchroniseur de figures introuvable ($SYNC) — build refusé (fail-closed)" >&2
  exit 1
fi
python3 "$SYNC" . "$FIGCANON"
for SRC in paper.md paper_fr.md; do
  OUT="${SRC%.md}"
  pandoc "$SRC" -o "$OUT.raw.tex" "${OPTS[@]}"
  # 3ᵉ argument = header.tex : la sonde XeLaTeX des largeurs mesure dans le
  # CONTEXTE du build réel (mappings Unicode, \texttt cassables).
  python3 "$POST" "$OUT.raw.tex" "$OUT.tex" header.tex
  xelatex -interaction=nonstopmode -halt-on-error "$OUT.tex" > "$OUT.xelatex.log" \
    || { echo "ERREUR xelatex sur $OUT.tex :" >&2; tail -25 "$OUT.xelatex.log" >&2; exit 1; }
  rm -f "$OUT.raw.tex"
done
# Gate de mise en page MESURÉ sur les PDF finis (boîtes de mots vs marge
# droite) : le claim « 0 overfull » n'était jamais vérifié, et des chemins
# monospace rognés au bord de page sont passés en publication le 2026-10-01.
GATE=../../../../scripts/check_pdf_layout.py
if [ ! -f "$GATE" ]; then
  echo "ERREUR: gate layout introuvable ($GATE) — build refusé (fail-closed)" >&2
  exit 1
fi
python3 "$GATE" paper.pdf paper_fr.pdf
# Gate figures MESURÉ sur le PDF fini et comparé à la SOURCE CANONIQUE (jamais à la
# copie du dossier figures/ : périmé contre périmé, l'égalité est triviale et c'est
# exactement ce qui a laissé passer les 14 figures périmées du 2026-10-03).
EMBGATE=../../../../scripts/check_figures_embedded.py
if [ ! -f "$EMBGATE" ]; then
  echo "ERREUR: gate figures embarquées introuvable ($EMBGATE) — build refusé (fail-closed)" >&2
  exit 1
fi
python3 "$EMBGATE" paper.pdf paper.md "$FIGCANON"
python3 "$EMBGATE" paper_fr.pdf paper_fr.md "$FIGCANON"
# Gate PAGINATION : le nombre de pages annoncé dans les README est MESURÉ par pdfinfo sur
# les PDF qui viennent d'être buildés, jamais recopié. Défaut trouvé le 2026-10-04 : le
# README du paper4 annonçait « 15 p. » pour les deux langues alors que paper_fr.pdf en
# faisait 16, et le CHANGELOG du paper3 annonçait « 16 -> 15 pages » alors que les deux
# PDF en font 16 — les comptes étaient écrits en dur dans la prose du stager.
PAGES=../../../../scripts/sync_bundle_pages.py
if [ ! -f "$PAGES" ]; then
  echo "ERREUR: synchroniseur de pagination introuvable ($PAGES) — build refusé (fail-closed)" >&2
  exit 1
fi
python3 "$PAGES" .
echo "built paper.pdf + paper_fr.pdf (letter, 1.7cm margins, Liberation Serif, tables non cassables, figures du canon)"
