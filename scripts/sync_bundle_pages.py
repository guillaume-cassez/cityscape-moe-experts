#!/usr/bin/env python3
"""sync_bundle_pages.py — le nombre de pages écrit dans les README est MESURÉ, pas recopié.

Défaut mesuré le 2026-10-04 avant dépôt :

    pdfinfo papers/paper3/publish/repo/paper.pdf     -> Pages: 16
    pdfinfo papers/paper3/publish/repo/paper_fr.pdf  -> Pages: 16
    pdfinfo papers/paper4/publish/repo/paper.pdf     -> Pages: 15
    pdfinfo papers/paper4/publish/repo/paper_fr.pdf  -> Pages: 16

    README/README_fr du paper4 annonçaient « 15 p. » pour les DEUX langues  -> FR faux
    CHANGELOG du paper3 annonçait « Pagination : 16 -> 15 pages (EN comme FR) » -> faux

Cause : les comptes étaient écrits **en dur** dans la prose du stager
(`scripts/p3_stage_bundle.py`, 10 occurrences) alors que la pagination dépend du
moteur XeLaTeX, de la largeur des colonnes de tableaux et de la hauteur des figures
— trois choses qui ont bougé le 2026-10-03/04. Un README public qui annonce la
mauvaise pagination est un défaut de même nature que l'étiquette de famille de Holm
fautive qui a valu l'erratum v1.1.0 du paper4 (le gate anti-régression de ce bundle
interdit la literal fautive jusque dans ses docstrings — d'où cette périphrase).

Ce script rend le compte à sa source : il lit `pdfinfo` sur les PDF **déjà buildés**
du bundle et réécrit chaque mention dans README.md / README_fr.md. `--check` ne
réécrit rien et sort 1 au premier écart : c'est le mode gate, appelé par
`build.sh` après le build et par `p3/p4_stage_bundle.py --verify-only`.

Usage :
    python3 scripts/sync_bundle_pages.py papers/paper3/publish/repo            # réécrit
    python3 scripts/sync_bundle_pages.py papers/paper3/publish/repo --check    # gate
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (motif, langue, gabarit de remplacement). La langue décide quel PDF fait autorité.
MOTIFS = [
    (re.compile(r"\(EN, \d+ p\.\)"), "en", "(EN, {n} p.)"),
    (re.compile(r"\(FR, \d+ p\.\)"), "fr", "(FR, {n} p.)"),
    (re.compile(r"English \(\d+ p\.\)"), "en", "English ({n} p.)"),
    (re.compile(r"French \(\d+ p\.\)"), "fr", "French ({n} p.)"),
    (re.compile(r"anglais \(\d+ p\.\)"), "en", "anglais ({n} p.)"),
    (re.compile(r"français \(\d+ p\.\)"), "fr", "français ({n} p.)"),
]

READMES = ("README.md", "README_fr.md")


def pages(pdf: Path) -> int:
    """Nombre de pages lu par pdfinfo — la seule autorité (jamais un log LaTeX)."""
    if not pdf.is_file():
        raise SystemExit(f"[FAIL] PDF absent du bundle : {pdf} — builder avant de synchroniser")
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    if not m:
        raise SystemExit(f"[FAIL] pdfinfo ne rapporte pas de `Pages:` pour {pdf}")
    return int(m.group(1))


def synchroniser(bundle: Path, ecrire: bool) -> list[str]:
    """Réécrit (ou vérifie) les mentions de pagination d'un bundle. Retourne le rapport."""
    bundle = Path(bundle)
    comptes = {"en": pages(bundle / "paper.pdf"), "fr": pages(bundle / "paper_fr.pdf")}
    rapport = [f"{bundle} : pages MESURÉES  EN={comptes['en']}  FR={comptes['fr']}"]
    for nom in READMES:
        f = bundle / nom
        if not f.is_file():
            raise SystemExit(f"[FAIL] {nom} absent du bundle {bundle}")
        avant = f.read_text(encoding="utf-8")
        apres, n_trouvees, n_fausses = avant, 0, 0
        for motif, langue, gabarit in MOTIFS:
            cible = gabarit.format(n=comptes[langue])
            for trouve in motif.findall(apres):
                n_trouvees += 1
            for m in motif.finditer(apres):
                if m.group(0) != cible:
                    n_fausses += 1
            apres = motif.sub(cible, apres)
        if n_fausses == 0:
            rapport.append(f"  [OK] {nom} : {n_trouvees} mention(s) de pagination, "
                           f"toutes conformes au PDF mesuré")
        elif ecrire:
            rapport.append(f"  [ÉCART CORRIGÉ] {nom} : {n_fausses}/{n_trouvees} mention(s) "
                           f"divergeaient du PDF, réécrites depuis pdfinfo")
            f.write_text(apres, encoding="utf-8")
        else:
            rapport.append(f"  [ÉCART] {nom} : {n_fausses}/{n_trouvees} mention(s) divergent "
                           f"du PDF mesuré — non corrigées (mode --check)")
    return rapport


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+", help="dossier(s) publish/repo")
    ap.add_argument("--check", action="store_true",
                    help="ne rien écrire ; sortir 1 si une mention diverge du PDF")
    a = ap.parse_args()
    rc = 0
    for b in a.bundles:
        lignes = synchroniser(Path(b), ecrire=not a.check)
        print("\n".join(lignes))
        if a.check and any("[ÉCART]" in l for l in lignes):
            rc = 1
    print("SYNC PAGES : exit", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
