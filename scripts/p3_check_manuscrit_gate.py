#!/usr/bin/env python3
"""p3_check_manuscrit_gate.py — rejoue le gate DOI global sur les manuscrits REELLEMENT publiés.

Homologue de ``scripts/p4_check_manuscrit_gate.py`` pour le paper3 (MoE-V3-CS). Même cause
mesurée le 2026-0061 sur paper4 : le gate qui bloque un vrai dépôt Zenodo est
``~/.jarvis/tools/prepublish_check.py``, chargé par ``backend.publications.prepublish.ensure_publishable``
(fail-closed). Son ``SKIP_DIRS`` contient ``'publish'``, donc pour la disposition
``papers/paperN/publish/repo/`` de ce dépôt il ne lit JAMAIS les manuscrits publiés : trois de
ses cinq checks (marqueurs d'incomplétude, métrique primaire pré-enregistrée, corpus) sont
aveugles au texte qui part en DOI.

Ce script ferme l'angle mort SANS toucher au gate global (outil partagé, hors de ce dépôt) :
il applique aux manuscrits du bundle les MÊMES regex que le gate global, recopiées à l'identique
de ``~/.jarvis/tools/prepublish_check.py`` lignes 99-113, plus la liste ``## Forbidden`` du SPEC
du papier, lue dynamiquement pour rester à une seule source de vérité.

Usage :
    python3 scripts/p3_check_manuscrit_gate.py            # paper3 (défaut)
    python3 scripts/p3_check_manuscrit_gate.py papers/paper4
Sortie : une ligne par manuscrit + le total. Exit 1 à la première classe de défaut.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- regex recopiées à l'identique du gate global (lignes 99-113) -------------
# (4) unresolved authoring markers — case-SENSITIVE (uppercase = intentional
#     annotation; "first draft" in a sentence must not trip it).
RE_MARKERS = re.compile(r"\b(TODO|FIXME|XXX|DRAFT)\b")
# (5) primary-metric pre-registration = a "primary" notion AND a
#     "pre-specified / pre-registered" notion co-occurring.
RE_PRIMARY = re.compile(
    r"primary\s+(metric|endpoint|ranking)|m[ée]trique\s+primaire|"
    r"crit[èe]re\s+primaire|primary[-\s]?endpoint",
    re.IGNORECASE,
)
RE_PREREG = re.compile(
    r"pre[-\s]?regist|pre[-\s]?specif|pr[ée][-\s]?enregist|pr[ée][-\s]?sp[ée]cif|"
    r"pre[-\s]?registered|definition[-\s]?of[-\s]?done",
    re.IGNORECASE,
)

MANUSCRIPTS = ("paper.md", "paper_fr.md")


def forbidden_from_spec(spec_path: Path) -> list[tuple[str, re.Pattern]]:
    """Extrait les regex de la section '## Forbidden' du SPEC (source de vérité)."""
    if not spec_path.exists():
        return []
    txt = spec_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^##\s+Forbidden\s*$(.*?)(?=^##\s|\Z)", txt, re.M | re.S)
    if not m:
        return []
    out: list[tuple[str, re.Pattern]] = []
    for raw in re.findall(r"^- `([^`]+)`", m.group(1), re.M):
        try:
            out.append((raw, re.compile(raw, re.IGNORECASE)))
        except re.error as exc:  # une regex invalide du SPEC ne doit pas passer inaperçue
            print(f"[FAIL] regex Forbidden invalide dans {spec_path.name}: {raw!r} ({exc})")
            sys.exit(1)
    return out


def main(argv: list[str]) -> int:
    paper_dir = Path(argv[1]) if len(argv) > 1 else ROOT / "papers" / "paper3"
    paper_dir = paper_dir if paper_dir.is_absolute() else ROOT / paper_dir
    bundle = paper_dir / "publish" / "repo"
    spec = paper_dir / "SPEC.md"

    if not bundle.is_dir():
        print(f"[FAIL] bundle absent : {bundle}")
        return 1

    forb = forbidden_from_spec(spec)
    print(f"Bundle : {bundle.relative_to(ROOT) if bundle.is_relative_to(ROOT) else bundle}")
    print(f"SPEC   : {spec.name if spec.exists() else 'ABSENT'} — "
          f"{len(forb)} regex Forbidden lues dynamiquement")

    failures = 0
    primary_total = 0
    for name in MANUSCRIPTS:
        path = bundle / name
        if not path.exists():
            print(f"[FAIL] {name} : manquant du bundle")
            failures += 1
            continue
        txt = path.read_text(encoding="utf-8", errors="replace")

        marks = RE_MARKERS.findall(txt)
        n_primary = len(RE_PRIMARY.findall(txt))
        n_prereg = len(RE_PREREG.findall(txt))
        primary_total += n_primary
        hits = [(raw, len(rx.findall(txt))) for raw, rx in forb if rx.search(txt)]

        statut = "PASS"
        if marks:
            statut = "FAIL"
            failures += len(marks)
            print(f"[FAIL] {name} : {len(marks)} marqueur(s) d'incomplétude {sorted(set(marks))}")
        if hits:
            statut = "FAIL"
            for raw, n in hits:
                failures += n
                print(f"[FAIL] {name} : regex Forbidden {raw!r} -> {n} occurrence(s)")
        print(f"[{statut}] {name} : {len(txt)} car. | marqueurs {len(marks)} | "
              f"primary {n_primary} | pre-reg {n_prereg} | Forbidden {sum(n for _, n in hits)}")

    # Le critère primaire pré-enregistré doit être nommé dans la langue du manuscrit qui
    # part en DOI (EN) ; le gate global exige la co-occurrence primary + pre-registration.
    en = bundle / "paper.md"
    if en.exists():
        txt = en.read_text(encoding="utf-8", errors="replace")
        if not (RE_PRIMARY.search(txt) and RE_PREREG.search(txt)):
            print("[FAIL] primary_metric_preregistered : le manuscrit EN ne co-déclare pas "
                  "'primary endpoint/metric' ET un marqueur de pré-enregistrement")
            failures += 1
        if not RE_PREREG.search(txt):
            failures += 1

    if failures:
        print(f"\nÉCHEC — {failures} défaut(s) sur les manuscrits du bundle.")
        return 1
    print(f"\nOK — manuscrits du bundle conformes au gate global "
          f"({len(MANUSCRIPTS)} fichiers, {primary_total} déclarations du critère primaire, "
          f"0 marqueur, 0 regex Forbidden).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
