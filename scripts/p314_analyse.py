#!/usr/bin/env python3
"""
P3.14 — Analyse + figures + tables du Paper 3 (MoE-V3-CS).

Deux volets, CPU-only (E-cores) — ne mord PAS sur la fenêtre GPU :

(A) MASTER TABLE Δ appariés : empile les confusions par-image de TOUS les bras disponibles
    (experts gelés A/B/C/D/Cp/Dp/G + consensus veto + contrôle-sans-MoE + MoE-V3-CS dès que
    leurs .cm.npy existent), bootstrap APPARIÉ (mêmes 500 positions, seeds moyennés dans
    chaque réplicat, B=10000, p bilatéral + Holm) via src.moe.bootstrap. Références au choix
    (--ref, défaut = contrôle-sans-MoE s'il existe, sinon B baseline). Réutilisable : quand
    P3.11 (contrôle) et P3.12 (MoE) auront écrit leurs cm, la table s'étend toute seule.

(B) COURBES DE ROUTAGE + VERDICT thèse « la porte ne choisit pas » : parse les JSONL
    results/moe_v3_cs/routing_<run>.jsonl (part_max, entropy_norm, entropy_token,
    gamma_absmean, frac par expert, noise_std), figures PNG par run + agrégat, et verdict
    chiffré vs les bornes BRATS V3 (part_max 0.507–0.583, entropy_norm 0.892–0.995,
    0 expert mort). Si seul le SMOKE (1 époque) existe, l'agrégat est écrit et la courbe
    sautée (dégénérée) — le script reste exécutable de bout en bout.

Sorties : results/moe_v3_cs/p314/{master_table.csv, master_table.md, routing_diagnostic.json,
         routing_<run>.png, routing_agregat.png}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ser/Bureau/City_Scape")
sys.path.insert(0, str(REPO))
from src.moe.bootstrap import (  # noqa: E402
    paired_bootstrap, load_val_order, miou_from_cm, DEFAULT_B, DEFAULT_BOOTSTRAP_SEED,
)

SEEDS = (42, 123, 456)
NC = 19
# Bornes BRATS V3 (NOTE_DECISION_V3, papier 3) pour le verdict « la porte ne choisit pas »
BRATS_V3 = {"part_max": (0.507, 0.583), "entropy_norm": (0.892, 0.995), "dead_experts": 0}

# (dir, suffix) par famille de bras — les deux conventions de nommage coexistent
ARM_SOURCES = [
    (REPO / "results/perimage_cm", "__epoch_160.cm.npy"),   # experts gelés + consensus fused
    (REPO / "results/moe_v3_cs/harness", ".cm.npy"),        # G + (futur) dense/moe via harness
]
# Noms de bras « contrôle » / « MoE » attendus pour choisir la référence par défaut
CONTROL_HINTS = ("control_nomoe", "controle", "control")
MOE_HINTS = ("moe_v3cs", "v3cs", "moe")


def discover_arms(extra_dirs=()) -> dict:
    """{arm: {seed: path}} pour tous les bras ayant les 3 seeds. Auto-détecté.

    `extra_dirs` : répertoires SUPPLÉMENTAIRES scannés avec le suffixe harness
    (`.cm.npy`) — self-tests/dry-runs sans écrire dans les répertoires réels.
    """
    found = {}
    sources = list(ARM_SOURCES) + [(Path(d), ".cm.npy") for d in extra_dirs]
    for d, suffix in sources:
        if not d.exists():
            continue
        for p in sorted(d.glob(f"*{suffix}")):
            stem = p.name[: -len(suffix)]          # ex: B_seed42  ou  G_seed42
            if "_seed" not in stem:
                continue
            arm, _, sseed = stem.rpartition("_seed")
            if not arm or not sseed.isdigit():
                continue
            found.setdefault(arm, {})[int(sseed)] = p
    # ne garder que les bras COMPLETS (3 seeds) pour l'appariement
    complete = {}
    incomplete = {}
    for arm, per_seed in found.items():
        if all(s in per_seed for s in SEEDS):
            complete[arm] = {s: per_seed[s] for s in SEEDS}
        else:
            incomplete[arm] = sorted(per_seed)
    return complete, incomplete


def load_stack(per_seed: dict) -> np.ndarray:
    """(S, N, 19, 19) empilé dans l'ordre fixe des seeds ; vérifie N cohérent."""
    arrs = [np.load(per_seed[s]) for s in SEEDS]
    n = arrs[0].shape[0]
    for a, s in zip(arrs, SEEDS):
        if a.shape != (n, NC, NC):
            raise ValueError(f"seed {s}: shape {a.shape} != {(n, NC, NC)}")
    return np.stack(arrs, 0)


def pick_ref(arms, want: str | None) -> str:
    if want and want in arms:
        return want
    for hint in CONTROL_HINTS:
        for a in arms:
            if hint in a.lower():
                return a
    if "B" in arms:
        return "B"
    return sorted(arms)[0]


def build_master_table(arms_stacks: dict, ref: str, B: int, seed: int, out_dir: Path):
    pairs = [(a, ref) for a in arms_stacks if a != ref]
    table = paired_bootstrap(arms_stacks, B=B, rng_seed=seed, pairs=pairs)
    rows = []
    for a in arms_stacks:
        key = f"{a}_vs_{ref}"
        pw = table["pairwise"].get(key)
        rows.append({
            "arm": a,
            "miou_point": table["point"][a],
            "miou_ci_lo": table["ci95"][a][0],
            "miou_ci_hi": table["ci95"][a][1],
            "delta_vs_ref": (pw["delta"] if pw else 0.0),
            "delta_ci_lo": (pw["ci95"][0] if pw else None),
            "delta_ci_hi": (pw["ci95"][1] if pw else None),
            "p_two_sided": (pw["p_two_sided"] if pw else None),
            "p_holm": (pw["p_holm"] if pw else None),
            "significant": (pw["significant"] if pw else False),
        })
    # trier par Δ décroissant (le ref en dernier, delta=0)
    rows.sort(key=lambda r: (r["delta_vs_ref"] if r["arm"] != ref else -9), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "master_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # markdown (unités en %, 2 déc.)
    md = [f"# P3.14 — Master table Δ appariés (réf = **{ref}**)", "",
          f"Bootstrap apparié : n={table['n_images']} images (mêmes positions), "
          f"B={B}, seeds {list(SEEDS)} moyennés dans chaque réplicat, p bilatéral + Holm.", "",
          "| Rang | Bras | mIoU % [IC95] | Δ vs réf (pt) [IC95] | p_holm | Signif |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lo, hi = r["miou_ci_lo"] * 100, r["miou_ci_hi"] * 100
        if r["arm"] == ref:
            md.append(f"| {i} | **{r['arm']}** (réf) | {r['miou_point']*100:.2f} "
                      f"[{lo:.2f}, {hi:.2f}] | — (référence) | — | — |")
            continue
        dlo = r["delta_ci_lo"] * 100
        dhi = r["delta_ci_hi"] * 100
        ph = r["p_holm"]
        sig = "✅" if r["significant"] else "✗"
        d = r["delta_vs_ref"] * 100
        md.append(f"| {i} | **{r['arm']}** | {r['miou_point']*100:.2f} "
                  f"[{lo:.2f}, {hi:.2f}] | {d:+.2f} [{dlo:+.2f}, {dhi:+.2f}] | "
                  f"{ph:.3f} | {sig} |")
    (out_dir / "master_table.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "master_table.json", "w") as f:
        json.dump(table, f, indent=2)
    return table, rows


MOE_COLOR = "#e8871a"


def _is_moe(arm: str) -> bool:
    low = arm.lower()
    return any(h in low for h in MOE_HINTS)


def plot_master_bar(table: dict, ref: str, out_dir: Path):
    """Barres Δ appariés vs référence (pt, IC95) — la figure P3.14 des bras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    arms = [a for a in table["arms"] if a != ref]
    items = []
    for a in arms:
        pw = table["pairwise"].get(f"{a}_vs_{ref}")
        if pw is None:
            continue
        items.append((a, pw["delta"] * 100, pw["ci95"][0] * 100, pw["ci95"][1] * 100,
                      pw["significant"]))
    items.sort(key=lambda t: t[1])
    labels = [f"{t[0]}  <- MoE V3 CS (ce travail)" if _is_moe(t[0]) else t[0]
              for t in items]
    deltas = [t[1] for t in items]
    errs = [[d - lo for _, d, lo, _, _ in items], [hi - d for _, d, _, hi, _ in items]]
    colors = [MOE_COLOR if _is_moe(t[0]) else ("#2a7" if t[4] else "#89a")
              for t in items]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(items) + 2))
    ypos = np.arange(len(items))
    ax.barh(ypos, deltas, xerr=errs, color=colors, alpha=.9, capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlabel(f"Δ mIoU apparié vs {ref} (points, IC95 bootstrap)")
    ax.set_title(f"P3.14 — bras évalués (n={table['n_images']}, "
                 f"B={table['B']}, Holm)")
    ax.legend(handles=[Patch(color=MOE_COLOR, label="MoE-V3-CS (ce travail)"),
                       Patch(color="#2a7", label="IC95 exclut 0 (significatif)"),
                       Patch(color="#89a", label="IC95 inclut 0")],
              loc="lower right", framealpha=.9)
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "master_delta_bar.png", dpi=130)
    fig.savefig(out_dir / "master_delta_bar.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# (B) Routage
# --------------------------------------------------------------------------- #
def parse_routing(routing_dir: Path) -> dict:
    """{run: [records triés par epoch]} depuis routing_<run>.jsonl."""
    runs = {}
    for p in sorted(routing_dir.glob("routing_*.jsonl")):
        run = p.stem[len("routing_"):]
        recs = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        recs.sort(key=lambda r: r.get("epoch", 0))
        runs[run] = recs
    return runs


def routing_diagnostic(runs: dict, out_dir: Path, allow_smoke: bool):
    """Agrégat + verdict thèse « la porte ne choisit pas ». Figures si ≥2 époques."""
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = {"runs": {}, "verdict_these": None, "ref_brats_v3": BRATS_V3}
    real_runs = {r: v for r, v in runs.items()
                 if (allow_smoke or "SMOKE" not in r.upper()) and len(v) >= 1}
    for run, recs in runs.items():
        if not recs:
            continue
        last = recs[-1]
        n_ep = len(recs)
        fracs = last.get("frac", [])
        dead = sum(1 for fr in fracs if fr < 0.01) if fracs else None
        d = {
            "n_epochs_logged": n_ep,
            "is_smoke": ("SMOKE" in run.upper()),
            "last_epoch": last.get("epoch"),
            "part_max_last": last.get("part_max"),
            "entropy_norm_last": last.get("entropy_norm"),
            "entropy_token_last": last.get("entropy_token"),
            "gamma_absmean_last": last.get("gamma_absmean"),
            "noise_std_last": last.get("noise_std"),
            "frac_last": fracs,
            "dead_experts_last": dead,
        }
        # moyennes sur les époques post-recuit (noise_std==0) si dispo, sinon toutes
        post = [r for r in recs if r.get("noise_std", 1) == 0]
        pool = post if post else recs
        d["part_max_mean_postanneal"] = float(np.mean([r["part_max"] for r in pool
                                                       if "part_max" in r])) if pool else None
        d["entropy_norm_mean_postanneal"] = float(np.mean([r["entropy_norm"] for r in pool
                                                           if "entropy_norm" in r])) if pool else None
        diag["runs"][run] = d
    # verdict thèse : sur les runs RÉELS uniquement (post-recuit), pas le SMOKE
    verdict_pool = [d for r, d in diag["runs"].items() if not d["is_smoke"]
                    and d.get("part_max_mean_postanneal") is not None]
    if verdict_pool:
        pm = float(np.mean([d["part_max_mean_postanneal"] for d in verdict_pool]))
        en = float(np.mean([d["entropy_norm_mean_postanneal"] for d in verdict_pool]))
        dead = int(np.sum([d["dead_experts_last"] or 0 for d in verdict_pool]))
        # « la porte ne choisit pas » = part_max proche du floor 1/E (=0.25), entropie haute
        E = 4
        diag["verdict_these"] = {
            "part_max_mean": pm, "part_max_floor_1_over_E": 1.0 / E,
            "entropy_norm_mean": en, "dead_experts_total": dead,
            "these_gate_does_not_choose": bool(en > 0.85 and dead == 0 and pm < 0.6),
            "note": ("thèse BRATS V3 = le gain vient de l'INIT experts, pas du routage ; "
                     "CS : part_max proche de 1/E + entropie haute + 0 expert mort → confirme"),
        }
    with open(out_dir / "routing_diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2)
    # figures (≥2 époques pour une courbe lisible)
    plotted = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for run, recs in real_runs.items():
            if len(recs) < 2:
                continue
            ep = [r.get("epoch", i) for i, r in enumerate(recs)]
            fig, ax = plt.subplots(2, 2, figsize=(11, 7))
            fig.suptitle(f"P3.14 routage — {run}")
            for a, key, lab in [
                (ax[0, 0], "part_max", "part_max (floor 1/E=0.25)"),
                (ax[0, 1], "entropy_norm", "entropy_norm"),
                (ax[1, 0], "entropy_token", "entropy_token"),
                (ax[1, 1], "gamma_absmean", "gamma_absmean (ReZero)"),
            ]:
                a.plot(ep, [r.get(key) for r in recs], marker=".")
                a.set_xlabel("epoch"); a.set_ylabel(key); a.set_title(lab); a.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"routing_{run}.png", dpi=110)
            plt.close(fig)
            plotted.append(run)
    except Exception as e:  # matplotlib absent / non-bloquant
        diag["plot_error"] = str(e)
    diag["runs_plotted"] = plotted
    with open(out_dir / "routing_diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="bras de référence (défaut contrôle sinon B)")
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--allow-smoke", action="store_true",
                    help="inclure le run SMOKE dans le diagnostic routage (self-test)")
    ap.add_argument("--skip-table", action="store_true")
    ap.add_argument("--skip-routing", action="store_true")
    ap.add_argument("--extra-arms-dir", action="append", default=[],
                    help="répertoire(s) SUPPLÉMENTAIRE(s) de .cm.npy (dry-run/self-test, "
                         "zéro écriture dans les répertoires réels)")
    ap.add_argument("--out-dir", default=str(REPO / "results/moe_v3_cs/p314"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_table:
        complete, incomplete = discover_arms(extra_dirs=args.extra_arms_dir)
        print(f"[P3.14-A] bras complets (3 seeds) : {sorted(complete)}", flush=True)
        if incomplete:
            print(f"[P3.14-A] bras INCOMPLETS (ignorés) : {incomplete}", flush=True)
        if not complete:
            print("[P3.14-A] aucun bras complet — table sautée.", flush=True)
        else:
            ref = pick_ref(complete, args.ref)
            stacks = {a: load_stack(ps) for a, ps in complete.items()}
            n = next(iter(stacks.values())).shape[1]
            print(f"[P3.14-A] {len(stacks)} bras × n={n} images, réf={ref}, "
                  f"bootstrap B={args.B}...", flush=True)
            table, rows = build_master_table(stacks, ref, args.B, args.bootstrap_seed, out_dir)
            print(f"[P3.14-A] table écrite : {out_dir/'master_table.md'} "
                  f"({len(rows)} bras)", flush=True)
            try:
                plot_master_bar(table, ref, out_dir)
                print(f"[P3.14-A] figure : {out_dir/'master_delta_bar.png'}", flush=True)
            except Exception as e:
                print(f"[P3.14-A] figure NON tracée (non bloquant) : {e}", flush=True)
            for r in rows[:6]:
                print(f"    {r['arm']:>16}  mIoU={r['miou_point']*100:.2f}%  "
                      f"Δ={r['delta_vs_ref']*100:+.2f}pt  p_holm="
                      f"{(r['p_holm'] if r['p_holm'] is not None else float('nan')):.3f}"
                      f"  {'SIGNIF' if r['significant'] else ''}", flush=True)

    if not args.skip_routing:
        runs = parse_routing(REPO / "results/moe_v3_cs")
        print(f"[P3.14-B] runs routage détectés : {list(runs)}", flush=True)
        diag = routing_diagnostic(runs, out_dir, allow_smoke=args.allow_smoke)
        print(f"[P3.14-B] diagnostic : {out_dir/'routing_diagnostic.json'}", flush=True)
        v = diag.get("verdict_these")
        if v:
            print(f"    verdict thèse : part_max={v['part_max_mean']:.3f} "
                  f"entropy_norm={v['entropy_norm_mean']:.3f} "
                  f"dead={v['dead_experts_total']} → "
                  f"{'CONFIRMÉE' if v['these_gate_does_not_choose'] else 'INFIRMÉE'}", flush=True)
        else:
            print("    verdict thèse : en attente de runs MoE réELS (seul SMOKE dispo).", flush=True)
        if diag.get("runs_plotted"):
            print(f"    figures tracées : {diag['runs_plotted']}", flush=True)


if __name__ == "__main__":
    main()
