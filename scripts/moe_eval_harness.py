#!/usr/bin/env python3
"""P3.10 — Harness d'éval MULTI-BRAS du paper 3 : tous les bras sur le MÊME holdout apparié.

Intègre dans un seul évaluateur les 8 familles de bras du plan (PLAN_TRANSFERT_BRATS_V3.md
§4, DESIGN.md §10) : singles experts, mean-ensemble, argmax/majority-vote, consensus
CC-veto, late-fusion (gate appris, rétrogradée baseline), contrôle-sans-MoE, MoE-V3-CS,
et n'importe quel bras « npy » déjà dumpé (results/perimage_cm). Pour chaque bras il
produit les confusions PAR IMAGE (N,19,19) alignées sur le même ordre d'images, puis la
table bootstrap APPARIÉE complète (IC 95 %, p bilatéral, Holm) via src.moe.bootstrap —
la source statistique unique de P3.09.

Spécification des bras : `--arm NOM=KIND:PAYLOAD` (répétable) :
  npy:TEMPLATE       charge des .cm.npy existants (placeholder {seed} optionnel) — 0 calcul :
                      ex. `B=npy:results/perimage_cm/B_seed{seed}__epoch_160.cm.npy`
  dense:TEMPLATE      forward d'un checkpoint dense (contrôle-sans-MoE, expert…) ;
                      {seed} optionnel ; GPU ; protocole IDENTIQUE à dump_perimage_cm.py
                      (channels_last, autocast BF16, TF32, cudnn.benchmark)
  moe:TEMPLATE        forward d'un checkpoint MoE-V3-CS ENTRAÎNÉ (format P3.12 :
                      model_state_dict + config + moe_kwargs [+ provenance]) ; stats de
                      routage moyennées et journalisées dans la table
  mean:E1,E2,…        mean-ensemble des logits depuis le CACHE fp16 (results/moe/cache)
  argmax:E1,E2,…      majority-vote des argmax experts (src.postprocessing.consensus)
  veto:P/V            consensus CC-veto P⊘V (cc_veto, protect_classes par défaut)
  lf:GATE:E1,E2       late-fusion CNNGate (state_dict entraîné par train_moe.py) sur les
                      logits cachés, dans l'ordre d'entraînement des experts.
                      ⚠️ Fuite si évaluée sur des images de son split gate-fit : passer
                      --indices gate-test:N_FIT,SPLIT_SEED (mêmes valeurs que le run).

Holdout : par défaut les `--max-images` premières images du val (ordre du loader,
shuffle=False — l'ordre historique de val_image_order.json), ou `--indices
gate-test:N_FIT,SPLIT_SEED` pour le complement du fit (DESIGN §8). TOUS les bras sont
tronqués/filtrés sur la MÊME liste de positions : c'est ce qui rend le bootstrap apparié
légitime. Les bras live (fusion/dense/moe) sont cachés dans --cm-out (idempotent).

Sorties : <out> (JSON : sources, mIoU par seed, table bootstrap mIoU, table Boundary F1
si --bf1, routage MoE, notes) + table texte stdout. Les .cm.npy restent réutilisables
par tout autre appel (mêmes conventions de nommage que dump_perimage_cm.py).

Exemples (Tour, depuis la racine du dépôt) :
  # Table CPU immédiate sur les bras déjà dumpés + fusions depuis le cache seed 42 :
  taskset -c 16-27 python3 scripts/moe_eval_harness.py \
      --arm 'B=npy:results/perimage_cm/B_seed{seed}__epoch_160.cm.npy' \
      --arm 'fused_DvetoB=npy:results/perimage_cm/fused_DvetoB_seed{seed}__epoch_160.cm.npy' \
      --arm 'meanBD=mean:B,D' --arm 'argmaxBD=argmax:B,D' --arm 'vetoDB=veto:D/B' \
      --seeds 42,123,456 --cache-seeds 42 --max-images 500 --B 10000 \
      --out results/moe_v3_cs/harness/table_paper3_seed42cache.json
  # Bras GPU (contrôle + V3-CS entraînés, P3.11/P3.12) contre le consensus :
  /home/ser/brats-venv/bin/python scripts/moe_eval_harness.py \
      --arm 'controle=dense:checkpoints/controle_nomoe_seed{seed}/epoch_XXX.pth' \
      --arm 'v3cs=moe:checkpoints/moe_v3_cs_seed{seed}/best.pth' \
      --arm 'fused_DvetoB=npy:results/perimage_cm/fused_DvetoB_seed{seed}__epoch_160.cm.npy' \
      --seeds 42 --ref controle --B 10000 --out results/moe_v3_cs/harness/table_v3cs.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.moe.bootstrap import (  # noqa: E402
    NC, IGNORE, per_image_cm, miou_from_cm, paired_bootstrap,
    paired_bootstrap_values, save_arm_cms, format_report,
    DEFAULT_B, DEFAULT_BOOTSTRAP_SEED,
)
from src.moe.data import split_val_indices  # noqa: E402
from src.postprocessing.consensus import cc_veto, majority_vote  # noqa: E402

VALID_KINDS = ("npy", "dense", "moe", "mean", "argmax", "veto", "lf")
FUSION_KINDS = ("mean", "argmax", "veto", "lf")
CM_SUFFIX = ".cm.npy"          # suffixe des caches du harness (≠ dumps officiels epoch_160)


# --------------------------------------------------------------------------- #
# Spécification des bras
# --------------------------------------------------------------------------- #

def parse_arm_spec(spec: str) -> dict:
    """'NOM=KIND:PAYLOAD' → dict(name, kind, payload). Erreurs explicites."""
    if "=" not in spec or ":" not in spec:
        raise ValueError(f"--arm attend NOM=KIND:PAYLOAD, reçu {spec!r}")
    name, rest = spec.split("=", 1)
    kind, payload = rest.split(":", 1)
    name, kind = name.strip(), kind.strip()
    if not name or kind not in VALID_KINDS:
        raise ValueError(f"bras {spec!r} : kind {kind!r} inconnu (attendu {VALID_KINDS})")
    if kind in FUSION_KINDS and not payload.strip():
        raise ValueError(f"bras {name!r} : payload vide pour kind {kind}")
    return {"name": name, "kind": kind, "payload": payload.strip()}


def _fmt_template(template: str, seed: int) -> str:
    return template.replace("{seed}", str(seed)) if "{seed}" in template else template


def _fusion_experts(kind: str, payload: str):
    """Payload fusion → liste ordonnée d'experts ; lf → (gate_path, experts)."""
    if kind == "veto":
        parts = payload.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"veto attend P/V, reçu {payload!r}")
        return parts
    if kind == "lf":
        if ":" not in payload:
            raise ValueError(f"lf attend GATE_CKPT:E1,E2, reçu {payload!r}")
        gate, experts = payload.rsplit(":", 1)
        experts = [e.strip() for e in experts.split(",") if e.strip()]
        if not experts:
            raise ValueError(f"lf : aucun expert dans {payload!r}")
        return gate, experts
    experts = [e.strip() for e in payload.split(",") if e.strip()]
    if not experts:
        raise ValueError(f"{kind} : aucun expert dans {payload!r}")
    return experts


# --------------------------------------------------------------------------- #
# Holdout partagé
# --------------------------------------------------------------------------- #

def cache_prefix_len(cache_root: str, experts, seed: int) -> int:
    """Plus grand M tel que {0..M-1}.pt existent pour TOUS les experts du seed."""
    m = None
    for e in experts:
        d = Path(cache_root) / f"{e}_seed{seed}"
        k = 0
        while (d / f"{k:04d}.pt").exists():
            k += 1
        m = k if m is None else min(m, k)
    return int(m or 0)


def resolve_holdout(args, specs) -> tuple:
    """Liste des POSITIONS du holdout partagé + description (pour la table JSON).

    Règle d'alignement : la longueur est bornée par (a) --max-images, (b) la plus
    courte source (préfixes cache des bras fusion, N des bras npy) — sinon deux bras
    ne couvriraient pas les mêmes images et le bootstrap apparié serait illégitime.
    `gate-test` filtre ensuite sur le complément du split de fit (DESIGN §8).
    """
    limit = args.max_images or None
    why = [f"max_images={args.max_images}"] if args.max_images else []
    for spec in specs:
        if spec["kind"] == "npy":
            # mesure sur le premier seed demandé (les dumps partagent l'ordre et N)
            first_seed = args.seeds[0]
            p = Path(_fmt_template(spec["payload"], first_seed))
            if not p.exists():
                raise FileNotFoundError(f"npy introuvable : {p}")
            n = np.load(p, mmap_mode="r").shape[0]
            if limit is None or n < limit:
                limit = n
                why.append(f"npy[{spec['name']}]={n}")
        elif spec["kind"] in FUSION_KINDS:
            experts = _fusion_experts(spec["kind"], spec["payload"])
            experts = experts[1] if spec["kind"] == "lf" else experts
            for s in args.cache_seeds:
                m = cache_prefix_len(args.cache_root, experts, s)
                if m <= 0:
                    continue
                if limit is None or m < limit:
                    limit = m
                    why.append(f"cache[{spec['name']}@{s}]={m}")
    if limit is None:
        raise ValueError("impossible de déterminer N (aucun bras npy/fusion, --max-images requis "
                         "pour les bras dense/moe seuls)")
    positions = list(range(limit))
    desc = f"first:{limit}"
    if args.indices.startswith("gate-test"):
        n_fit, split_seed = (int(x) for x in args.indices.split(":", 1)[1].split(","))
        _fit, eval_idx = split_val_indices(max(limit, n_fit + 1), n_fit, split_seed)
        keep = set(eval_idx)
        positions = [p for p in positions if p in keep]
        desc = f"gate-test:{n_fit},{split_seed}∩first:{limit}"
        why.append(f"indices={args.indices}")
    elif args.indices != "all":
        raise ValueError(f"--indices inconnu : {args.indices!r} (all | gate-test:N_FIT,SPLIT_SEED)")
    return positions, {"description": desc, "n": len(positions), "bornes": why}


# --------------------------------------------------------------------------- #
# Bras npy (aucun calcul)
# --------------------------------------------------------------------------- #

def arm_from_npy(spec, seeds, positions, cm_out) -> dict:
    per_seed = {}
    for s in seeds:
        p = Path(_fmt_template(spec["payload"], s))
        if not p.exists():
            raise FileNotFoundError(f"bras {spec['name']} seed {s} : {p} absent")
        arr = np.load(p)
        if arr.ndim != 3 or arr.shape[1:] != (NC, NC):
            raise ValueError(f"bras {spec['name']} seed {s} : shape {arr.shape} ≠ (N,19,19)")
        arr = np.ascontiguousarray(arr[positions])
        per_seed[s] = {"cm": arr, "bf1": None, "source": str(p)}
    return {"kind": "npy", "seeds": per_seed, "meta": {"template": spec["payload"]}}


# --------------------------------------------------------------------------- #
# Bras fusion depuis le cache de logits fp16 (CPU, zéro GPU)
# --------------------------------------------------------------------------- #

def _stacked_from_cache(cache_root, experts, seed, idx):
    import torch
    from src.moe.cache import load_stacked_from_cache
    return load_stacked_from_cache(cache_root, experts, seed, idx, torch.device("cpu"))


def _gt_dataset(cfg_ckpt_expert, seed, repo_root):
    from src.moe.data import cfg_from_checkpoint, build_val_dataset
    from src.moe.experts import resolve_expert_path
    cfg = cfg_from_checkpoint(resolve_expert_path(cfg_ckpt_expert, seed, repo_root))
    return build_val_dataset(cfg), cfg


def fused_labels(kind, payload, stacked, gate=None):
    """Labels fusionnés (H,W) int64 depuis les logits empilés (K,19,H,W)."""
    if kind == "mean":
        return stacked.mean(0).argmax(0).numpy().astype(np.int64)
    if kind == "argmax":
        votes = [stacked[k].argmax(0).numpy().astype(np.int64) for k in range(stacked.shape[0])]
        out = majority_vote(votes, num_classes=NC, ignore_index=IGNORE)
        if (out == IGNORE).any():
            raise AssertionError("vote avec pixels void inattendus ( logits cache invalides)")
        return out.astype(np.int64)
    if kind == "veto":
        prim = stacked[0].argmax(0).numpy().astype(np.int64)
        veto = stacked[1].argmax(0).numpy().astype(np.int64)
        return cc_veto(prim, veto, num_classes=NC, ignore_index=IGNORE).astype(np.int64)
    if kind == "lf":
        import torch
        with torch.no_grad():
            g = gate(stacked.unsqueeze(0))                       # (1,K,H,W)
            mixed = (g.unsqueeze(2) * stacked.unsqueeze(0)).sum(1)  # (1,19,H,W)
        return mixed[0].argmax(0).numpy().astype(np.int64)
    raise ValueError(kind)


def arm_from_fusion(spec, seeds, positions, cm_out, cache_root, repo_root,
                    want_bf1, overwrite, log=print):
    import torch
    kind = spec["kind"]
    gate = None
    if kind == "lf":
        gate_path, experts = _fusion_experts(kind, spec["payload"])
        from src.moe.gate import CNNGate
        gate = CNNGate(len(experts), n_classes=NC)
        sd = torch.load(gate_path, map_location="cpu", weights_only=True)
        gate.load_state_dict(sd)
        gate.eval()
    else:
        experts = _fusion_experts(kind, spec["payload"])

    val_ds, _cfg = _gt_dataset("B", 42, repo_root)
    if len(val_ds) < max(positions) + 1:
        raise ValueError(f"positions jusqu'à {max(positions)} > val {len(val_ds)}")

    per_seed = {}
    for s in seeds:
        m = cache_prefix_len(cache_root, experts, s)
        cov = [p for p in positions if p < m]
        if len(cov) < len(positions):
            log(f"  [{spec['name']}] seed {s} : cache incomplet ({m} < {max(positions)+1}) → skip")
            continue
        cached = Path(cm_out) / f"{spec['name']}_seed{s}{CM_SUFFIX}"
        if cached.exists() and not overwrite:
            arr = np.load(cached)
            if arr.shape[0] == len(positions):
                per_seed[s] = {"cm": arr, "bf1": None, "source": f"cache:{cached}"}
                log(f"  [{spec['name']}] seed {s} : réemploi du cache {cached.name}")
                continue
        t0 = time.time()
        cms, bf1s = [], []
        for j, i in enumerate(cov):
            stacked = _stacked_from_cache(cache_root, experts, s, i)
            pred = fused_labels(kind, spec["payload"], stacked, gate=gate)
            gt = np.asarray(val_ds[int(i)]["label"]).astype(np.int64)
            cms.append(per_image_cm(pred, gt))
            if want_bf1:
                from src.metrics.segmentation_metrics import compute_boundary_f1
                bf1s.append(float(compute_boundary_f1(pred, gt)["boundary_f1"]))
            if (j + 1) % 50 == 0:
                log(f"  [{spec['name']}] seed {s} : {j+1}/{len(cov)} ({time.time()-t0:.0f}s)")
        arr = np.stack(cms)
        save_arm_cms(arr, cm_out, spec["name"], s, suffix=CM_SUFFIX)
        per_seed[s] = {"cm": arr,
                       "bf1": np.asarray(bf1s, dtype=np.float32) if want_bf1 else None,
                       "source": f"logits_cache_fp16:{'/'.join(experts)}@seed{s}"}
        log(f"  [{spec['name']}] seed {s} : {len(cov)} images, {time.time()-t0:.0f}s "
            f"(mIoU={miou_from_cm(arr.sum(0)):.4f})")
    if not per_seed:
        raise FileNotFoundError(f"bras {spec['name']} : aucune seed exploitable (cache {experts})")
    return {"kind": kind, "seeds": per_seed, "meta": {"experts": experts,
                                                      "gate": (str(gate is not None))}}


# --------------------------------------------------------------------------- #
# Bras GPU : dense (contrôle / expert) et moe (V3-CS entraîné)
# --------------------------------------------------------------------------- #

def _setup_runtime():
    import torch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def _loader_of_cfg(cfg):
    from src.data import build_dataloaders
    return build_dataloaders(cfg)["val"]


def arm_from_checkpoint(spec, seeds, positions, cm_out, device, want_bf1,
                        overwrite, log=print):
    """dense: et moe: — protocole IDENTIQUE à dump_perimage_cm.py / moe_epoch0_check.py."""
    import torch
    from omegaconf import OmegaConf
    from src.models import build_model
    from src.moe.moe_model import attach_patch_moe, find_patch_moe
    from src.metrics.segmentation_metrics import compute_boundary_f1

    _setup_runtime()
    device = torch.device(device)
    keep = set(positions)
    per_seed = {}
    for s in seeds:
        path = _fmt_template(spec["payload"], s)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"bras {spec['name']} seed {s} : checkpoint absent {path}")
        cached = Path(cm_out) / f"{spec['name']}_seed{s}{CM_SUFFIX}"
        if cached.exists() and not overwrite:
            arr = np.load(cached)
            if arr.shape[0] == len(positions):
                per_seed[s] = {"cm": arr, "bf1": None, "source": f"cache:{cached}"}
                log(f"  [{spec['name']}] seed {s} : réemploi du cache {cached.name}")
                continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ck["config"])
        OmegaConf.set_struct(cfg, False)
        cfg.model.backbone.pretrained = "none"
        model = build_model(cfg)
        moe = None
        if spec["kind"] == "moe":
            moe_kwargs = ck.get("moe_kwargs")
            if moe_kwargs is None:
                # Repli défensif : provenance imbriquée (checkpoints d'avant le contrat
                # top-level du 2026-09-22 ; le trainer P3.12 sauvegarde les deux).
                moe_kwargs = (ck.get("moe_provenance") or {}).get("moe_kwargs")
            if moe_kwargs is None:
                raise KeyError(f"{path} : 'moe_kwargs' absent (top-level ET provenance) — le "
                               f"trainer P3.12 doit les sauvegarder (recette exacte de "
                               f"l'attach, cf. plan §7)")
            moe_kwargs = {k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in dict(moe_kwargs).items()}
            moe = attach_patch_moe(model, **moe_kwargs)
            load = model.load_state_dict(ck["model_state_dict"], strict=True)
            assert not load.missing_keys and not load.unexpected_keys
        else:
            model.load_state_dict(ck["model_state_dict"], strict=True)
        model = model.to(device).eval().to(memory_format=torch.channels_last)
        channels_last = bool(cfg.training.get("channels_last", True))

        loader = _loader_of_cfg(cfg)
        cms, bf1s, routing, pos = [], [], {k: [] for k in
                                           ("entropy_norm", "entropy_token", "top_expert_share",
                                            "gamma_absmean")}, 0
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                for b in range(batch["label"].shape[0]):
                    if pos in keep:
                        images = batch["image"][b:b + 1].to(device, non_blocking=True)
                        if channels_last:
                            images = images.contiguous(memory_format=torch.channels_last)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
                            out = model(images)
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            elif isinstance(out, dict):
                                out = out["main"]
                        pred = out.argmax(1)[0].to(torch.uint8).cpu().numpy().astype(np.int64)
                        gt = batch["label"][b].numpy().astype(np.int64)
                        cms.append(per_image_cm(pred, gt))
                        if want_bf1:
                            bf1s.append(float(compute_boundary_f1(pred, gt)["boundary_f1"]))
                        if moe is not None and moe.last_stats:
                            for k in routing:
                                if k in moe.last_stats:
                                    routing[k].append(float(moe.last_stats[k]))
                    pos += 1
                    if len(cms) == len(positions):
                        break
                if len(cms) == len(positions):
                    break
        if len(cms) != len(positions):
            raise ValueError(f"bras {spec['name']} seed {s} : {len(cms)} images ≠ "
                             f"{len(positions)} attendues (loader plus court ?)")
        arr = np.stack(cms)
        save_arm_cms(arr, cm_out, spec["name"], s, suffix=CM_SUFFIX)
        entry = {"cm": arr, "bf1": np.asarray(bf1s, dtype=np.float32) if want_bf1 else None,
                 "source": f"forward:{path}"}
        if moe is not None:
            entry["routing"] = {k: (float(np.mean(v)) if v else None) for k, v in routing.items()}
            entry["provenance"] = ck.get("provenance") or ck.get("moe_provenance")
            entry["epoch"] = int(ck.get("epoch", -1)) + 1
        per_seed[s] = entry
        log(f"  [{spec['name']}] seed {s} : {len(cms)} images, {time.time()-t0:.0f}s "
            f"(mIoU={miou_from_cm(arr.sum(0)):.4f})")
        del model, ck
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"kind": spec["kind"], "seeds": per_seed,
            "meta": {"template": spec["payload"]}}


# --------------------------------------------------------------------------- #
# Assemblage + table bootstrap
# --------------------------------------------------------------------------- #

def build_stacks(arms: dict, positions_n: int) -> dict:
    stacks = {}
    for name, arm in arms.items():
        for s, d in arm["seeds"].items():
            if d["cm"].shape[0] != positions_n:
                raise ValueError(f"bras {name} seed {s} : N={d['cm'].shape[0]} ≠ {positions_n} — "
                                 f"holdout non aligné (bug d'alignement, ne PAS bootstrapper)")
        stacks[name] = np.stack([arm["seeds"][s]["cm"] for s in sorted(arm["seeds"])])
    return stacks


def build_bf1_values(arms: dict) -> dict:
    out = {}
    for name, arm in arms.items():
        if any(d["bf1"] is None for d in arm["seeds"].values()):
            continue
        out[name] = np.stack([arm["seeds"][s]["bf1"].astype(np.float64)
                              for s in sorted(arm["seeds"])])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="NOM=KIND:PAYLOAD (npy|dense|moe|mean|argmax|veto|lf), répétable")
    ap.add_argument("--seeds", default="42,123,456", help="seeds des bras npy/dense/moe")
    ap.add_argument("--cache-seeds", default="42", help="seeds des bras fusion (cache logits)")
    ap.add_argument("--indices", default="all",
                    help="all | gate-test:N_FIT,SPLIT_SEED (holdout partagé, DESIGN §8)")
    ap.add_argument("--max-images", type=int, default=0, help="0 = tout le holdout disponible")
    ap.add_argument("--B", type=int, default=DEFAULT_B, help="réplicats bootstrap")
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--ref", default=None,
                    help="bras de référence : paires limitées à (ref, x) au lieu de toutes")
    ap.add_argument("--bf1", action="store_true", help="Boundary F1 par image (bras live)")
    ap.add_argument("--cache-root", default=str(REPO_ROOT / "results/moe/cache"))
    ap.add_argument("--cm-out", default=str(REPO_ROOT / "results/moe_v3_cs/harness"))
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true", help="recalculer les caches de bras")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.cache_seeds = [int(s) for s in args.cache_seeds.split(",")]
    specs = [parse_arm_spec(a) for a in args.arm]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("noms de bras dupliqués")

    log = lambda *a: print(*a, flush=True)
    positions, holdout = resolve_holdout(args, specs)
    log(f"[harness] holdout {holdout['description']} (n={holdout['n']}) — bornes : {holdout['bornes']}")

    arms = {}
    for spec in specs:
        t0 = time.time()
        if spec["kind"] == "npy":
            arms[spec["name"]] = arm_from_npy(spec, args.seeds, positions, args.cm_out)
        elif spec["kind"] in FUSION_KINDS:
            fusion_seeds = sorted(set(args.seeds) & set(args.cache_seeds))
            if not fusion_seeds:
                fusion_seeds = args.cache_seeds
            arms[spec["name"]] = arm_from_fusion(
                spec, fusion_seeds, positions, args.cm_out, args.cache_root,
                args.repo_root, args.bf1, args.overwrite, log=log)
        else:
            arms[spec["name"]] = arm_from_checkpoint(
                spec, args.seeds, positions, args.cm_out, args.device, args.bf1,
                args.overwrite, log=log)
        log(f"[harness] bras {spec['name']} ({spec['kind']}) : "
            f"seeds {sorted(arms[spec['name']]['seeds'])}, {time.time()-t0:.0f}s")

    stacks = build_stacks(arms, len(positions))
    pairs = None
    if args.ref:
        if args.ref not in arms:
            raise KeyError(f"--ref {args.ref!r} n'est pas un bras déclaré")
        pairs = [(args.ref, n) for n in names if n != args.ref]
    table = paired_bootstrap(stacks, B=args.B, rng_seed=args.bootstrap_seed, pairs=pairs)

    bf1_table, bf1_note = None, None
    if args.bf1:
        vals = build_bf1_values(arms)
        if len(vals) >= 2:
            bpairs = ([(args.ref, a) for a in vals if a != args.ref]
                      if (args.ref and args.ref in vals) else None)
            bf1_table = paired_bootstrap_values(vals, B=args.B, rng_seed=args.bootstrap_seed,
                                                pairs=bpairs)
        else:
            bf1_note = (f"--bf1 demandé mais <2 bras avec Boundary F1 (les bras npy historiques "
                        f"n'en ont pas) : {sorted(vals)} — table BF1 ignorée")
            log("[harness] " + bf1_note)

    notes = [f"holdout partagé : {holdout['description']}",
             "mIoU dataset-level, bootstrap APPARIÉ (mêmes positions pour tous les bras), "
             "seeds moyennés dans chaque réplicat"]
    for name, arm in arms.items():
        for s, d in arm["seeds"].items():
            if "logits_cache_fp16" in d["source"]:
                notes.append(f"bras {name}@{s} : fusion sur logits fp16 CACHÉS (train_moe.py) — "
                             f"différence d'arrondi possible vs forward frais")
        if arm["kind"] == "lf":
            notes.append(f"bras {name} : LATE-FUSION — le gate a été entraîné sur un split de "
                         f"FIT du val ; si le holdout recoupe ce split, le chiffre est FUITÉ "
                         f"(baseline rétrogradée, plan §4.5). Utiliser --indices gate-test:… "
                         f"pour un chiffre propre.")

    out = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "holdout": holdout,
        "B": args.B, "bootstrap_seed": args.bootstrap_seed, "ref": args.ref,
        "arms": {n: {"kind": a["kind"], "meta": a["meta"],
                     "seeds": sorted(a["seeds"]),
                     "source_par_seed": {str(s): a["seeds"][s]["source"]
                                         for s in a["seeds"]},
                     "miou_dataset_par_seed": {str(s): miou_from_cm(a["seeds"][s]["cm"].sum(0))
                                               for s in a["seeds"]},
                     "routing": {str(s): a["seeds"][s].get("routing")
                                 for s in a["seeds"] if "routing" in a["seeds"][s]},
                     "provenance": {str(s): a["seeds"][s].get("provenance")
                                    for s in a["seeds"] if a["seeds"][s].get("provenance")},
                     "epoch": {str(s): a["seeds"][s].get("epoch")
                               for s in a["seeds"] if a["seeds"][s].get("epoch")}}
                 for n, a in arms.items()},
        "bootstrap_miou": table,
        "bootstrap_bf1": bf1_table,
        "bf1_note": bf1_note,
        "notes": notes,
    }
    out_path = args.out or str(Path(args.cm_out) /
                               f"table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

    print("\n=== Table mIoU (bootstrap apparié, n=%d, B=%d) ===" % (table["n_images"], args.B))
    print(format_report(table))
    if bf1_table:
        print("\n=== Table Boundary F1 (apparié) ===")
        print(format_report(bf1_table))
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
