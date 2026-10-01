#!/usr/bin/env python3
"""P3.12 — Trainer du run MoE-V3-CS (transfert BRATS V3, sprint #75).

Bra APPARIÉ au contrôle-sans-MoE P3.11 : MÊME recette d'entraînement (section
`training:` de `configs/experiment/moe_v3cs_Binit.yaml` identique à
`control_nomoe_Binit.yaml`, N=80), la SEULE différence est la présence de la
couche PatchMoE2D attachée après `head.fpn_bottleneck` et de sa loss switch.

Ce que ce trainer ajoute à `scripts/train.py` (dont il réutilise les helpers :
optimisations Blackwell, reprise `last.pth`, scheduler Poly, sauvegarde atomique
fsync) :

1. INIT — `build_moe_model_from_baseline(seed)` : architecture reconstruite
   depuis la config EMBARQUÉE du checkpoint B (jamais depuis la config hydra,
   garantie de correspondance des clés), backbone+head ← B seed s, experts 0-3 ←
   `head.fpn_convs.0` de B/D/Dp/G AU MÊME seed (appariement strict, audit P3.06,
   copie vérifiée bit-à-bit), gate ← aléatoire, γ résiduel init 0 (garde-fou
   MESURÉ P3.08 : sans lui −28.1 pt à l'ép.0 ; avec lui l'ép.0 est bit-exacte au
   contrôle). La `provenance` renvoyée est OBLIGATOIREMENT sauvegardée dans
   chaque checkpoint (`moe_provenance`) — traçabilité BRATS.

2. LOSS — `criterion(main) + 0.4·criterion(aux) + moe.aux_loss` (la switch-loss
   déjà pondérée par `balance_weight=1e-3` dans la couche).

3. RECUIT — `moe.set_epoch(epoch)` au DÉBUT de chaque époque : le bruit de
   Shazeer se DÉDUIT du numéro d'époque (1.0 → 0 en 40 époques), donc une reprise
   après crash est sûre par construction (leçon BRATS 2026-08-06).

4. LOGS ROUTAGE (plan §4 — diagnostics de la thèse « la porte ne choisit pas ») :
   `part_max`, `entropy_norm` (équilibrage de charge), `entropy_token` (netteté de
   décision, bruitée) ET `entropy_token_clean` (sans bruit — distingue « gate
   indécis » de « bruit qui masque une préférence »), `gamma_absmean` (le MoE
   sert-il ?), fractions par expert, switch-loss pondérée, noise_std courant.
   Moyennés par époque → stdout + wandb (`moe/*`, projet city-scape) + JSONL
   durable fsync `results/moe_v3_cs/routing_<run>.jsonl` (les courbes P3.14 se
   tracent depuis ce fichier même sans wandb).

Usage (Tour, repo root — le launcher P3.11/P3.12 le fait tout seul) :
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 /home/ser/brats-venv/bin/python \
      scripts/train_moe_v3cs.py +experiment=moe_v3cs_Binit experiment.seeds="[42]"

Smoke GPU court (2 batches, basse résolution — ne remplace PAS le run) :
    ... scripts/train_moe_v3cs.py +experiment=moe_v3cs_Binit experiment.seeds="[42]" \
      training.epochs=1 training.smoke_max_batches=2 \
      augmentation.train.resize.height=128 augmentation.train.resize.width=256 \
      hardware.num_workers=2
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import hydra


def _load_control_module():
    """Charge `scripts/train.py` en direct depuis son fichier (helpers du contrôle P3.11).

    `import scripts.train` est IMPOSSIBLE sur la Tour : un paquet RÉGULIER `scripts/`
    installé dans `~/.local/lib/python3.10/site-packages` (mesuré 2026-09-16,
    scripts/__init__.py) shadow le namespace package PEP 420 du repo. Le chargement
    par chemin de fichier est déterministe ; les tests réutilisent cette poignée.
    """
    import importlib.util
    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_control_p312", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_control_p312"] = mod
    spec.loader.exec_module(mod)
    return mod


# Helpers mutualisés avec le contrôle P3.11 (même format de checkpoint, même
# reprise, même scheduler) — la comparaison appariée exige des états homogènes.
train_ctl = _load_control_module()
_build_scheduler = train_ctl._build_scheduler  # noqa: E402
_load_checkpoint = train_ctl._load_checkpoint  # noqa: E402
_setup_blackwell_optims = train_ctl._setup_blackwell_optims  # noqa: E402
_wandb_finish = train_ctl._wandb_finish  # noqa: E402
_wandb_log = train_ctl._wandb_log  # noqa: E402
from src.data import build_dataloaders  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.moe.moe_model import build_moe_model_from_baseline, find_patch_moe  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

ROUTING_DIR = Path("results") / "moe_v3_cs"

# Clés de last_stats (floats) agrégées par époque + clé dérivée.
_STAT_FLOAT_KEYS = (
    "entropy_norm", "entropy_token", "entropy_token_clean",
    "top_expert_share", "gamma_absmean",
)


def _durable_append_jsonl(path: Path, obj: dict):
    """Ajoute une ligne JSON durable (fsync) — journal de routage par époque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _aggregate_routing_stats(moe, n_experts: int) -> dict:
    """Convertit `moe.last_stats` courant en floats sérialisables (frac → liste)."""
    ls = moe.last_stats or {}
    stats = {k: float(ls[k]) for k in _STAT_FLOAT_KEYS if k in ls}
    frac = ls.get("frac")
    if frac is not None:
        stats["frac"] = [float(v) for v in frac.detach().reshape(-1)]
    stats["aux_loss_weighted"] = float(moe.aux_loss.detach()) if moe.aux_loss is not None else 0.0
    stats["noise_std"] = float(moe.noise_std)
    stats["n_experts"] = int(n_experts)
    return stats


def _epoch_mean(rows) -> dict:
    """Moyenne par clé sur les batches (frac = moyenne composante à composante)."""
    if not rows:
        return {}
    out = {}
    keys = [k for k in rows[0] if k != "frac"]
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[k] = float(np.mean(vals)) if vals else None
    if "frac" in rows[0]:
        arr = np.array([r["frac"] for r in rows], dtype=np.float64)
        out["frac"] = [float(v) for v in arr.mean(axis=0)]
        out["part_max"] = float(max(out["frac"]))  # diagnostic plan §4, redondant avec top_expert_share moyen
    return out


def train_one_epoch_moe(model, moe, loader, criterion, optimizer, scaler, device,
                        accum_steps, channels_last, epoch, max_batches=None):
    """Une époque V3-CS : recuit déduit de `epoch`, aux switch dans la loss, stats routage.

    `scaler=None` et CPU sont supportés (tests unitaires / smoke CPU) ; sur GPU le
    scaler BF16 standard du projet est utilisé. Retourne (loss moyenne, stats époque).
    """
    # Recuit du bruit de Shazeer DÉDUIT du numéro d'époque (reprise sûre, P3.07).
    moe.set_epoch(epoch)

    model.train()
    total_loss = 0.0
    num_batches = 0
    routing_rows = []
    optimizer.zero_grad()

    use_cuda_amp = device.type == "cuda"
    amp_ctx = (lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)) \
        if use_cuda_amp else (lambda: torch.autocast(device_type="cpu", enabled=False))

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if channels_last and use_cuda_amp:
            images = images.contiguous(memory_format=torch.channels_last)

        with amp_ctx():
            output = model(images)
            if isinstance(output, dict):
                raise RuntimeError(
                    "sortie dict (distmap) inattendue : le run V3-CS est apparié à la "
                    "recette B (ce_dice) — config du checkpoint B corrompue ?")
            if isinstance(output, tuple):
                main_out, aux_out = output
                loss_main, _ = criterion(main_out, labels)
                loss_aux, _ = criterion(aux_out, labels)
                seg_loss = loss_main + 0.4 * loss_aux
            else:
                seg_loss, _ = criterion(output, labels)
            # Switch-loss d'équilibrage (déjà pondérée balance_weight dans la couche).
            if moe.aux_loss is None:
                raise RuntimeError("moe.aux_loss non produit par le forward — attach cassé ?")
            loss = seg_loss + moe.aux_loss.to(seg_loss.dtype)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        routing_rows.append(_aggregate_routing_stats(moe, moe.n_experts))
        total_loss += loss.item() * accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1), _epoch_mean(routing_rows)


class _NullScaler:
    """Le `scaler` factice accepté par _load_checkpoint quand on tourne sans CUDA."""

    def load_state_dict(self, state):
        return None

    def state_dict(self):
        return {}


def _require_moe_kwargs(provenance: dict) -> dict:
    """Recette de l'attach pour le contrat top-level du checkpoint (bras `moe` P3.10).

    `build_moe_model_from_baseline` met `moe_kwargs` dans la provenance — absents =
    bug d'init, et le harness relèverait KeyError des jours plus tard : on échoue ICI.
    Listes (pas tuples) pour la sérialisation ; le harness re-tuple à la lecture.
    """
    kw = provenance.get("moe_kwargs")
    if not kw:
        raise AssertionError(
            "provenance sans 'moe_kwargs' — checkpoint MoE illisible par le harness P3.10 "
            "(bras moe) ; corriger l'init AVANT d'écrire un checkpoint")
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(kw).items()}


def _save_checkpoint_moe(path: Path, *, epoch, model, optimizer, scheduler, scaler,
                         best_miou, cfg, provenance, routing_stats, noise_std):
    """Sauvegarde atomique fsync au FORMAT de scripts/train.py + clés MoE.

    Mêmes clés standard (epoch/model/optimizer/scheduler/scaler/rng/config) pour
    que `_load_checkpoint` du projet relise le run sans adaptation, PLUS :
    `moe_provenance` (traçabilité exigée par P3.08), `moe_kwargs` + `provenance`
    au TOP-LEVEL (contrat du bras `moe` du harness P3.10 — recette exacte de
    l'attach pour reconstruire la couche, sinon reconstruction FAUSSE aux défauts
    PatchMoE2D), `moe_routing_last`, `moe_noise_std`. Le scaler n'est sauvegardé
    que s'il existe (tests CPU).
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_miou": best_miou,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
        "moe_provenance": provenance,
        # Contrat P3.10 (bras `moe` du harness) : la recette EXACTE de l'attach et la
        # provenance doivent être lisibles au TOP-LEVEL du checkpoint. Sans `moe_kwargs`,
        # le harness lève KeyError — et une reconstruction silencieuse avec les défauts
        # PatchMoE2D (sans γ=0) serait FAUSSE. Vérifié de bout en bout le 2026-09-22
        # (self-test harness sur checkpoint au format P3.12, tests CPU Tour).
        "moe_kwargs": _require_moe_kwargs(provenance),
        "provenance": provenance,
        "moe_routing_last": routing_stats,
        "moe_noise_std": float(noise_std),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def train_one_seed_moe(cfg: DictConfig, seed: int, device: torch.device):
    print(f"\n{'='*80}\nMoE-V3-CS training — seed {seed} (apparié B/D/Dp/G)\n{'='*80}")
    deterministic = cfg.training.get("deterministic", False)
    set_seed(seed, deterministic)
    _setup_blackwell_optims(deterministic)

    channels_last = cfg.training.get("channels_last", True)
    if cfg.model.get("distmap", {}).get("enabled", False):
        raise RuntimeError("V3-CS est apparié à la recette B (ce_dice) : distmap doit rester désactivé")

    dataloaders = build_dataloaders(cfg)

    # INIT V3 (P3.08) : architecture = config EMBARQUÉE de B, experts ← spécialistes
    # du MÊME seed (copie vérifiée bit-à-bit, checkpoint manquant = arrêt du run).
    model, moe, provenance, cfg_arch = build_moe_model_from_baseline(
        seed, repo_root=str(REPO_ROOT))
    if cfg_arch.model.get("distmap", {}).get("enabled", False):
        raise RuntimeError("le checkpoint B embarque une config distmap — attendue recette B pure")
    model = model.to(device)
    moe = find_patch_moe(model)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = build_loss(cfg, class_frequencies=dataloaders["class_frequencies"]).to(device)

    opt_cfg = cfg.training.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),   # TOUT est entraîné bout-en-bout (experts, gate, γ) — recette V3 BRATS
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
    )

    total_epochs = cfg.training.epochs
    scheduler = _build_scheduler(optimizer, cfg, total_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    run_name = f"{cfg.experiment.name}_seed{seed}"
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pth"
    routing_path = ROUTING_DIR / f"routing_{run_name}.jsonl"

    start_epoch = 0
    best_miou = 0.0
    if last_path.exists():
        start_epoch, best_miou = _load_checkpoint(
            last_path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler if scaler is not None else _NullScaler(), device=device)
        ck_resume = torch.load(last_path, map_location="cpu", weights_only=False)
        # La provenance AUTORITAIRE est celle du checkpoint (celle du démarrage du
        # run), pas celle recalculée par le rebuild de tout à l'heure.
        provenance = ck_resume.get("moe_provenance", provenance)
        if scheduler.total_iters != total_epochs:
            raise RuntimeError(
                "reprise avec training.epochs différent (%d → %d) : le recuit Poly ET le "
                "recuit du bruit sont calibrés sur N — relancer avec le N du run"
                % (scheduler.total_iters, total_epochs))
        print(f"  [resume] epoch {start_epoch}/{total_epochs} — bruit/recuit déduits de l'époque")
        if start_epoch >= total_epochs:
            print(f"  [resume] déjà terminé, seed {seed} ignoré")
            return best_miou

    try:
        import wandb
        wandb.init(
            project=cfg.experiment.wandb_project,
            name=run_name, id=run_name, resume="allow",
            config={**OmegaConf.to_container(cfg, resolve=True),
                    "moe_provenance": provenance},
            tags=list(cfg.experiment.get("wandb_tags", [])),
        )
    except Exception as e:
        print(f"  wandb init failed ({e}), continuing without wandb")

    accum_steps = cfg.training.get("gradient_accumulation_steps", 1)
    ckpt_every = cfg.evaluation.get("checkpoint_every_epochs", 10)
    max_batches = cfg.training.get("smoke_max_batches", None)
    if max_batches:
        print(f"  [SMOKE] époques tronquées à {max_batches} batches — PAS un run réel")

    t_start = time.time()
    last_stats_epoch = {}
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        train_loss, stats = train_one_epoch_moe(
            model, moe, dataloaders["train"], criterion, optimizer, scaler, device,
            accum_steps, channels_last, epoch, max_batches=max_batches)
        scheduler.step()
        elapsed = time.time() - t0
        vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        last_stats_epoch = stats

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "vram_peak_gb": vram_gb,
            "epoch_time_s": elapsed,
        }
        for k in ("entropy_norm", "entropy_token", "entropy_token_clean", "top_expert_share",
                  "gamma_absmean", "aux_loss_weighted", "noise_std", "part_max"):
            if stats.get(k) is not None:
                log[f"moe/{k}"] = stats[k]
        if "frac" in stats:
            for e_i, f in enumerate(stats["frac"]):
                log[f"moe/frac_e{e_i}"] = f

        print(
            "Epoch %d/%d | loss=%.4f (switch=%.4f) | lr=%.2e | γ=%.4g | part_max=%.3f | "
            "H_norm=%.3f H_tok=%.3f H_clean=%.3f | noise=%.2f | VRAM=%.1fGB | %.0fs"
            % (epoch + 1, total_epochs, train_loss, stats.get("aux_loss_weighted", float("nan")),
               log["lr"], stats.get("gamma_absmean", float("nan")),
               stats.get("part_max", stats.get("top_expert_share", float("nan"))),
               stats.get("entropy_norm", float("nan")),
               stats.get("entropy_token", float("nan")),
               stats.get("entropy_token_clean", float("nan")),
               stats.get("noise_std", float("nan")), vram_gb, elapsed)
        )

        _durable_append_jsonl(routing_path, {"epoch": epoch, "run": run_name, "seed": seed,
                                             "train_loss": train_loss, **{k: v for k, v in stats.items()}})

        _save_checkpoint_moe(
            last_path, epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_miou=best_miou, cfg=cfg, provenance=provenance,
            routing_stats=stats, noise_std=moe.noise_std)

        is_last = epoch == total_epochs - 1
        if (epoch + 1) % ckpt_every == 0 or is_last:
            versioned = checkpoint_dir / f"epoch_{epoch+1:03d}.pth"
            _save_checkpoint_moe(
                versioned, epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, best_miou=best_miou, cfg=cfg, provenance=provenance,
                routing_stats=stats, noise_std=moe.noise_std)
            print(f"  [ckpt] {versioned.name} written")
            _keep = int(cfg.evaluation.get("keep_last_checkpoints", 1))
            if _keep > 0:
                for _old in sorted(checkpoint_dir.glob("epoch_*.pth"))[:-_keep]:
                    try:
                        _old.unlink()
                        print(f"  [ckpt] purged obsolete {_old.name}")
                    except OSError:
                        pass

        _wandb_log(log)

    summary = {
        "run": run_name, "seed": int(seed), "epochs": int(total_epochs),
        "elapsed_s": time.time() - t_start,
        "final_epoch_stats": last_stats_epoch,
        "provenance": provenance,
        "routing_jsonl": str(routing_path),
        "smoke_max_batches": int(max_batches) if max_batches else None,
    }
    sum_path = ROUTING_DIR / f"train_{run_name}_summary.json"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sum_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, sum_path)

    print(f"\nSeed {seed} done after {total_epochs} epochs — summary {sum_path}")
    _wandb_finish()
    return best_miou


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = cfg.experiment.get("seeds", [42])
    for seed in seeds:
        train_one_seed_moe(cfg, seed, device)


if __name__ == "__main__":
    main()
