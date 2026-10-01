"""P3.08 — Accroche du MoE-V3-CS sur UPerNet + initialisation depuis les 4 experts.

Deux fonctions, calquées sur la recette GAGNANTE du papier 3 BRATS (V3, 1er/29 bras
à +0.00566, `nnUNetTrainerMedNeXtMoEPatchV3.py`) :

1. `attach_patch_moe` — insère une `PatchMoE2D` comme DERNIER élément du Sequential
   `head.fpn_bottleneck` d'UPerNet (plan §3.1 : l'analogue du `dec_block_0` MedNeXt).
   Leçon BRATS du 2026-08-05 : on ÉTEND le Sequential au lieu de l'emballer dans un
   second — les blocs originaux gardent leurs indices (0=conv, 1=BN, 2=ReLU), le MoE
   prend l'indice 3, donc le checkpoint du contrôle B se charge en `strict=False`
   sans AUCUNE clé inattendue ni manquante hors couche MoE. Le forward d'UPerNet
   n'est pas modifié : `fused = fpn_bottleneck(cat)` devient
   `relu(bn(conv(cat)))` puis `moe(fused) = fused + experts(fused)` — le résiduel
   est géré par PatchMoE2D lui-même.

2. `initialize_from_experts` — l'ingrédient que BRATS a mesuré décisif (le gain vient
   de l'init, pas du routage) :
   - backbone + head ← checkpoint du Baseline **B**, `load_state_dict(strict=False)` :
     les clés manquantes doivent être EXACTEMENT celles de la couche MoE et il doit
     rester 0 clé inattendue (sinon l'architecture a dérivé → on arrête) ;
   - chaque expert e ← le bloc `head.fpn_convs.0` du spécialiste e du MÊME seed
     (verrouillé par l'audit P3.06 : distance inter-méthodes 0.42-0.57 sur ce bloc,
     et drift de spécialisation NON reproductible entre seeds → appariement strict,
     analogue des folds appariés BRATS ; jamais de mélange inter-seeds) ;
   - gate ← aléatoire (c'est lui qu'on veut observer en train d'apprendre, ou de ne
     PAS apprendre — thèse « la porte ne choisit pas ») ;
   - TRAÇABILITÉ : chaque copie est vérifiée bit-à-bit (écart max au checkpoint
     source = 0, sinon AssertionError) et journalisée ; un checkpoint manquant lève
     FileNotFoundError — le run serait FAUX, il doit être arrêté (principe BRATS).
   La correspondance expert→checkpoint est renvoyée dans un dict `provenance` que le
   trainer (P3.12) DOIT sauvegarder dans le checkpoint du run.

Alignement des experts (§2.1 du plan, miroir EXACT de B/D/K/G de BRATS — décision Guillaume
2026-09-17 : le 4e spécialiste devient le blob de Kofler, C sort de l'init et reste bras
gelé pour late-fusion/consensus) :
    expert 0 ← B  Baseline (CE+Dice)       — contrôle + source du backbone
    expert 1 ← D  Kervadec (CE+Boundary)   — spécialisation frontière (≈ K BRATS)
    expert 2 ← Dp DistMap (CE+DistMap)     — 2e géométrie de frontière (≈ D BRATS)
    expert 3 ← G  Blob (CE+Dice+0.5·blob)  — spécialisation instance-équilibre (≈ G BRATS)
"""

import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models import build_model
from src.moe.experts import EXPERT_LABELS, resolve_expert_path
from src.moe.patch_moe2d import MOE_V3_CS, PatchMoE2D

# Recette figée par le plan (P3.05/P3.06) + décision Guillaume 2026-09-17 (expert 3 = G blob,
# miroir B/D/K/G de BRATS) — ne pas changer sans re-mesurer.
V3CS_EXPERT_METHODS = ("B", "D", "Dp", "G")
V3CS_BACKBONE_SOURCE = "B"
V3CS_EXPERT_SRC_BLOCK = "head.fpn_convs.0"   # audit P3.06 : seul bloc fortement différencié
V3CS_ATTACH_SEQ = "fpn_bottleneck"           # plan §3.1 : dernier bloc conv avant classifier

# MOE_V3_CS (recette BRATS brute) + le garde-fou résiduel γ=0 MESURÉ nécessaire ici
# (P3.08, 2026-09-15) : sans γ, la recette brute coûte −28.1 pt de mIoU à l'époque 0
# (accord pixel 91.5 %, classes fines devastées : camion −87, cavalier −78, feu −70)
# parce que les experts initialisés depuis `head.fpn_convs.0` voient au point
# d'accroche une distribution qu'ils n'ont jamais traitée (sortie de fpn_bottleneck
# au lieu de lateral+upsampled), sans aucune renormalisation avant le classifier.
# BRATS n'avait pas ce problème (blocs MedNeXt identiques partout + GroupNorm).
# Avec γ=0 apprenable par canal : époque 0 BIT-EXACTE au contrôle B, et |γ| devient
# un diagnostic de la thèse « la porte ne choisit pas » (γ→0 = MoE inutile).
MOE_V3_CS_INIT = dict(MOE_V3_CS, residual_scale=0.0)


def find_patch_moe(model: nn.Module, required: bool = True):
    """Renvoie l'unique couche PatchMoE2D du modèle (None si absente et non requise)."""
    layers = [m for m in model.modules() if isinstance(m, PatchMoE2D)]
    if len(layers) > 1:
        raise AssertionError("attendu au plus 1 couche PatchMoE2D, mesuré %d" % len(layers))
    if required and not layers:
        raise AssertionError("aucune couche PatchMoE2D dans le modèle — attach_patch_moe d'abord")
    return layers[0] if layers else None


def attach_patch_moe(model: nn.Module, **moe_kwargs) -> PatchMoE2D:
    """Insère une PatchMoE2D en dernier élément de `model.head.fpn_bottleneck`.

    Args:
        model: SegmentationModel dont la head est une UPerNetHead.
        **moe_kwargs: transmis à PatchMoE2D (défaut du trainer V3-CS : `MOE_V3_CS`).

    Returns:
        la couche, pour que le trainer lise `aux_loss` et `last_stats` après forward.
    """
    head = getattr(model, "head", None)
    seq = getattr(head, V3CS_ATTACH_SEQ, None)
    if not isinstance(seq, nn.Sequential):
        raise TypeError(
            "attach_patch_moe attend une UPerNetHead (head.%s = nn.Sequential), obtenu %r"
            % (V3CS_ATTACH_SEQ, type(seq).__name__))
    if find_patch_moe(model, required=False) is not None:
        raise RuntimeError("une PatchMoE2D est déjà attachée à ce modèle")

    channels = seq[0].out_channels   # Conv2d(len(in_channels)*C -> C), sortie C=512
    moe = PatchMoE2D(channels=channels, **moe_kwargs)

    children = list(seq.children())
    # Préfixe ABSOLU (depuis la racine du modèle) des clés du MoE dans le state_dict :
    # sert à l'assert « missing == clés MoE » de initialize_from_experts.
    moe.state_dict_prefix = "head.%s.%d." % (V3CS_ATTACH_SEQ, len(children))
    setattr(head, V3CS_ATTACH_SEQ, nn.Sequential(*children, moe))
    return moe


def initialize_from_experts(model: nn.Module, seed: int, repo_root: str,
                            methods=V3CS_EXPERT_METHODS,
                            backbone_source: str = V3CS_BACKBONE_SOURCE,
                            log=print) -> dict:
    """Initialisation V3 : backbone ← Baseline, experts ← bloc spécialiste, gate ← aléa.

    Tous les checkpoints sont résolus au MÊME seed (appariement P3.06 §3.3-2b).
    Toute anomalie (checkpoint absent, clé manquante/inattendue, forme différente,
    copie non bit-exacte) lève et DOIT arrêter le run : une init silencieusement
    partielle fausserait la comparaison au contrôle.

    Returns: provenance (dict sérialisable, à sauvegarder dans le checkpoint du run).
    """
    moe = find_patch_moe(model)
    methods = tuple(methods)
    if moe.n_experts != len(methods):
        raise ValueError("la couche a %d experts, %d méthodes déclarées (%s)"
                         % (moe.n_experts, len(methods), methods))
    prefix = getattr(moe, "state_dict_prefix", None)
    if not prefix:
        raise RuntimeError("state_dict_prefix absent — attach_patch_moe n'a pas été utilisé")

    log("[MoE-V3-CS] === initialisation depuis les experts, seed %d (apparié, P3.06 §2b) ===" % seed)

    # 1) Backbone + head ← Baseline B. Les clés du MoE (experts + gate) manquent, et
    #    RIEN d'autre ne doit manquer ni dépasser : sinon l'architecture a dérivé.
    path_b = resolve_expert_path(backbone_source, seed, repo_root)
    if not os.path.isfile(path_b):
        raise FileNotFoundError("checkpoint backbone (%s seed %d) introuvable : %s"
                                % (backbone_source, seed, path_b))
    ck_b = torch.load(path_b, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck_b["model_state_dict"], strict=False)
    clefs_moe = {prefix + k for k in moe.state_dict()}
    # ⊆ et non == : torch exclut num_batches_tracked des missing_keys pour un
    # state_dict sans metadata de version (cas des checkpoints du projet). Buffer
    # sans effet ici (BN à momentum exponentiel, pas cumulatif).
    extra_missing = sorted(set(missing) - clefs_moe)
    if extra_missing:
        raise AssertionError("clés manquantes ≠ couche MoE (%d) : %s"
                             % (len(extra_missing), extra_missing[:8]))
    if unexpected:
        raise AssertionError("clés inattendues du backbone : %s" % sorted(unexpected)[:8])
    log("[MoE-V3-CS] backbone+head ← %s (%s) seed %d : %s — %d clés MoE manquantes (normal), "
        "0 inattendue" % (backbone_source, EXPERT_LABELS.get(backbone_source, "?"), seed,
                          path_b, len(clefs_moe)))
    provenance = {
        "backbone": {"method": backbone_source, "seed": int(seed), "path": path_b,
                     "epoch": int(ck_b.get("epoch", -1)) + 1},
        "experts": [], "gate": "random",
        "source_block": V3CS_EXPERT_SRC_BLOCK,
        "seed": int(seed),
    }
    del ck_b

    # 2) Chaque expert ← le bloc source du checkpoint spécialiste du MÊME seed.
    for e, nom in enumerate(methods):
        path = resolve_expert_path(nom, seed, repo_root)
        if not os.path.isfile(path):
            raise FileNotFoundError("checkpoint expert %d (%s) introuvable : %s — run FAUX, "
                                    "arrêt obligatoire" % (e, nom, path))
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        expert = moe.experts[e]

        # Correspondance structurelle exacte : chaque clé de l'expert (paramètres ET
        # buffers BN) doit exister dans le checkpoint source, avec la même forme.
        esd = expert.state_dict()
        src_sd = {}
        for k, v in esd.items():
            cle = "%s.%s" % (V3CS_EXPERT_SRC_BLOCK, k)
            if cle not in sd:
                raise KeyError("expert %d (%s) : clé %s absente du checkpoint" % (e, nom, cle))
            if tuple(v.shape) != tuple(sd[cle].shape):
                raise AssertionError("expert %d (%s) : clé %s forme %s attendue %s"
                                     % (e, nom, cle, tuple(sd[cle].shape), tuple(v.shape)))
            src_sd[k] = sd[cle]
        expert.load_state_dict(src_sd, strict=True)

        # Preuve dans le journal (leçon BRATS) : l'écart maximal au checkpoint source
        # doit être EXACTEMENT nul — une copie partielle ferait un expert hybride
        # sans le dire. Tenseurs source sur CPU, expert potentiellement sur GPU.
        pire = 0.0
        apres = expert.state_dict()
        for k, src in src_sd.items():
            d = float((apres[k].detach().to(src.device).double() - src.double()).abs().max())
            pire = max(pire, d)
        if pire != 0.0:
            raise AssertionError("expert %d (%s) mal copié (écart %.3g)" % (e, nom, pire))
        log("[MoE-V3-CS] expert %d ← %s (%s) seed %d — bloc %s — écart max au source %.3e"
            % (e, nom, EXPERT_LABELS.get(nom, "?"), seed, V3CS_EXPERT_SRC_BLOCK, pire))
        provenance["experts"].append(
            {"expert": e, "method": nom, "label": EXPERT_LABELS.get(nom, "?"),
             "seed": int(seed), "path": path, "max_ecart": pire})
        del sd, src_sd, apres

    # 3) Gate ← aléatoire : ne PAS l'initialiser (c'est lui qu'on veut voir apprendre).
    log("[MoE-V3-CS] gate ← aléatoire (non initialisé — thèse « la porte ne choisit pas »)")
    return provenance


def build_moe_model_from_baseline(seed: int, repo_root: str, device=None, log=print,
                                  **moe_kwargs):
    """Construit le modèle MoE-V3-CS complet, initialisé depuis les experts du seed.

    L'architecture est celle du checkpoint B (config embarquée — garantie de
    correspondance des clés), le MoE prend les défauts `MOE_V3_CS_INIT` (recette du
    bras V3 gagnant BRATS + garde-fou résiduel γ=0 mesuré nécessaire, cf. plus haut)
    surchargeables par `**moe_kwargs`.

    Returns: (model, moe, provenance, cfg) — model sur CPU si device=None.
    """
    path_b = resolve_expert_path(V3CS_BACKBONE_SOURCE, seed, repo_root)
    if not os.path.isfile(path_b):
        raise FileNotFoundError("checkpoint Baseline introuvable : %s" % path_b)
    ck = torch.load(path_b, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.backbone.pretrained = "none"  # poids tous écrasés ; jamais de download timm
    del ck
    model = build_model(cfg)

    kwargs = dict(MOE_V3_CS_INIT)
    kwargs.update(moe_kwargs)
    moe = attach_patch_moe(model, **kwargs)
    provenance = initialize_from_experts(model, seed, repo_root, log=log)
    provenance["moe_kwargs"] = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in kwargs.items()}

    if device is not None:
        model.to(device)
    return model, moe, provenance, cfg
