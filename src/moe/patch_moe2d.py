"""Couche Mixture-of-Experts 2D à routage patch-wise, pour la tête UPerNet.

Port 2D fidèle de `BRATS/src/models/moe_patch3d.py` (P3.07 du plan
`PLAN_TRANSFERT_BRATS_V3.md` §3.2) — la brique GAGNANTE du papier 3 BRATS :
un MoE sparse INTERNE au réseau (pas une fusion convexe de réseaux gelés, que
BRATS a mesurée bornée à son meilleur expert), inséré en résiduel dans le
décodeur, avec chaque expert INITIALISÉ depuis un spécialiste de loss déjà
entraîné (P3.08) puis tout ré-entraîné bout-en-bout.

Recette d'origine : Pavlitska et al., "Design and Behavior of Sparse Mixture-of-Experts
Layers in CNN-based Semantic Segmentation" (arXiv 2604.13761) — ablation menée EN 2D sur
Cityscapes/BDD100K, donc ce port est un RETOUR à la dimension native de la recette.
Ses conclusions reprises telles quelles : UNE seule couche MoE en fin de décodeur ;
routage par PATCH (grille ~3x3), jamais image-entière ; 4 experts top-2 ; gate
« 2Conv-GAP » ; pas d'expert partagé ; switch-loss. Réserve explicite : les gains
décroissent quand le réseau grossit (ENet +3.90 mIoU vs DeepLabv3+ +0.35) — avec le
ConvNeXt-V2-Base full-res on s'attend à un gain MODESTE, à mesurer, jamais à supposer.

Hyperparamètres par défaut = ceux du bras V3 de BRATS (`MOE_V3` du trainer
`nnUNetTrainerMedNeXtMoEPatchV3.py`, 1er/29 bras à +0.00566) : 4 experts, top-2,
grille 3x3 (au lieu de 3x3x3), switch-loss 0.001, bruit de Shazeer noise_std=1.0
recuit sur 40 époques. Le dict `MOE_V3_CS` les fige pour le trainer (P3.12).

Deux écarts assumés vs le papier, hérités de BRATS :
  - HALO. Un patch est extrait avec une bordure de `halo` pixels prise chez ses
    voisins, puis la sortie est recadrée. Sans halo, chaque frontière de patch
    verrait un padding zéro et imprimerait la grille dans la segmentation. Avec
    halo >= (k//2) par conv spatiale de l'expert, la convolution est identique à
    celle du même bloc appliqué à la carte entière — la non-régression
    « 1 expert + top-1 == bloc plein-volume » est testée dans
    `tests/test_patch_moe2d.py`. Nuance : la BatchNorm des experts normalise sur
    l'étendue spatiale, donc ses statistiques d'entraînement restent LOCALES au
    lot de patches — c'est intrinsèque au routage par patch, pas un bug (en eval,
    les running stats figées rendent l'égalité exacte).
  - RESIDUEL. La couche est AJOUTÉE à la tête UPerNet (après `fpn_bottleneck`,
    cf. plan §3.1), pas substituée à un bloc existant. On sort `x + moe(x)` pour
    qu'à l'initialisation le réseau reste proche du Baseline B et que la
    comparaison au contrôle parte du même point.

Écart vs BRATS : l'expert par défaut n'est plus un `MedNeXtBlock` (exp_r, 3D)
mais la structure EXACTE du bloc `head.fpn_convs.0` d'UPerNet
(Conv2d 3x3 C->C bias=False + BatchNorm2d + ReLU) : c'est le bloc retenu par
l'audit P3.06 (`AUDIT_DIVERSITE_EXPERTS.md`, distance inter-méthodes 0.42-0.57)
comme source d'initialisation des experts, et une structure identique permet au
P3.08 de copier le state_dict du spécialiste dans l'expert sans la moindre
adaptation de forme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.moe.patch_tiling2d import from_patches, pad_to_grid, to_patches

# Recette V3 de BRATS, transposée 2D (les valeurs du bras gagnant, 1er/29).
MOE_V3_CS = dict(
    n_experts=4, top_k=2, grid=(3, 3), kernel_size=3,
    balance='switch', balance_weight=0.001, noise_std=1.0, noise_anneal=40,
)


def fpn_block_expert(channels, kernel_size=3):
    """Expert par défaut = structure exacte du bloc `head.fpn_convs.0` d'UPerNet.

    `nn.Sequential(Conv2d(C,C,3,padding=1,bias=False), BatchNorm2d(C), ReLU)` :
    les clés (`0.weight`, `1.weight`, `1.bias`, `1.running_*`) correspondent
    une à une à celles d'un checkpoint ConvNeXt-V2+UPerNet du projet, donc
    l'init depuis les spécialistes (P3.08) est un simple `load_state_dict`.
    """
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class PatchMoE2D(nn.Module):
    """MoE sparse 2D : la carte est découpée en gh x gw patchs, chacun routé vers top_k experts.

    Args:
        channels: canaux d'entrée = de sortie (la couche est résiduelle). À l'accroche
            prévue (sortie de `fpn_bottleneck` UPerNet, plan §3.1) : 512.
        n_experts: nombre d'experts (recette V3 BRATS : 4 ; variante V2 aléatoire : 8).
        top_k: experts actifs par patch (recette : 2).
        grid: (gh, gw) — découpage spatial du routage (recette : 3x3).
        kernel_size: noyau des convs spatiales de l'expert par défaut ; pilote le halo.
        halo: bordure prise chez les voisins ; None => kernel_size // 2.
        balance: 'switch' (Fedus 2021, stabilité) | 'entropy' (équilibre de la moyenne) | 'none'.
        balance_weight: poids de la loss auxiliaire, exposée via `aux_loss` (V3 BRATS : 0.001).
        noise_std / noise_anneal: bruit de Shazeer et son recuit linéaire (V3 BRATS : 1.0 / 40).
        expert_factory: callable(channels, kernel_size) -> nn.Module, pour tester avec
            un expert trivial ou remplacer le bloc par défaut.
        residual_scale: None (défaut) = recette BRATS brute, `x + moe(x)` ; float =
            résiduel à échelle APPRENABLE par canal initialisée à cette valeur,
            `x + γ⊙moe(x)` (ReZero/LayerScale). Garde-fou Cityscapes MESURÉ (P3.08,
            2026-09-15) : la recette brute coûte −28.1 pt de mIoU à l'époque 0 sur
            val (accord pixel 91.5 %) — les experts sont initialisés depuis
            `head.fpn_convs.0`, un bloc entraîné sur UNE AUTRE distribution
            (lateral+upsampled) que celle du point d'accroche (sortie de
            `fpn_bottleneck`), et AUCUNE normalisation ne suit avant le classifier
            1×1 ; BRATS tenait à l'ép.0 (pseudo-dice 0.937 vs 0.81 baseline neuve)
            parce que MedNeXt empile des blocs IDENTIQUES (expert = copie du bloc
            voisin, même distribution) et renormalise (GroupNorm) à chaque étage.
            Avec γ=0, l'époque 0 est BIT-EXACTEMENT le contrôle B ; γ (journalisé
            via `last_stats['gamma_absmean']`) devient un diagnostic de la thèse :
            γ→0 = le MoE ne sert à rien, γ qui croît = le réseau s'appuie dessus.
    """

    def __init__(self, channels, n_experts=4, top_k=2, grid=(3, 3), kernel_size=3,
                 halo=None, balance='switch', balance_weight=0.001,
                 noise_std=1.0, noise_anneal=40, expert_factory=None,
                 residual_scale=None):
        super().__init__()
        assert 1 <= top_k <= n_experts
        assert balance in ('switch', 'entropy', 'none')
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.n_experts = n_experts
        self.top_k = top_k
        self.grid = tuple(grid)
        self.halo = kernel_size // 2 if halo is None else int(halo)
        self.balance = balance
        self.balance_weight = float(balance_weight)
        self.noise_std = float(noise_std)
        # Recuit du bruit (leçon BRATS 2026-08-06). Le bruit de Shazeer vaut
        # noise_std x std(logits) : il SUIT l'échelle des logits et ne s'efface
        # donc PAS quand le gate se décide. À bruit constant, rien ne rend le gate
        # responsable de ses choix (mesuré côté BRATS : routage décisif pour le
        # score mais gate sans préférence). Ramener le bruit à zéro en fin de
        # recuit rend le routage déterministe.
        self.noise_std0 = float(noise_std)
        self.noise_anneal = int(noise_anneal)

        factory = expert_factory or fpn_block_expert
        self.experts = nn.ModuleList([factory(channels, kernel_size) for _ in range(n_experts)])

        # Échelle résiduelle apprenable par canal (garde-fou Cityscapes, voir docstring
        # de residual_scale) — absente par défaut pour rester bit-compatible BRATS.
        if residual_scale is not None:
            self.gamma = nn.Parameter(torch.full((self.channels,), float(residual_scale)))
        else:
            self.gamma = None

        # Gate « 2Conv-GAP » : deux convolutions puis moyenne par patch (le GAP du
        # papier devient un pooling adaptatif à la résolution de la grille => un
        # vote par patch).
        hidden = max(8, channels // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, n_experts, kernel_size=1),
        )

        self.aux_loss = None      # tenseur scalaire, lu par le trainer après le forward
        self.last_stats = {}      # diagnostics de routage (fractions, entropies, part du 1er)

    def set_epoch(self, epoch):
        """Recuit linéaire du bruit : noise_std0 -> 0 sur `noise_anneal` époques.

        À appeler au début de chaque époque par le trainer, y compris après une
        reprise : la valeur se DÉDUIT du numéro d'époque au lieu de s'accumuler,
        sans quoi un run repris repartirait avec le bruit de l'époque 0.
        """
        if self.noise_anneal > 0:
            self.noise_std = self.noise_std0 * max(0.0, 1.0 - epoch / float(self.noise_anneal))

    # -- routage ----------------------------------------------------------------

    def _route(self, x):
        """Renvoie (poids (B*N, top_k), indices (B*N, top_k), probas, probas sans bruit)."""
        logits = self.gate(x)
        logits = F.adaptive_avg_pool2d(logits.float(), self.grid)         # (B, E, gh, gw)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.n_experts)   # (B*N, E)
        # Avant bruit : c'est cette distribution-là qui dit si le gate a une
        # préférence ; celle d'après mélange préférence et tirage aléatoire.
        clean = F.softmax(logits, dim=-1)
        if self.training and self.noise_std > 0:
            # Noisy top-k (Shazeer 2017). Sans lui, le gate est quasi constant à
            # l'initialisation : tous les patchs partent vers les MÊMES experts et
            # les autres ne reçoivent jamais de gradient. Le bruit suit std(logits)
            # (leçon BRATS 2026-08-06) : il ne décroît PAS avec la confiance —
            # c'est le recuit `set_epoch` qui l'éteint.
            logits = logits + self.noise_std * logits.std().detach() * torch.randn_like(logits)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        weights = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights, top_i, probs, clean

    @staticmethod
    def _entropy_per_token(p):
        """Entropie moyenne d'un patch pris isolément, en nats."""
        return -(p.clamp_min(1e-9).log() * p).sum(dim=-1).mean()

    def _balance_loss(self, top_i, probs, probs_clean=None):
        """Loss d'équilibrage + diagnostics de routage."""
        n_tok = probs.shape[0]
        onehot = F.one_hot(top_i, self.n_experts).sum(dim=1).clamp_max(1).float()  # (B*N, E)
        frac = onehot.mean(dim=0)              # part des patchs traités par chaque expert
        mean_p = probs.mean(dim=0)             # proba moyenne attribuée à chaque expert

        ent = -(mean_p.clamp_min(1e-9).log() * mean_p).sum()
        max_ent = torch.log(torch.tensor(float(self.n_experts)))
        # Deux entropies, et elles ne disent pas la même chose : celle de la
        # distribution MOYENNE mesure l'équilibrage de charge (ce que la switch
        # loss optimise), celle d'un patch pris seul mesure la netteté de la
        # décision. Spécialisation = entropy_norm haute ET entropy_token basse ;
        # les deux hautes = gate indécis. Ce sont LES diagnostics de la thèse
        # « la porte ne choisit pas » (plan §4) — le trainer DOIT les logger.
        max_ent_t = max_ent.to(probs.device)
        stats = {
            'frac': frac.detach(),
            'entropy_norm': float(ent.detach() / max_ent),
            'entropy_token': float(self._entropy_per_token(probs).detach() / max_ent_t),
            'top_expert_share': float(frac.max().detach()),
            'n_tokens': int(n_tok),
        }
        if probs_clean is not None:
            # Sans le bruit de Shazeer : distingue « gate indécis » de « bruit qui
            # masque une préférence ». Les deux donnent entropy_token ~ 1 sur la
            # version bruitée.
            stats['entropy_token_clean'] = float(
                self._entropy_per_token(probs_clean).detach() / max_ent_t)
        self.last_stats = stats

        if self.balance == 'switch':
            return self.n_experts * (frac * mean_p).sum()
        if self.balance == 'entropy':
            return -ent  # maximiser l'entropie de la distribution MOYENNE = équilibrer
        return probs.new_zeros(())

    # -- forward ----------------------------------------------------------------

    def forward(self, x):
        h_, w = x.shape[2:]
        xg = pad_to_grid(x, self.grid)
        b = xg.shape[0]

        weights, top_i, probs, probs_clean = self._route(xg)
        aux = self._balance_loss(top_i, probs, probs_clean)
        self.aux_loss = self.balance_weight * aux

        patches, shape = to_patches(xg, self.grid, self.halo)
        ph, pw = shape
        hal = self.halo
        out = torch.zeros(patches.shape[0], patches.shape[1], ph, pw,
                          dtype=x.dtype, device=x.device)

        for e, expert in enumerate(self.experts):
            sel = (top_i == e).any(dim=-1).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            y = expert(patches[sel])
            if hal:
                y = y[:, :, hal:hal + ph, hal:hal + pw]
            w_e = (weights * (top_i == e)).sum(dim=-1)[sel].to(y.dtype)
            out.index_add_(0, sel, (y * w_e.view(-1, 1, 1, 1)).to(out.dtype))

        out = from_patches(out, b, self.grid, shape)[:, :, :h_, :w]
        if self.gamma is not None:
            # x + γ⊙moe(x), γ apprenable (ReZero/LayerScale) — garde-fou mesuré P3.08.
            out = self.gamma.view(1, -1, 1, 1) * out
            self.last_stats['gamma_absmean'] = float(self.gamma.detach().abs().mean())
        return x + out
