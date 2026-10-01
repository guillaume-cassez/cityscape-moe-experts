"""Tests CPU P3.12 — scripts/train_moe_v3cs.py (trainer du run MoE-V3-CS apparié).

Sans GPU ni dataset Cityscapes ni checkpoints experts : le MODÈLE des tests est un
petit réseau synthétique contenant une vraie PatchMoE2D, le loader est une liste de
batches, le criterion renvoie (loss, {}) comme le composite du projet. Ce qui est
testé, ce sont les CONTRATS du trainer (les invariants dont dépend la validité du
run réel) :

  1. la switch-loss de la couche EST incluse dans la loss journalisée et rétro-
     propagée (balance_weight énorme ⇒ loss journalisée ≫ loss seg seule) ;
  2. le recuit du bruit se DÉDUIT du numéro d'époque via set_epoch appelé par
     train_one_epoch_moe (reprise après crash sûre : ep ≥ noise_anneal ⇒ bruit 0) ;
  3. le contrat stats routage du plan §4 : part_max/entropy_norm/entropy_token/
     entropy_token_clean/gamma_absmean/frac/aux_loss_weighted/noise_std présentes,
     bornées, et entropy_token_clean présent même à bruit non nul ;
  4. γ reçoit un gradient dès γ=0 (ReZero : le MoE peut se réveiller) ;
  5. roundtrip checkpoint `_save_checkpoint_moe` → `_load_checkpoint` (scripts/train)
     : epoch de reprise, γ préservé, `moe_provenance` conservé, state_dict lisible ;
  6. le JSONL de routage s'append durablement (une ligne par époque, relisible).

Exécution (Tour) :
    /home/ser/brats-venv/bin/python -m pytest tests/test_train_moe_v3cs.py -q
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pytest  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import train_moe_v3cs as T  # noqa: E402
from src.moe.patch_moe2d import PatchMoE2D  # noqa: E402

# scripts/train.py chargé PAR FICHIER (le paquet ~/.local/lib/.../scripts shadow le
# namespace du repo sur la Tour) — même poignée que le trainer.
_build_scheduler = T.train_ctl._build_scheduler
_load_checkpoint = T.train_ctl._load_checkpoint

DEV = torch.device("cpu")
N_CLASSES = 5


class TinySeg(nn.Module):
    """Conv → PatchMoE2D (comme après fpn_bottleneck) → classifier ; sort (main, aux)."""

    def __init__(self, **moe_kwargs):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.moe = PatchMoE2D(channels=8, **moe_kwargs)
        self.cls = nn.Conv2d(8, N_CLASSES, 1)

    def forward(self, x):
        h = self.moe(self.stem(x))
        main = self.cls(h)
        return main, main  # tuple (main, aux) comme UPerNet + aux head


class CELoss:
    """Même contrat que le composite du projet : (loss, dict)."""

    def __call__(self, out, labels, **kw):
        return F.cross_entropy(out.float(), labels), {}


def make_loader(n_batches=3, batch=2, hw=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {"image": torch.randn(batch, 3, hw, hw, generator=g),
         "label": torch.randint(0, N_CLASSES, (batch, hw, hw), generator=g)}
        for _ in range(n_batches)
    ]


V3_LIKE = dict(n_experts=4, top_k=2, grid=(3, 3), balance="switch",
               noise_std=0.0, noise_anneal=40, residual_scale=0.0)


def _optimizer(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr)


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_aux_switch_incluse_dans_loss_et_backward():
    """balance_weight énorme ⇒ la loss journalisée dépasse largement la loss seg seule."""
    torch.manual_seed(0)
    loader = make_loader(seed=1)
    kw = dict(V3_LIKE, balance_weight=137.0)

    torch.manual_seed(7)
    m_big = TinySeg(**kw).to(DEV)
    torch.manual_seed(7)
    m_zero = TinySeg(**dict(kw, balance_weight=0.0)).to(DEV)

    for m in (m_big, m_zero):
        m.eval()  # BN/routes déterministes pour la comparaison

    # accum_steps > n_batches : aucun step/zero_grad, les gradients restent visibles
    loss_big, stats_big = T.train_one_epoch_moe(
        m_big, m_big.moe, loader, CELoss(), _optimizer(m_big, 0.0), None, DEV,
        accum_steps=4, channels_last=False, epoch=0)
    loss_zero, stats_zero = T.train_one_epoch_moe(
        m_zero, m_zero.moe, loader, CELoss(), _optimizer(m_zero, 0.0), None, DEV,
        accum_steps=4, channels_last=False, epoch=0)

    # Modèles identiques (même seed), aux seule différence :
    assert loss_big - loss_zero > 1.0, (loss_big, loss_zero)
    assert stats_big["aux_loss_weighted"] > 0.0
    assert stats_zero["aux_loss_weighted"] < 1e-9
    # γ reçoit un gradient dès γ=0 (ReZero — le MoE peut se réveiller) :
    assert m_big.moe.gamma.grad is not None
    assert float(m_big.moe.gamma.grad.abs().sum()) > 0.0


def test_recuit_bruit_deduit_de_epoque():
    """set_epoch dans train_one_epoch_moe : ep=2/4 ⇒ 0.5, ep≥4 ⇒ 0 (reprise sûre)."""
    torch.manual_seed(0)
    model = TinySeg(**dict(V3_LIKE, noise_std=1.0, noise_anneal=4, balance_weight=0.001)).to(DEV)
    loader = make_loader(n_batches=1)

    T.train_one_epoch_moe(model, model.moe, loader, CELoss(), _optimizer(model, 0.0), None,
                          DEV, accum_steps=1, channels_last=False, epoch=0)
    assert model.moe.noise_std == pytest.approx(1.0)

    T.train_one_epoch_moe(model, model.moe, loader, CELoss(), _optimizer(model, 0.0), None,
                          DEV, accum_steps=1, channels_last=False, epoch=2)
    assert model.moe.noise_std == pytest.approx(0.5)

    # Reprise après crash à une époque POST-recuit : bruit 0, jamais 1 (leçon BRATS).
    T.train_one_epoch_moe(model, model.moe, loader, CELoss(), _optimizer(model, 0.0), None,
                          DEV, accum_steps=1, channels_last=False, epoch=41)
    assert model.moe.noise_std == 0.0


def test_contrat_stats_routage_plan_s4():
    torch.manual_seed(3)
    model = TinySeg(**dict(V3_LIKE, noise_std=1.0, noise_anneal=40, balance_weight=0.001)).to(DEV)
    model.train()
    _, stats = T.train_one_epoch_moe(
        model, model.moe, make_loader(n_batches=2), CELoss(), _optimizer(model, 0.0), None,
        DEV, accum_steps=1, channels_last=False, epoch=5)

    for k in ("entropy_norm", "entropy_token", "entropy_token_clean", "gamma_absmean",
              "aux_loss_weighted", "noise_std", "part_max"):
        assert k in stats, k
    assert len(stats["frac"]) == 4
    # top_k=2 : chaque patch est traité par 2 experts DISTINCTS (topk), donc la somme
    # des fractions par expert vaut top_k, pas 1 (invariant dérivé de _balance_loss).
    assert sum(stats["frac"]) == pytest.approx(float(model.moe.top_k), abs=1e-6)
    assert 0.25 <= stats["part_max"] <= 1.0
    assert 0.0 <= stats["entropy_norm"] <= 1.0 + 1e-9
    assert 0.0 <= stats["entropy_token"] <= 1.0 + 1e-9
    assert 0.0 <= stats["entropy_token_clean"] <= 1.0 + 1e-9
    # bruit recuit à ep 5/40, pas éteint :
    assert stats["noise_std"] == pytest.approx(1.0 * (1 - 5 / 40))
    assert stats["gamma_absmean"] == pytest.approx(0.0)


def test_checkpoint_roundtrip_provenance_gamma(tmp_path):
    torch.manual_seed(4)
    model = TinySeg(**V3_LIKE).to(DEV)
    with torch.no_grad():
        model.moe.gamma.fill_(0.375)
    opt = _optimizer(model, 1e-3)
    cfg = OmegaConf.create({"training": {"scheduler": {"power": 1.0}}, "x": 1})
    sched = _build_scheduler(opt, cfg, 80)
    prov = {"backbone": {"method": "B", "seed": 42, "path": "/x/b.pth", "epoch": 160},
            "experts": [{"expert": 0, "method": "B"}], "gate": "random", "seed": 42,
            "moe_kwargs": {"n_experts": 4, "top_k": 2, "patch_size": [3, 3],
                           "residual_scale": 0.0}}

    ck = tmp_path / "last.pth"
    T._save_checkpoint_moe(ck, epoch=17, model=model, optimizer=opt, scheduler=sched,
                           scaler=None, best_miou=0.0, cfg=cfg, provenance=prov,
                           routing_stats={"part_max": 0.5}, noise_std=0.55)

    raw = torch.load(ck, map_location="cpu", weights_only=False)
    assert raw["epoch"] == 17
    assert raw["moe_provenance"] == prov
    assert raw["moe_noise_std"] == pytest.approx(0.55)
    # Contrat top-level du bras `moe` du harness P3.10 (bug du 2026-09-22 : le harness
    # lisait ck["moe_kwargs"]/ck["provenance"], le trainer ne sauvait que moe_provenance) :
    assert raw["provenance"] == prov
    assert raw["moe_kwargs"] == {"n_experts": 4, "top_k": 2, "patch_size": [3, 3],
                                 "residual_scale": 0.0}
    assert "scaler_state_dict" not in raw  # CPU : pas d'état scaler fantôme

    # provenance SANS moe_kwargs = checkpoint illisible par le harness → échec IMMÉDIAT
    with pytest.raises(AssertionError):
        T._save_checkpoint_moe(tmp_path / "bad.pth", epoch=1, model=model, optimizer=opt,
                               scheduler=sched, scaler=None, best_miou=0.0, cfg=cfg,
                               provenance={"gate": "random"},
                               routing_stats={}, noise_std=0.0)

    torch.manual_seed(99)
    model2 = TinySeg(**V3_LIKE).to(DEV)          # γ différent (0.0)
    opt2 = _optimizer(model2, 1e-3)
    sched2 = _build_scheduler(opt2, cfg, 80)
    start, _best = _load_checkpoint(ck, model=model2, optimizer=opt2, scheduler=sched2,
                                    scaler=T._NullScaler(), device=DEV)
    assert start == 18
    gkey = [k for k in model2.state_dict() if k.endswith("moe.gamma")][0]
    assert float(model2.state_dict()[gkey].mean()) == pytest.approx(0.375)


def test_jsonl_routage_append_durable(tmp_path):
    p = tmp_path / "routing_run.jsonl"
    T._durable_append_jsonl(p, {"epoch": 0, "part_max": 0.6})
    T._durable_append_jsonl(p, {"epoch": 1, "part_max": 0.4, "frac": [0.25] * 4})
    lines = [json.loads(l) for l in p.read_text().splitlines()]
    assert [l["epoch"] for l in lines] == [0, 1]
    assert lines[1]["part_max"] == pytest.approx(0.4)
    assert lines[1]["frac"] == [0.25] * 4
