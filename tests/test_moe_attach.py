"""Tests non-régression P3.08 — attach_patch_moe + initialize_from_experts (CPU).

Vérifient sur modèles/checkpoints SYNTHÉTIQUES (rapides, sans timm ni données) :
  1. l'attach ÉTEND le Sequential head.fpn_bottleneck sans décaler les clés
     (leçon BRATS 2026-08-05) : le state_dict du contrôle se charge en strict=False
     avec missing == exactement les clés MoE et 0 inattendue ;
  2. initialize_from_experts copie chaque expert depuis head.fpn_convs.0 de SON
     spécialiste bit-à-bit (écart max == 0), charge le backbone depuis B, laisse le
     gate aléatoire, journalise la correspondance (traçabilité) et renvoie la
     provenance ;
  3. tout manquement ARRÊTE : checkpoint manquant (FileNotFoundError), clé absente
     (KeyError), forme divergente (AssertionError), nb experts ≠ nb méthodes
     (ValueError), attach double (RuntimeError) ;
  4. appariement de seed : les chemins de la provenance portent tous le même seed ;
  5. après init, le forward passe, le résiduel MoE alimente le graphe (grad sur les
     experts) et les diagnostics de routage sont peuplés.

Exécution (Tour ou toute machine avec torch) :
    python3 -m pytest tests/test_moe_attach.py -q
"""

import copy
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.moe.experts import resolve_expert_path  # noqa: E402
from src.moe.moe_model import (  # noqa: E402
    V3CS_EXPERT_METHODS,
    V3CS_EXPERT_SRC_BLOCK,
    attach_patch_moe,
    find_patch_moe,
    initialize_from_experts,
)
from src.moe.patch_moe2d import PatchMoE2D  # noqa: E402
from src.models.builder import SegmentationModel  # noqa: E402
from src.models.heads.upernet import UPerNetHead  # noqa: E402

CH = (8, 16, 32, 64)          # canaux de stages minuscules (pas de timm en test)
HEAD_CH = 8                   # canaux de la head => expert 8->8, gate 8->4
SEED = 42


class TinyBackbone(nn.Module):
    """Backbone synthétique : une conv par stage, strides 4/8/16/32."""

    def __init__(self, channels=CH):
        super().__init__()
        self.channels = tuple(channels)
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, c, 3, padding=1, stride=s) for c, s in zip(channels, (4, 8, 16, 32))])

    def forward(self, x):
        return [c(x) for c in self.convs]


def tiny_model():
    head = UPerNetHead(in_channels=list(CH), channels=HEAD_CH, num_classes=19)
    return SegmentationModel(TinyBackbone(), head)


def silent_log():
    msgs = []
    return msgs, msgs.append


# -- dépôts synthétiques (layout real : checkpoints/ + reeval_slim/, cf. EXPERT_SPECS) --

def make_synthetic_repo(root, seed=SEED, methods=V3CS_EXPERT_METHODS):
    """Écrit un mini-checkpoint par méthode : bloc source tagué d'une valeur
    reconnaissable (e+1), classifier de B tagué 7.0 (preuve de la source backbone)."""
    base_sd = tiny_model().state_dict()
    paths = {}
    for i, m in enumerate(methods):
        sd = copy.deepcopy(base_sd)
        for k in sd:
            if k.startswith(V3CS_EXPERT_SRC_BLOCK + "."):
                sd[k].fill_(float(i + 1))          # marqueur par méthode
        if m == "B":
            sd["head.classifier.bias"].fill_(7.0)  # marqueur backbone-source
        path = resolve_expert_path(m, seed, str(root))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model_state_dict": sd, "epoch": 159, "config": {}}, path)
        paths[m] = path
    return paths


def attached_tiny(n_experts=4, **kw):
    model = tiny_model()
    moe = attach_patch_moe(model, n_experts=n_experts, grid=(3, 3), **kw)
    return model, moe


# --------------------------------------------------------------------------- attach

class TestAttach:
    def testClefsControleInchangees(self):
        model = tiny_model()
        sd_ctrl = copy.deepcopy(model.state_dict())
        moe = attach_patch_moe(model, n_experts=4, grid=(3, 3))
        sd_new = model.state_dict()

        # Aucune clé existante décalée ni modifiée (leçon BRATS : pas de ré-emballage).
        for k, v in sd_ctrl.items():
            assert k in sd_new, "clé contrôle disparue : %s" % k
            assert torch.equal(v, sd_new[k]), "valeur altérée : %s" % k

        # Clés ajoutées == exactement celles du MoE, sous le préfixe attendu.
        assert moe.state_dict_prefix == "head.fpn_bottleneck.3."
        ajoutees = set(sd_new) - set(sd_ctrl)
        assert ajoutees == {moe.state_dict_prefix + k for k in moe.state_dict()}
        assert ajoutees, "aucune clé MoE ajoutée ?"

    def testChargementControleStrictFalse(self):
        model = tiny_model()
        sd_ctrl = copy.deepcopy(model.state_dict())
        moe = attach_patch_moe(model, n_experts=4)
        missing, unexpected = model.load_state_dict(sd_ctrl, strict=False)
        clefs_moe = {moe.state_dict_prefix + k for k in moe.state_dict()}
        # NB : torch EXCLUT num_batches_tracked de missing_keys quand le state_dict
        # n'a pas de metadata de version (cas des checkpoints du projet) — d'où le
        # ⊆ ci-dessous, même tolérance que l'assert de initialize_from_experts.
        # Ce buffer ne sert qu'au momentum cumulatif (BN momentum=None), jamais ici.
        assert set(missing) <= clefs_moe
        assert clefs_moe - set(missing) <= {k for k in clefs_moe if k.endswith("num_batches_tracked")}
        assert any(k.endswith(".experts.0.0.weight") for k in missing)
        assert not unexpected

    def testRefuseDoubleAttach(self):
        model, _ = attached_tiny()
        with pytest.raises(RuntimeError):
            attach_patch_moe(model, n_experts=4)

    def testRefuseHeadNonUPerNet(self):
        with pytest.raises(TypeError):
            attach_patch_moe(nn.Linear(2, 2), n_experts=4)

    def testCanauxPrisSurLeBottleneck(self):
        _, moe = attached_tiny(n_experts=4)
        # fpn_bottleneck[0] : Conv2d(len(CH)*HEAD_CH -> HEAD_CH) => expert HEAD_CH.
        assert moe.channels == HEAD_CH
        assert moe.experts[0][0].in_channels == HEAD_CH == moe.experts[0][0].out_channels


# ------------------------------------------------------------------- initialize

class TestInitialize:
    def testCopieBitExacteEtTracabilite(self, tmp_path):
        make_synthetic_repo(tmp_path)
        model, moe = attached_tiny(n_experts=4)
        gate_avant = copy.deepcopy(moe.gate.state_dict())
        msgs, log = silent_log()
        prov = initialize_from_experts(model, SEED, str(tmp_path), log=log)

        # Chaque expert == le marqueur de SA méthode (e+1), bit-à-bit, buffers inclus.
        for e in range(4):
            for k, v in moe.experts[e].state_dict().items():
                assert torch.all(v == float(e + 1)), "expert %d clé %s mal copiée" % (e, k)
        # Backbone ← B : le marqueur classifier de B est en place.
        assert torch.all(model.head.classifier.bias == 7.0)
        # Gate inchangé (aléatoire d'origine — c'est celui qu'on veut observer).
        for k, v in gate_avant.items():
            assert torch.equal(v, moe.gate.state_dict()[k]), "gate modifié (%s)" % k

        # Provenance : ordre, seeds appariés, écart nul, chemins réels.
        assert [x["method"] for x in prov["experts"]] == list(V3CS_EXPERT_METHODS)
        assert prov["backbone"]["method"] == "B" and prov["backbone"]["seed"] == SEED
        assert prov["gate"] == "random" and prov["source_block"] == V3CS_EXPERT_SRC_BLOCK
        for x in prov["experts"]:
            assert x["seed"] == SEED and x["max_ecart"] == 0.0
            assert os.path.isfile(x["path"])
        # Journal : une ligne de correspondance par expert + backbone + gate.
        texte = "\n".join(msgs)
        for e, m in enumerate(V3CS_EXPERT_METHODS):
            assert ("expert %d ← %s" % (e, m)) in texte, "trace absente pour expert %d" % e
        assert "backbone+head ← B" in texte and "gate ← aléatoire" in texte

    def testSeedApparie(self, tmp_path):
        # Un dépôt au seed 123 : la provenance ne doit contenir QUE des chemins seed123.
        make_synthetic_repo(tmp_path, seed=123)
        model, _ = attached_tiny(n_experts=4)
        msgs, log = silent_log()
        prov = initialize_from_experts(model, 123, str(tmp_path), log=log)
        chemins = [prov["backbone"]["path"]] + [x["path"] for x in prov["experts"]]
        assert all("seed123" in p for p in chemins), chemins

    def testCheckpointManquantArrete(self, tmp_path):
        paths = make_synthetic_repo(tmp_path)
        os.remove(paths["Dp"])                       # expert 2 absent => run FAUX
        model, _ = attached_tiny(n_experts=4)
        with pytest.raises(FileNotFoundError, match="Dp|expert 2"):
            initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])

    def testBackboneManquantArrete(self, tmp_path):
        paths = make_synthetic_repo(tmp_path)
        os.remove(paths["B"])
        model, _ = attached_tiny(n_experts=4)
        with pytest.raises(FileNotFoundError, match="backbone"):
            initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])

    def testCleSourceAbsenteArrete(self, tmp_path):
        paths = make_synthetic_repo(tmp_path)
        ck = torch.load(paths["G"], map_location="cpu", weights_only=False)
        del ck["model_state_dict"][V3CS_EXPERT_SRC_BLOCK + ".0.weight"]
        torch.save(ck, paths["G"])
        model, _ = attached_tiny(n_experts=4)
        with pytest.raises(KeyError, match="absente"):
            initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])

    def testFormeDivergenteArrete(self, tmp_path):
        paths = make_synthetic_repo(tmp_path)
        ck = torch.load(paths["D"], map_location="cpu", weights_only=False)
        cle = V3CS_EXPERT_SRC_BLOCK + ".1.bias"
        ck["model_state_dict"][cle] = torch.zeros(HEAD_CH + 1)
        torch.save(ck, paths["D"])
        model, _ = attached_tiny(n_experts=4)
        with pytest.raises(AssertionError, match="forme"):
            initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])

    def testNbExpertsIncoherentArrete(self, tmp_path):
        make_synthetic_repo(tmp_path)
        model, _ = attached_tiny(n_experts=8)        # V2 : 8 experts, 4 méthodes
        with pytest.raises(ValueError, match="experts"):
            initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])

    def testAttachManquantArrete(self, tmp_path):
        make_synthetic_repo(tmp_path)
        with pytest.raises(AssertionError, match="attach"):
            initialize_from_experts(tiny_model(), SEED, str(tmp_path), log=silent_log()[1])

    def testReinitIdempotente(self, tmp_path):
        # Deux inits successives (ex. reprise à chaud) donnent le même état exact.
        make_synthetic_repo(tmp_path)
        model, moe = attached_tiny(n_experts=4)
        q = silent_log()[1]
        initialize_from_experts(model, SEED, str(tmp_path), log=q)
        snap = copy.deepcopy(moe.experts.state_dict())
        initialize_from_experts(model, SEED, str(tmp_path), log=q)
        for k, v in snap.items():
            assert torch.equal(v, moe.experts.state_dict()[k]), k


# ------------------------------------------------------------------- bout en bout

class TestApresInit:
    def testForwardEtGradient(self, tmp_path):
        make_synthetic_repo(tmp_path)
        model, moe = attached_tiny(n_experts=4)
        initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])
        x = torch.randn(1, 3, 64, 128)

        model.eval()
        out = model(x)
        assert out.shape == (1, 19, 64, 128)
        assert torch.isfinite(out).all()
        assert "entropy_token" in moe.last_stats and "entropy_norm" in moe.last_stats

        model.train()
        moe.set_epoch(0)
        # batch=2 : la BatchNorm du PPM (pool 1x1) refuse le batch=1 en mode train
        # (comportement UPerNetHead préexistant, sans rapport avec le MoE).
        x2 = torch.randn(2, 3, 64, 128)
        out = model(x2)
        loss = out.mean() + moe.aux_loss
        loss.backward()
        # Au moins un expert DOIT recevoir du gradient (le routage top-2 sur 4 peut
        # légitimement en écarter un sur un petit lot — d'où le test sur la somme).
        grads = [moe.experts[e][0].weight.grad for e in range(moe.n_experts)]
        n_grad = sum(g is not None and g.abs().sum() > 0 for g in grads)
        assert n_grad >= 1, "aucun expert initialisé ne reçoit de gradient"
        assert all(g is None or torch.isfinite(g).all() for g in grads)
        assert moe.gate[0].weight.grad is not None

    def testUneSeuleCoucheTrouvable(self, tmp_path):
        model, moe = attached_tiny(n_experts=4)
        assert find_patch_moe(model) is moe
        assert find_patch_moe(tiny_model(), required=False) is None


# ------------------------------------------------------- garde-fou résiduel γ (P3.08)

class TestResidualScale:
    """γ=0 (MOE_V3_CS_INIT) : mesure P3.08 — la recette BRATS brute coûte −28.1 pt à
    l'ép.0 (experts initialisés depuis un bloc d'UNE AUTRE distribution, classifier
    sans renormalisation). γ apprenable par canal ⇒ ép.0 bit-exacte au contrôle."""

    def testGammaZeroBitExactAuControle(self, tmp_path):
        paths = make_synthetic_repo(tmp_path)
        ctrl = tiny_model()
        ck_b = torch.load(paths["B"], map_location="cpu", weights_only=False)
        ctrl.load_state_dict(ck_b["model_state_dict"], strict=True)
        ctrl.eval()

        model, moe = attached_tiny(n_experts=4, residual_scale=0.0)
        initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])
        model.eval()

        x = torch.randn(1, 3, 64, 128)
        with torch.no_grad():
            diff = (model(x) - ctrl(x)).abs().max().item()
        assert diff == 0.0, "γ=0 doit donner EXACTEMENT la sortie du contrôle, écart %g" % diff
        assert moe.gamma is not None and float(moe.gamma.abs().max()) == 0.0
        assert "gamma_absmean" in moe.last_stats

    def testGammaReguEtGradient(self, tmp_path):
        make_synthetic_repo(tmp_path)
        model, moe = attached_tiny(n_experts=4, residual_scale=0.0)
        initialize_from_experts(model, SEED, str(tmp_path), log=silent_log()[1])
        model.train()
        moe.set_epoch(0)
        out = model(torch.randn(2, 3, 64, 128))
        (out.mean() + moe.aux_loss).backward()
        assert moe.gamma.grad is not None
        assert torch.isfinite(moe.gamma.grad).all()
        assert moe.gamma.grad.abs().sum() > 0, "γ ne reçoit aucun gradient — ReZero cassé"

    def testGammaUnEgalRecetteBrute(self, tmp_path):
        # residual_scale=1.0 ≡ recette BRATS brute x + moe(x) (aucune régression de
        # comportement pour qui veut la rejouer, ex. ablation P3.13).
        make_synthetic_repo(tmp_path)
        torch.manual_seed(7)
        m_brut, moe_brut = attached_tiny(n_experts=4)                    # residual_scale=None
        torch.manual_seed(7)
        m_gamma, moe_gamma = attached_tiny(n_experts=4, residual_scale=1.0)
        q = silent_log()[1]
        initialize_from_experts(m_brut, SEED, str(tmp_path), log=q)
        initialize_from_experts(m_gamma, SEED, str(tmp_path), log=q)
        assert moe_brut.gamma is None
        m_brut.eval(); m_gamma.eval()
        x = torch.randn(1, 3, 64, 128)
        with torch.no_grad():
            diff = (m_brut(x) - m_gamma(x)).abs().max().item()
        assert diff == 0.0, "γ=1 doit reproduire la recette brute, écart %g" % diff

    def testDefautBrutInchange(self):
        # Sans residual_scale : PAS de paramètre γ (comportement P3.07 bit-compatible).
        _, moe = attached_tiny(n_experts=4)
        assert moe.gamma is None
        assert not any(n == "gamma" for n, _ in moe.named_parameters())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
