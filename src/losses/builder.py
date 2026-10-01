"""
Loss builder: constructs composite loss functions from config.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


def build_loss(cfg: DictConfig, class_frequencies=None) -> nn.Module:
    """Build composite loss from config."""
    components = []
    weights = []

    for comp_cfg in cfg.loss.components:
        loss_type = comp_cfg.type
        weight = comp_cfg.weight

        if loss_type == "cross_entropy":
            loss = CrossEntropyLoss(
                class_weights=comp_cfg.get("class_weights"),
                class_frequencies=class_frequencies,
                ignore_index=comp_cfg.get("ignore_index", 255),
            )
        elif loss_type == "dice":
            loss = DiceLoss(
                smooth=comp_cfg.get("smooth", 1.0),
                ignore_index=comp_cfg.get("ignore_index", 255),
            )
        elif loss_type == "boundary":
            from src.losses.boundary_loss import BoundaryLoss
            loss = BoundaryLoss(
                theta0=comp_cfg.get("theta0", 3),
                ignore_index=comp_cfg.get("ignore_index", 255),
            )
        elif loss_type == "blob":
            from src.losses.blob_loss import BlobLoss
            loss = BlobLoss(
                eps=comp_cfg.get("eps", 1.0),
                ignore_index=comp_cfg.get("ignore_index", 255),
                min_blob_pixels=comp_cfg.get("min_blob_pixels", 0),
            )
        elif loss_type == "adaptive_focal":
            from src.losses.adaptive_focal import AdaptiveFocalLoss
            loss = AdaptiveFocalLoss(
                alpha=comp_cfg.get("alpha", 0.25),
                gamma_base=comp_cfg.get("gamma_base", 2.0),
                gamma_scale=comp_cfg.get("gamma_scale", 1.5),
                size_threshold=comp_cfg.get("size_threshold", 0.01),
                ignore_index=comp_cfg.get("ignore_index", 255),
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        components.append(loss)
        weights.append(weight)

    return CompositeLoss(components, weights)


class CompositeLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Optional `sdt` kwarg routed to BoundaryLoss only, optional `blob` kwarg (paquetage
    d'instances pré-calculé du dataloader) routed to BlobLoss only.
    """

    def __init__(self, components: list, weights: list):
        super().__init__()
        self.components = nn.ModuleList(components)
        self.weights = weights

    def forward(self, pred, target, sdt=None, blob=None):
        from src.losses.boundary_loss import BoundaryLoss
        from src.losses.blob_loss import BlobLoss
        total = 0.0
        loss_dict = {}
        for comp, w in zip(self.components, self.weights):
            if isinstance(comp, BoundaryLoss):
                l = comp(pred, target, sdt=sdt)
            elif isinstance(comp, BlobLoss):
                l = comp(pred, target, blob=blob)
            else:
                l = comp(pred, target)
            total = total + w * l
            loss_dict[comp.__class__.__name__] = l.item()
        loss_dict["total"] = total.item()
        return total, loss_dict


class CrossEntropyLoss(nn.Module):
    """Cross-entropy with optional ISNS class weighting."""

    def __init__(self, class_weights=None, class_frequencies=None, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

        weight = None
        if class_weights == "isns" and class_frequencies is not None:
            # Inverse Square Root Number of Samples
            freq = torch.tensor(class_frequencies, dtype=torch.float32)
            weight = 1.0 / torch.sqrt(freq + 1e-6)
            weight = weight / weight.sum() * len(weight)

        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.ce(pred, target)


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""

    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        pred_soft = torch.softmax(pred, dim=1)

        mask = target != self.ignore_index
        target_clean = target.clone()
        target_clean[~mask] = 0

        target_onehot = torch.zeros_like(pred_soft)
        target_onehot.scatter_(1, target_clean.unsqueeze(1), 1)

        mask_expanded = mask.unsqueeze(1).expand_as(pred_soft)
        pred_masked = pred_soft * mask_expanded
        target_masked = target_onehot * mask_expanded

        dims = (0, 2, 3)
        intersection = (pred_masked * target_masked).sum(dims)
        cardinality = (pred_masked + target_masked).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()
