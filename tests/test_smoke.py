from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from spavr.backbone import CJepaBackbone
from spavr.model import RewardRiskModel
from train.common import build_stable_pretraining_module


class FakeEncoder(nn.Module):
    def forward(self, pixels):
        return {"features": pixels.mean(dim=(-1, -2))}


class FakeInitializer(nn.Module):
    def forward(self, batch_size):
        return torch.zeros(batch_size, 4, 128)


class FakeProcessor(nn.Module):
    def forward(self, initial, features):
        batch_size, num_steps = features.shape[:2]
        return {
            "state": initial[:, None]
            .expand(batch_size, num_steps, 4, 128)
            .clone()
        }


def make_backbone():
    with patch(
        "spavr.backbone.AutoModel.from_pretrained",
        return_value=nn.Identity(),
    ), patch(
        "spavr.backbone.MapOverTime",
        return_value=FakeEncoder(),
    ), patch(
        "spavr.backbone.RandomInit",
        return_value=FakeInitializer(),
    ), patch(
        "spavr.backbone.ScanOverTime",
        return_value=FakeProcessor(),
    ):
        return CJepaBackbone(
            depth=1,
            heads=2,
            dim_head=16,
            mlp_dim=64,
            dropout=0.0,
        )


def make_training_config(anchor_weight):
    return OmegaConf.create(
        {
            "loss": {"anchor_weight": anchor_weight},
            "optim": {
                "head_lr": 5e-4,
                "dynamics_lr": 5e-5,
                "weight_decay": 1e-4,
                "scheduler": "CosineAnnealingLR",
            },
        }
    )


def test_plain_model_accepts_pixels_and_pre_extracted_slots():
    model = RewardRiskModel(backbone=make_backbone())
    actions = torch.randn(2, 10, 6)

    raw_quantiles = model(
        {
            "pixels": torch.randn(2, 5, 3, 8, 8),
            "proprio": torch.randn(2, 5, 4),
            "action": actions,
        }
    )
    slot_quantiles = model(
        {
            "pixels_embed": torch.randn(2, 5, 4, 128),
            "proprio": torch.randn(2, 5, 4),
            "action": actions,
        }
    )

    assert raw_quantiles.shape == (2, 3)
    assert slot_quantiles.shape == (2, 3)
    assert torch.all(raw_quantiles[:, 1:] >= raw_quantiles[:, :-1])
    assert torch.all(slot_quantiles[:, 1:] >= slot_quantiles[:, :-1])


def test_stable_pretraining_frozen_and_predictor_strategies():
    actions = torch.randn(2, 10, 6)
    raw_batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "proprio": torch.randn(2, 5, 4),
        "action": actions,
        "return_to_go": torch.randn(2),
    }
    slot_batch = {
        "pixels_embed": torch.randn(2, 8, 4, 128),
        "proprio": torch.randn(2, 8, 4),
        "action": actions,
        "return_to_go": torch.randn(2),
    }

    frozen = build_stable_pretraining_module(
        RewardRiskModel(backbone=make_backbone()),
        make_training_config(anchor_weight=0.0),
        strategy="frozen",
    )
    frozen.log_dict = lambda *args, **kwargs: None
    frozen_state = frozen(raw_batch, stage="fit")

    assert frozen_state["loss"].ndim == 0
    assert not any(parameter.requires_grad for parameter in frozen.model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in frozen.model.head.parameters())

    finetune = build_stable_pretraining_module(
        RewardRiskModel(backbone=make_backbone()),
        make_training_config(anchor_weight=0.1),
        strategy="predictor",
    )
    finetune.log_dict = lambda *args, **kwargs: None
    finetune_state = finetune(slot_batch, stage="fit")
    finetune_state["loss"].backward()
    world_model = finetune.model.backbone.world_model

    assert "loss_anchor" in finetune_state
    assert world_model.predictor.mask_token.grad is not None
    assert not any(parameter.requires_grad for parameter in world_model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in finetune.model.head.parameters())
