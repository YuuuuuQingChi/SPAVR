import pickle
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from spavr.backbone import CJepaBackbone
from spavr.heads import pinball_loss
from spavr.model import RewardRiskModel
from third_party.cjepa.src.custom_codes.custom_dataset import PushTSlotDataset


def build_reward_model(cfg):
    """Build SPAVR and optionally load an official C-JEPA model object."""
    backbone = CJepaBackbone(
        history_size=cfg.model.history_size,
        predicted_size=cfg.model.predicted_size,
        num_objects=cfg.model.num_objects,
        slot_dim=cfg.model.slot_dim,
        action_dim=cfg.model.action_dim,
        proprio_dim=cfg.model.proprio_dim,
        frameskip=cfg.model.frameskip,
        num_masked_slots=cfg.model.num_masked_slots,
    )
    if cfg.cjepa_model_object:
        loaded = torch.load(
            cfg.cjepa_model_object,
            map_location="cpu",
            weights_only=False,
        )
        pretrained_model = loaded.model if hasattr(loaded, "model") else loaded
        backbone.world_model.load_state_dict(pretrained_model.state_dict())

    return RewardRiskModel(
        backbone=backbone,
        hidden_dim=cfg.model.hidden_dim,
        taus=tuple(cfg.model.taus),
    )


def build_slot_data_module(cfg):
    """Build the official PushT slot datasets and Stable-Pretraining DataModule."""
    with open(cfg.data.embedding_dir, "rb") as file:
        slot_data = pickle.load(file)

    train_dataset = PushTSlotDataset(
        slot_data["train"],
        "train",
        cfg.model.history_size,
        cfg.model.predicted_size,
        cfg.data.action_dir,
        cfg.data.proprio_dir,
        state_dir=cfg.data.state_dir,
        frameskip=cfg.model.frameskip,
        seed=cfg.seed,
    )
    val_dataset = PushTSlotDataset(
        slot_data["val"],
        "val",
        cfg.model.history_size,
        cfg.model.predicted_size,
        cfg.data.action_dir,
        cfg.data.proprio_dir,
        state_dir=cfg.data.state_dir,
        frameskip=cfg.model.frameskip,
        seed=cfg.seed,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


def build_stable_pretraining_module(model, cfg, strategy):
    """Apply one fixed SPAVR strategy and return its Stable-Pretraining module."""
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    optim = {
        "head_opt": {
            "modules": r"^model\.head",
            "optimizer": {
                "type": "AdamW",
                "lr": cfg.optim.head_lr,
                "weight_decay": cfg.optim.weight_decay,
            },
            "scheduler": cfg.optim.scheduler,
        }
    }

    if strategy == "frozen":
        model.backbone = spt.backbone.EvalOnly(model.backbone)
    else:
        world_model = model.backbone.world_model
        world_model.encoder = spt.backbone.EvalOnly(world_model.encoder)
        world_model.slot_attention = spt.backbone.EvalOnly(world_model.slot_attention)
        world_model.initializer = spt.backbone.EvalOnly(world_model.initializer)
        world_model.predictor.requires_grad_(True)
        world_model.action_encoder.requires_grad_(True)
        world_model.proprio_encoder.requires_grad_(True)
        optim["dynamics_opt"] = {
            "modules": (
                r"^model\.backbone\.world_model\."
                r"(predictor|action_encoder|proprio_encoder)"
            ),
            "optimizer": {
                "type": "AdamW",
                "lr": cfg.optim.dynamics_lr,
                "weight_decay": cfg.optim.weight_decay,
            },
            "scheduler": cfg.optim.scheduler,
        }

    def forward(self, batch, stage):
        state = dict(batch)
        quantiles = self.model(batch)
        state["quantiles"] = quantiles
        if stage == "predict":
            return state

        quantile_loss = pinball_loss(
            quantiles,
            batch["return_to_go"],
            self.model.head.taus,
        )
        loss = quantile_loss
        state["loss_quantile"] = quantile_loss
        if cfg.loss.anchor_weight:
            anchor_loss = self.model.backbone.anchor_loss(batch)
            state["loss_anchor"] = anchor_loss
            loss = loss + cfg.loss.anchor_weight * anchor_loss

        state["loss"] = loss
        prefix = {"fit": "train", "validate": "val", "test": "test"}.get(
            stage, stage
        )
        metrics = {
            f"{prefix}/loss": loss.detach(),
            f"{prefix}/loss_quantile": quantile_loss.detach(),
        }
        if "loss_anchor" in state:
            metrics[f"{prefix}/loss_anchor"] = state["loss_anchor"].detach()
        self.log_dict(metrics, on_step=stage == "fit", on_epoch=True, sync_dist=True)
        return state

    return spt.Module(
        model=model,
        forward=forward,
        optim=optim,
        hparams=cfg,
    )


def run_training(cfg, strategy):
    """Build data, model, Trainer and Stable-Pretraining Manager for one strategy."""
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module = build_stable_pretraining_module(
        build_reward_model(cfg),
        cfg,
        strategy,
    )
    data = build_slot_data_module(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=cfg.output_name + "-{epoch:03d}",
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback],
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        ckpt_path=str(output_dir / (cfg.output_name + "-resume.ckpt")),
        seed=cfg.seed,
    )()
