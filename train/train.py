from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from spavr.backbone import CJepaBackbone
from spavr.model import RewardRiskModel
from train.dataset import SPAVRDynamicsWindowDataset, SPAVRWindowDataset
from train.tensorboard_utils import log_frame_strip


LOSS_NAMES = (
    "loss_total",
    "loss_quantile",
    "loss_future_object",
    "loss_future_proprio",
    "loss_masked_history",
)
PHASE_NAMES = {0: "stable", 1: "crossing", 2: "decline"}
TRAINING_STAGES = {"dynamics", "quantile_head", "joint"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPAVR from frozen object slots")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--object-slots-root", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["_config_path"] = str(path.resolve())
    return config


def get_training_stage(config):
    stage = str(config.get("training", {}).get("stage", "joint"))
    if stage not in TRAINING_STAGES:
        raise ValueError(
            f"training.stage must be one of {sorted(TRAINING_STAGES)}, got {stage!r}"
        )
    return stage


def make_dataset(config, split):
    data = config["data"]
    if get_training_stage(config) == "dynamics":
        return SPAVRDynamicsWindowDataset(
            root=data["root"],
            split=split,
            history_size=data["history_size"],
            predicted_size=data["predicted_size"],
            object_slots_root=data.get("object_slots_root"),
        )
    return SPAVRWindowDataset(
        root=data["root"],
        split=split,
        history_size=data["history_size"],
        predicted_size=data["predicted_size"],
        action_horizon=data["action_horizon"],
        reward_gamma=data["reward_gamma"],
        object_slots_root=data.get("object_slots_root"),
    )


def make_loader(dataset, config, train):
    data = config["data"]
    return DataLoader(
        dataset,
        batch_size=int(data["batch_size"]),
        shuffle=train,
        num_workers=int(data["num_workers"]),
        pin_memory=bool(data["pin_memory"] and torch.cuda.is_available()),
        persistent_workers=int(data["num_workers"]) > 0,
        drop_last=train and len(dataset) >= int(data["batch_size"]),
    )


def validate_slot_dataset(dataset, model_config):
    if dataset.object_slots_root is None:
        return
    metadata_path = dataset.object_slots_root / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing slot metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    expected = (
        int(model_config.get("num_objects", 4)),
        int(model_config.get("slot_dim", 128)),
    )
    actual = (int(metadata["num_slots"]), int(metadata["slot_dim"]))
    if actual != expected:
        raise ValueError(f"slot metadata S,D={actual}, model expects {expected}")
    missing = [
        episode["episode_id"]
        for episode in dataset.episodes
        if not (
            dataset.object_slots_root / "files" / f"{episode['episode_id']}.npy"
        ).is_file()
    ]
    if missing:
        raise ValueError(
            f"slot extraction is incomplete for split={dataset.split}: "
            f"{len(missing)} files missing; first={missing[0]}"
        )


def make_model(config):
    data = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    backbone = CJepaBackbone(
        num_objects=int(model_cfg.get("num_objects", 4)),
        slot_dim=int(model_cfg.get("slot_dim", 128)),
        history_size=int(data["history_size"]),
        predicted_size=int(data["predicted_size"]),
        action_dim=7,
        proprio_dim=7,
        num_masked_slots=int(model_cfg.get("num_masked_slots", 1)),
        seed=int(config["seed"]),
        depth=int(model_cfg.get("depth", 6)),
        heads=int(model_cfg.get("heads", 16)),
        dim_head=int(model_cfg.get("dim_head", 64)),
        mlp_dim=int(model_cfg.get("mlp_dim", 2048)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        visual_input_mode=model_cfg.get("visual_input_mode", "object_slots"),
        videosaur_checkpoint=model_cfg.get("videosaur_checkpoint"),
        dino_model=model_cfg.get("dino_model", "facebook/dinov2-small"),
    )
    return RewardRiskModel(
        backbone=backbone,
        hidden_dim=int(model_cfg["hidden_dim"]),
        taus=tuple(model_cfg["taus"]),
        predicted_size=int(data["predicted_size"]),
        quantile_loss_weight=loss_cfg["quantile_weight"],
        future_object_loss_weight=loss_cfg["future_object_weight"],
        future_proprio_loss_weight=loss_cfg["future_proprio_weight"],
        masked_history_loss_weight=loss_cfg["masked_history_weight"],
    )


def configure_trainable_parameters(model, stage):
    """Freeze modules according to the 3A/3B/3C training contract."""
    model.requires_grad_(False)
    world_model = model.backbone.world_model

    if stage in {"dynamics", "joint"}:
        world_model.predictor.requires_grad_(True)
        world_model.action_encoder.requires_grad_(True)
        world_model.proprio_encoder.requires_grad_(True)
    if stage in {"quantile_head", "joint"}:
        model.head.requires_grad_(True)

    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError(f"training stage {stage!r} has no trainable parameters")
    return trainable


def load_initial_weights(model, checkpoint_path, device):
    """Load model weights for a new stage without restoring optimizer state."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise ValueError(f"initial checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "spavr_reward_risk":
        raise ValueError(f"not an SPAVR reward-risk checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint_path.resolve()


def make_optimizer(model, config, stage):
    optim_cfg = config["optim"]
    weight_decay = float(optim_cfg["weight_decay"])
    if stage != "joint":
        learning_rate_key = (
            "head_learning_rate" if stage == "quantile_head" else "learning_rate"
        )
        learning_rate = float(
            optim_cfg.get(learning_rate_key, optim_cfg.get("learning_rate", 1e-4))
        )
        return torch.optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": learning_rate,
                    "name": stage,
                }
            ],
            weight_decay=weight_decay,
        )

    world_model = model.backbone.world_model
    dynamics_parameters = [
        parameter
        for module in (
            world_model.predictor,
            world_model.action_encoder,
            world_model.proprio_encoder,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {
                "params": dynamics_parameters,
                "lr": float(
                    optim_cfg.get(
                        "dynamics_learning_rate",
                        optim_cfg.get("learning_rate", 1e-4),
                    )
                ),
                "name": "dynamics",
            },
            {
                "params": [
                    parameter
                    for parameter in model.head.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(
                    optim_cfg.get(
                        "head_learning_rate",
                        optim_cfg.get("learning_rate", 1e-4),
                    )
                ),
                "name": "quantile_head",
            },
        ],
        weight_decay=weight_decay,
    )


def make_scheduler(optimizer, config):
    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("training.warmup_epochs must satisfy 0 <= warmup < epochs")
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(config["training"].get("warmup_start_factor", 0.1)),
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as error:
            print(f"CUDA unavailable at allocation time; falling back to CPU: {error}")
    return torch.device("cpu")


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if torch.is_tensor(value) and key != "current_step"
    }


def empty_metrics():
    return {
        "loss_sums": {name: 0.0 for name in LOSS_NAMES},
        "count": 0,
        "reward_count": 0,
        "mae_q50": 0.0,
        "interval_width": 0.0,
        "coverage": [0.0, 0.0, 0.0],
        "phase_count": {phase: 0 for phase in PHASE_NAMES},
        "phase_mae": {phase: 0.0 for phase in PHASE_NAMES},
    }


def update_metrics(metrics, output, batch):
    batch_size = len(batch["proprio"])
    metrics["count"] += batch_size
    for name in LOSS_NAMES:
        metrics["loss_sums"][name] += float(output[name].detach()) * batch_size
    if output["quantiles"] is None:
        return

    target = batch["return_to_go"]
    quantiles = output["quantiles"].detach()
    metrics["reward_count"] += batch_size
    error = (quantiles[:, 1] - target).abs()
    metrics["mae_q50"] += float(error.sum())
    metrics["interval_width"] += float((quantiles[:, 2] - quantiles[:, 0]).sum())
    for index in range(3):
        metrics["coverage"][index] += float((target <= quantiles[:, index]).sum())
    if "phase" in batch:
        for phase in PHASE_NAMES:
            selected = batch["phase"] == phase
            selected_count = int(selected.sum())
            metrics["phase_count"][phase] += selected_count
            if selected_count:
                metrics["phase_mae"][phase] += float(error[selected].sum())


def finalize_metrics(metrics):
    count = metrics["count"]
    if count == 0:
        raise RuntimeError("epoch received no batches")
    result = {
        name: value / count for name, value in metrics["loss_sums"].items()
    }
    reward_count = metrics["reward_count"]
    if reward_count:
        result.update(
            {
                "mae_q50": metrics["mae_q50"] / reward_count,
                "interval_width_q90_q10": (
                    metrics["interval_width"] / reward_count
                ),
                "coverage_q10": metrics["coverage"][0] / reward_count,
                "coverage_q50": metrics["coverage"][1] / reward_count,
                "coverage_q90": metrics["coverage"][2] / reward_count,
            }
        )
    for phase, name in PHASE_NAMES.items():
        phase_count = metrics["phase_count"][phase]
        if phase_count:
            result[f"phase_{name}_mae_q50"] = (
                metrics["phase_mae"][phase] / phase_count
            )
            result[f"phase_{name}_windows"] = phase_count
    return result


def run_epoch(
    model,
    loader,
    device,
    stage,
    optimizer,
    scaler,
    gradient_clip_norm,
    writer=None,
    global_step=0,
    log_every_steps=20,
    max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    if training and stage == "quantile_head":
        world_model = model.backbone.world_model
        world_model.predictor.eval()
        world_model.action_encoder.eval()
        world_model.proprio_encoder.eval()
    if training and stage == "dynamics":
        model.head.eval()
    metrics = empty_metrics()
    amp_enabled = scaler is not None and scaler.is_enabled()
    started = time.perf_counter()

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model.compute_loss(batch)

            grad_norm = None
            if training:
                scaler.scale(output["loss_total"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

            update_metrics(metrics, output, batch)
            if training and writer is not None and (
                global_step == 1 or global_step % log_every_steps == 0
            ):
                for name in LOSS_NAMES:
                    writer.add_scalar(f"train_step/{name}", float(output[name]), global_step)
                writer.add_scalar("train_step/gradient_norm", float(grad_norm), global_step)
                for group_index, group in enumerate(optimizer.param_groups):
                    group_name = group.get("name", f"group_{group_index}")
                    writer.add_scalar(
                        f"train_step/learning_rate_{group_name}",
                        group["lr"],
                        global_step,
                    )
                elapsed = max(time.perf_counter() - started, 1e-8)
                writer.add_scalar(
                    "train_step/samples_per_second",
                    metrics["count"] / elapsed,
                    global_step,
                )
                if device.type == "cuda":
                    writer.add_scalar(
                        "system/gpu_memory_allocated_mb",
                        torch.cuda.memory_allocated(device) / 2**20,
                        global_step,
                    )

    return finalize_metrics(metrics), global_step


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    config,
    normalization,
    epoch,
    best_score,
    selection_metric,
    global_step,
    epochs_without_improvement,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "spavr_reward_risk",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "normalization": normalization,
            "epoch": epoch,
            "best_val": best_score,
            "best_score": best_score,
            "selection_metric": selection_metric,
            "global_step": global_step,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_derived_metrics(metrics, config):
    """Add checkpoint-selection metrics that are meaningful across stages."""
    loss_cfg = config["loss"]
    metrics["loss_dynamics"] = (
        float(loss_cfg["future_object_weight"])
        * metrics["loss_future_object"]
        + float(loss_cfg["future_proprio_weight"])
        * metrics["loss_future_proprio"]
    )
    return metrics


def validate_stage_config(config, stage):
    weights = {
        name: float(value) for name, value in config["loss"].items()
    }
    if stage == "dynamics" and weights["quantile_weight"] != 0.0:
        raise ValueError("dynamics stage requires loss.quantile_weight=0")
    if stage == "quantile_head" and any(
        weights[name] != 0.0
        for name in (
            "future_object_weight",
            "future_proprio_weight",
            "masked_history_weight",
        )
    ):
        raise ValueError("quantile_head stage requires all dynamics loss weights=0")
    selection_metric = str(config["training"].get("selection_metric", "loss_total"))
    allowed_metrics = {"loss_total", "loss_quantile", "loss_dynamics", "mae_q50"}
    if selection_metric not in allowed_metrics:
        raise ValueError(
            f"training.selection_metric must be one of {sorted(allowed_metrics)}"
        )
    if stage == "dynamics" and selection_metric in {"loss_quantile", "mae_q50"}:
        raise ValueError("dynamics stage cannot select checkpoints by reward metrics")
    return selection_metric


@torch.no_grad()
def log_prediction_examples(writer, dataset, model, device, epoch, num_examples):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))
    indices = indices[: min(num_examples, len(indices))]
    was_training = model.training
    model.eval()
    for example_number, index in enumerate(indices):
        sample = base_dataset[index]
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
            if key in {"pixels", "object_slots", "proprio", "action", "return_to_go"}
        }
        quantiles = model(batch)[0].float().cpu().tolist()
        frames = base_dataset.visualization_frames(index)
        tag = f"val_examples/example_{example_number:02d}"
        log_frame_strip(writer, f"{tag}/frames", frames, epoch)
        writer.add_text(
            f"{tag}/prediction",
            "  \n".join(
                [
                    f"episode: `{sample['episode_id']}`",
                    f"current step: {int(sample['current_step'])}",
                    f"phase: {PHASE_NAMES[int(sample['phase'])]}",
                    f"reward sequence: {sample['reward_sequence'].tolist()}",
                    f"target return: {float(sample['return_to_go']):.5f}",
                    f"q10 / q50 / q90: {quantiles}",
                ]
            ),
            global_step=epoch,
        )
    model.train(was_training)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.object_slots_root is not None:
        config["data"]["object_slots_root"] = str(args.object_slots_root.resolve())
    if config.get("logging", {}).get("backend") != "tensorboard":
        raise ValueError("SPAVR training logging.backend must be tensorboard")
    stage = get_training_stage(config)
    selection_metric = validate_stage_config(config, stage)
    if config["model"].get("visual_input_mode", "object_slots") != "object_slots":
        raise ValueError("three-stage SPAVR training requires frozen object_slots")
    seed_everything(int(config["seed"]))

    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")
    if not args.smoke_test:
        validate_slot_dataset(train_dataset, config["model"])
        validate_slot_dataset(val_dataset, config["model"])
    if args.smoke_test:
        train_dataset = Subset(train_dataset, range(min(4, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(4, len(val_dataset))))

    train_loader = make_loader(train_dataset, config, train=True)
    val_loader = make_loader(val_dataset, config, train=False)
    device = resolve_device(config["training"]["device"])
    model = make_model(config).to(device)
    initial_checkpoint = None
    configured_initial_checkpoint = config["training"].get("init_checkpoint")
    if args.resume is None and configured_initial_checkpoint:
        initial_checkpoint = load_initial_weights(
            model, configured_initial_checkpoint, device
        )
    elif (
        args.resume is None
        and stage in {"quantile_head", "joint"}
        and bool(config["training"].get("require_init_checkpoint", False))
    ):
        raise ValueError(f"{stage} stage requires training.init_checkpoint")

    trainable_names = configure_trainable_parameters(model, stage)
    optimizer = make_optimizer(model, config, stage)
    scheduler = make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(config["training"]["amp"]),
    )

    start_epoch = 0
    global_step = 0
    best_score = math.inf
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_stage = get_training_stage(checkpoint.get("config", {}))
        if checkpoint_stage != stage:
            raise ValueError(
                f"cannot resume {checkpoint_stage!r} checkpoint in {stage!r} stage; "
                "use training.init_checkpoint when starting a new stage"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", start_epoch * len(train_loader)))
        best_score = float(
            checkpoint.get("best_score", checkpoint.get("best_val", math.inf))
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

    output_dir = Path(config["training"]["output_dir"])
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(output_dir / "tensorboard"),
        purge_step=global_step if global_step else None,
    )
    writer.add_text("config/yaml", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}\n```", 0)
    end_epoch = start_epoch + 1 if args.smoke_test else int(config["training"]["epochs"])
    max_batches = 1 if args.smoke_test else None
    gradient_clip_norm = float(config["training"]["gradient_clip_norm"])
    log_every_steps = int(config["logging"]["log_every_steps"])
    visualize_every_epochs = int(config["logging"].get("visualize_every_epochs", 1))
    visualization_examples = int(config["logging"].get("visualization_examples", 4))
    early_stopping_patience = int(
        config["training"].get("early_stopping_patience", 0)
    )
    normalization = (
        train_dataset.dataset.normalization
        if isinstance(train_dataset, Subset)
        else train_dataset.normalization
    )
    print(
        json.dumps(
            {
                "stage": stage,
                "device": str(device),
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "trainable_tensors": len(trainable_names),
                "init_checkpoint": (
                    None if initial_checkpoint is None else str(initial_checkpoint)
                ),
                "selection_metric": selection_metric,
                "tensorboard": str((output_dir / "tensorboard").resolve()),
                "smoke_test": args.smoke_test,
            },
            ensure_ascii=False,
        )
    )

    try:
        for epoch in range(start_epoch, end_epoch):
            train_metrics, global_step = run_epoch(
                model,
                train_loader,
                device,
                stage,
                optimizer,
                scaler,
                gradient_clip_norm,
                writer=writer,
                global_step=global_step,
                log_every_steps=log_every_steps,
                max_batches=max_batches,
            )
            val_metrics, _ = run_epoch(
                model,
                val_loader,
                device,
                stage,
                None,
                scaler,
                gradient_clip_norm,
                max_batches=max_batches,
            )
            add_derived_metrics(train_metrics, config)
            add_derived_metrics(val_metrics, config)
            scheduler.step()
            for name, value in train_metrics.items():
                writer.add_scalar(f"train_epoch/{name}", value, epoch)
            for name, value in val_metrics.items():
                writer.add_scalar(f"val_epoch/{name}", value, epoch)
            if (
                config["loss"]["quantile_weight"] > 0.0
                and (
                    epoch == start_epoch
                    or (epoch + 1) % visualize_every_epochs == 0
                )
            ):
                log_prediction_examples(
                    writer,
                    val_dataset,
                    model,
                    device,
                    epoch,
                    visualization_examples,
                )
            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "stage": stage,
                "selection_metric": selection_metric,
                "train": train_metrics,
                "val": val_metrics,
            }
            print(json.dumps(metrics, ensure_ascii=False))
            append_jsonl(output_dir / "metrics.jsonl", metrics)

            current_score = float(val_metrics[selection_metric])
            improved = current_score < best_score
            if improved:
                best_score = current_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                config,
                normalization,
                epoch,
                best_score,
                selection_metric,
                global_step,
                epochs_without_improvement,
            )
            if improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    normalization,
                    epoch,
                    best_score,
                    selection_metric,
                    global_step,
                    epochs_without_improvement,
                )
            writer.flush()
            if (
                not args.smoke_test
                and early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "early_stopped": True,
                            "epoch": epoch,
                            "selection_metric": selection_metric,
                            "best_score": best_score,
                        },
                        ensure_ascii=False,
                    )
                )
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
