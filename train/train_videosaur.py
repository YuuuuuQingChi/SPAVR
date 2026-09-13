"""Pretrain the VideoSAUR object encoder on processed SPAVR videos."""

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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from spavr.videosaur import (
    load_videosaur_pretrainer_initialization,
    make_videosaur_pretrainer,
)
from train.dataset import SPAVRVideoClipDataset
from train.tensorboard_utils import log_slot_visualization


LOSS_NAMES = (
    "loss_total",
    "loss_featrec",
    "loss_timesim",
    "timesim_target_entropy",
    "timesim_kl",
    "timesim_effective_patches",
    "timesim_target_max_probability",
    "slot_feature_std",
    "slot_cosine_similarity",
    "mask_entropy",
    "slot_usage_min",
    "slot_usage_max",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain VideoSAUR on SPAVR videos")
    parser.add_argument("--config", type=Path, default=Path("configs/train_videosaur.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="override initialization.checkpoint without resuming optimizer state",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(config, split):
    data = config["data"]
    return SPAVRVideoClipDataset(
        root=data["root"],
        split=split,
        clip_length=data["clip_length"],
        frame_stride=data["frame_stride"],
        clip_step=data["clip_step"],
        image_size=config["model"]["image_size"],
        horizontal_flip_probability=(
            data["horizontal_flip_probability"] if split == "train" else 0.0
        ),
    )


def make_loader(dataset, config, train, num_workers=None):
    data = config["data"]
    if num_workers is None:
        num_workers = int(data["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(data["batch_size"]),
        shuffle=train,
        num_workers=num_workers,
        pin_memory=bool(data["pin_memory"] and torch.cuda.is_available()),
        persistent_workers=num_workers > 0,
        drop_last=train and len(dataset) >= int(data["batch_size"]),
    )


def schedule_factor(step, warmup_steps, decay_steps, decay_rate):
    if warmup_steps > 0 and step < warmup_steps:
        return max(step, 1) / warmup_steps
    elapsed = max(step - warmup_steps, 0)
    duration = max(decay_steps - warmup_steps, 1)
    return decay_rate ** (elapsed / duration)


def scalar_metrics(output):
    return {name: float(output[name].detach()) for name in LOSS_NAMES}


@torch.no_grad()
def validate(model, loader, device, amp_enabled, max_batches):
    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    visual_batch = None
    visual_output = None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        video = batch["video"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model.compute_loss(video)
        batch_size = len(video)
        count += batch_size
        for name, value in scalar_metrics(output).items():
            totals[name] += value * batch_size
        if visual_batch is None:
            visual_batch = video.detach()
            visual_output = output
    if count == 0:
        raise RuntimeError("VideoSAUR validation received no batches")
    return (
        {name: value / count for name, value in totals.items()},
        visual_batch,
        visual_output,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    config,
    global_step,
    best_val,
    initialization,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "spavr_videosaur",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "model_config": config["model"],
            "loss_config": config["loss"],
            "global_step": global_step,
            "best_val": best_val,
            "initialization": initialization,
        },
        path,
    )


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["_config_path"] = str(args.config.resolve())
    seed_everything(int(config["seed"]))

    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")
    if args.smoke_test:
        train_dataset = Subset(train_dataset, range(min(2, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(2, len(val_dataset))))
    # A smoke test has only two samples and should not require worker-process
    # IPC, which is commonly disabled in containers and restricted shells.
    smoke_test_workers = 0 if args.smoke_test else None
    train_loader = make_loader(
        train_dataset, config, train=True, num_workers=smoke_test_workers
    )
    val_loader = make_loader(
        val_dataset, config, train=False, num_workers=smoke_test_workers
    )

    device = resolve_device(config["training"]["device"])
    model = make_videosaur_pretrainer(config["model"], config["loss"])
    initialization = {"mode": "scratch"}
    initialization_checkpoint = args.init_checkpoint
    if initialization_checkpoint is None:
        configured_checkpoint = config.get("initialization", {}).get("checkpoint")
        if configured_checkpoint:
            initialization_checkpoint = Path(configured_checkpoint)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    if initialization_checkpoint is not None and not args.resume:
        initialization = load_videosaur_pretrainer_initialization(
            model,
            initialization_checkpoint,
            map_location="cpu",
        )
    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(config["optim"]["learning_rate"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    max_steps = 1 if args.smoke_test else int(config["training"]["max_steps"])
    scheduler = LambdaLR(
        optimizer,
        lambda step: schedule_factor(
            step,
            int(config["optim"]["warmup_steps"]),
            max_steps,
            float(config["optim"]["decay_rate"]),
        ),
    )
    amp_enabled = device.type == "cuda" and bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_step = 0
    best_val = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint["best_val"])
        initialization = checkpoint.get(
            "initialization",
            {"mode": "resumed", "checkpoint": str(args.resume.resolve())},
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
    writer.add_text(
        "config/initialization",
        f"```json\n{json.dumps(initialization, ensure_ascii=False, indent=2)}\n```",
        global_step,
    )
    metrics_path = output_dir / "metrics.jsonl"

    print(
        json.dumps(
            {
                "stage": "videosaur_pretrain",
                "device": str(device),
                "train_clips": len(train_dataset),
                "val_clips": len(val_dataset),
                "max_steps": max_steps,
                "initialization": initialization,
                "tensorboard": str((output_dir / "tensorboard").resolve()),
                "smoke_test": args.smoke_test,
            },
            ensure_ascii=False,
        )
    )

    accumulation = int(config["training"]["gradient_accumulation_steps"])
    gradient_clip = float(config["training"]["gradient_clip_norm"])
    val_every = 1 if args.smoke_test else int(config["training"]["val_every_steps"])
    checkpoint_every = int(config["training"]["checkpoint_every_steps"])
    visualize_every = 1 if args.smoke_test else int(
        config["training"]["visualize_every_steps"]
    )
    max_val_batches = 1 if args.smoke_test else int(config["training"]["max_val_batches"])

    train_iterator = iter(train_loader)
    model.train()
    try:
        while global_step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            accumulated = {name: 0.0 for name in LOSS_NAMES}
            last_video = None
            last_output = None
            started = time.perf_counter()
            for _ in range(accumulation):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                video = batch["video"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model.compute_loss(video)
                    scaled_loss = output["loss_total"] / accumulation
                scaler.scale(scaled_loss).backward()
                for name, value in scalar_metrics(output).items():
                    accumulated[name] += value / accumulation
                last_video = video.detach()
                last_output = output

            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(parameters, max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            elapsed = max(time.perf_counter() - started, 1e-8)

            for name, value in accumulated.items():
                writer.add_scalar(f"train/{name}", value, global_step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train/gradient_norm", float(grad_norm), global_step)
            writer.add_scalar(
                "train/clips_per_second",
                int(config["data"]["batch_size"]) * accumulation / elapsed,
                global_step,
            )
            if device.type == "cuda":
                writer.add_scalar(
                    "system/gpu_memory_allocated_mb",
                    torch.cuda.memory_allocated(device) / 2**20,
                    global_step,
                )

            if global_step == 1 or global_step % visualize_every == 0:
                log_slot_visualization(
                    writer,
                    "train/slot_visualization",
                    last_video,
                    last_output["decoder_masks"],
                    global_step,
                )
                log_slot_visualization(
                    writer,
                    "train/grouping_slot_visualization",
                    last_video,
                    last_output["grouping_masks"],
                    global_step,
                )

            if global_step % val_every == 0 or global_step == max_steps:
                val_metrics, val_video, val_output = validate(
                    model,
                    val_loader,
                    device,
                    amp_enabled,
                    max_val_batches,
                )
                for name, value in val_metrics.items():
                    writer.add_scalar(f"val/{name}", value, global_step)
                log_slot_visualization(
                    writer,
                    "val/slot_visualization",
                    val_video,
                    val_output["decoder_masks"],
                    global_step,
                )
                log_slot_visualization(
                    writer,
                    "val/grouping_slot_visualization",
                    val_video,
                    val_output["grouping_masks"],
                    global_step,
                )
                record = {
                    "step": global_step,
                    "train": accumulated,
                    "val": val_metrics,
                }
                print(json.dumps(record, ensure_ascii=False))
                append_jsonl(metrics_path, record)
                improved = val_metrics["loss_total"] < best_val
                if improved:
                    best_val = val_metrics["loss_total"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        global_step,
                        best_val,
                        initialization,
                    )
                save_checkpoint(
                    output_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    global_step,
                    best_val,
                    initialization,
                )
                model.train()

            if global_step % checkpoint_every == 0 or global_step == max_steps:
                save_checkpoint(
                    output_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    global_step,
                    best_val,
                    initialization,
                )
            writer.flush()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
