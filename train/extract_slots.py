"""Freeze a trained VideoSAUR encoder and pre-extract slots for every episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from spavr.videosaur import make_videosaur_pretrainer
from train.dataset import decode_video, normalized_pixels
from train.tensorboard_utils import log_slot_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen VideoSAUR slots")
    parser.add_argument("--config", type=Path, default=Path("configs/extract_slots.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-episodes-per-split", type=int)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_seed(base_seed, episode_id):
    digest = hashlib.sha256(f"{base_seed}:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_root = Path(config["data_root"]).expanduser().resolve()
    checkpoint_path = (args.checkpoint or Path(config["checkpoint"])).expanduser().resolve()
    output_root = (args.output_root or Path(config["output_root"])).expanduser().resolve()
    files_root = output_root / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = file_sha256(checkpoint_path)
    metadata_path = output_root / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_metadata.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(
                "slot output was created from a different checkpoint; choose a new "
                "output_root or pass --overwrite"
            )

    device = resolve_device(config["device"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "spavr_videosaur":
        raise ValueError(f"not an SPAVR VideoSAUR checkpoint: {checkpoint_path}")
    model = make_videosaur_pretrainer(
        checkpoint["model_config"], checkpoint["loss_config"]
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.object_encoder.requires_grad_(False)
    model.object_encoder.eval().to(device)
    # Extraction does not need the feature decoder.
    del model.decoder

    with (data_root / "manifest.jsonl").open(encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file if line.strip()]
    requested_splits = set(config.get("splits", ["train", "val", "test"]))
    episodes = [episode for episode in episodes if episode["split"] in requested_splits]
    if args.max_episodes_per_split is not None:
        split_counts = {split: 0 for split in requested_splits}
        selected = []
        for episode in episodes:
            split = episode["split"]
            if split_counts[split] < args.max_episodes_per_split:
                selected.append(episode)
                split_counts[split] += 1
        episodes = selected
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    writer = SummaryWriter(log_dir=str(output_root / "tensorboard"))
    dtype_name = config.get("storage_dtype", "float16")
    if dtype_name not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    storage_dtype = np.float16 if dtype_name == "float16" else np.float32
    frame_batch_size = int(config["frame_batch_size"])
    visualization_examples = int(config.get("visualization_examples", 8))
    base_seed = int(config.get("seed", 42))
    amp_enabled = device.type == "cuda" and bool(config.get("amp", True))

    records = []
    for index, episode in enumerate(episodes):
        destination = files_root / f"{episode['episode_id']}.npy"
        if destination.exists() and not args.overwrite:
            slots = np.load(destination, mmap_mode="r")
            if slots.shape[0] != int(episode["num_steps"]):
                raise ValueError(f"incomplete slot file: {destination}")
            records.append(
                {
                    "episode_id": episode["episode_id"],
                    "split": episode["split"],
                    "shape": list(slots.shape),
                    "file": destination.relative_to(output_root).as_posix(),
                }
            )
            continue

        raw_video = decode_video(str(data_root / episode["video"]))
        if len(raw_video) != int(episode["num_steps"]):
            raise ValueError(
                f"{episode['episode_id']}: video has {len(raw_video)} frames, "
                f"expected {episode['num_steps']}"
            )
        video = normalized_pixels(
            raw_video,
            image_size=checkpoint["model_config"].get("image_size"),
        ).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model.object_encoder.extract_episode(
                video,
                frame_batch_size=frame_batch_size,
                seed=episode_seed(base_seed, episode["episode_id"]),
            )
        slots = output["slots"].float().cpu().numpy().astype(storage_dtype)
        if not np.isfinite(slots).all():
            raise ValueError(f"non-finite slots for {episode['episode_id']}")
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as file:
            np.save(file, slots)
        temporary.replace(destination)

        if index < visualization_examples:
            log_slot_visualization(
                writer,
                f"examples/{index:02d}_{episode['episode_id']}",
                video.unsqueeze(0),
                output["grouping_masks"].unsqueeze(0),
                step=0,
            )
        writer.add_scalar("extraction/episodes_complete", index + 1, index + 1)
        records.append(
            {
                "episode_id": episode["episode_id"],
                "split": episode["split"],
                "shape": list(slots.shape),
                "file": destination.relative_to(output_root).as_posix(),
            }
        )
        print(
            json.dumps(
                {
                    "episode": index + 1,
                    "total": len(episodes),
                    "episode_id": episode["episode_id"],
                    "shape": list(slots.shape),
                },
                ensure_ascii=False,
            )
        )

    metadata = {
        "schema_version": "1.0",
        "source_dataset": str(data_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": int(checkpoint["global_step"]),
        "num_episodes": len(records),
        "num_slots": int(checkpoint["model_config"]["num_slots"]),
        "slot_dim": int(checkpoint["model_config"]["slot_dim"]),
        "storage_dtype": dtype_name,
        "splits": sorted(requested_splits),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "index.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    writer.add_text("extraction/metadata", f"```json\n{json.dumps(metadata, indent=2)}\n```", 0)
    writer.close()
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
