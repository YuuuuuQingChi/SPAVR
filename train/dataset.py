from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DINO_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(
    1, 3, 1, 1
)
DINO_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(
    1, 3, 1, 1
)


@dataclass(frozen=True)
class Window:
    episode_index: int
    current_step: int


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=128)
def load_trajectory(path: str):
    with np.load(path) as trajectory:
        return {name: trajectory[name].copy() for name in trajectory.files}


@lru_cache(maxsize=16)
def decode_video(path: str):
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames)


@lru_cache(maxsize=128)
def load_object_slots(path: str):
    slots = np.load(path)
    if slots.ndim != 3:
        raise ValueError(f"object slots must have shape (T,S,D): {path}")
    return slots


def normalized_pixels(frames, image_size=None):
    pixels = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    pixels = pixels.to(torch.float32).div_(255.0)
    if image_size is not None:
        image_size = int(image_size)
        if image_size < 1:
            raise ValueError("image_size must be positive")
        if pixels.shape[-2:] != (image_size, image_size):
            # Match the official VideoSAUR Push-T preprocessing: bicubic resize
            # followed by ImageNet normalization.
            pixels = F.interpolate(
                pixels,
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0.0, 1.0)
    return (pixels - DINO_MEAN) / DINO_STD


class SPAVRVideoClipDataset(Dataset):
    """Contiguous episode-video clips for self-supervised VideoSAUR training."""

    def __init__(
        self,
        root,
        split,
        clip_length=6,
        frame_stride=2,
        clip_step=1,
        image_size=None,
        horizontal_flip_probability=0.0,
        max_episodes=None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.clip_length = int(clip_length)
        self.frame_stride = int(frame_stride)
        self.clip_step = int(clip_step)
        self.image_size = None if image_size is None else int(image_size)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        if self.clip_length < 2 or self.frame_stride < 1 or self.clip_step < 1:
            raise ValueError("clip_length>=2, frame_stride>=1 and clip_step>=1 are required")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0,1]")

        with (self.root / "manifest.jsonl").open(encoding="utf-8") as file:
            all_episodes = [json.loads(line) for line in file if line.strip()]
        episodes = [episode for episode in all_episodes if episode["split"] == split]
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        self.episodes = episodes
        span = (self.clip_length - 1) * self.frame_stride + 1
        self.clips = []
        for episode_index, episode in enumerate(episodes):
            self.clips.extend(
                (episode_index, start)
                for start in range(
                    0,
                    max(int(episode["num_steps"]) - span + 1, 0),
                    self.clip_step,
                )
            )
        if not self.clips:
            raise ValueError(
                f"split={split!r} has no VideoSAUR clips with length="
                f"{self.clip_length}, stride={self.frame_stride}"
            )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        episode_index, start = self.clips[index]
        episode = self.episodes[episode_index]
        video = decode_video(str(self.root / episode["video"]))
        indices = start + np.arange(self.clip_length) * self.frame_stride
        pixels = normalized_pixels(video[indices], image_size=self.image_size)
        if (
            self.horizontal_flip_probability > 0.0
            and torch.rand(()) < self.horizontal_flip_probability
        ):
            pixels = pixels.flip(-1)
        return {
            "video": pixels,
            "episode_id": episode["episode_id"],
            "start_step": torch.tensor(start, dtype=torch.int64),
        }


class SPAVRWindowDataset(Dataset):
    """Build causal state/action/consequence windows from processed episodes."""

    def __init__(
        self,
        root,
        split,
        history_size,
        predicted_size,
        action_horizon,
        reward_gamma=1.0,
        object_slots_root=None,
        max_episodes=None,
        window_mode="reward",
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.history_size = int(history_size)
        self.predicted_size = int(predicted_size)
        self.action_horizon = int(action_horizon)
        self.reward_gamma = float(reward_gamma)
        self.window_mode = str(window_mode)
        if self.window_mode not in {"reward", "dynamics"}:
            raise ValueError("window_mode must be 'reward' or 'dynamics'")
        self.object_slots_root = None
        if object_slots_root:
            self.object_slots_root = Path(object_slots_root).expanduser()
            if not self.object_slots_root.is_absolute():
                self.object_slots_root = (self.root / self.object_slots_root).resolve()
            if not self.object_slots_root.is_dir():
                raise ValueError(
                    f"object slots root does not exist: {self.object_slots_root}"
                )

        if self.history_size < 1 or self.predicted_size < 1:
            raise ValueError("history_size and predicted_size must be positive")
        if (
            self.window_mode == "reward"
            and self.action_horizon < self.history_size + self.predicted_size
        ):
            raise ValueError(
                "action_horizon must be at least history_size + predicted_size "
                "for consequence supervision"
            )
        if not 0.0 < self.reward_gamma <= 1.0:
            raise ValueError("reward_gamma must satisfy 0 < gamma <= 1")

        manifest_path = self.root / "manifest.jsonl"
        with manifest_path.open(encoding="utf-8") as file:
            all_episodes = [json.loads(line) for line in file if line.strip()]
        episodes = [episode for episode in all_episodes if episode["split"] == split]
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        if not episodes:
            raise ValueError(f"no episodes found for split={split!r}")
        self.episodes = episodes

        normalization = load_json(self.root / "normalization.json")
        if normalization["computed_from_split"] != "train":
            raise ValueError("normalization statistics must come from train split")
        self.normalization = normalization
        self.proprio_mean = np.asarray(
            normalization["proprio"]["mean"], dtype=np.float32
        )
        self.proprio_std = np.asarray(
            normalization["proprio"]["std"], dtype=np.float32
        )
        self.action_mean = np.asarray(
            normalization["action"]["mean"], dtype=np.float32
        )
        self.action_std = np.asarray(
            normalization["action"]["std"], dtype=np.float32
        )

        self.windows = []
        h, p, length = (
            self.history_size,
            self.predicted_size,
            self.action_horizon,
        )
        for episode_index, episode in enumerate(self.episodes):
            num_steps = int(episode["num_steps"])
            first_step = h - 1
            if self.window_mode == "dynamics":
                # Use every legal H -> P window. Reward training additionally
                # requires a complete action/reward horizon and therefore has
                # fewer legal windows near the end of an episode.
                last_step = num_steps - p - 1
            else:
                last_step = min(num_steps - p - 1, num_steps - length)
            self.windows.extend(
                Window(episode_index, current_step)
                for current_step in range(first_step, last_step + 1)
            )
        if not self.windows:
            raise ValueError(
                f"split={split!r} has no legal H={h}, P={p}, L={length} windows"
            )

        self.reward_weights = None
        if self.window_mode == "reward":
            discounts = self.reward_gamma ** np.arange(
                self.action_horizon, dtype=np.float32
            )
            self.reward_weights = discounts / discounts.sum()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        episode = self.episodes[window.episode_index]
        trajectory = load_trajectory(str(self.root / episode["trajectory"]))

        num_steps = int(episode["num_steps"])

        t = window.current_step
        history = np.arange(t - self.history_size + 1, t + 1)
        future = np.arange(t + 1, t + self.predicted_size + 1)
        state_indices = np.concatenate((history, future))
        proprio = (
            trajectory["proprio"][state_indices] - self.proprio_mean
        ) / self.proprio_std
        sample = {
            "proprio": torch.from_numpy(proprio.astype(np.float32, copy=False)),
            "episode_id": episode["episode_id"],
            "current_step": torch.tensor(t, dtype=torch.int64),
        }
        if self.window_mode == "dynamics":
            # Dynamics training consumes the same H+P contiguous span as the
            # state targets, so it can retain the final legal windows.
            action = (
                trajectory["action"][state_indices] - self.action_mean
            ) / self.action_std
            sample["action"] = torch.from_numpy(
                action.astype(np.float32, copy=False)
            )
        else:
            action_slice = slice(t, t + self.action_horizon)
            action = (
                trajectory["action"][action_slice] - self.action_mean
            ) / self.action_std
            rewards = trajectory["reward"][action_slice]
            return_to_go = np.dot(rewards, self.reward_weights)

            onset = episode.get("failure_onset_step")
            action_end = t + self.action_horizon - 1
            if onset is None or action_end < int(onset):
                phase = 0  # Stable/prefix window.
            elif t < int(onset):
                phase = 1  # Candidate plan crosses the failure onset.
            else:
                phase = 2  # Candidate plan starts at/after the onset.

            sample.update(
                {
                    "action": torch.from_numpy(
                        action.astype(np.float32, copy=False)
                    ),
                    "return_to_go": torch.tensor(
                        return_to_go, dtype=torch.float32
                    ),
                    "reward_sequence": torch.from_numpy(
                        rewards.astype(np.float32, copy=False)
                    ),
                    "phase": torch.tensor(phase, dtype=torch.int64),
                }
            )
        if self.object_slots_root is None:
            video = decode_video(str(self.root / episode["video"]))
            if len(video) != num_steps:
                raise ValueError(
                    f"{episode['episode_id']}: video has {len(video)} frames, "
                    f"manifest declares {num_steps}"
                )
            sample["pixels"] = normalized_pixels(video[state_indices])
        else:
            slots_path = self.object_slots_root / "files" / f"{episode['episode_id']}.npy"
            slots = load_object_slots(str(slots_path))
            if len(slots) != num_steps:
                raise ValueError(
                    f"{episode['episode_id']}: {len(slots)} slot frames, expected {num_steps}"
                )
            sample["object_slots"] = torch.from_numpy(
                slots[state_indices].astype(np.float32, copy=False)
            )
        return sample

    def visualization_frames(self, index):
        """Return unnormalized frames for a fixed TensorBoard example."""
        window = self.windows[index]
        episode = self.episodes[window.episode_index]
        video = decode_video(str(self.root / episode["video"]))
        t = window.current_step
        state_indices = np.concatenate(
            (
                np.arange(t - self.history_size + 1, t + 1),
                np.arange(t + 1, t + self.predicted_size + 1),
            )
        )
        return torch.from_numpy(video[state_indices].copy()).permute(0, 3, 1, 2).float() / 255.0


class SPAVRDynamicsWindowDataset(SPAVRWindowDataset):
    """All stride-one H -> P windows used by stage 3A dynamics training."""

    def __init__(
        self,
        root,
        split,
        history_size,
        predicted_size,
        object_slots_root,
        max_episodes=None,
    ):
        super().__init__(
            root=root,
            split=split,
            history_size=history_size,
            predicted_size=predicted_size,
            action_horizon=history_size + predicted_size,
            reward_gamma=1.0,
            object_slots_root=object_slots_root,
            max_episodes=max_episodes,
            window_mode="dynamics",
        )
