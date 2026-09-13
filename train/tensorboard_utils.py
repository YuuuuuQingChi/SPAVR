"""Small TensorBoard visualizations shared by SPAVR training stages."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from train.dataset import DINO_MEAN, DINO_STD


SLOT_COLORS = torch.tensor(
    [
        [0.90, 0.20, 0.20],
        [0.20, 0.75, 0.25],
        [0.20, 0.45, 0.95],
        [0.95, 0.75, 0.15],
        [0.75, 0.25, 0.90],
        [0.15, 0.80, 0.80],
        [0.95, 0.45, 0.10],
        [0.55, 0.55, 0.55],
    ],
    dtype=torch.float32,
)


def denormalize_video(video):
    mean = DINO_MEAN.to(video.device, video.dtype)
    std = DINO_STD.to(video.device, video.dtype)
    return (video * std + mean).clamp(0.0, 1.0)


def masks_to_overlay(video, masks, alpha=0.45):
    """Overlay hard slot assignments on normalized video frames.

    Args:
        video: ``(B,T,3,H,W)`` DINO-normalized pixels.
        masks: ``(B,T,S,P)`` soft slot masks over square patch grids.
    """
    if masks.ndim != 4:
        raise ValueError(f"expected masks (B,T,S,P), got {tuple(masks.shape)}")
    patch_side = math.isqrt(masks.shape[-1])
    if patch_side * patch_side != masks.shape[-1]:
        raise ValueError("slot visualization requires a square patch grid")
    assignment = masks.argmax(dim=2)
    colors = SLOT_COLORS.to(video.device, video.dtype)
    if masks.shape[2] > len(colors):
        repeats = math.ceil(masks.shape[2] / len(colors))
        colors = colors.repeat(repeats, 1)
    color_map = colors[assignment].reshape(
        assignment.shape[0], assignment.shape[1], patch_side, patch_side, 3
    ).permute(0, 1, 4, 2, 3)
    color_map = F.interpolate(
        color_map.flatten(0, 1),
        size=video.shape[-2:],
        mode="nearest",
    ).unflatten(0, video.shape[:2])
    pixels = denormalize_video(video)
    return pixels * (1.0 - alpha) + color_map * alpha


@torch.no_grad()
def log_slot_visualization(writer, tag, video, masks, step, max_frames=6):
    """Log input/overlay rows and each soft slot mask as TensorBoard images."""
    video = video.detach()[:1].float()
    masks = masks.detach()[:1].float()
    frame_count = min(max_frames, video.shape[1])
    video = video[:, :frame_count]
    masks = masks[:, :frame_count]

    pixels = denormalize_video(video)[0].cpu()
    overlay = masks_to_overlay(video, masks)[0].cpu()
    writer.add_image(
        f"{tag}/frames_and_slots",
        make_grid(torch.cat([pixels, overlay]), nrow=frame_count),
        global_step=step,
    )

    patch_side = math.isqrt(masks.shape[-1])
    soft_masks = masks[0].permute(1, 0, 2).reshape(
        masks.shape[2] * frame_count, 1, patch_side, patch_side
    )
    soft_masks = F.interpolate(
        soft_masks,
        size=video.shape[-2:],
        mode="nearest",
    ).cpu()
    writer.add_image(
        f"{tag}/soft_masks",
        make_grid(soft_masks, nrow=frame_count, normalize=True),
        global_step=step,
    )


def log_frame_strip(writer, tag, frames, step):
    frames = frames.detach().float().cpu().clamp(0.0, 1.0)
    writer.add_image(tag, make_grid(frames, nrow=len(frames)), global_step=step)
