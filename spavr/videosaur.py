"""VideoSAUR object encoder and self-supervised pretraining objective.

The C-JEPA paper trains this visual frontend before training the world model.
DINOv2 remains frozen; the projection, recurrent Slot Attention processor and
feature decoder are learned with feature reconstruction and temporal
similarity supervision.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from third_party.cjepa import src as _cjepa_src

sys.modules.setdefault("src", _cjepa_src)

from third_party.cjepa.src.third_party.videosaur.videosaur.modules.decoders import (
    SlotMixerDecoder,
)
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.encoders import (
    FrameEncoder,
)
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.groupers import (
    SlotAttention,
)
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.initializers import (
    RandomInit,
)
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.networks import (
    MLP,
    TransformerEncoder,
)
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.utils import (
    FeatureTimeSimilarity,
)
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.video import (
    LatentProcessor,
    MapOverTime,
    ScanOverTime,
)


def build_object_encoder(
    model_name="facebook/dinov2-small",
    num_slots=4,
    slot_dim=128,
    feature_dim=384,
):
    """Build the VideoSAUR frontend shared by all three training stages."""
    frame_encoder = FrameEncoder(
        backbone=AutoModel.from_pretrained(model_name),
        output_transform=MLP(
            inp_dim=feature_dim,
            outp_dim=slot_dim,
            hidden_dims=[feature_dim * 2],
            initial_layer_norm=True,
        ),
    )
    frame_encoder.backbone.requires_grad_(False)
    return VideoSAURObjectEncoder(
        encoder=MapOverTime(frame_encoder),
        initializer=RandomInit(n_slots=num_slots, dim=slot_dim),
        processor=ScanOverTime(
            LatentProcessor(
                corrector=SlotAttention(
                    inp_dim=slot_dim,
                    slot_dim=slot_dim,
                    n_iters=2,
                    use_mlp=False,
                ),
                predictor=TransformerEncoder(
                    dim=slot_dim,
                    n_blocks=1,
                    n_heads=4,
                ),
                first_step_corrector_args={"n_iters": 3},
            )
        ),
    )


class VideoSAURObjectEncoder(nn.Module):
    """Frozen-DINOv2 VideoSAUR frontend producing temporally aligned slots."""

    def __init__(self, encoder, initializer, processor):
        super().__init__()
        self.encoder = encoder
        self.initializer = initializer
        self.processor = processor

    @property
    def backbone(self):
        return self.encoder.module.backbone

    def train(self, mode=True):
        super().train(mode)
        # The paper keeps DINOv2 frozen and in inference mode while learning
        # the remaining VideoSAUR modules.
        self.backbone.eval()
        return self

    def forward(self, video):
        encoder_output = self.encoder(video)
        slots_initial = self.initializer(batch_size=video.shape[0])
        processor_output = self.processor(slots_initial, encoder_output["features"])
        return {
            "encoder": encoder_output,
            "processor": processor_output,
            "slots": processor_output["state"],
            "grouping_masks": processor_output["corrector"]["masks"],
        }

    @torch.no_grad()
    def extract_episode(self, video, frame_batch_size=16, seed=0):
        """Extract one complete episode while batching only DINOv2 frames.

        Running the recurrent slot processor after concatenating all DINO
        features preserves slot identity across the complete episode.
        """
        if video.ndim != 4:
            raise ValueError(f"expected video (T,C,H,W), got {tuple(video.shape)}")
        encoded_chunks = []
        for start in range(0, len(video), frame_batch_size):
            encoded_chunks.append(self.encoder.module(video[start : start + frame_batch_size]))
        encoder_output = {
            key: torch.cat([chunk[key] for chunk in encoded_chunks], dim=0).unsqueeze(0)
            for key in encoded_chunks[0]
        }

        devices = []
        if video.device.type == "cuda":
            devices = [video.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            slots_initial = self.initializer(batch_size=1)
            processor_output = self.processor(
                slots_initial,
                encoder_output["features"],
            )
        return {
            "slots": processor_output["state"].squeeze(0),
            "grouping_masks": processor_output["corrector"]["masks"].squeeze(0),
        }


class VideoSAURPretrainer(nn.Module):
    """VideoSAUR frontend plus the paper's self-supervised decoder losses."""

    def __init__(
        self,
        model_name="facebook/dinov2-small",
        num_slots=4,
        slot_dim=128,
        feature_dim=384,
        num_patches=256,
        feature_reconstruction_weight=1.0,
        time_similarity_weight=0.25,
        time_similarity_temperature=0.25,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.feature_dim = int(feature_dim)
        self.num_patches = int(num_patches)
        self.feature_reconstruction_weight = float(feature_reconstruction_weight)
        self.time_similarity_weight = float(time_similarity_weight)

        self.object_encoder = build_object_encoder(
            model_name=model_name,
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=feature_dim,
        )
        self.decoder = MapOverTime(
            SlotMixerDecoder(
                inp_dim=slot_dim,
                embed_dim=slot_dim,
                outp_dim=feature_dim + num_patches,
                n_patches=num_patches,
                allocator=TransformerEncoder(
                    dim=slot_dim,
                    memory_dim=slot_dim,
                    n_blocks=3,
                    n_heads=4,
                ),
                renderer=MLP(
                    inp_dim=slot_dim,
                    outp_dim=1024,
                    hidden_dims=[1024, 1024],
                    final_activation=True,
                ),
                renderer_dim=1024,
                use_layer_norms=True,
                pos_embed_mode="add",
            )
        )
        self.time_similarity = FeatureTimeSimilarity(
            softmax=True,
            temperature=time_similarity_temperature,
            threshold=0.0,
        )

    def forward(self, video):
        encoded = self.object_encoder(video)
        decoded = self.decoder(encoded["slots"])
        return {**encoded, "decoder": decoded}

    def compute_loss(self, video):
        """Compute the complete VideoSAUR objective inside the model."""
        output = self(video)
        reconstruction = output["decoder"]["reconstruction"]
        target_features = output["encoder"]["backbone_features"].detach()
        loss_featrec = F.mse_loss(
            reconstruction[..., : self.feature_dim],
            target_features,
        )

        if video.shape[1] < 2:
            raise ValueError("temporal similarity loss requires at least two frames")
        target_similarity = self.time_similarity(
            output["encoder"]["vit_block_keys12"].detach()
        )
        similarity_logits = reconstruction[
            :, :-1, :, self.feature_dim : self.feature_dim + self.num_patches
        ]
        loss_timesim = -(
            target_similarity * F.log_softmax(similarity_logits, dim=-1)
        ).sum(dim=-1).mean()
        # Cross-entropy alone is hard to interpret when the DINO similarity
        # target is nearly uniform.  Log its entropy and the remaining KL gap
        # separately so a value close to log(num_patches) is not mistaken for
        # a failed optimization run.
        target_similarity_float = target_similarity.float()
        timesim_target_entropy = -(
            target_similarity_float
            * target_similarity_float.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1).mean()
        timesim_kl = loss_timesim.float() - timesim_target_entropy
        timesim_effective_patches = timesim_target_entropy.exp()
        timesim_target_max_probability = target_similarity_float.max(dim=-1).values.mean()
        loss_total = (
            self.feature_reconstruction_weight * loss_featrec
            + self.time_similarity_weight * loss_timesim
        )

        slots = output["slots"]
        normalized_slots = F.normalize(slots, dim=-1)
        slot_similarity = torch.einsum(
            "btsd,btkd->btsk", normalized_slots, normalized_slots
        )
        off_diagonal = ~torch.eye(
            self.num_slots, dtype=torch.bool, device=slots.device
        )
        mean_slot_similarity = slot_similarity[..., off_diagonal].mean()

        masks = output["decoder"]["masks"].clamp_min(1e-8)
        normalized_mask_entropy = -(
            masks * masks.log()
        ).sum(dim=2).mean() / math.log(self.num_slots)
        slot_usage = masks.mean(dim=(0, 1, 3))

        return {
            "loss_total": loss_total,
            "loss_featrec": loss_featrec,
            "loss_timesim": loss_timesim,
            "timesim_target_entropy": timesim_target_entropy,
            "timesim_kl": timesim_kl,
            "timesim_effective_patches": timesim_effective_patches,
            "timesim_target_max_probability": timesim_target_max_probability,
            "slot_feature_std": slots.std(unbiased=False),
            "slot_cosine_similarity": mean_slot_similarity,
            "mask_entropy": normalized_mask_entropy,
            "slot_usage_min": slot_usage.min(),
            "slot_usage_max": slot_usage.max(),
            "slots": slots,
            "decoder_masks": output["decoder"]["masks"],
            "grouping_masks": output["grouping_masks"],
        }


def checkpoint_model_state(checkpoint):
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def remap_official_videosaur_state(state):
    """Map the released C-JEPA VideoSAUR keys to this pretrainer layout.

    The official Lightning module stores ``encoder``, ``initializer`` and
    ``processor`` at its root. SPAVR groups those modules under
    ``object_encoder`` so the same encoder can later be frozen and exported.
    """
    object_prefixes = ("encoder.", "initializer.", "processor.")
    return {
        f"object_encoder.{key}" if key.startswith(object_prefixes) else key: value
        for key, value in state.items()
    }


def load_videosaur_pretrainer_initialization(
    model, checkpoint_path, map_location="cpu"
):
    """Initialize pretraining from an SPAVR or official C-JEPA checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    state = checkpoint_model_state(checkpoint)
    if any(key.startswith("object_encoder.") for key in state):
        source_format = "spavr"
    elif any(key.startswith(("encoder.", "initializer.", "processor.")) for key in state):
        source_format = "official_cjepa"
        state = remap_official_videosaur_state(state)
    else:
        raise ValueError(
            f"unrecognized VideoSAUR checkpoint layout: {checkpoint_path}"
        )

    # Strict loading catches accidental changes to image size, slot count, or
    # slot dimension before an expensive fine-tuning run starts.
    model.load_state_dict(state, strict=True)
    return {
        "mode": "pretrained_finetune",
        "source_format": source_format,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_global_step": int(checkpoint.get("global_step", -1)),
        "loaded_tensors": len(state),
    }


def load_videosaur_object_encoder(object_encoder, checkpoint_path, map_location="cpu"):
    """Load only the reusable encoder part from a pretraining checkpoint."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=map_location, weights_only=False
    )
    state = checkpoint_model_state(checkpoint)
    prefix = "object_encoder."
    if any(key.startswith(prefix) for key in state):
        state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
    else:
        reusable_prefixes = ("encoder.", "initializer.", "processor.")
        state = {
            key: value
            for key, value in state.items()
            if key.startswith(reusable_prefixes)
        }
    object_encoder.load_state_dict(state, strict=True)
    return checkpoint


def make_videosaur_pretrainer(model_config, loss_config):
    image_size = int(model_config.get("image_size", 224))
    patch_size = int(model_config.get("patch_size", 14))
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    return VideoSAURPretrainer(
        model_name=model_config.get("dino_model", "facebook/dinov2-small"),
        num_slots=int(model_config.get("num_slots", 4)),
        slot_dim=int(model_config.get("slot_dim", 128)),
        feature_dim=int(model_config.get("feature_dim", 384)),
        num_patches=(image_size // patch_size) ** 2,
        feature_reconstruction_weight=float(
            loss_config.get("feature_reconstruction_weight", 1.0)
        ),
        time_similarity_weight=float(loss_config.get("time_similarity_weight", 0.25)),
        time_similarity_temperature=float(
            loss_config.get("time_similarity_temperature", 0.25)
        ),
    )
