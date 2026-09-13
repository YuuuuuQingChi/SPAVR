import sys

import torch
import torch.nn.functional as F
from torch import nn

# Register the vendored C-JEPA source package for its internal absolute imports.
from third_party.cjepa import src as _cjepa_src

sys.modules.setdefault("src", _cjepa_src)

from third_party.cjepa.src.cjepa_predictor import MaskedSlot_AP_Predictor
from third_party.cjepa.src.world_models.dinowm_causal_AP_node import (
    CausalWM_AP,
    Embedder,
)
from spavr.videosaur import build_object_encoder, load_videosaur_object_encoder


class RandomMaskedSlotAPPredictor(MaskedSlot_AP_Predictor):
    """C-JEPA predictor whose shared object mask changes on every forward pass.

    The upstream implementation recreates ``RandomState(seed)`` inside every
    call, which selects the same object slot forever. Global torch RNGs are
    already seeded by the training entry point, so ``randperm`` remains
    reproducible while advancing between batches.
    """

    def get_mask_indices(self, batch_size, device):
        del batch_size  # The upstream masking layout is shared across a batch.
        num_objects = self.num_slots - 2
        if not 0 <= self.num_masked_slots <= num_objects:
            raise ValueError(
                "num_masked_slots must be between 0 and the number of objects"
            )
        masked_indices = torch.randperm(num_objects, device=device)[
            : self.num_masked_slots
        ]
        is_slot_masked = torch.zeros(
            self.num_slots, dtype=torch.bool, device=device
        )
        is_slot_masked[masked_indices] = True
        return is_slot_masked, masked_indices


class CJepaBackbone(nn.Module):
    def __init__(
        self,
        num_objects=4,  # S: number of object slots
        slot_dim=128,  # D
        history_size=5,
        predicted_size=3,
        action_dim=7,
        proprio_dim=7,
        num_masked_slots=2,
        seed=42,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        visual_input_mode="pixels",
        videosaur_checkpoint=None,
        dino_model="facebook/dinov2-small",
    ):
        super().__init__()
        self.history_size = history_size
        self.predicted_size = predicted_size
        self.action_input_dim = action_dim
        self.num_objects = num_objects  # S
        self.num_slots = num_objects + 2  # S + 2 (objects + proprio + action)
        self.slot_dim = slot_dim  # D
        self.visual_input_mode = visual_input_mode
        if visual_input_mode not in {"pixels", "object_slots"}:
            raise ValueError("visual_input_mode must be 'pixels' or 'object_slots'")

        predictor = RandomMaskedSlotAPPredictor(
            num_slots=self.num_slots,
            slot_dim=slot_dim,
            history_frames=history_size,
            pred_frames=predicted_size,
            num_masked_slots=num_masked_slots,
            seed=seed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        if visual_input_mode == "pixels":
            if videosaur_checkpoint is None:
                raise ValueError(
                    "pixel mode requires a pretrained videosaur_checkpoint; "
                    "untrained Slot Attention is not a valid C-JEPA encoder"
                )
            object_encoder = build_object_encoder(
                model_name=dino_model,
                num_slots=num_objects,
                slot_dim=slot_dim,
            )
            load_videosaur_object_encoder(object_encoder, videosaur_checkpoint)
            object_encoder.requires_grad_(False)
            object_encoder.eval()
            encoder = object_encoder.encoder
            initializer = object_encoder.initializer
            processor = object_encoder.processor
        else:
            # These modules are bypassed by encode_inputs. Avoid loading DINOv2
            # during efficient training from pre-extracted object slots.
            encoder = nn.Identity()
            initializer = nn.Identity()
            processor = nn.Identity()

        self.world_model = CausalWM_AP(
            encoder=encoder,
            slot_attention=processor,
            initializer=initializer,
            predictor=predictor,
            action_encoder=Embedder(
                in_chans=self.action_input_dim,
                emb_dim=slot_dim,
            ),
            proprio_encoder=Embedder(in_chans=proprio_dim, emb_dim=slot_dim),
            history_size=self.history_size,
            num_pred=self.predicted_size,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.visual_input_mode == "pixels":
            self.world_model.encoder.eval()
            self.world_model.initializer.eval()
            self.world_model.slot_attention.eval()
        return self

    def encode_inputs(self, batch, num_steps):
        """Build C-JEPA embeddings from raw pixels or pre-extracted object slots."""
        actions = batch["action"][:, :num_steps]
        proprio = batch["proprio"][:, :num_steps]

        if "object_slots" in batch:
            object_slots = batch["object_slots"][:, :num_steps].float()
            if object_slots.shape[-2:] != (self.num_objects, self.slot_dim):
                raise ValueError(
                    "object_slots must end in "
                    f"({self.num_objects},{self.slot_dim}), got {tuple(object_slots.shape)}"
                )
            proprio_slots = self.world_model.proprio_encoder(proprio.float()).unsqueeze(
                2
            )
            action_slots = self.world_model.action_encoder(actions.float()).unsqueeze(2)
            return torch.cat([object_slots, proprio_slots, action_slots], dim=2)

        if self.visual_input_mode != "pixels":
            raise ValueError("this model was configured for pre-extracted object_slots")

        info = self.world_model.encode(
            {
                "pixels": batch["pixels"][:, :num_steps],
                "action": actions,
                "proprio": proprio,
            },
            pixels_key="pixels",
            proprio_key="proprio",
            action_key="action",
            target="embed",
        )
        return info["embed"]

    def forward(self, x):
        """Collect the full rollout trace ``(B, R, P, S+2, D)`` over all rounds.

        ``R = ceil((L-H)/P)+1`` predictor calls. Each round's raw prediction is
        collected before its action slot is overwritten for the next round.
        """
        h, p = self.history_size, self.predicted_size
        actions = x["action"]
        history = self.encode_inputs(x, h)

        predictor = self.world_model.predictor
        saved_num_masked_slots = predictor.num_masked_slots
        predictor.num_masked_slots = 0
        trace = []
        try:
            current_step = h
            while current_step < actions.shape[1]:
                future = self.world_model.predict(history)[0][:, h : h + p]
                trace.append(future)
                steps_this_round = min(p, actions.shape[1] - current_step)
                future = self.world_model.replace_action_in_embedding(
                    future[:, :steps_this_round].unsqueeze(1),
                    actions[
                        :, current_step : current_step + steps_this_round
                    ].unsqueeze(1),
                ).squeeze(1)
                history = torch.cat([history, future], dim=1)[:, -h:]
                current_step += steps_this_round

            trace.append(self.world_model.predict(history)[0][:, h : h + p])
        finally:
            predictor.num_masked_slots = saved_num_masked_slots
        return torch.stack(trace, dim=1)

    def consequence_losses(self, batch, mask_history=True):
        """Return separately logged C-JEPA representation losses.

        The first ``H`` states are the observed context.  The following ``P``
        states are used only as detached targets for the consequences predicted
        from that context and the candidate action plan.
        """
        h, p, s = self.history_size, self.predicted_size, self.num_objects
        embedding = self.encode_inputs(batch, h + p)
        history = embedding[:, :h]
        target = embedding[:, h : h + p].detach()

        predictor = self.world_model.predictor
        saved_num_masked_slots = predictor.num_masked_slots
        if not mask_history:
            predictor.num_masked_slots = 0
        try:
            pred, mask_indices = self.world_model.predict(history)
        finally:
            predictor.num_masked_slots = saved_num_masked_slots
        pred_history = pred[:, :h]
        pred_future = pred[:, h : h + p]

        if len(mask_indices):
            loss_masked_history = F.mse_loss(
                pred_history[:, :, mask_indices],
                history[:, :, mask_indices].detach(),
            )
        else:
            loss_masked_history = pred_history.sum() * 0.0

        return {
            "loss_future_object": F.mse_loss(
                pred_future[:, :, :s], target[:, :, :s]
            ),
            "loss_future_proprio": F.mse_loss(
                pred_future[:, :, s : s + 1], target[:, :, s : s + 1]
            ),
            "loss_masked_history": loss_masked_history,
        }

    def anchor_loss(self, batch):
        """Backward-compatible sum of the separately exposed losses."""
        return sum(self.consequence_losses(batch).values())
