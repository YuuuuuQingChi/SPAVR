import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from third_party.cjepa.src.cjepa_predictor import MaskedSlot_AP_Predictor
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
from third_party.cjepa.src.third_party.videosaur.videosaur.modules.video import (
    LatentProcessor,
    MapOverTime,
    ScanOverTime,
)
from third_party.cjepa.src.world_models.dinowm_causal_AP_node import (
    CausalWM_AP,
    Embedder,
)


class CJepaBackbone(nn.Module):
    def __init__(
        self,
        num_objects=4,  # S: number of object slots (PushT)
        slot_dim=128,  # D
        history_size=5,
        predicted_size=3,
        action_dim=2,
        proprio_dim=4,
        frameskip=3,
        num_masked_slots=2,
        seed=42,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.history_size = history_size
        self.predicted_size = predicted_size
        self.action_input_dim = frameskip * action_dim
        self.num_objects = num_objects  # S
        self.num_slots = num_objects + 2  # S + 2 (objects + proprio + action)
        self.slot_dim = slot_dim  # D

        predictor = MaskedSlot_AP_Predictor(
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
        encoder = MapOverTime(
            FrameEncoder(
                backbone=AutoModel.from_pretrained("facebook/dinov2-small"),
                output_transform=MLP(
                    inp_dim=384,
                    outp_dim=slot_dim,
                    hidden_dims=[768],
                    initial_layer_norm=True,
                ),
            )
        )
        initializer = RandomInit(n_slots=num_objects, dim=slot_dim)
        processor = ScanOverTime(
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
        )

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

    def encode_inputs(self, batch, num_steps):
        """Build C-JEPA embeddings from raw pixels or pre-extracted object slots."""
        actions = batch["action"][:, :num_steps]
        proprio = batch["proprio"][:, :num_steps]

        if "object_slots" in batch:
            object_slots = batch["object_slots"][:, :num_steps].float()
            proprio_slots = self.world_model.proprio_encoder(proprio.float()).unsqueeze(
                2
            )
            action_slots = self.world_model.action_encoder(actions.float()).unsqueeze(2)
            return torch.cat([object_slots, proprio_slots, action_slots], dim=2)

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

    def anchor_loss(self, batch):
        """Compute the masked-history and future-state anchor loss."""
        h, p, s = self.history_size, self.predicted_size, self.num_objects
        embedding = self.encode_inputs(batch, h + p)
        history = embedding[:, :h]
        target = embedding[:, h : h + p].detach()

        pred, mask_indices = self.world_model.predict(history)
        pred_history = pred[:, :h]
        pred_future = pred[:, h : h + p]

        loss = F.mse_loss(
            pred_history[:, :, mask_indices], history[:, :, mask_indices].detach()
        )
        loss = loss + F.mse_loss(pred_future[:, :, :s], target[:, :, :s])
        loss = loss + F.mse_loss(pred_future[:, :, s : s + 1], target[:, :, s : s + 1])
        return loss
