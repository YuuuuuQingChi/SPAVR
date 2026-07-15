import torch
import torch.nn.functional as F
from torch import nn

import stable_worldmodel as swm
from src.cjepa_predictor import MaskedSlot_AP_Predictor
from src.world_models.dinowm_causal_AP_node import CausalWM_AP


class CJepaBackbone(nn.Module):
    def __init__(
        self,
        num_objects=4,        # S: number of object slots (PushT)
        slot_dim=128,         # D
        history_size=5,
        num_preds=3,
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
        self.num_preds = num_preds
        self.num_objects = num_objects                # S
        self.num_slots = num_objects + 2              # S + 2 (objects + proprio + action)
        self.slot_dim = slot_dim                      # D

        predictor = MaskedSlot_AP_Predictor(
            num_slots=self.num_slots,
            slot_dim=slot_dim,
            history_frames=history_size,
            pred_frames=num_preds,
            num_masked_slots=num_masked_slots,
            seed=seed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        action_encoder = swm.wm.dinowm.Embedder(in_chans=frameskip * action_dim, emb_dim=slot_dim)
        proprio_encoder = swm.wm.dinowm.Embedder(in_chans=proprio_dim, emb_dim=slot_dim)

        # CausalWM_AP only consumes slots on this path; the visual front-end lives in model.py.
        self.world_model = CausalWM_AP(
            encoder=None,
            slot_attention=None,
            initializer=None,
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            history_size=self.history_size,
            num_pred=self.num_preds,
        )

    def build_embedding(self, slots, action, proprio):
    
        wm = self.world_model
        proprio_slot = wm.proprio_encoder(proprio.float()).unsqueeze(2)
        action_slot = wm.action_encoder(action.float()).unsqueeze(2)
        return torch.cat([slots, proprio_slot, action_slot], dim=2)

    def forward_repr(self, slots, action, proprio):
        history = self.build_embedding(slots, action, proprio)[:, :self.history_size]
        predictor = self.world_model.predictor
        saved = predictor.num_masked_slots
        predictor.num_masked_slots = 0
        try:
            out, _ = predictor(history)
        finally:
            predictor.num_masked_slots = saved
        return out[:, self.history_size - 1]

    def anchor_loss(self, slots, action, proprio):
      
        embedding = self.build_embedding(slots, action, proprio)
        h, p, s = self.history_size, self.num_preds, self.num_objects
        history = embedding[:, :h]
        target = embedding[:, h:h + p].detach()

        pred, mask_indices = self.world_model.predictor(history)
        pred_history = pred[:, :h]
        pred_future = pred[:, h:h + p]

        loss = F.mse_loss(pred_history[:, :, mask_indices], history[:, :, mask_indices].detach())
        loss = loss + F.mse_loss(pred_future[:, :, :s], target[:, :, :s])
        loss = loss + F.mse_loss(pred_future[:, :, s:s + 1], target[:, :, s:s + 1])
        return loss
