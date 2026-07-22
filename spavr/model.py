from torch import nn

from spavr.backbone import CJepaBackbone
from spavr.heads import QuantileHead


class RewardRiskModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        taus=(0.1, 0.5, 0.9),
        history_size=5,
        predicted_size=3,
        action_dim=2,
        proprio_dim=4,
        frameskip=3,
        backbone=None,
    ):
        super().__init__()
        self.backbone = backbone or CJepaBackbone(
            history_size=history_size,
            predicted_size=predicted_size,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            frameskip=frameskip,
        )
        self.head = QuantileHead(
            self.backbone.slot_dim,
            self.backbone.num_objects,
            predicted_size,
            hidden_dim,
            taus,
        )

    def forward(self, x):
        """Score one batch -> quantiles ``(B, K)``.

        ``pixels`` and ``proprio`` contain state history. ``action`` contains the
        complete future sequence ``(B, L, action_input_dim)``. The backbone rolls
        over all L actions and returns the full rollout trace ``(B, R, P, S+2, D)``.
        """
        rollout_trace = self.backbone(x)
        return self.head(rollout_trace)
