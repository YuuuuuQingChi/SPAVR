from torch import nn

from spavr.backbone import CJepaBackbone
from spavr.heads import QuantileHead, pinball_loss


class RewardRiskModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        taus=(0.1, 0.5, 0.9),
        history_size=5,
        predicted_size=3,
        action_dim=7,
        proprio_dim=7,
        backbone=None,
        quantile_loss_weight=1.0,
        future_object_loss_weight=1.0,
        future_proprio_loss_weight=1.0,
        masked_history_loss_weight=0.0,
    ):
        super().__init__()
        self.backbone = backbone or CJepaBackbone(
            history_size=history_size,
            predicted_size=predicted_size,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
        )
        self.head = QuantileHead(
            self.backbone.slot_dim,
            self.backbone.num_objects,
            predicted_size,
            hidden_dim,
            taus,
        )
        self.loss_weights = {
            "loss_quantile": float(quantile_loss_weight),
            "loss_future_object": float(future_object_loss_weight),
            "loss_future_proprio": float(future_proprio_loss_weight),
            "loss_masked_history": float(masked_history_loss_weight),
        }

    def forward(self, x):
        """Score one batch -> quantiles ``(B, K)``.

        ``pixels`` and ``proprio`` contain state history. ``action`` contains the
        complete future sequence ``(B, L, 7)``. The backbone rolls
        over all L actions and returns the full rollout trace ``(B, R, P, S+2, D)``.
        """
        rollout_trace = self.backbone(x)
        return self.head(rollout_trace)

    def compute_loss(self, batch):
        """Compute only the losses enabled for the current training stage."""
        reference = next(self.parameters())
        zero = reference.sum() * 0.0

        quantiles = None
        loss_quantile = zero
        if self.loss_weights["loss_quantile"] > 0.0:
            if "return_to_go" not in batch:
                raise ValueError("quantile loss requires return_to_go")
            quantiles = self(batch)
            loss_quantile = pinball_loss(
                quantiles,
                batch["return_to_go"],
                self.head.taus,
            )

        consequence_names = (
            "loss_future_object",
            "loss_future_proprio",
            "loss_masked_history",
        )
        if any(self.loss_weights[name] > 0.0 for name in consequence_names):
            consequence_losses = self.backbone.consequence_losses(
                batch,
                # Random slot masking is a training augmentation. Validation
                # uses complete history so checkpoint scores are comparable
                # across epochs.
                mask_history=(
                    self.training
                    and self.loss_weights["loss_masked_history"] > 0.0
                ),
            )
        else:
            consequence_losses = {name: zero for name in consequence_names}

        losses = {"loss_quantile": loss_quantile, **consequence_losses}
        loss_total = sum(
            self.loss_weights[name] * value for name, value in losses.items()
        )
        return {
            "quantiles": quantiles,
            **losses,
            "loss_total": loss_total,
        }
